"""Browser (WebAssembly) front end for the segmentation framework.

This is the in-browser counterpart of web/server.py. It runs inside Pyodide (CPython compiled to
WebAssembly) in a Web Worker, so the whole demo works without a server.

The framework is used exactly as the README shows and is left untouched:

    s = segmentation(imname=..., algo=..., dt=...)
    s.execute()

It was written for a desktop matplotlib window: it draws with pyplot, animates with plt.pause(), and
asks for a freehand region with a LassoSelector followed by plt.show(). As in web/server.py, those
three calls are redirected to the page instead:

  * plt.pause()      -> render the current figure to a PNG and send it to the page
  * LassoSelector    -> remember the axes and callback the framework wants a region for
  * plt.show()       -> the starting region comes with the run request: the first call raises a stop
                        so the page can ask for it, and the run is started again with the polygon.
                        At the control algorithms' steering checkpoints (every 50 iterations) it hands
                        over any regions drawn since the last checkpoint, or returns at once.

The worker (worker.js) provides these globals, which are the only things this module needs from JS:
  emitEvent(json, png_bytes_or_null)   send an event to the page
  isCancelled() -> bool                has the page pressed Stop?
  pollSteer()   -> str                 JSON list of steering polygons drawn since the last call
"""

import contextlib
import io
import json
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # frames are rendered off screen and sent to the page
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.widgets  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import js  # noqa: E402  (Pyodide's bridge to the worker's JavaScript)
from pyodide.ffi import to_js  # noqa: E402

PANEL_INCHES = 4.5  # size of each subplot; the figure is resized to fit the framework's subplot grid
DPI = 100
MIN_FRAME_GAP = 0.12  # seconds between frames sent to the page


class Cancelled(Exception):
    """Raised inside the framework (from plt.pause / plt.show) to stop a run."""


class NeedRegion(Cancelled):
    """The framework asked for the starting region and the run has none yet."""


class Run:
    def __init__(self, imname, algo, dt, polygon):
        self.imname, self.algo, self.dt = imname, algo, dt
        self.polygon = polygon  # starting region in PNG pixels, or None on the first pass
        self.pending_lasso = None
        self.lasso_count = 0
        self.steer_ready = False
        self.frames = 0
        self.last_frame_at = 0.0
        self.started_at = time.time()


_run = None


def emit(payload, png=None):
    js.emitEvent(json.dumps(payload), to_js(png) if png is not None else None)


def current():
    if _run is None:
        raise Cancelled()
    return _run


def check_cancel():
    if js.isCancelled():
        raise Cancelled()


def render(fig):
    """Draws the figure; returns the PNG cropped to the subplots that show something, plus what's
    needed to map a point on that PNG back to matplotlib display coordinates."""
    axes = fig.get_axes()
    if axes and axes[0].get_subplotspec() is not None:
        rows, cols = axes[0].get_subplotspec().get_gridspec().get_geometry()
        fig.set_size_inches(cols * PANEL_INCHES, rows * PANEL_INCHES)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    height, width = rgba.shape[:2]

    # Crop to the panels with content (the start-region figure only uses the left half of a 1x2 grid).
    boxes = [ax.get_tightbbox(renderer) for ax in axes if ax.has_data()]
    if boxes:
        pad = 6
        x0 = max(int(min(b.x0 for b in boxes)) - pad, 0)
        x1 = min(int(max(b.x1 for b in boxes)) + pad, width)
        top = max(int(height - max(b.y1 for b in boxes)) - pad, 0)
        bottom = min(int(height - min(b.y0 for b in boxes)) + pad, height)
    else:
        x0, x1, top, bottom = 0, width, 0, height
    out = io.BytesIO()
    Image.fromarray(rgba[top:bottom, x0:x1]).convert("RGB").save(out, format="PNG", optimize=False)
    return {"png": out.getvalue(), "width": x1 - x0, "height": bottom - top,
            "x0": x0, "top": top, "fig_height": height}


def emit_frame(run, fig, force=False):
    now = time.time()
    if not force and now - run.last_frame_at < MIN_FRAME_GAP:
        return None
    run.last_frame_at = now
    run.frames += 1
    frame = render(fig)
    emit({"type": "frame", "n": run.frames, "width": frame["width"], "height": frame["height"],
          "elapsed": round(now - run.started_at, 1)}, frame["png"])
    return frame


class _BrowserLasso:
    """Stands in for matplotlib's LassoSelector: the region is drawn in the page instead."""

    def __init__(self, ax, onselect=None, *args, **kwargs):
        self.ax, self.onselect = ax, onselect
        current().pending_lasso = self


def _pause(interval=0):
    run = current()
    check_cancel()
    if plt.get_fignums():
        emit_frame(run, plt.gcf())


def _apply(lasso, frame, points):
    # PNG pixels (origin top-left, cropped) -> matplotlib display pixels (origin bottom-left) -> data coords.
    # Steering regions are drawn on an earlier frame; the panel layout is the same in every frame of a run.
    display = [(x + frame["x0"], frame["fig_height"] - (y + frame["top"])) for x, y in points]
    verts = lasso.ax.transData.inverted().transform(display)
    lasso.onselect([tuple(v) for v in verts])


def _show(*args, **kwargs):
    run = current()
    check_cancel()
    lasso, run.pending_lasso = run.pending_lasso, None
    if lasso is None:
        return
    first = run.lasso_count == 0
    run.lasso_count += 1
    frame = emit_frame(run, lasso.ax.figure, force=True)
    # Where the target axes sit inside the PNG, so the page can outline the area to draw in.
    box = lasso.ax.get_window_extent()
    area = [round(float(v), 1) for v in (box.x0 - frame["x0"], frame["fig_height"] - box.y1 - frame["top"],
                                          box.x1 - frame["x0"], frame["fig_height"] - box.y0 - frame["top"])]

    if first:
        if run.polygon is None:
            emit({"type": "lasso", "area": area, "prompt": "Draw the starting region"})
            raise NeedRegion()
        _apply(lasso, frame, run.polygon)
        return

    # Steering checkpoint (every seg_w_control._steer_every iterations). Don't wait for the user: hand over
    # whatever regions they drew since the last checkpoint, or carry on, like closing the window.
    if not run.steer_ready:
        run.steer_ready = True
        emit({"type": "steer_ready", "area": area})
    polygons = json.loads(js.pollSteer() or "[]")
    for points in polygons:
        _apply(lasso, frame, points)
    if polygons:
        n = len(polygons)
        emit({"type": "steer_applied", "count": n})
        emit({"type": "log", "text": f"(applied {n} steering region{'s' if n > 1 else ''})"})


class _LogWriter(io.TextIOBase):
    """Sends the framework's print() lines to the page."""

    def __init__(self):
        self.buf = ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            emit({"type": "log", "text": line})
        return len(s)


# Installed before the framework is imported, so its `from matplotlib.widgets import LassoSelector` picks this up.
matplotlib.widgets.LassoSelector = _BrowserLasso
plt.show = _show
plt.pause = _pause

from seg_fwk import segmentation  # noqa: E402
import schemes  # noqa: E402

# In the page steering doesn't pause the run, so check for new regions more often than the desktop default (200).
schemes.seg_w_control._steer_every = 50


def start_run(imname, algo, dt, polygon_json=""):
    """Runs the framework. Ends with a `done` event whose status is finished, stopped, error, or
    needs_region (the page must send the starting region and call this again with it)."""
    global _run
    polygon = json.loads(polygon_json) if polygon_json else None
    _run = Run(imname, algo, float(dt), polygon)
    status = "finished"
    plt.close("all")
    if polygon is None:
        emit({"type": "log", "text": f">>> s = segmentation(imname={imname!r}, algo={algo!r}, dt={_run.dt})"})
        emit({"type": "log", "text": ">>> s.execute()"})
    try:
        with contextlib.redirect_stdout(_LogWriter()):
            s = segmentation(imname=imname, algo=algo, dt=_run.dt)
            s.execute()
        if plt.get_fignums():
            emit_frame(_run, plt.gcf(), force=True)
    except NeedRegion:
        status = "needs_region"
    except Cancelled:
        status = "stopped"
    except Exception as exc:  # show framework errors in the page instead of a dead worker
        status = "error"
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
    finally:
        plt.close("all")
        emit({"type": "done", "status": status, "elapsed": round(time.time() - _run.started_at, 1)})
