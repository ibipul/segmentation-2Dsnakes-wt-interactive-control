"""Browser front end for the segmentation framework.

Run from the repo root:  uvicorn web.server:app --port 8080

The framework is used exactly as the README shows:

    s = segmentation(imname=..., algo=..., dt=...)
    s.execute()

It was written for a desktop matplotlib window: it draws with pyplot, animates with plt.pause(),
and asks for a freehand region with a LassoSelector followed by a blocking plt.show(). This module
leaves the framework untouched and redirects those three calls to the browser instead:

  * plt.pause()      -> render the current figure to a PNG and stream it to the page
  * LassoSelector    -> remember the axes and callback the framework wants a region for
  * plt.show()       -> for the starting region: send the page a lasso request, wait for the polygon the
                        user draws, convert it to data coordinates and hand it to the framework's callback.
                        At the control algorithms' steering checkpoints (every 50 iterations here) it doesn't wait: it hands over any
                        regions drawn since the last checkpoint, or returns at once.
"""

import asyncio
import base64
import contextlib
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no desktop window; frames are rendered off screen and sent to the browser
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.widgets  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

IMAGES = {"twoObj": "twoObj.bmp", "zebra": "zebra.bmp", "airplane": "airplane.png"}
ALGOS = {
    "chanvesse": "Chan-Vese: separates mean intensity inside vs outside",
    "yezzi": "Yezzi: mean separation weighted by region area",
    "bhattacharya": "Bhattacharyya: separates the intensity histograms",
    "ctrl1": "Chan-Vese with interactive feedback control (steer any time; applied every 50 iterations)",
    "ctrl_bhattacharya": "Bhattacharyya with interactive feedback control (steer any time; applied every 50 iterations)",
}
# Button headings that differ from the algo name passed to the framework.
TITLES = {"ctrl1": "Control + chanvesse", "ctrl_bhattacharya": "Control + Bhattacharya"}

PANEL_INCHES = 4.5  # size of each subplot; the figure is resized to fit the framework's subplot grid
DPI = 100
MIN_FRAME_GAP = 0.12  # seconds; at most ~8 frames per second go to the browser


class Cancelled(Exception):
    """Raised inside the framework's thread (from plt.pause / plt.show) to stop a run."""


class Run:
    def __init__(self, imname: str, algo: str, dt: float):
        self.id = uuid.uuid4().hex
        self.imname, self.algo, self.dt = imname, algo, dt
        self.events: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()  # the starting-region polygon sent back by the page
        self.steering: queue.Queue = queue.Queue()  # steering polygons drawn any time, applied at the next checkpoint
        self.steer_ready = False
        self.cancel = threading.Event()
        self.pending_lasso = None
        self.lasso_count = 0
        self.frames = 0
        self.last_frame_at = 0.0
        self.started_at = time.time()
        self.thread = threading.Thread(target=self._work, name=f"seg-{self.id[:6]}", daemon=True)

    def emit(self, payload: dict):
        self.events.put(payload)

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()

    def emit_frame(self, fig, force=False):
        now = time.time()
        if not force and now - self.last_frame_at < MIN_FRAME_GAP:
            return None
        self.last_frame_at = now
        self.frames += 1
        frame = render(fig)
        self.emit({"type": "frame", "n": self.frames, "png": frame["png"],
                   "width": frame["width"], "height": frame["height"], "elapsed": round(now - self.started_at, 1)})
        return frame

    def _work(self):
        status = "finished"
        plt.close("all")
        self.emit({"type": "log", "text": f">>> s = segmentation(imname={self.imname!r}, algo={self.algo!r}, dt={self.dt})"})
        try:
            with contextlib.redirect_stdout(_LogWriter(self)):
                s = segmentation(imname=self.imname, algo=self.algo, dt=self.dt)
                self.emit({"type": "log", "text": ">>> s.execute()"})
                s.execute()
            if plt.get_fignums():
                self.emit_frame(plt.gcf(), force=True)
        except Cancelled:
            status = "stopped"
        except Exception as exc:  # show framework errors in the page instead of a dead stream
            status = "error"
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()})
        finally:
            plt.close("all")
            self.emit({"type": "done", "status": status, "elapsed": round(time.time() - self.started_at, 1)})


class _LogWriter(io.TextIOBase):
    """Sends the framework's print() lines to the page, and still echoes them in the terminal."""

    def __init__(self, run: Run):
        self.run, self.buf = run, ""

    def write(self, s):
        sys.__stdout__.write(s)
        # redirect_stdout is process-wide; only the run's own thread is the framework talking.
        if threading.current_thread() is not self.run.thread:
            return len(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.run.emit({"type": "log", "text": line})
        return len(s)


def render(fig) -> dict:
    """Draws the figure and returns it as a PNG cropped to the subplots that show something.

    Also returns what's needed to map a click on that PNG back to matplotlib display coordinates.
    """
    axes = fig.get_axes()
    if axes and axes[0].get_subplotspec() is not None:
        rows, cols = axes[0].get_subplotspec().get_gridspec().get_geometry()
        fig.set_size_inches(cols * PANEL_INCHES, rows * PANEL_INCHES)
    fig.set_dpi(DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    height, width = rgba.shape[:2]

    # Crop to the panels with content (e.g. the start-region figure only uses the left half of a 1x2 grid).
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
    return {"png": base64.b64encode(out.getvalue()).decode(), "width": x1 - x0, "height": bottom - top,
            "x0": x0, "top": top, "fig_height": height}


class _BrowserLasso:
    """Stands in for matplotlib's LassoSelector: the region is drawn in the browser instead."""

    def __init__(self, ax, onselect=None, *args, **kwargs):
        self.ax, self.onselect = ax, onselect
        current().pending_lasso = self


def _pause(interval):
    run = current()
    run.check_cancel()
    if plt.get_fignums():
        run.emit_frame(plt.gcf())


def _show(*args, **kwargs):
    run = current()
    run.check_cancel()
    lasso, run.pending_lasso = run.pending_lasso, None
    if lasso is None:
        return
    first = run.lasso_count == 0
    run.lasso_count += 1
    frame = run.emit_frame(lasso.ax.figure, force=True)
    # Where the target axes sit inside the PNG, so the page can outline the area to draw in.
    box = lasso.ax.get_window_extent()
    area = [round(v, 1) for v in (box.x0 - frame["x0"], frame["fig_height"] - box.y1 - frame["top"],
                                  box.x1 - frame["x0"], frame["fig_height"] - box.y0 - frame["top"])]

    if not first:
        # Steering checkpoint (every seg_w_control._steer_every iterations). Don't wait for the user:
        # hand over whatever regions they drew since the last checkpoint, or carry on, like closing the window.
        if not run.steer_ready:
            run.steer_ready = True
            run.emit({"type": "steer_ready", "area": area})
        applied = 0
        while True:
            try:
                points = run.steering.get_nowait()
            except queue.Empty:
                break
            _apply(lasso, frame, points)
            applied += 1
        if applied:
            run.emit({"type": "steer_applied", "count": applied})
            run.emit({"type": "log", "text": f"(applied {applied} steering region{'s' if applied > 1 else ''})"})
        return

    run.emit({"type": "lasso", "n": run.frames, "area": area, "prompt": "Draw the starting region"})
    while True:
        run.check_cancel()
        try:
            points = run.replies.get(timeout=0.25)
            break
        except queue.Empty:
            continue
    _apply(lasso, frame, points)


def _apply(lasso, frame, points):
    # PNG pixels (origin top-left, cropped) -> matplotlib display pixels (origin bottom-left) -> data coords.
    # Steering regions are drawn on an earlier frame; the panel layout is the same in every frame of a run.
    display = [(x + frame["x0"], frame["fig_height"] - (y + frame["top"])) for x, y in points]
    verts = lasso.ax.transData.inverted().transform(display)
    lasso.onselect([tuple(v) for v in verts])


# Installed before the framework is imported, so its `from matplotlib.widgets import LassoSelector` picks this up.
matplotlib.widgets.LassoSelector = _BrowserLasso
plt.show = _show
plt.pause = _pause

# The framework loads its images by bare file name, so it has to run from the repo root.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from seg_fwk import segmentation  # noqa: E402
import schemes  # noqa: E402

# In the page steering doesn't pause the run, so check for new regions more often than the desktop default (200).
# Progress prints stay at the default (seg_w_control._log_every).
schemes.seg_w_control._steer_every = 50

_lock = threading.Lock()
_run: Run | None = None


def current() -> Run:
    if _run is None:
        raise Cancelled()
    return _run


app = FastAPI(title="Level-set segmentation")


class RunRequest(BaseModel):
    imname: str
    algo: str
    dt: float = 0.5


class LassoReply(BaseModel):
    run_id: str
    points: list[list[float]]


class StopRequest(BaseModel):
    run_id: str


def _line(payload: dict) -> bytes:
    return (json.dumps(payload) + "\n").encode()


@app.get("/")
async def index():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.get("/api/options")
async def options():
    images = []
    for name, file in IMAGES.items():
        with Image.open(ROOT / file) as im:
            out = io.BytesIO()
            im.convert("L").save(out, format="PNG")
            images.append({"name": name, "file": file, "width": im.width, "height": im.height,
                           "png": base64.b64encode(out.getvalue()).decode()})
    return JSONResponse({"images": images, "algos": [{"name": k, "title": TITLES.get(k, k), "label": v}
                                                     for k, v in ALGOS.items()]})


@app.post("/api/run")
async def start(req: RunRequest):
    global _run
    if req.imname not in IMAGES:
        raise HTTPException(400, f"Unknown image {req.imname!r}")
    if req.algo not in ALGOS:
        raise HTTPException(400, f"Unknown algorithm {req.algo!r}")
    if not 0 < req.dt <= 5:
        raise HTTPException(400, "dt must be between 0 and 5")

    with _lock:
        # pyplot is global state, so only one run at a time: stop the previous one first.
        if _run is not None and _run.thread.is_alive():
            _run.cancel.set()
            await asyncio.to_thread(_run.thread.join, 30)
            if _run.thread.is_alive():
                raise HTTPException(409, "The previous run is still stopping. Try again in a moment.")
        run = _run = Run(req.imname, req.algo, req.dt)
        run.thread.start()

    async def stream():
        yield _line({"type": "run", "run_id": run.id, "imname": run.imname, "algo": run.algo, "dt": run.dt})
        try:
            while True:
                try:
                    event = await asyncio.to_thread(run.events.get, True, 0.5)
                except queue.Empty:
                    continue
                yield _line(event)
                if event["type"] == "done":
                    break
        finally:
            run.cancel.set()  # page closed or reloaded: stop the run instead of computing for nobody

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/lasso")
async def lasso(reply: LassoReply):
    if _run is None or _run.id != reply.run_id:
        raise HTTPException(404, "That run is no longer active.")
    if len(reply.points) < 3:
        raise HTTPException(400, "A region needs at least 3 points.")
    _run.replies.put(reply.points)
    return {"ok": True}


@app.post("/api/steer")
async def steer(reply: LassoReply):
    if _run is None or _run.id != reply.run_id or not _run.steer_ready:
        raise HTTPException(404, "That run is no longer taking steering regions.")
    if len(reply.points) < 3:
        raise HTTPException(400, "A region needs at least 3 points.")
    _run.steering.put(reply.points)
    return {"queued": _run.steering.qsize()}


@app.post("/api/stop")
async def stop(req: StopRequest):
    if _run is not None and _run.id == req.run_id:
        _run.cancel.set()
    return {"ok": True}
