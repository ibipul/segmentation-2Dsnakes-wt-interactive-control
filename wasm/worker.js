// Runs the Python framework in WebAssembly (Pyodide), off the page's main thread.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const FILES = ["seg_fwk.py", "schemes.py", "initialize_phi.py", "redistance.py", "curvature_term.py",
               "airplane.png", "zebra.bmp", "twoObj.bmp"];
const ROOT = new URL("../", import.meta.url);
const NSLOT = 8, MAXPT = 400;

let pyodide, ctl, lens, pts;

// Shared with the page (when cross-origin isolated): [cancel flag] + steering polygon slots.
function attach(sab) {
  if (!sab) return;
  ctl = new Int32Array(sab, 0, 4);
  lens = new Int32Array(sab, 16, NSLOT);
  pts = new Float32Array(sab, 64, NSLOT * MAXPT * 2);
}

globalThis.emitEvent = (json, png) => {
  const ev = JSON.parse(json);
  if (png) { const bytes = png.slice(); ev.png = bytes; postMessage(ev, [bytes.buffer]); }
  else postMessage(ev);
};
globalThis.isCancelled = () => !!ctl && Atomics.load(ctl, 0) === 1;
globalThis.pollSteer = () => {
  if (!lens) return "[]";
  const out = [];
  for (let i = 0; i < NSLOT; i++) {
    const n = Atomics.load(lens, i);
    if (n <= 0) continue;
    const poly = [];
    for (let k = 0; k < n; k++) poly.push([pts[(i * MAXPT + k) * 2], pts[(i * MAXPT + k) * 2 + 1]]);
    Atomics.store(lens, i, 0);
    out.push(poly);
  }
  return JSON.stringify(out);
};

async function init(sab) {
  attach(sab);
  const status = (text) => postMessage({ type: "loading", text });
  status("Starting the Python runtime…");
  pyodide = await loadPyodide({ stdout: () => {}, stderr: (s) => console.warn(s) });
  status("Loading NumPy, SciPy and Matplotlib (about 40 MB the first time, cached after)…");
  await pyodide.loadPackage(["numpy", "scipy", "matplotlib", "pillow"]);
  status("Loading the segmentation framework…");
  const files = await Promise.all([...FILES.map((f) => new URL(f, ROOT)), new URL("wasm/bridge.py", ROOT)]
    .map(async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: ${r.status}`); return [u, new Uint8Array(await r.arrayBuffer())]; }));
  pyodide.FS.mkdirTree("/home/pyodide/seg");
  for (const [u, data] of files) pyodide.FS.writeFile("/home/pyodide/seg/" + u.pathname.split("/").pop(), data);
  pyodide.FS.chdir("/home/pyodide/seg"); // the framework loads its images by bare file name
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide/seg')\nimport bridge");
  postMessage({ type: "ready" });
}

function run({ imname, algo, dt, polygon }) {
  if (ctl) Atomics.store(ctl, 0, 0);
  const fn = pyodide.globals.get("bridge").start_run;
  try { fn(imname, algo, dt, polygon ? JSON.stringify(polygon) : ""); }
  catch (e) { postMessage({ type: "error", message: String(e), traceback: String(e) }); postMessage({ type: "done", status: "error", elapsed: 0 }); }
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") await init(m.sab);
    else if (m.type === "run") run(m);
  } catch (err) { postMessage({ type: "fatal", message: String(err) }); }
};
