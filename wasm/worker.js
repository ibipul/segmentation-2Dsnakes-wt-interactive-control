// Runs the Python framework in WebAssembly (Pyodide), off the page's main thread.
// Nothing here lists the framework's files, images or algorithms: modules are found by following the
// `import` lines from seg_fwk.py, images and algorithms come from the registry in seg_fwk.py, and the
// Python packages come from the imports of those files. Developing the framework needs no edit here.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const ROOT = new URL("../", import.meta.url);
const NSLOT = 8, MAXPT = 400;
const HOME = "/home/pyodide/seg";

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

async function fetchBytes(url) {
  const r = await fetch(url, { cache: "no-cache" }); // revalidate, so edits show up on a normal refresh
  if (!r.ok) return null;
  return new Uint8Array(await r.arrayBuffer());
}

// Every repo-root module the framework needs: start at seg_fwk.py and follow `import x` / `from x import ...`.
// A name with no x.py at the repo root (numpy, matplotlib, ...) is a package, not one of ours.
async function collectModules() {
  const found = new Map(), seen = new Set(), queue = ["seg_fwk"];
  while (queue.length) {
    const name = queue.shift();
    if (seen.has(name)) continue;
    seen.add(name);
    const bytes = await fetchBytes(new URL(name + ".py", ROOT));
    if (!bytes) continue;
    found.set(name + ".py", bytes);
    const src = new TextDecoder().decode(bytes);
    for (const m of src.matchAll(/^[ \t]*(?:from[ \t]+([A-Za-z_]\w*)[^\n]*\bimport\b|import[ \t]+([^\n#]+))/gm)) {
      if (m[1]) queue.push(m[1]);
      else for (const part of m[2].split(",")) queue.push(part.trim().split(/\s+/)[0].split(".")[0]);
    }
  }
  return found;
}

async function init(sab) {
  attach(sab);
  const status = (text) => postMessage({ type: "loading", text });
  status("Starting the Python runtime…");
  pyodide = await loadPyodide({ stdout: () => {}, stderr: (s) => console.warn(s) });

  status("Loading the segmentation framework…");
  const modules = await collectModules();
  const bridge = await fetchBytes(new URL("wasm/bridge.py", ROOT));
  modules.set("bridge.py", bridge);
  pyodide.FS.mkdirTree(HOME);
  const sources = [];
  for (const [file, data] of modules) { pyodide.FS.writeFile(`${HOME}/${file}`, data); sources.push(new TextDecoder().decode(data)); }
  pyodide.FS.chdir(HOME); // the framework loads its images by bare file name

  status("Loading NumPy, SciPy and Matplotlib (about 40 MB the first time, cached after)…");
  await pyodide.loadPackagesFromImports(sources.join("\n")); // numpy, scipy, matplotlib, pillow, ... as the code imports them

  pyodide.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(HOME)})\nimport bridge`);
  const options = JSON.parse(pyodide.globals.get("bridge").options_json());
  for (const im of options.images) {
    const data = await fetchBytes(new URL(im.file, ROOT));
    if (!data) throw new Error(`Image ${im.file} listed in seg_fwk.py was not found`);
    pyodide.FS.writeFile(`${HOME}/${im.file}`, data);
  }
  postMessage({ type: "ready", options });
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
