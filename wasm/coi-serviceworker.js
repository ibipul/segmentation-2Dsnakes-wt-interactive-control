/* Cross-origin isolation for GitHub Pages, which cannot set response headers itself.
   Adds COOP/COEP headers to every response so the page gets SharedArrayBuffer; the demo uses it so the
   Python running in the worker can see Stop and steering regions while it computes.
   The first visit registers this worker and reloads once. If it can't register, the demo still runs
   (without live steering). */
if (typeof window === "undefined") {
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
  self.addEventListener("fetch", (e) => {
    const r = e.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;
    e.respondWith(
      fetch(r).then((resp) => {
        if (resp.status === 0) return resp;
        const h = new Headers(resp.headers);
        h.set("Cross-Origin-Embedder-Policy", "require-corp");
        h.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
      }).catch((err) => { console.error(err); return Response.error(); })
    );
  });
} else {
  (async () => {
    if (window.crossOriginIsolated || !window.isSecureContext || !("serviceWorker" in navigator)) return;
    try {
      const reg = await navigator.serviceWorker.register(document.currentScript.src);
      if (sessionStorage.getItem("coi-reloaded")) return; // never loop
      sessionStorage.setItem("coi-reloaded", "1");
      if (reg.active && !navigator.serviceWorker.controller) location.reload();
      else reg.addEventListener("updatefound", () => {
        const w = reg.installing;
        w && w.addEventListener("statechange", () => { if (w.state === "activated") location.reload(); });
      });
    } catch (e) { console.warn("cross-origin isolation unavailable:", e); }
  })();
}
