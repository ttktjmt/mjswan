/**
 * ONNX Runtime Web's environment, configured once for every module that creates a session.
 *
 * The library build points ORT at the runtime wasm it emits beside `dist/mjswan.js`, so a
 * host serving `dist/` from its own origin executes and fetches nothing it did not serve
 * (issue #123). The SPA build leaves the default alone — Vite resolves ORT's assets for it.
 */
import * as ort from 'onnxruntime-web';

ort.env.wasm.proxy = false;
// Load-bearing for `wasmPaths` below, not only a preference: off the bundle's own origin,
// ORT keeps its inlined loader only while the wasm is overridden *and* single-threaded.
// More threads would send it after the `.mjs` this package does not ship.
ort.env.wasm.numThreads = 1;

/**
 * The co-located wasm's name, content hash included, defined only by vite.lib.config.ts
 * (absent in the SPA, which lets Vite resolve ORT's own default reference instead).
 */
declare const __ORT_WASM_FILE__: string | undefined;

if (typeof __ORT_WASM_FILE__ !== 'undefined') {
  // Names the wasm rather than a prefix, and that distinction is the whole fix: given a
  // prefix ORT dynamic-imports `ort-wasm-simd-threaded.jsep.mjs` from it, while naming
  // only the wasm keeps ORT on the loader already inlined in `ort.bundle.min.mjs` — which
  // is why that `.mjs` need not ship.
  // Resolved against a variable rather than `import.meta.url` in place, so Vite reads it
  // as a runtime URL — wherever dist/ is served from — not a build-time asset reference.
  const bundleUrl = import.meta.url;
  ort.env.wasm.wasmPaths = { wasm: new URL(__ORT_WASM_FILE__, bundleUrl).href };
}
