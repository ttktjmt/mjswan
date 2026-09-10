/**
 * ONNX Runtime Web's environment, configured once for every module that creates a session.
 *
 * The library build points ORT at the runtime wasm it emits into `dist/assets/`, so a host
 * serving `dist/` from its own origin executes and fetches nothing it did not serve (issue
 * #123). The SPA build leaves the default alone: Vite resolves ORT's assets for it.
 */
import * as ort from 'onnxruntime-web';

ort.env.wasm.proxy = false;
// Load-bearing for `wasmPaths` below, not a preference: ORT keeps its inlined loader only
// while the wasm is overridden *and* single-threaded. More threads fetch the `.mjs` the
// package does not ship.
ort.env.wasm.numThreads = 1;

/** The emitted wasm's hashed name, defined only by vite.lib.config.ts (absent in the SPA). */
declare const __ORT_WASM_FILE__: string | undefined;

if (typeof __ORT_WASM_FILE__ !== 'undefined') {
  // Names the wasm, never a prefix: given a prefix ORT dynamic-imports
  // `ort-wasm-simd-threaded.jsep.mjs` from it, while naming only the wasm keeps ORT on the
  // loader already inlined in `ort.bundle.min.mjs`, so that `.mjs` need not ship.
  // Resolved against a variable so Vite reads it as a runtime URL, not a build-time asset.
  const bundleUrl = import.meta.url;
  ort.env.wasm.wasmPaths = { wasm: new URL(__ORT_WASM_FILE__, bundleUrl).href };
}
