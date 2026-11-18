/**
 * ONNX Runtime Web, loaded when the first session is made and configured once for every
 * module that creates one; a scene without a policy never downloads it.
 *
 * The library build points ORT at the runtime wasm it emits into `dist/assets/`, so a host
 * serving `dist/` from its own origin executes and fetches nothing it did not serve (issue
 * #123). The SPA build leaves the default alone: Vite resolves ORT's assets for it.
 *
 * Every import is `onnxruntime-web/wasm`, the CPU-only build: see `ORT_ENTRY` in vite.wasm.ts.
 * Elsewhere it is imported for its types only.
 */
import type * as Ort from 'onnxruntime-web/wasm';

/** The emitted wasm's hashed name, defined only by vite.lib.config.ts (absent in the SPA). */
declare const __ORT_WASM_FILE__: string | undefined;

let loaded: typeof Ort | null = null;
let loading: Promise<typeof Ort> | null = null;

/** ORT, imported and configured once. Awaited before any session is created. */
export function loadOrt(): Promise<typeof Ort> {
  loading ??= import('onnxruntime-web/wasm').then(
    (ort) => {
      ort.env.wasm.proxy = false;
      // Load-bearing for `wasmPaths` below, not a preference: ORT keeps its inlined loader
      // only while the wasm is overridden *and* single-threaded. More threads fetch the
      // `.mjs` the package does not ship.
      ort.env.wasm.numThreads = 1;
      if (typeof __ORT_WASM_FILE__ !== 'undefined') {
        // Names the wasm, never a prefix: given a prefix ORT dynamic-imports
        // `ort-wasm-simd-threaded.mjs` from it, while naming only the wasm keeps ORT on the
        // loader already inlined in `ort.wasm.bundle.min.mjs`, so that `.mjs` need not ship.
        // Resolved against a variable so Vite reads it as a runtime URL, not a build-time asset.
        const bundleUrl = import.meta.url;
        ort.env.wasm.wasmPaths = { wasm: new URL(__ORT_WASM_FILE__, bundleUrl).href };
      }
      loaded = ort;
      return ort;
    },
    (error: unknown) => {
      // A chunk that failed to arrive may arrive for the next policy.
      loading = null;
      throw error;
    },
  );
  return loading;
}

/**
 * ORT once `loadOrt()` has resolved, for what runs every step and must not wait: building
 * tensors. Every caller holds a session, which `loadOrt()` came before.
 */
export function getOrt(): typeof Ort {
  if (!loaded) throw new Error('ONNX Runtime was used before loadOrt() resolved.');
  return loaded;
}
