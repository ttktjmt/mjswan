/**
 * What the library build assumes about onnxruntime-web's runtime files.
 *
 * Its own module so `vite.lib.config.ts` can name the file without importing
 * `src/core/onnx/` (and `onnxruntime-web` with it), and so a test can pin the assumptions
 * without loading the Vite config. The runtime half is `src/core/onnx/ortEnv.ts`.
 */

/**
 * The only runtime file ORT fetches. Under the `import` condition `onnxruntime-web`
 * resolves to `dist/ort.bundle.min.mjs`, the build carrying its `.mjs` loader inlined, so
 * co-locating this wasm beside `dist/mjswan.js` is enough — see
 * `src/core/onnx/__tests__/ortRuntimeFiles.test.ts`, which fails if either changes.
 */
export const ORT_WASM_FILE = 'ort-wasm-simd-threaded.jsep.wasm';

/** The entry `import * as ort from 'onnxruntime-web'` must keep resolving to. */
export const ORT_BUNDLED_ENTRY = './dist/ort.bundle.min.mjs';
