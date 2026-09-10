/**
 * The WebAssembly both Vite builds co-locate, and what the library build assumes about
 * onnxruntime-web.
 *
 * Its own module so `vite.lib.config.ts` can name these files without importing
 * `src/core/onnx/` (and `onnxruntime-web` with it), and so a test can pin the assumptions
 * without loading a Vite config. The runtime half is `src/core/onnx/ortEnv.ts`.
 *
 * The two builds write into one `dist/`, so they have to agree on these paths or the same
 * bytes ship twice. Both name each file `<source basename>-<content hash>.wasm`, flat in
 * `dist/`: the SPA through `assetFileNames` (vite.config.ts), the library build by matching
 * each de-inlined asset's digest against the sources below. Identical bytes hash the same
 * in either build, so the two land on one file.
 */

/** Wasm the builds emit from node_modules, as paths under it. */
export const UPSTREAM_WASM = [
  'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm',
  'mujoco/mujoco.wasm',
  'mujoco/mt/mujoco.wasm',
] as const;

/**
 * The only runtime file ORT fetches. Under the `import` condition `onnxruntime-web`
 * resolves to `dist/ort.bundle.min.mjs`, the build carrying its `.mjs` loader inlined, so
 * co-locating this wasm beside `dist/mjswan.js` is enough — see
 * `src/core/onnx/__tests__/ortRuntimeFiles.test.ts`, which fails if either changes.
 */
export const ORT_WASM_FILE = 'ort-wasm-simd-threaded.jsep.wasm';

/** The entry `import * as ort from 'onnxruntime-web'` must keep resolving to. */
export const ORT_BUNDLED_ENTRY = './dist/ort.bundle.min.mjs';

/**
 * Stands in for the emitted ORT wasm's name in `ortEnv.ts`, which needs it in source
 * while the content hash is only known once the bundle is generated. `vite.lib.config.ts`
 * defines it as `__ORT_WASM_FILE__` and rewrites it in place afterwards.
 */
export const ORT_WASM_TOKEN = '__MJSWAN_ORT_WASM__';
