/**
 * The WebAssembly both Vite builds co-locate, and what the library build assumes about
 * onnxruntime-web. Its own module so `vite.lib.config.ts` can name these files without
 * importing `src/core/onnx/` (and `onnxruntime-web` with it). The runtime half is
 * `src/core/onnx/ortEnv.ts`.
 *
 * The builds write into one `dist/` and must agree on these paths or the same bytes ship
 * twice. Both name each file `<source basename>-<content hash>.wasm` under `dist/assets/`:
 * the SPA by Vite's default asset naming, the library build by matching each de-inlined
 * asset's digest against the sources below. Identical bytes hash the same either way.
 */

/** Wasm the builds emit from node_modules, as paths under it. */
export const UPSTREAM_WASM = [
  'onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm',
  'mujoco/mujoco.wasm',
  'mujoco/mt/mujoco.wasm',
] as const;

/**
 * The only runtime file ORT fetches: under the `import` condition `onnxruntime-web`
 * resolves to `dist/ort.bundle.min.mjs`, which carries its `.mjs` loader inlined, so
 * shipping this wasm is enough. `__tests__/ortRuntimeFiles.test.ts` fails if either changes.
 */
export const ORT_WASM_FILE = 'ort-wasm-simd-threaded.jsep.wasm';

/** The entry `import * as ort from 'onnxruntime-web'` must keep resolving to. */
export const ORT_BUNDLED_ENTRY = './dist/ort.bundle.min.mjs';

/**
 * Stands in for the emitted ORT wasm's path in `ortEnv.ts`: its content hash is only known
 * once the bundle exists, so `vite.lib.config.ts` rewrites it in `generateBundle`.
 */
export const ORT_WASM_TOKEN = '__MJSWAN_ORT_WASM__';
