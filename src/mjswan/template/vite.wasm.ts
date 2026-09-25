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

/**
 * The onnxruntime-web entry every module imports. Not the package root: that resolves
 * to the JSEP build, whose WebGPU-carrying wasm (26.5 MiB) is over Cloudflare Pages'
 * 25 MiB per-file limit. Same API, CPU backend only.
 */
export const ORT_ENTRY = 'onnxruntime-web/wasm';

/** Wasm the builds emit from node_modules, as paths under it. */
export const UPSTREAM_WASM = [
  'onnxruntime-web/dist/ort-wasm-simd-threaded.wasm',
  'mujoco/mujoco.wasm',
  'mujoco/mt/mujoco.wasm',
] as const;

/**
 * The only runtime file ORT fetches: under the `import` condition `ORT_ENTRY` resolves to
 * `dist/ort.wasm.bundle.min.mjs`, which carries its `.mjs` loader inlined, so shipping
 * this wasm is enough. `__tests__/ortRuntimeFiles.test.ts` fails if either changes.
 */
export const ORT_WASM_FILE = 'ort-wasm-simd-threaded.wasm';

/** The entry `import * as ort from 'onnxruntime-web/wasm'` must keep resolving to. */
export const ORT_BUNDLED_ENTRY = './dist/ort.wasm.bundle.min.mjs';

/**
 * Cloudflare Pages' per-file limit. Asserted per source in `ortRuntimeFiles.test.ts`: a
 * wasm over it fails the deploy, or, if the build prunes it, deploys green and 404s.
 */
export const MAX_ASSET_BYTES = 25 * 1024 * 1024;

/**
 * Stands in for the emitted ORT wasm's path in `ortEnv.ts`: its content hash is only known
 * once the bundle exists, so `vite.lib.config.ts` rewrites it in `generateBundle`.
 */
export const ORT_WASM_TOKEN = '__MJSWAN_ORT_WASM__';
