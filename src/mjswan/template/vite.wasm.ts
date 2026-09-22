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
 * The onnxruntime-web entry every module imports: the `./wasm` subpath, not the package
 * root. The root resolves to the JSEP build, whose wasm carries the WebGPU kernels and
 * weighs 26.5 MiB: over Cloudflare Pages' 25 MiB per-file limit, and twice this one for a
 * GPU path these policies do not want (see `src/core/onnx/session.ts`). Same API, CPU
 * backend only.
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
 * Cloudflare Pages refuses any file above this, and a host that takes one still makes
 * every visitor download it. Asserted per source in `__tests__/ortRuntimeFiles.test.ts`,
 * because an upgrade that crosses it deploys green and 404s the wasm at runtime.
 */
export const MAX_ASSET_BYTES = 25 * 1024 * 1024;

/**
 * Stands in for the emitted ORT wasm's path in `ortEnv.ts`: its content hash is only known
 * once the bundle exists, so `vite.lib.config.ts` rewrites it in `generateBundle`.
 */
export const ORT_WASM_TOKEN = '__MJSWAN_ORT_WASM__';
