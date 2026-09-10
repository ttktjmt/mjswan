import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';
import { createHash } from 'crypto';
import path from 'path';
import fs from 'fs';
import { minify } from './vite.shared';
import { ORT_WASM_FILE, ORT_WASM_TOKEN, UPSTREAM_WASM } from './vite.wasm';

// Library build: emits a single self-contained ESM (`dist/mjswan.js`) exposing
// `createEngine(element, options?)` (the headless engine; no React/Mantine),
// with every dependency bundled and the MuJoCo / ONNX WASM in `dist/assets/`
// beside the SPA build's, resolved relative to the bundle wherever it is served
// from. See src/engine/ and docs/adr/0004-headless-engine-core.md.

function getVersionFromPython(): string {
  const initPath = path.resolve(__dirname, '../__init__.py');
  try {
    const content = fs.readFileSync(initPath, 'utf-8');
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) {
      return match[1];
    }
  } catch {
    // Fall through to package.json.
  }
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pkg = require('./package.json');
  return pkg.version || '0.0.0';
}

// Where every build puts what only it references; the entry alone sits at the root.
const ASSETS_DIR = 'assets';

const sha256 = (source: Uint8Array | string): string =>
  createHash('sha256').update(source).digest('hex');

// A specifier for `target` (a dist-relative path) as written from `fromDir`: the entry sits
// at the root and everything else under assets/, so the depth differs per referrer.
function specifierFromDir(fromDir: string, target: string): string {
  const relative = path.posix.relative(fromDir, target);
  return relative.startsWith('.') ? relative : `./${relative}`;
}
const specifierFrom = (fromFile: string, target: string): string =>
  specifierFromDir(path.posix.dirname(fromFile), target);

// digest → emitted name: de-inlined assets arrive as bytes with the source file name long
// gone, so content is all there is to match the SPA build's naming on. See vite.wasm.ts.
function upstreamWasmNames(): Map<string, string> {
  const names = new Map<string, string>();
  for (const source of UPSTREAM_WASM) {
    const file = path.resolve(__dirname, 'node_modules', source);
    names.set(sha256(fs.readFileSync(file)), path.basename(file));
  }
  return names;
}

// Vite library mode force-inlines `new URL('x.wasm', import.meta.url)` as base64
// `data:` URLs (ignoring assetsInlineLimit), bloating the bundle to ~70 MB. This
// extracts each back to a co-located dist/ file resolved via `import.meta.url` so
// it loads relative to the bundle. See mjswan-cloud ADR 0001.
//
// Each extracted file keeps its upstream name, so the SPA build's copy of the same bytes
// lands on one path. ORT's is what src/core/onnx/ortEnv.ts points `wasmPaths` at, so its
// emitted name is written back into the code here, where the content hash is known.
function extractInlinedWasmPlugin(): Plugin {
  const B64 = '([A-Za-z0-9+/=]+)';
  // Two inlined shapes, each with a base arg re-checked below: normally quoted
  // (`"`/`'`/backtick, backreferenced), and — inside a dormant pthread-worker
  // string — escaped `\"data:...\"` (keep the escaping so the literal stays valid).
  const QUOTED = new RegExp(
    `new URL\\(\\s*(["'\`])data:application/wasm;base64,${B64}\\1\\s*,\\s*([^)]*)\\)`,
    'g'
  );
  const ESCAPED = new RegExp(`new URL\\(\\s*\\\\"data:application/wasm;base64,${B64}\\\\"\\s*,\\s*[^)]*\\)`, 'g');

  return {
    name: 'mjswan-extract-wasm',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      const upstream = upstreamWasmNames();
      const emitted = new Map<string, string>(); // base64 → emitted fileName
      let ortWasm: string | undefined; // emitted fileName of ORT_WASM_FILE
      const fileFor = (b64: string): string => {
        let fileName = emitted.get(b64);
        if (!fileName) {
          const source = Buffer.from(b64, 'base64');
          // Anything the SPA build does not also emit is ours, and named as such.
          const name = upstream.get(sha256(source)) ?? 'mjswan-engine.wasm';
          fileName = this.getFileName(this.emitFile({ type: 'asset', name, source }));
          emitted.set(b64, fileName);
          if (name === ORT_WASM_FILE) ortWasm = fileName;
        }
        return fileName;
      };
      const deinline = (code: string, fileName: string): string =>
        code
          .replace(
            QUOTED,
            (m, _quote: string, b64: string, base: string) => {
              // Only de-inline main-thread wasm (base `import.meta.url`). A
              // `self.location.href` base means a classic Blob worker (Spark's
              // Splat sort) where `import.meta` is a syntax error — leave inline.
              if (!/import\.meta\.url/.test(base)) return m;
              const specifier = specifierFrom(fileName, fileFor(b64));
              return `new URL(${JSON.stringify(specifier)}, import.meta.url)`;
            }
          )
          .replace(
            ESCAPED,
            (m, b64: string) => {
              // Keep small active-worker wasm (Spark) inline; only extract large dormant wasm.
              if (b64.length < 1_000_000) return m;
              // Against the worker's own URL, which the worker output options put in ASSETS_DIR.
              const specifier = specifierFromDir(ASSETS_DIR, fileFor(b64));
              return `new URL(\\"${specifier}\\", self.location.href)`;
            }
          );

      // Emitted worker modules arrive as JS assets rather than chunks; rewriting either
      // means reading and writing a different field, so both passes below go through here.
      const rewriteJs = (visit: (code: string, fileName: string) => string | null): void => {
        for (const [fileName, chunk] of Object.entries(bundle)) {
          if (chunk.type === 'chunk') {
            const next = visit(chunk.code, fileName);
            if (next !== null) chunk.code = next;
          } else if (chunk.type === 'asset' && fileName.endsWith('.js')) {
            const source =
              typeof chunk.source === 'string'
                ? chunk.source
                : Buffer.from(chunk.source).toString('utf-8');
            const next = visit(source, fileName);
            if (next !== null) chunk.source = next;
          }
        }
      };

      rewriteJs((code, fileName) =>
        code.includes('data:application/wasm') ? deinline(code, fileName) : null
      );

      if (!ortWasm) {
        // A Vite that emits the asset instead of inlining it already has the bytes here.
        const ortDigest = [...upstream].find(([, name]) => name === ORT_WASM_FILE)?.[0];
        for (const [fileName, item] of Object.entries(bundle)) {
          if (item.type === 'asset' && fileName.endsWith('.wasm') && sha256(item.source) === ortDigest) {
            ortWasm = fileName;
          }
        }
      }
      if (!ortWasm) {
        // Most likely an onnxruntime-web upgrade changing which build the `import`
        // condition resolves to (vite.wasm.ts).
        throw new Error(`${ORT_WASM_FILE} is in the bundle neither inlined nor emitted, so ortEnv.ts has nothing to name`);
      }

      const ortFile = ortWasm;
      let substituted = 0;
      rewriteJs((code, fileName) => {
        if (!code.includes(ORT_WASM_TOKEN)) return null;
        substituted += 1;
        return code.replaceAll(ORT_WASM_TOKEN, specifierFrom(fileName, ortFile));
      });
      if (substituted === 0) {
        // A minifier that re-encoded the literal, a `define` that stopped applying, or
        // ortEnv.ts out of the graph. Shipping would 404 every policy.
        throw new Error(`${ORT_WASM_TOKEN} appears in no emitted JS, so the ORT wasm path was never written`);
      }
    },
  };
}

// Inline all bundled CSS (including vanilla-extract output) into the JS entry so
// a single `import('mjswan.js')` brings its own styles — no separate stylesheet
// the host page must remember to load.
function cssInjectedByJsPlugin(): Plugin {
  return {
    name: 'mjswan-css-inject',
    apply: 'build',
    enforce: 'post',
    generateBundle(_options, bundle) {
      let css = '';
      for (const [fileName, chunk] of Object.entries(bundle)) {
        if (chunk.type === 'asset' && fileName.endsWith('.css')) {
          css +=
            typeof chunk.source === 'string'
              ? chunk.source
              : Buffer.from(chunk.source).toString('utf-8');
          delete bundle[fileName];
        }
      }
      if (!css) return;
      const entry = Object.values(bundle).find(
        (chunk) => chunk.type === 'chunk' && chunk.isEntry
      );
      if (entry && entry.type === 'chunk') {
        const injector =
          `(function(){try{` +
          `var d=document;if(!d||d.getElementById('mjswan-styles'))return;` +
          `var s=d.createElement('style');s.id='mjswan-styles';` +
          `s.textContent=${JSON.stringify(css)};` +
          `(d.head||d.documentElement).appendChild(s);` +
          `}catch(e){console.error('mjswan: failed to inject styles',e);}})();\n`;
        entry.code = injector + entry.code;
      }
    },
  };
}

export default defineConfig({
  // The engine loads from a versioned CDN path (`/npm/mjswan@<v>/dist/`), where Vite's
  // default `/` base would send a worker's `new URL` to the serving origin's root.
  base: './',
  plugins: [
    react(),
    vanillaExtractPlugin(),
    extractInlinedWasmPlugin(),
    cssInjectedByJsPlugin(),
  ],
  define: {
    __APP_VERSION__: JSON.stringify(getVersionFromPython()),
    // Library mode (unlike app mode) does NOT replace `process.env.NODE_ENV`, so
    // React/Mantine's bare `process` references would throw `ReferenceError` when
    // loaded from a CDN. Fold to "production"; other `process`/`Buffer` refs are
    // runtime-guarded (`typeof process < "u"`) Node paths that never run here.
    'process.env.NODE_ENV': JSON.stringify('production'),
    // The co-located ORT wasm `ortEnv.ts` resolves against the bundle URL; a placeholder
    // until generateBundle knows the emitted name and the depth of its referrer.
    __ORT_WASM_FILE__: JSON.stringify(ORT_WASM_TOKEN),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  optimizeDeps: {
    exclude: ['mujoco', 'mujoco/mt'],
  },
  assetsInclude: ['**/*.wasm'],
  worker: {
    format: 'es',
    // Workers are dormant single-threaded (no SharedArrayBuffer); emit them into
    // assets/ with everything else the build generates (the extract plugin runs
    // over them as JS assets of the main bundle).
    rollupOptions: {
      output: {
        entryFileNames: `${ASSETS_DIR}/[name]-[hash].js`,
        chunkFileNames: `${ASSETS_DIR}/[name]-[hash].js`,
        assetFileNames: `${ASSETS_DIR}/[name]-[hash][extname]`,
      },
    },
  },
  build: {
    outDir: 'dist',
    // The SPA build (vite build) runs first and empties dist/; the lib build
    // runs second and must preserve those files alongside mjswan.js.
    emptyOutDir: false,
    // Discourage inlining of small assets. NB: lib mode still force-inlines the
    // multi-MB WASM regardless of this value — extractInlinedWasmPlugin above is
    // what actually pulls them back out into co-located files.
    assetsInlineLimit: 0,
    sourcemap: false,
    chunkSizeWarningLimit: 11000,
    // Bundle all CSS into one asset so the inject plugin can hoist it into JS.
    cssCodeSplit: false,
    lib: {
      // The engine entry (createEngine) — no React/Mantine in the CDN bundle.
      entry: path.resolve(__dirname, 'src/engine/index.ts'),
      formats: ['es'],
      fileName: () => 'mjswan.js',
    },
    rollupOptions: {
      // Bundle everything — nothing is external. A bare import left in the
      // output would be unresolvable from jsDelivr.
      external: [],
      output: {
        // Only the entry sits at the root of dist/, where it is addressed from outside
        // (the npm entry, the URL Cloud pins). Everything it generates goes into assets/
        // with the SPA build's, so identical files share one path. See vite.wasm.ts.
        entryFileNames: 'mjswan.js',
        chunkFileNames: `${ASSETS_DIR}/[name]-[hash].js`,
        assetFileNames: `${ASSETS_DIR}/[name]-[hash][extname]`,
        minify,
      },
      onwarn(warning, warn) {
        if (
          warning.message.includes('mujoco') &&
          warning.message.includes('externalized for browser compatibility')
        )
          return;
        warn(warning);
      },
    },
  },
});
