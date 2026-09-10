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
// with every dependency bundled and the MuJoCo / ONNX WASM co-located flat in
// `dist/` so they resolve relative to the bundle wherever it is served from.
// See src/engine/ and docs/adr/0004-headless-engine-core.md.

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

const sha256 = (buffer: Buffer): string => createHash('sha256').update(buffer).digest('hex');

// digest → emitted name, for the upstream WASM the SPA build names the same way. Matching
// by content is what recovers the name: the de-inlined assets arrive as bytes with the
// source file name long gone. See vite.wasm.ts.
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
// lands on the same path and ships once. ORT's is also the one src/core/onnx/ortEnv.ts
// points `ort.env.wasm.wasmPaths` at, so its emitted name is written back into the code
// here, where the content hash is finally known (issue #123).
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
      const emittedFor = new Map<string, string>(); // upstream name → emitted fileName
      const fileFor = (b64: string): string => {
        let fileName = emitted.get(b64);
        if (!fileName) {
          const source = Buffer.from(b64, 'base64');
          // Anything the SPA build does not also emit is ours, and named as such.
          const name = upstream.get(sha256(source)) ?? 'mjswan-engine.wasm';
          fileName = this.getFileName(this.emitFile({ type: 'asset', name, source }));
          emitted.set(b64, fileName);
          emittedFor.set(name, fileName);
        }
        return fileName;
      };
      const deinline = (code: string): string =>
        code
          .replace(
            QUOTED,
            (m, _quote: string, b64: string, base: string) => {
              // Only de-inline main-thread wasm (base `import.meta.url`). A
              // `self.location.href` base means a classic Blob worker (Spark's
              // Splat sort) where `import.meta` is a syntax error — leave inline.
              if (!/import\.meta\.url/.test(base)) return m;
              return `new URL(${JSON.stringify('./' + fileFor(b64))}, import.meta.url)`;
            }
          )
          .replace(
            ESCAPED,
            (m, b64: string) => {
              // Keep small active-worker wasm (Spark) inline; only extract large dormant wasm.
              if (b64.length < 1_000_000) return m;
              return `new URL(\\"./${fileFor(b64)}\\", self.location.href)`;
            }
          );

      // Emitted worker modules arrive as JS assets rather than chunks; rewriting either
      // means reading and writing a different field, so both passes below go through here.
      const rewriteJs = (visit: (code: string) => string | null): void => {
        for (const [fileName, chunk] of Object.entries(bundle)) {
          if (chunk.type === 'chunk') {
            const next = visit(chunk.code);
            if (next !== null) chunk.code = next;
          } else if (chunk.type === 'asset' && fileName.endsWith('.js')) {
            const source =
              typeof chunk.source === 'string'
                ? chunk.source
                : Buffer.from(chunk.source).toString('utf-8');
            const next = visit(source);
            if (next !== null) chunk.source = next;
          }
        }
      };

      rewriteJs((code) => (code.includes('data:application/wasm') ? deinline(code) : null));

      const ortWasm = emittedFor.get(ORT_WASM_FILE);
      if (!ortWasm) {
        // Without it `ortEnv.ts` has no file to name and every policy would 404 at
        // runtime, so fail here instead. Most likely cause: an onnxruntime-web upgrade
        // changed which build the `import` condition resolves to (vite.wasm.ts).
        throw new Error(`${ORT_WASM_FILE} was not inlined in the bundle, so it cannot be emitted beside it`);
      }
      rewriteJs((code) =>
        code.includes(ORT_WASM_TOKEN) ? code.replaceAll(ORT_WASM_TOKEN, ortWasm) : null
      );
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
    // Lib-build only: the co-located ORT wasm `ortEnv.ts` resolves against the bundle
    // URL. A placeholder until generateBundle knows the emitted name's content hash.
    __ORT_WASM_FILE__: JSON.stringify(`./${ORT_WASM_TOKEN}`),
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
    // Workers are dormant single-threaded (no SharedArrayBuffer); emit them flat
    // in dist/ so they sit alongside the extracted WASM (the extract plugin runs
    // over them as JS assets of the main bundle).
    rollupOptions: {
      output: {
        entryFileNames: '[name]-[hash].js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: '[name]-[hash][extname]',
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
        // Flat layout in dist/: mjswan.js next to its chunks and *.wasm, so
        // `new URL('./x.wasm', import.meta.url)` resolves on the CDN.
        entryFileNames: 'mjswan.js',
        chunkFileNames: '[name]-[hash].js',
        assetFileNames: '[name]-[hash][extname]',
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
