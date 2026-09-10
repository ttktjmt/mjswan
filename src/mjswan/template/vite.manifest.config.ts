import { defineConfig } from 'vite';
import path from 'path';

// Builds `dist/manifest.js` — the `mjswan/manifest` parser as a standalone ESM
// so it can be imported from a CDN (jsDelivr) alongside the engine bundle, e.g.
// mjswan Cloud: `import('https://cdn.jsdelivr.net/npm/mjswan@<v>/dist/manifest.js')`.
// The parser is framework-free with only type-level imports, so it bundles to a
// tiny self-contained module (no React, no WASM, no bare imports). Runs after the
// SPA + engine builds with emptyOutDir:false so it sits beside mjswan.js.
export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    sourcemap: false,
    lib: {
      entry: path.resolve(__dirname, 'src/manifest/index.ts'),
      formats: ['es'],
      fileName: () => 'manifest.js',
    },
    rollupOptions: {
      external: [],
      output: { entryFileNames: 'manifest.js', chunkFileNames: 'assets/[name]-[hash].js' },
    },
  },
});
