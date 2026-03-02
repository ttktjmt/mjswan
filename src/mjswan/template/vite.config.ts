import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';
import path from 'path';
import fs from 'fs';

// Extract version from Python package (source of truth)
function getVersionFromPython(): string {
  const initPath = path.resolve(__dirname, '../__init__.py');
  try {
    const content = fs.readFileSync(initPath, 'utf-8');
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) {
      return match[1];
    }
  } catch {
    // Fallback to package.json if Python file not found
  }
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pkg = require('./package.json');
  return pkg.version || '0.0.0';
}

function gtmPlugin(gtmId: string | undefined) {
  if (!gtmId) return null;
  return {
    name: 'mjswan-gtm',
    transformIndexHtml(html: string) {
      const headScript =
        `<!-- Google Tag Manager -->\n` +
        `    <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':\n` +
        `    new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],\n` +
        `    j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=\n` +
        `    'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);\n` +
        `    })(window,document,'script','dataLayer','${gtmId}');</script>\n` +
        `    <!-- End Google Tag Manager -->`;
      const bodyNoscript =
        `<!-- Google Tag Manager (noscript) -->\n` +
        `    <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=${gtmId}"\n` +
        `    height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>\n` +
        `    <!-- End Google Tag Manager -->`;
      return html
        .replace('</head>', `    ${headScript}\n  </head>`)
        .replace('<body>', `<body>\n    ${bodyNoscript}`);
    },
  };
}

export default defineConfig({
  plugins: [react(), vanillaExtractPlugin(), gtmPlugin(process.env.MJSWAN_GTM_ID)],
  base: process.env.MJSWAN_BASE_PATH || '/',
  define: {
    __APP_VERSION__: JSON.stringify(getVersionFromPython()),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      'mujoco': path.resolve(__dirname, './src/mujoco/mujoco_wasm'),
    },
  },
  optimizeDeps: {
    exclude: ['mujoco'],
  },
  assetsInclude: ['**/*.wasm'],
  server: {
    port: 8000,
    host: true,
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 11000,
    rollupOptions: {
      input: path.resolve(__dirname, 'index.html'),
    },
  },
  worker: {
    format: 'es',
  },
});
