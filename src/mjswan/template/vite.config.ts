import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { vanillaExtractPlugin } from '@vanilla-extract/vite-plugin';
import path from 'path';
import fs from 'fs';
import postcss from 'postcss';
import { minify } from './vite.shared';

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

const isMt = process.env.MJSWAN_MT === '1';
const coiSwPath = path.resolve(__dirname, '_mt/coi-serviceworker.js');

function mtPlugin(enabled: boolean): Plugin | null {
  if (!enabled) return null;
  return {
    name: 'mjswan-mt',
    configureServer(server) {
      // Serve the SW file during `vite dev` so the browser can register it.
      server.middlewares.use('/coi-serviceworker.js', (_req, res) => {
        res.setHeader('Content-Type', 'application/javascript');
        res.end(fs.readFileSync(coiSwPath, 'utf-8'));
      });
    },
    generateBundle() {
      // Emit the SW file only when building the mt variant.
      this.emitFile({
        type: 'asset',
        fileName: 'coi-serviceworker.js',
        source: fs.readFileSync(coiSwPath, 'utf-8'),
      });
    },
    transformIndexHtml(html: string) {
      // Register the COOP/COEP service worker early in <head>.
      // Required for GitHub Pages hosting (cannot set response headers).
      // On Netlify / Cloudflare Pages / Vercel the _headers file is used instead.
      const swScript =
        `<script>\n` +
        `    /* Register COOP/COEP service worker — required for GitHub Pages hosting */\n` +
        `    if (!window.crossOriginIsolated && 'serviceWorker' in navigator) {\n` +
        `      document.documentElement.style.display = 'none';\n` +
        `      navigator.serviceWorker.register('coi-serviceworker.js').then(function() {\n` +
        `        window.location.reload();\n` +
        `      });\n` +
        `    }\n` +
        `  </script>`;
      return html.replace('<meta charset="utf-8" />', `<meta charset="utf-8" />\n  ${swScript}`);
    },
  };
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

const MANTINE_ESM = path.resolve(__dirname, 'node_modules/@mantine/core/esm');

/** The `m_` names in the class maps of the Mantine components src/'s .tsx files reach. */
function mantineNamesInUse(): Set<string> {
  const entries = new Map<string, string>();
  const index = fs.readFileSync(path.join(MANTINE_ESM, 'index.mjs'), 'utf-8');
  for (const [, names, from] of index.matchAll(/export \{([^}]*)\} from '([^']+)'/g)) {
    for (const name of names.split(',')) entries.set(name.trim(), path.resolve(MANTINE_ESM, from));
  }
  const used = new Set<string>();
  const seen = new Set<string>();
  const walk = (file: string): void => {
    if (seen.has(file)) return;
    seen.add(file);
    const code = fs.readFileSync(file, 'utf-8');
    if (file.endsWith('.module.css.mjs')) {
      for (const [, name] of code.matchAll(/"(m_[0-9a-f]+)"/g)) used.add(name);
      return;
    }
    // Named imports only: the bundler drops Mantine's bare `import './x.mjs'` lines too.
    for (const [, from] of code.matchAll(/(?:import|export) (?:\w+|\{[^}]*\}) from '(\.[^']+)'/g)) {
      walk(path.resolve(path.dirname(file), from));
    }
  };
  const src = path.resolve(__dirname, 'src');
  for (const file of fs.readdirSync(src, { recursive: true, encoding: 'utf-8' })) {
    if (!file.endsWith('.tsx') || file.includes('__tests__') || file.endsWith('.test.tsx')) continue;
    const code = fs.readFileSync(path.join(src, file), 'utf-8');
    for (const [, names] of code.matchAll(/import \{([^}]*)\} from ['"]@mantine\/core['"]/g)) {
      for (const name of names.split(',')) {
        const entry = entries.get(name.trim().split(/\s+as\s+/)[0]);
        if (entry) walk(entry);
      }
    }
  }
  return used;
}

// Drops each `@mantine/core/styles.css` rule no component in use names, before Vite hashes
// the CSS. Not Mantine's per-component files: they size in px where styles.css scales by
// rem, which index.css's root font size relies on.
function mantineCssPlugin(): Plugin {
  let used = new Set<string>();
  const unused = (names: string[]) => names.length > 0 && names.every((name) => !used.has(name));
  return {
    name: 'mjswan-mantine-css',
    enforce: 'pre',
    buildStart() {
      used = mantineNamesInUse();
      if (used.size === 0) this.error('found no Mantine component in use under src/');
    },
    transform(code, id) {
      if (!/[\\/]@mantine[\\/]core[\\/]styles\.css$/.test(id.split('?')[0])) return null;
      const root = postcss.parse(code);
      root.walkRules((rule) => {
        if (unused([...rule.selector.matchAll(/\.(m_[0-9a-f]+)(?![\w-])/g)].map((m) => m[1]))) rule.remove();
      });
      root.walkAtRules('keyframes', (rule) => {
        if (unused(rule.params.match(/^m_[0-9a-f]+$/) ?? [])) rule.remove();
      });
      for (let emptied = true; emptied; ) {
        emptied = false;
        root.walkAtRules((rule) => {
          if (rule.nodes?.length === 0) {
            rule.remove();
            emptied = true;
          }
        });
      }
      return { code: root.toString(), map: null };
    },
  };
}

// What `build.license` writes, for thirdPartyLicensesPlugin to read and drop.
const LICENSES_JSON = '.vite/licenses.json';

interface BundledLicense {
  name: string;
  version: string;
  identifier?: string;
  text?: string;
}

// Bundled from no npm package, so `build.license` cannot list it: src/assets/'s UI font.
const VENDORED_LICENSES: BundledLicense[] = [{ name: 'Inter', version: '3.19', identifier: 'OFL-1.1' }];

// Minify drops the notices bundled packages carry, so dist/ carries them in one file. A
// package that publishes no LICENSE, and anything vendored, takes its text from
// licenses/<name>/LICENSE.
function thirdPartyLicensesPlugin(): Plugin {
  return {
    name: 'mjswan-third-party-licenses',
    apply: 'build',
    generateBundle: {
      // vite:license emits LICENSES_JSON from its own generateBundle.
      order: 'post',
      handler(_options, bundle) {
        const json = bundle[LICENSES_JSON];
        if (json?.type !== 'asset') this.error(`build.license emitted no ${LICENSES_JSON}`);
        delete bundle[LICENSES_JSON];
        const source =
          typeof json.source === 'string' ? json.source : Buffer.from(json.source).toString('utf-8');
        const missing: string[] = [];
        const bundled = [...(JSON.parse(source) as BundledLicense[]), ...VENDORED_LICENSES];
        const sections = bundled.map(
          ({ name, version, identifier, text }) => {
            const own = path.resolve(__dirname, 'licenses', name, 'LICENSE');
            text ??= fs.existsSync(own) ? fs.readFileSync(own, 'utf-8').trim() : undefined;
            if (!text) missing.push(name);
            return `## ${name} ${version}${identifier ? ` (${identifier})` : ''}\n\n${text}\n`;
          }
        );
        if (missing.length) {
          this.error(`No license text for ${missing.join(', ')}: add it as licenses/<package>/LICENSE`);
        }
        this.emitFile({
          type: 'asset',
          fileName: 'THIRD_PARTY_LICENSES.md',
          source: `# Third-party licenses\n\nThe packages bundled into this build, with their licenses.\n\n${sections.join('\n')}`,
        });
      },
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    vanillaExtractPlugin(),
    mtPlugin(isMt),
    gtmPlugin(process.env.MJSWAN_GTM_ID),
    mantineCssPlugin(),
    thirdPartyLicensesPlugin(),
  ],
  base: process.env.MJSWAN_BASE_PATH || '/',
  define: {
    __APP_VERSION__: JSON.stringify(getVersionFromPython()),
    __MUJOCO_MT__: JSON.stringify(isMt),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      // Single-threaded apps never load `mujoco/mt`; the stub keeps its 10 MB build out of `dist/`.
      ...(isMt ? {} : { 'mujoco/mt': path.resolve(__dirname, 'src/engine/mujocoMtStub.ts') }),
    },
  },
  optimizeDeps: {
    exclude: ['mujoco', 'mujoco/mt'],
  },
  assetsInclude: ['**/*.wasm'],
  server: {
    port: 8000,
    host: true,
    headers: isMt ? {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    } : {},
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: false,
    chunkSizeWarningLimit: 11000,
    license: { fileName: LICENSES_JSON },
    rollupOptions: {
      input: path.resolve(__dirname, 'index.html'),
      output: { minify },
      onwarn(warning, warn) {
        // mujoco.js is an Emscripten-generated file that imports Node.js built-ins
        // (module, worker_threads) behind runtime environment checks that are never
        // reached in the browser.  Suppress the harmless externalization warnings.
        if (
          warning.message.includes('mujoco') &&
          warning.message.includes('externalized for browser compatibility')
        ) return;
        warn(warning);
      },
    },
  },
  worker: {
    format: 'es',
  },
});
