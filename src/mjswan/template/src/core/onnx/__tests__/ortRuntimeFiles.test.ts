/**
 * The onnxruntime-web assumptions the library build's ORT wiring rests on; an upgrade
 * breaking one would otherwise surface as `no available backend found` in a released
 * engine. See vite.wasm.ts and src/core/onnx/ortEnv.ts.
 */
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

import * as ort from 'onnxruntime-web/wasm';
import { describe, expect, it } from 'vitest';

import {
  MAX_ASSET_BYTES,
  ORT_BUNDLED_ENTRY,
  ORT_ENTRY,
  ORT_WASM_FILE,
  UPSTREAM_WASM,
} from '../../../../vite.wasm';
import '../ortEnv';

const NODE_MODULES = join(__dirname, '../../../../node_modules');
const ORT_PKG = join(NODE_MODULES, 'onnxruntime-web');

describe('onnxruntime-web runtime files', () => {
  it('resolves to the build with its .mjs loader inlined', () => {
    const pkg = JSON.parse(readFileSync(join(ORT_PKG, 'package.json'), 'utf-8'));
    const subpath = ORT_ENTRY.replace('onnxruntime-web', '.');
    // An external-loader build would dynamic-import a `.mjs` the engine does not ship.
    expect(pkg.exports[subpath].import.default).toBe(ORT_BUNDLED_ENTRY);
  });

  it('still ships the wasm the bundle emits beside itself', () => {
    expect(existsSync(join(ORT_PKG, 'dist', ORT_WASM_FILE))).toBe(true);
    // Named in ORT's own source, so a rename upstream lands here and not in a 404.
    const entry = readFileSync(join(ORT_PKG, ORT_BUNDLED_ENTRY), 'utf-8');
    expect(entry).toContain(ORT_WASM_FILE);
  });
});

describe('ortEnv', () => {
  it('keeps ORT single-threaded, which the inlined-loader path depends on', () => {
    expect(ort.env.wasm.numThreads).toBe(1);
  });
});

describe('co-located WASM sources', () => {
  // Matched by digest to recover each emitted name, so a moved source silently ships a
  // second copy of bytes the SPA build already emitted.
  it.each(UPSTREAM_WASM)('%s is where the build looks for it', (source) => {
    expect(existsSync(join(NODE_MODULES, source))).toBe(true);
  });

  // A file over the limit uploads nowhere: Cloudflare Pages rejects the deploy, and a
  // build command that prunes it instead leaves the page fetching a wasm that 404s.
  it.each(UPSTREAM_WASM)('%s stays inside a static host per-file limit', (source) => {
    expect(statSync(join(NODE_MODULES, source)).size).toBeLessThanOrEqual(MAX_ASSET_BYTES);
  });
});
