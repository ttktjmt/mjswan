/**
 * The two facts the library build's ORT wiring rests on, checked against the installed
 * package rather than a published bundle — an onnxruntime-web upgrade that breaks either
 * would otherwise only show up as `no available backend found` in a released engine.
 *
 * See vite.wasm.ts and src/core/onnx/ortEnv.ts.
 */
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { ORT_BUNDLED_ENTRY, ORT_WASM_FILE, UPSTREAM_WASM } from '../../../../vite.wasm';

const ORT_PKG = join(__dirname, '../../../../node_modules/onnxruntime-web');

describe('onnxruntime-web runtime files', () => {
  it('resolves to the build with its .mjs loader inlined', () => {
    const pkg = JSON.parse(readFileSync(join(ORT_PKG, 'package.json'), 'utf-8'));
    // A build with an external loader would dynamic-import the `.mjs` from `wasmPaths`,
    // which the engine does not ship — naming only the wasm would stop working.
    expect(pkg.exports['.'].import.default).toBe(ORT_BUNDLED_ENTRY);
  });

  it('still ships the wasm the bundle emits beside itself', () => {
    expect(existsSync(join(ORT_PKG, 'dist', ORT_WASM_FILE))).toBe(true);
    // Named in ORT's own source, so a rename upstream lands here and not in a 404.
    const entry = readFileSync(join(ORT_PKG, ORT_BUNDLED_ENTRY), 'utf-8');
    expect(entry).toContain(ORT_WASM_FILE);
  });
});

describe('co-located WASM sources', () => {
  // The library build matches these by content digest to recover each emitted name. A
  // package that moves its wasm would silently send that file back to `mjswan-engine.wasm`
  // and ship a second copy of bytes the SPA build already emitted.
  it.each(UPSTREAM_WASM)('%s is where the build looks for it', (source) => {
    expect(existsSync(join(__dirname, '../../../../node_modules', source))).toBe(true);
  });
});
