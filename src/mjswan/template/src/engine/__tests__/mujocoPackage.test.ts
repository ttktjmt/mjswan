/**
 * The engine imports the bare `mujoco` specifier, and a `.mjb` scene loads only in the
 * MuJoCo that wrote it, so that specifier must resolve to the package package.json pins.
 * A transitive dependency can alias another package to the same name.
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const ROOT = join(__dirname, '../../..');

const readJson = (path: string): Record<string, unknown> =>
  JSON.parse(readFileSync(join(ROOT, path), 'utf-8'));

describe('mujoco', () => {
  it('resolves to the pinned package', () => {
    const pinned = (readJson('package.json').dependencies as Record<string, string>).mujoco;
    const installed = readJson('node_modules/mujoco/package.json');
    expect(`npm:${installed.name}@${installed.version}`).toBe(pinned);
  });
});
