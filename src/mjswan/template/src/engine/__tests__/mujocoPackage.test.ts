/**
 * The engine imports the bare `mujoco` specifier, which must resolve to the version
 * package.json pins, since a `.mjb` loads only in the MuJoCo that wrote it. A transitive
 * dependency can alias another package to that name.
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
