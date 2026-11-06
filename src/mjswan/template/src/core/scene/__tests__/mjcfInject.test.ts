import { describe, expect, it } from 'vitest';
import { isEntityPrefixed } from '../mjcfInject';

describe('isEntityPrefixed', () => {
  it('reads a slash in a name value', () => {
    expect(isEntityPrefixed('<body name="robot/torso"/>')).toBe(true);
    expect(isEntityPrefixed('<body name="torso"/>')).toBe(false);
    expect(isEntityPrefixed('<geom name="a" class="b/c"/>')).toBe(false);
  });

  it('scans an unclosed run of slashes in linear time', () => {
    const xml = `<mujoco/><!-- name="${'/'.repeat(200_000)} -->`;
    const start = performance.now();
    expect(isEntityPrefixed(xml)).toBe(false);
    expect(performance.now() - start).toBeLessThan(100);
  });
});
