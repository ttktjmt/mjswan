/** The one table three places read: the manager, the engine descriptors and the panel. */
import { describe, expect, it } from 'vitest';

import {
  DEFAULT_INTERACTION_MODE,
  INTERACTION_MODES,
  INTERACTION_MODE_IDS,
  clampParam,
  defaultParams,
  modeSpec,
} from '../params';

describe('interaction parameter table', () => {
  it('declares every id exactly once, in the published order', () => {
    expect(INTERACTION_MODES.map((m) => m.id)).toEqual([...INTERACTION_MODE_IDS]);
    expect(new Set(INTERACTION_MODE_IDS).size).toBe(INTERACTION_MODE_IDS.length);
    expect(INTERACTION_MODE_IDS).toContain(DEFAULT_INTERACTION_MODE);
  });

  // A default outside its own slider is a control that moves the moment it is touched.
  it('gives every parameter a default inside its range and a usable step', () => {
    for (const mode of INTERACTION_MODES) {
      expect(mode.params.length, mode.id).toBeGreaterThan(0);
      const names = mode.params.map((p) => p.name);
      expect(new Set(names).size, mode.id).toBe(names.length);
      for (const param of mode.params) {
        const where = `${mode.id}.${param.name}`;
        expect(param.min, where).toBeLessThan(param.max);
        expect(param.default, where).toBeGreaterThanOrEqual(param.min);
        expect(param.default, where).toBeLessThanOrEqual(param.max);
        expect(param.step, where).toBeGreaterThan(0);
        expect(param.step, where).toBeLessThanOrEqual(param.max - param.min);
      }
    }
  });

  it('starts every mode at its declared defaults', () => {
    const values = defaultParams();
    for (const mode of INTERACTION_MODES) {
      for (const param of mode.params) {
        expect(values[mode.id][param.name], `${mode.id}.${param.name}`).toBe(param.default);
      }
      expect(Object.keys(values[mode.id]).sort()).toEqual(mode.params.map((p) => p.name).sort());
    }
  });

  it('clamps rather than refuses, and never lets NaN through', () => {
    const gain = modeSpec('pull').params.find((p) => p.name === 'gain')!;
    expect(clampParam(gain, -50)).toBe(gain.min);
    expect(clampParam(gain, 1e9)).toBe(gain.max);
    expect(clampParam(gain, 250)).toBe(250);
    expect(clampParam(gain, Number.NaN)).toBe(gain.default);
    expect(clampParam(gain, Number.POSITIVE_INFINITY)).toBe(gain.default);
  });

  it('refuses an unknown mode loudly', () => {
    expect(() => modeSpec('nope' as never)).toThrow(/Unknown interaction mode/);
  });
});
