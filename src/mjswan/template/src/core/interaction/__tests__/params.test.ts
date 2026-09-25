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
  it('gives every parameter a default on its slider, inside limits that hold the slider', () => {
    for (const mode of INTERACTION_MODES) {
      expect(mode.params.length, mode.id).toBeGreaterThan(0);
      const names = mode.params.map((p) => p.name);
      expect(new Set(names).size, mode.id).toBe(names.length);
      for (const param of mode.params) {
        const where = `${mode.id}.${param.name}`;
        const softMin = param.softMin ?? param.min;
        const softMax = param.softMax ?? param.max;
        expect(param.min, where).toBeLessThan(param.max);
        expect(softMin, where).toBeLessThan(softMax);
        expect(softMin, where).toBeGreaterThanOrEqual(param.min);
        expect(softMax, where).toBeLessThanOrEqual(param.max);
        expect(param.default, where).toBeGreaterThanOrEqual(softMin);
        expect(param.default, where).toBeLessThanOrEqual(softMax);
        expect(param.step, where).toBeGreaterThan(0);
        expect(param.step, where).toBeLessThanOrEqual(softMax - softMin);
        // A negative gain or impulse would pull where a push was asked for.
        expect(param.min, where).toBeGreaterThanOrEqual(0);
        if (param.type === 'checkbox') {
          expect([param.min, param.max], where).toEqual([0, 1]);
          expect([0, 1], where).toContain(param.default);
        }
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

  it('clamps to the hard limits, not the slider, and never lets NaN through', () => {
    const gain = modeSpec('pull').params.find((p) => p.name === 'gain')!;
    expect(clampParam(gain, -50)).toBe(gain.min);
    expect(clampParam(gain, 1e9)).toBe(gain.max);
    // Past where the slider reaches, which is what the number box is for.
    const typed = (gain.softMax! + gain.max) / 2;
    expect(clampParam(gain, typed)).toBe(typed);
    expect(clampParam(gain, Number.NaN)).toBe(gain.default);
    expect(clampParam(gain, Number.POSITIVE_INFINITY)).toBe(gain.default);
  });

  it('holds a checkbox to 0 or 1', () => {
    const hold = modeSpec('weld').params.find((p) => p.name === 'torqueScale')!;
    expect(hold.type).toBe('checkbox');
    expect(clampParam(hold, 0.7)).toBe(1);
    expect(clampParam(hold, 0.2)).toBe(0);
    expect(clampParam(hold, 5)).toBe(1);
    expect(clampParam(hold, Number.NaN)).toBe(hold.default);
  });

  it('refuses an unknown mode loudly', () => {
    expect(() => modeSpec('nope' as never)).toThrow(/Unknown interaction mode/);
  });
});
