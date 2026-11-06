import { describe, expect, it } from 'vitest';
import { decimationFor, MAX_DECIMATION } from '../decimation';

describe('decimationFor', () => {
  it('rounds control_dt over the timestep, to at least one step', () => {
    expect(decimationFor(0.02, 0.002)).toBe(10);
    expect(decimationFor(0.02, 0.02 / MAX_DECIMATION)).toBe(MAX_DECIMATION);
    expect(decimationFor(0.001, 0.002)).toBe(1);
  });

  // Infinity is what JSON's 1e309 parses to.
  it.each([
    [Infinity, 0.002],
    [1e300, 0.002],
    [0.02, 1e-300],
    [0.02, 0.02 / (MAX_DECIMATION + 1)],
    [Number.NaN, 0.002],
  ])('refuses control_dt %s over timestep %s', (controlDt, timestep) => {
    expect(() => decimationFor(controlDt, timestep)).toThrow(/at most 1000/);
  });
});
