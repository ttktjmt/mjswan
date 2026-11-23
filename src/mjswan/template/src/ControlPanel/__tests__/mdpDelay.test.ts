import { describe, expect, it } from 'vitest';

import { freezeProgress, heldProgress, MAX_DELAY_MS, parseDelayMs, toMs } from '../mdpDelay';

describe('parseDelayMs', () => {
  it('reads whole milliseconds, with or without the unit', () => {
    expect(parseDelayMs('1500')).toBe(1500);
    expect(parseDelayMs(' 250 ms ')).toBe(250);
    expect(parseDelayMs('0')).toBe(0);
  });

  it('rounds a fraction and stops at the engine’s longest delay', () => {
    expect(parseDelayMs('12.6')).toBe(13);
    expect(parseDelayMs('90000')).toBe(MAX_DELAY_MS);
  });

  it('is nothing for what is not a number', () => {
    expect(parseDelayMs('')).toBeNull();
    expect(parseDelayMs('-5')).toBeNull();
    expect(parseDelayMs('1s')).toBeNull();
  });
});

describe('toMs', () => {
  it('writes seconds as whole milliseconds', () => {
    expect(toMs(1)).toBe(1000);
    expect(toMs(0.0005)).toBe(1);
  });
});

describe('heldProgress', () => {
  const term = { onFire: { delay: 1 }, resetAt: 1.6 } as const;

  it('fills a held delay as it runs', () => {
    expect(heldProgress(term, 0.6)).toBe(0);
    expect(heldProgress(term, 1.1)).toBeCloseTo(0.5);
    expect(heldProgress(term, 2)).toBe(1);
  });

  it('stays empty without a held reset or a delay', () => {
    expect(heldProgress({ ...term, resetAt: null }, 1.1)).toBe(0);
    expect(heldProgress({ onFire: 'freeze', resetAt: 1.6 }, 1.1)).toBe(0);
  });
});

describe('freezeProgress', () => {
  const frozenBy = { term: 'fell', at: 0.84, resumesAt: 13_000 };

  it('fills a timed freeze with the wall clock', () => {
    expect(freezeProgress(frozenBy, 'fell', { freeze: 2 }, 11_000)).toBe(0);
    expect(freezeProgress(frozenBy, 'fell', { freeze: 2 }, 12_000)).toBeCloseTo(0.5);
    expect(freezeProgress(frozenBy, 'fell', { freeze: 2 }, 14_000)).toBe(1);
  });

  it('stays empty for another term, a freeze until reset, or none', () => {
    expect(freezeProgress(frozenBy, 'other', { freeze: 2 }, 12_000)).toBe(0);
    expect(freezeProgress({ ...frozenBy, resumesAt: null }, 'fell', 'freeze', 12_000)).toBe(0);
    expect(freezeProgress(null, 'fell', { freeze: 2 }, 12_000)).toBe(0);
  });
});
