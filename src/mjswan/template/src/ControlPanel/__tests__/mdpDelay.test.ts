import { describe, expect, it } from 'vitest';

import { heldProgress, MAX_DELAY_MS, parseDelayMs, toMs } from '../mdpDelay';

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
    expect(heldProgress({ onFire: 'stop', resetAt: 1.6 }, 1.1)).toBe(0);
  });
});
