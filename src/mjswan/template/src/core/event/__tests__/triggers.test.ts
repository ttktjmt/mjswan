/**
 * Native Event-mode dispatch: the interval, reset and startup trigger *semantics* ported
 * from mjlab, expressed as scalars, not tensors.
 */
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../../rng';
import { IntervalTrigger, ResetTrigger, StartupTrigger } from '../triggers';

describe('IntervalTrigger', () => {
  it('does not fire before the interval elapses', () => {
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 1.0] }, new SeededRng(1));
    let fired = 0;
    for (let i = 0; i < 9; i++) if (t.tick(0.1)) fired++;
    expect(fired).toBe(0);
  });

  it('fires once the interval elapses, then resamples', () => {
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 1.0] }, new SeededRng(1));
    let fired = 0;
    // dt=0.25 is exact in binary, so 3.0s against a 1.0s interval fires exactly 3 times.
    // dt=0.1 would accumulate ~1e-16 and land a tick later — see the drift test below.
    for (let i = 0; i < 12; i++) if (t.tick(0.25)) fired++;
    expect(fired).toBe(3);
  });

  it('samples the interval from its range', () => {
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 3.0] }, new SeededRng(42));
    expect(t.secondsUntilNextFiring).toBeGreaterThanOrEqual(1.0);
    expect(t.secondsUntilNextFiring).toBeLessThan(3.0);
  });

  it('fires only once for a dt spanning several intervals (no catch-up burst)', () => {
    const t = new IntervalTrigger({ intervalRangeS: [0.1, 0.1] }, new SeededRng(3));
    expect(t.tick(1.0)).toBe(true);
    // The overshoot must not leave the timer negative and fire again next frame.
    expect(t.secondsUntilNextFiring).toBeGreaterThan(0);
  });

  it('carries overshoot so the average rate does not drift', () => {
    // dt=0.3 against 1.0s fires at 1.2, 2.1, 3.0: the remainder rolls forward.
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 1.0] }, new SeededRng(1));
    let fired = 0;
    for (let i = 0; i < 100; i++) if (t.tick(0.3)) fired++;
    // 30 seconds of playback at 1 Hz → 30 firings (±1 for phase).
    expect(fired).toBeGreaterThanOrEqual(29);
    expect(fired).toBeLessThanOrEqual(30);
  });

  it('does not fire while disarmed, and does not bank the wait', () => {
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 1.0] }, new SeededRng(1));
    t.setArmed(false);
    let fired = 0;
    for (let i = 0; i < 40; i++) if (t.tick(0.25)) fired++;
    expect(fired).toBe(0);
    expect(t.isArmed).toBe(false);

    t.setArmed(true);
    expect(t.isArmed).toBe(true);
    expect(t.secondsUntilNextFiring).toBeCloseTo(1.0, 6);
    for (let i = 0; i < 3; i++) expect(t.tick(0.25)).toBe(false);
    expect(t.tick(0.25)).toBe(true);
  });

  it('arming an already-armed timer leaves its countdown alone', () => {
    const t = new IntervalTrigger({ intervalRangeS: [1.0, 1.0] }, new SeededRng(1));
    t.tick(0.75);
    t.setArmed(true);
    expect(t.secondsUntilNextFiring).toBeCloseTo(0.25, 6);
  });

  it('per-episode timers restart on reset; global timers keep running', () => {
    const perEpisode = new IntervalTrigger(
      { intervalRangeS: [1.0, 1.0], isGlobalTime: false },
      new SeededRng(1),
    );
    perEpisode.tick(0.9);
    expect(perEpisode.secondsUntilNextFiring).toBeCloseTo(0.1, 6);
    perEpisode.onReset();
    expect(perEpisode.secondsUntilNextFiring).toBeCloseTo(1.0, 6);

    const global = new IntervalTrigger(
      { intervalRangeS: [1.0, 1.0], isGlobalTime: true },
      new SeededRng(1),
    );
    global.tick(0.9);
    global.onReset();
    expect(global.secondsUntilNextFiring).toBeCloseTo(0.1, 6);
  });

  it('is reproducible from a seed (replay)', () => {
    const run = (): number[] => {
      const t = new IntervalTrigger({ intervalRangeS: [1.0, 3.0] }, new SeededRng(7));
      const firings: number[] = [];
      for (let i = 0; i < 200; i++) if (t.tick(0.05)) firings.push(i);
      return firings;
    };
    expect(run()).toEqual(run());
  });
});

describe('StartupTrigger', () => {
  it('fires exactly once', () => {
    const t = new StartupTrigger();
    expect(t.take()).toBe(true);
    expect(t.take()).toBe(false);
    expect(t.take()).toBe(false);
    expect(t.hasFired).toBe(true);
  });
});

describe('ResetTrigger', () => {
  it('fires on every reset when ungated', () => {
    const t = new ResetTrigger();
    expect(t.take()).toBe(true);
    expect(t.take()).toBe(true);
  });

  it('suppresses a reset that arrives too soon (min_step_count_between_reset)', () => {
    const t = new ResetTrigger({ minStepCountBetweenReset: 5 });
    expect(t.take()).toBe(true); // first reset always allowed
    expect(t.take()).toBe(false); // immediately after → gated
    for (let i = 0; i < 5; i++) t.step();
    expect(t.take()).toBe(true); // enough steps have passed
  });
});
