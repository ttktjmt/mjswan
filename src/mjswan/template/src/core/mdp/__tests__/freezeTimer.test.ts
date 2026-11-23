/** A timed freeze ends once, at its time; set again, it is timed afresh; cancelled, never. */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { FreezeTimer } from '../freezeTimer';

describe('FreezeTimer', () => {
  beforeEach(() => vi.useFakeTimers({ now: 10_000 }));
  afterEach(() => vi.useRealTimers());

  it('ends a timed freeze at its time, once', () => {
    const end = vi.fn();
    new FreezeTimer(end).set(11_000);
    vi.advanceTimersByTime(999);
    expect(end).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    vi.advanceTimersByTime(5_000);
    expect(end).toHaveBeenCalledTimes(1);
  });

  it('takes the latest time, and none for a freeze that waits for a reset', () => {
    const end = vi.fn();
    const timer = new FreezeTimer(end);
    timer.set(11_000);
    timer.set(12_000);
    vi.advanceTimersByTime(1_500);
    expect(end).not.toHaveBeenCalled();
    vi.advanceTimersByTime(500);
    expect(end).toHaveBeenCalledTimes(1);
    timer.set(null);
    vi.advanceTimersByTime(60_000);
    expect(end).toHaveBeenCalledTimes(1);
  });

  it('ends at once for a time already past, and never once cancelled', () => {
    const end = vi.fn();
    const timer = new FreezeTimer(end);
    timer.set(9_000);
    vi.advanceTimersByTime(0);
    expect(end).toHaveBeenCalledTimes(1);
    timer.set(11_000);
    timer.cancel();
    vi.advanceTimersByTime(60_000);
    expect(end).toHaveBeenCalledTimes(1);
  });
});
