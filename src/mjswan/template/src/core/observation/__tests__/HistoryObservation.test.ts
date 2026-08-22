/**
 * `HistoryObservation`: per-term frame stacking (mjlab's `history_length`, plus the
 * sparse `history_steps` a tracking policy is trained on).
 *
 * The cases that matter are the ones a wrong buffer makes *plausible* rather than
 * obviously broken: the offset→frame mapping (an off-by-one reads last step's value
 * forever), the direction a dense count stacks in (reversing it keeps the width and
 * runs time backwards), priming after a reset (a history of zeros the policy never saw
 * in training), and sparse offsets sizing the term by the offsets they name rather
 * than by how far back they reach.
 */
import { describe, expect, it } from 'vitest';

import { HistoryObservation, historyOffsets } from '../HistoryObservation';
import { ObservationBase, type ObservationConfig } from '../ObservationBase';
import type { PolicyRunner } from '../../policy/PolicyRunner';
import type { PolicyState } from '../../policy/types';

const STATE = {} as PolicyState;
const RUNNER = {} as PolicyRunner;

/** A term whose value is `[n, n * 10]` on its n-th frame, so frames are identifiable. */
class Counter extends ObservationBase {
  frames = 0;
  resets = 0;

  get size(): number {
    return 2;
  }

  reset(): void {
    this.resets += 1;
  }

  compute(): Float32Array {
    this.frames += 1;
    return new Float32Array([this.frames, this.frames * 10]);
  }
}

function build(config: ObservationConfig): { history: HistoryObservation; base: Counter } {
  const base = new Counter(RUNNER, config);
  const offsets = historyOffsets(config);
  if (!offsets) throw new Error('expected offsets');
  return { history: new HistoryObservation(RUNNER, config, base, offsets), base };
}

describe('historyOffsets', () => {
  it('expands a dense length oldest-frame-first, as mjlab flattens its buffer', () => {
    expect(historyOffsets({ name: 'x', history_length: 3 })).toEqual([2, 1, 0]);
  });

  it('treats no history and a single frame alike — nothing to stack', () => {
    expect(historyOffsets({ name: 'x' })).toBeNull();
    expect(historyOffsets({ name: 'x', history_length: 1 })).toBeNull();
  });

  it('takes sparse offsets verbatim, over a dense length', () => {
    expect(
      historyOffsets({ name: 'x', history_length: 3, history_offsets: [0, 2, 8] }),
    ).toEqual([0, 2, 8]);
  });
});

describe('HistoryObservation', () => {
  it('sizes the term by frames stacked, not by how far back they reach', () => {
    const { history } = build({ name: 'x', history_offsets: [0, 4, 20] });
    // 3 offsets x width 2 — not the 21 frames the buffer has to hold.
    expect(history.size).toBe(6);
  });

  it('primes every slot with the first frame after a reset', async () => {
    const { history } = build({ name: 'x', history_length: 3 });
    expect(Array.from(await history.compute(STATE))).toEqual([1, 10, 1, 10, 1, 10]);
  });

  it('shifts frame-major, oldest first', async () => {
    const { history } = build({ name: 'x', history_length: 3 });
    await history.compute(STATE);
    expect(Array.from(await history.compute(STATE))).toEqual([1, 10, 1, 10, 2, 20]);
    expect(Array.from(await history.compute(STATE))).toEqual([1, 10, 2, 20, 3, 30]);
    // Frame 1 has now fallen off the start of a 3-deep buffer.
    expect(Array.from(await history.compute(STATE))).toEqual([2, 20, 3, 30, 4, 40]);
  });

  it('takes a newest-first stack from offsets, which a count no longer spells', async () => {
    const { history } = build({ name: 'x', history_offsets: [0, 1, 2] });
    for (let i = 0; i < 3; i++) await history.compute(STATE);
    expect(Array.from(await history.compute(STATE))).toEqual([4, 40, 3, 30, 2, 20]);
  });

  it('selects only the named sparse offsets', async () => {
    const { history } = build({ name: 'x', history_offsets: [0, 2, 3] });
    for (let i = 0; i < 4; i++) await history.compute(STATE);
    // Frame 4 is current; offsets 2 and 3 are frames 2 and 1.
    expect(Array.from(await history.compute(STATE))).toEqual([5, 50, 3, 30, 2, 20]);
  });

  it('re-primes after a reset instead of carrying the old episode', async () => {
    const { history, base } = build({ name: 'x', history_length: 2 });
    await history.compute(STATE);
    await history.compute(STATE);
    history.reset(STATE);
    expect(base.resets).toBe(1);
    expect(Array.from(await history.compute(STATE))).toEqual([3, 30, 3, 30]);
  });

  it('lays the stack out element-major when interleaved', async () => {
    const { history } = build({ name: 'x', history_length: 2, history_interleaved: true });
    await history.compute(STATE);
    // [e0_t-1, e0_t, e1_t-1, e1_t] rather than [e0_t-1, e1_t-1, e0_t, e1_t].
    expect(Array.from(await history.compute(STATE))).toEqual([1, 2, 10, 20]);
  });
});
