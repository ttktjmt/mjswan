/**
 * Orchestrator-owned seeded PRNG.
 *
 * The point of these tests is the *replay guarantee*: a recorded session must
 * reproduce bit-for-bit, so the sequence has to be a pure function of the seed
 * (and resumable from a state snapshot).
 */
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../rng';

describe('SeededRng', () => {
  it('is deterministic for a given seed', () => {
    const a = new SeededRng(12345);
    const b = new SeededRng(12345);
    const drawsA = Array.from({ length: 64 }, () => a.next());
    const drawsB = Array.from({ length: 64 }, () => b.next());
    expect(drawsA).toEqual(drawsB);
  });

  it('produces different streams for different seeds', () => {
    const a = new SeededRng(1);
    const b = new SeededRng(2);
    const drawsA = Array.from({ length: 16 }, () => a.next());
    const drawsB = Array.from({ length: 16 }, () => b.next());
    expect(drawsA).not.toEqual(drawsB);
  });

  it('stays within [0, 1)', () => {
    const rng = new SeededRng(7);
    for (let i = 0; i < 2000; i++) {
      const v = rng.next();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('uniform() respects its range', () => {
    const rng = new SeededRng(99);
    for (let i = 0; i < 1000; i++) {
      const v = rng.uniform(-0.5, 2.5);
      expect(v).toBeGreaterThanOrEqual(-0.5);
      expect(v).toBeLessThan(2.5);
    }
  });

  it('resumes exactly from a state snapshot (session replay)', () => {
    const rng = new SeededRng(4242);
    for (let i = 0; i < 10; i++) rng.next();
    const state = rng.getState();
    const expected = Array.from({ length: 8 }, () => rng.next());

    const resumed = new SeededRng(0);
    resumed.setState(state);
    const actual = Array.from({ length: 8 }, () => resumed.next());
    expect(actual).toEqual(expected);
  });

  it('randVector fills an ONNX rand input, scaling per-element ranges', () => {
    const rng = new SeededRng(5);
    const ranges: Array<readonly [number, number]> = [
      [-0.5, 0.5],
      [-0.5, 0.5],
      [-0.4, 0.4],
      [-0.52, 0.52],
      [-0.52, 0.52],
      [-0.78, 0.78],
    ];
    const vec = rng.randVector(6, ranges);
    expect(vec).toBeInstanceOf(Float32Array);
    expect(vec.length).toBe(6);
    for (let i = 0; i < 6; i++) {
      expect(vec[i]).toBeGreaterThanOrEqual(ranges[i][0]);
      expect(vec[i]).toBeLessThan(ranges[i][1]);
    }
  });

  it('randVector without ranges yields plain [0,1) draws', () => {
    const vec = new SeededRng(11).randVector(4);
    for (const v of vec) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('never returns the same value twice in a row over a long run', () => {
    const rng = new SeededRng(2026);
    let previous = rng.next();
    let repeats = 0;
    for (let i = 0; i < 5000; i++) {
      const v = rng.next();
      if (v === previous) repeats++;
      previous = v;
    }
    expect(repeats).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// What a seed alone does *not* buy.
//
// The tests above establish the half that holds: the sequence is a pure function of the
// seed, so startup DR and every resample schedule reproduce. *Bit-for-bit* replay does
// not, and is not pursued — it is not satisfiable alongside the non-blocking async
// boundary, since every traced term draws from one shared stream and each skips its run
// while a previous one is in flight. Whether a frame skips is wall-clock, so
// machine-dependent. Pinned here so the limitation stays visible.
// ---------------------------------------------------------------------------

describe('SeededRng — shared-stream coupling (documents a replay limitation)', () => {
  it('shifts every later draw when one consumer skips a frame', () => {
    // Two terms alternating draws off one stream, and the same two with term A skipping a
    // single frame — as it would on a machine where that frame's inference had not settled.
    const drawBoth = (skipFrame: number | null): number[] => {
      const rng = new SeededRng(4242);
      const fromB: number[] = [];
      for (let frame = 0; frame < 6; frame++) {
        if (frame !== skipFrame) rng.next(); // term A's `rand`
        fromB.push(rng.next()); // term B's `rand`
      }
      return fromB;
    };

    const steady = drawBoth(null);
    const withSkip = drawBoth(2);
    // Frames before the skip agree; after it, a term that did nothing differently diverges.
    expect(withSkip.slice(0, 2)).toEqual(steady.slice(0, 2));
    expect(withSkip.slice(2)).not.toEqual(steady.slice(2));
  });

  it('would be unaffected if draws were addressed by step instead of order', () => {
    // Records the shape that *would* decouple the streams — derive a draw from
    // (seed, term, step) so frame N gets the same numbers whether or not any other
    // term ran. Kept as documentation rather than as a plan: it closes the draw half
    // and not the trajectory half, since a termination verdict arriving a frame late
    // moves the reset frame no matter how the draws are addressed.
    const at = (term: string, step: number): number => {
      let h = 2166136261 >>> 0;
      for (const ch of `${4242}:${term}:${step}`) {
        h = (Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0) >>> 0;
      }
      return new SeededRng(h).next();
    };

    const steady = Array.from({ length: 6 }, (_, frame) => at('B', frame));
    // Term A skipping frame 2 cannot move term B's numbers: nothing is sequential.
    const withSkip = Array.from({ length: 6 }, (_, frame) => at('B', frame));
    expect(withSkip).toEqual(steady);
    // And the addressing is still seed-dependent, not a constant.
    expect(at('B', 0)).not.toEqual(at('B', 1));
    expect(at('A', 0)).not.toEqual(at('B', 0));
  });
});
