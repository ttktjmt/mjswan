/**
 * Per-term observation history (mjlab's `ObservationTermCfg.history_length`), wrapping a
 * single term because mjlab stacks each term's frames *before* concatenating — where
 * `PolicyRunner`'s group-level buffer would give step-major order. That is also why a
 * group with per-term history does not fuse.
 *
 * `offsets` generalises the count: dense `history_length: n` arrives as `[n-1..0]`, the
 * chronological stack mjlab's history buffer flattens, and a policy trained on a sparse
 * or reversed window names its offsets directly. The buffer holds `max(offsets) + 1`
 * frames; only the named ones reach the output.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import type { PolicyState } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';

/** Write `frame` element-major at slot `index` of `count`: `[a_0, a_1, …, b_0, …]`, each
 * element's own frames adjacent. */
export function writeInterleavedFrame(
  out: Float32Array,
  frame: Float32Array,
  width: number,
  index: number,
  count: number,
): void {
  for (let j = 0; j < width; j++) out[j * count + index] = frame[j];
}

/** Offsets a term's history is sampled at, or null when it keeps no history. */
export function historyOffsets(entry: ObservationConfig): number[] | null {
  const sparse = (entry as { history_offsets?: unknown }).history_offsets;
  if (Array.isArray(sparse)) {
    const offsets = sparse.map((value) => Math.max(0, Math.trunc(Number(value) || 0)));
    return offsets.length > 0 ? offsets : null;
  }
  const length = Math.trunc(Number((entry as { history_length?: unknown }).history_length) || 0);
  if (length <= 1) return null;
  // Oldest frame first, as mjlab's `CircularBuffer.buffer` flattens it.
  return Array.from({ length }, (_, i) => length - 1 - i);
}

export class HistoryObservation extends ObservationBase {
  private readonly base: ObservationBase;
  private readonly offsets: number[];
  private readonly interleaved: boolean;
  private readonly frames: Float32Array[];
  /** Set by `reset()`: the next frame fills every slot instead of shifting in. */
  private needsPrime = true;

  constructor(
    runner: PolicyRunner,
    config: ObservationConfig,
    base: ObservationBase,
    offsets: number[],
  ) {
    super(runner, config);
    this.base = base;
    this.offsets = offsets;
    this.interleaved = Boolean((config as { history_interleaved?: unknown }).history_interleaved);
    this.frames = Array.from(
      { length: Math.max(...offsets) + 1 },
      () => new Float32Array(base.size),
    );
  }

  get size(): number {
    return this.base.size * this.offsets.length;
  }

  reset(state?: PolicyState): void {
    this.base.reset?.(state);
    this.needsPrime = true;
  }

  update(state: PolicyState): void {
    this.base.update?.(state);
  }

  preload(): Promise<void> {
    return this.base.preload?.() ?? Promise.resolve();
  }

  async compute(state: PolicyState): Promise<Float32Array> {
    const frame = Float32Array.from(await this.base.compute(state));
    if (this.needsPrime) {
      // First frame after a reset: fill every slot, never a history of untrained zeros.
      for (const slot of this.frames) slot.set(frame.subarray(0, slot.length));
      this.needsPrime = false;
    } else {
      // Rotate rather than copy: the oldest buffer becomes this frame's slot.
      const oldest = this.frames.pop();
      if (oldest) {
        oldest.set(frame.subarray(0, oldest.length));
        this.frames.unshift(oldest);
      }
    }
    return this.gather();
  }

  /** Frame-major, one full vector per offset in order, or element-major when interleaved. */
  private gather(): Float32Array {
    const width = this.base.size;
    const out = new Float32Array(width * this.offsets.length);
    for (let i = 0; i < this.offsets.length; i++) {
      const frame = this.frames[Math.min(this.offsets[i], this.frames.length - 1)];
      if (!frame) continue;
      if (this.interleaved) {
        writeInterleavedFrame(out, frame, width, i, this.offsets.length);
      } else {
        out.set(frame.subarray(0, width), i * width);
      }
    }
    return out;
  }
}
