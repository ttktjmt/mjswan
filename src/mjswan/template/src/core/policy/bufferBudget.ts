/** Most frames one history, delay or contact ring may declare. */
export const MAX_FRAMES = 10_000;
/** Float-equivalents one policy load may buffer across all its rings (about 64 MB). */
export const MAX_BUFFERED = 16_000_000;
/** A frame object's heap beyond its data, in floats, so empty frames still count. */
const FRAME_OVERHEAD = 64;

/**
 * One policy load's allowance, drawn down before each ring a document sizes is allocated.
 * A per-count ceiling alone would not hold: counts under it still multiply and add up.
 */
export class BufferBudget {
  private remaining = MAX_BUFFERED;

  reserve(frames: number, width: number, label: string): void {
    const cost = frames * (width + FRAME_OVERHEAD);
    if (!(frames <= MAX_FRAMES) || !(cost <= this.remaining)) {
      throw new Error(`mjswan: ${label} declares ${frames} frames of width ${width}, past this policy's buffer limit.`);
    }
    this.remaining -= cost;
  }
}
