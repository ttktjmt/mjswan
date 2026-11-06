/** Most frames one buffer may declare. */
export const MAX_FRAMES = 10_000;
/** Floats one policy load may buffer in all (about 64 MB). */
export const MAX_BUFFERED = 16_000_000;
/** A frame object's heap beyond its data, in floats, so empty frames still count. */
const FRAME_OVERHEAD = 64;

/**
 * One policy load's allowance, drawn down for each buffer the document sizes.
 * A per-buffer ceiling alone would not hold: counts under it still multiply and add up.
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
