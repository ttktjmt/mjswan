/** Ends a timed freeze: calls `end` at `resumesAt` (a `Date.now()` time) unless set again or cancelled first. */
export class FreezeTimer {
  private handle: ReturnType<typeof setTimeout> | null = null;

  constructor(
    private readonly end: () => void,
    private readonly now: () => number = Date.now,
  ) {}

  /** Null waits for a reset: nothing is timed. */
  set(resumesAt: number | null): void {
    this.cancel();
    if (resumesAt === null) return;
    this.handle = setTimeout(() => {
      this.handle = null;
      this.end();
    }, Math.max(0, resumesAt - this.now()));
  }

  cancel(): void {
    if (this.handle !== null) clearTimeout(this.handle);
    this.handle = null;
  }
}
