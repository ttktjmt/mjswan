// Mirrors the `window.__harness` shape published by src/harness/e2e-entry.ts,
// for the Playwright specs (compiled separately from the app's tsconfig).
interface Window {
  __harness?: {
    ok: boolean;
    error?: string;
    running?: boolean;
    nonBlank?: boolean;
    luminanceRange?: [number, number];
    termSeed?: number;
  };
  /** Published by src/harness/interaction-entry.ts for the pointer-mode spec. */
  __engine?: unknown;
  __ready?: boolean;
  __probe?: {
    reset(): void;
    read(): { maxForce: number; welded: boolean };
  };
}
