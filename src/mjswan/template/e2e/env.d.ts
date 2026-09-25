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
}
