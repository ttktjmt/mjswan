/**
 * Browser glue for the Playwright engine E2E (issue #76 step 1i, runtime tier).
 *
 * React-free: imports ONLY `createEngine`, loads a fixture scene by bytes (`?scene=`, with
 * `?format=mjb` for a compiled model), lets the physics + render loops run, then reads back
 * `captureThumbnail()` and checks the frame is non-blank. Result is published on
 * `window.__harness` for the test.
 */
import { createEngine } from '../engine';
import type { MjswanEngineState } from '../engine';

export interface HarnessOutcome {
  ok: boolean;
  error?: string;
  /** True if any pushed snapshot reached the 'running' phase. */
  running?: boolean;
  /** True if the captured frame has luminance spread (not a flat/blank canvas). */
  nonBlank?: boolean;
  luminanceRange?: [number, number];
  /** The seed the engine reports back, for the ADR 0005 §2 replay path. */
  termSeed?: number;
}

declare global {
  interface Window {
    __harness?: HarnessOutcome;
  }
}

/** Arbitrary but not the built-in default, so a dropped option is visible. */
const HARNESS_SEED = 0xc0ffee;

async function main(): Promise<void> {
  const element = document.getElementById('app');
  if (!element) throw new Error('missing #app');

  const params = new URLSearchParams(location.search);
  const sceneUrl = params.get('scene') ?? '/fixtures/container.mjz';
  const modelFormat = params.get('format') === 'mjb' ? 'mjb' : 'mjz';
  const states: MjswanEngineState[] = [];

  // Caller-chosen, so the test checks the value survives option → runtime → snapshot.
  const engine = await createEngine(element, { termSeed: HARNESS_SEED });
  engine.subscribe((state) => states.push(state));

  const model = await (await fetch(sceneUrl)).arrayBuffer();
  await engine.loadScene({ model, modelFormat });

  // Let the setTimeout physics loop and the rAF render loop advance a few frames.
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const blob = await engine.captureThumbnail({ maxDim: 128 });
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('no 2d context');
  ctx.drawImage(bitmap, 0, 0);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  let min = 255;
  let max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
    if (lum < min) min = lum;
    if (lum > max) max = lum;
  }

  window.__harness = {
    ok: true,
    termSeed: engine.getState().termSeed,
    running: states.some((s) => s.phase === 'running'),
    nonBlank: max - min > 8,
    luminanceRange: [min, max],
  };
  engine.dispose();
}

main().catch((err) => {
  window.__harness = { ok: false, error: err instanceof Error ? err.message : String(err) };
});
