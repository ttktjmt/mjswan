/**
 * Browser glue for the pointer-interaction E2E: loads a scene and puts the engine on
 * `window` for Playwright. A push lasts one control step, so a sampler latches the effects
 * rather than the test reading them at a guessed moment.
 */
import { createEngine } from '../engine';

/** Fast enough to catch a one-control-step shove (20 ms at the default rate). */
const SAMPLE_MS = 4;

export interface InteractionProbe {
  /** Largest force any body has been under since the last `reset`: pull and push. */
  peakForce: number;
  /** Whether any equality has been active since the last `reset`: grab. */
  welded: boolean;
}

declare global {
  interface Window {
    __engine?: unknown;
    __ready?: boolean;
    __probe?: { reset(): void; read(): InteractionProbe };
  }
}

async function main(): Promise<void> {
  const element = document.getElementById('app');
  if (!element) throw new Error('missing #app');

  const sceneUrl = new URLSearchParams(location.search).get('scene') ?? '/fixtures/blocks.mjz';
  const engine = await createEngine(element, { termSeed: 0xc0ffee });
  const model = await (await fetch(sceneUrl)).arrayBuffer();
  await engine.loadScene({ model });
  window.__engine = engine;

  // Past the public API on purpose: it has no read of simulation state, and what the body
  // felt is what a mode test has to assert.
  const runtime = (
    engine as unknown as {
      runtime: {
        mjModel: { nbody: number; neq: number };
        mjData: { xfrc_applied: ArrayLike<number>; eq_active: ArrayLike<number> };
      };
    }
  ).runtime;

  let peakForce = 0;
  let welded = false;
  setInterval(() => {
    const { mjData, mjModel } = runtime;
    for (let body = 1; body < mjModel.nbody; body++) {
      const at = body * 6;
      const force = Math.hypot(mjData.xfrc_applied[at], mjData.xfrc_applied[at + 1], mjData.xfrc_applied[at + 2]);
      if (force > peakForce) peakForce = force;
    }
    for (let eq = 0; eq < mjModel.neq; eq++) {
      if (mjData.eq_active[eq]) welded = true;
    }
  }, SAMPLE_MS);

  window.__probe = {
    reset: () => {
      peakForce = 0;
      welded = false;
    },
    read: () => ({ peakForce, welded }),
  };

  // Let the physics and render loops advance before anything presses on the canvas.
  await new Promise((resolve) => setTimeout(resolve, 800));
  window.__ready = true;
}

void main();
