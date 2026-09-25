/**
 * Browser glue for the pointer-interaction E2E.
 *
 * The modes are the one part of the engine whose unit tests cannot reach what actually
 * breaks: they drive MuJoCo directly, with no pointer, no camera and no raycast. This
 * loads a scene and leaves the engine on `window` so Playwright can press a real mouse
 * against a real canvas and watch the simulation answer.
 *
 * A pull lasts as long as the drag but a push lasts one control step, so the effects are
 * *latched* by a sampler rather than read at a moment the test would have to guess.
 */
import { createEngine } from '../engine';

/** Fast enough to catch a one-control-step shove (20 ms at the default rate). */
const SAMPLE_MS = 4;

export interface InteractionProbe {
  /** Largest force any body has been under since the last `reset`: pull and push. */
  maxForce: number;
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

  // Reaching past the public API on purpose: there is no public read of simulation state,
  // and "did the body actually feel it" is the only honest assertion for a mode. Test-only
  // glue, which is what src/harness is.
  const runtime = (
    engine as unknown as {
      runtime: {
        mjModel: { nbody: number; neq: number };
        mjData: { xfrc_applied: ArrayLike<number>; eq_active: ArrayLike<number> };
      };
    }
  ).runtime;

  let maxForce = 0;
  let welded = false;
  setInterval(() => {
    const { mjData, mjModel } = runtime;
    for (let body = 1; body < mjModel.nbody; body++) {
      const at = body * 6;
      const force = Math.hypot(mjData.xfrc_applied[at], mjData.xfrc_applied[at + 1], mjData.xfrc_applied[at + 2]);
      if (force > maxForce) maxForce = force;
    }
    for (let eq = 0; eq < mjModel.neq; eq++) {
      if (mjData.eq_active[eq]) welded = true;
    }
  }, SAMPLE_MS);

  window.__probe = {
    reset: () => {
      maxForce = 0;
      welded = false;
    },
    read: () => ({ maxForce, welded }),
  };

  // Let the physics and render loops advance before anything presses on the canvas.
  await new Promise((resolve) => setTimeout(resolve, 800));
  window.__ready = true;
}

void main();
