/**
 * Browser glue for the pointer-interaction E2E.
 *
 * The modes are the one part of the engine whose unit tests cannot reach the thing that
 * actually breaks: they drive MuJoCo directly, with no pointer, no camera and no raycast.
 * This loads a scene and leaves the engine on `window` so Playwright can press the mouse
 * against a real canvas and watch the simulation answer.
 */
import { createEngine } from '../engine';

/** Parked slots sag by one step of gravity between writes; well under the parking height. */
const PARKED_Z = 50;

export interface PoolProbe {
  /** Height of every throwable box. Above {@link PARKED_Z} means it is still in the rack. */
  poolZ: number[];
  thrown: number;
}

declare global {
  interface Window {
    __engine?: unknown;
    __ready?: boolean;
    __pool?: () => PoolProbe;
  }
}

/** How many the engine compiles in by default; the probe reads that many off the tail. */
const POOL_SLOTS = 8;

async function main(): Promise<void> {
  const element = document.getElementById('app');
  if (!element) throw new Error('missing #app');

  const sceneUrl = new URLSearchParams(location.search).get('scene') ?? '/fixtures/container.mjz';
  const engine = await createEngine(element, { termSeed: 0xc0ffee });
  const model = await (await fetch(sceneUrl)).arrayBuffer();
  await engine.loadScene({ model });
  window.__engine = engine;

  // Reaching past the public API on purpose: there is no public read of simulation state,
  // and "did a box actually appear" is the only honest assertion for a throw. Test-only
  // glue, which is what src/harness is.
  const runtime = (
    engine as unknown as { runtime: { mjModel: { nq: number }; mjData: { qpos: ArrayLike<number> } } }
  ).runtime;
  window.__pool = () => {
    const { nq } = runtime.mjModel;
    const poolZ: number[] = [];
    // The pool's free joints are the last ones appended, seven qpos each.
    for (let i = 0; i < POOL_SLOTS; i++) {
      poolZ.push(runtime.mjData.qpos[nq - 7 * (POOL_SLOTS - i) + 2]);
    }
    return { poolZ, thrown: poolZ.filter((z) => z < PARKED_Z).length };
  };

  // Let the physics and render loops advance before anything presses on the canvas.
  await new Promise((resolve) => setTimeout(resolve, 800));
  window.__ready = true;
}

void main();
