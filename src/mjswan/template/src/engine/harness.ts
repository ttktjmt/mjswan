/**
 * Headless acceptance harness for `createEngine`.
 *
 * Imports ONLY the engine (no React, no Mantine) and drives the full lifecycle
 * (create → subscribe → loadScene(bytes) → play/pause → dispose), proving the public API
 * is sufficient to run a simulation React-free. Run it in a browser context (WebGL +
 * WASM required); it is not a Node unit test.
 *
 * Usage (browser context, importing the in-repo module):
 *   import { runHarness } from './engine/harness';
 *   const model = await (await fetch('scene.mjz')).arrayBuffer();
 *   await runHarness(document.getElementById('app')!, { model });
 */
import { createEngine } from './createEngine';
import type { MjswanEngineState, SceneInput } from './types';

export interface HarnessResult {
  /** Every state snapshot pushed to subscribe, in order. */
  states: MjswanEngineState[];
  /** The final snapshot after the scripted run. */
  final: MjswanEngineState;
}

/** Drive one scene through the engine's verbs and return the observed states. */
export async function runHarness(
  element: HTMLElement,
  scene: SceneInput,
  options?: { multithreaded?: boolean },
): Promise<HarnessResult> {
  const states: MjswanEngineState[] = [];
  const engine = await createEngine(element, options);
  const off = engine.subscribe((state) => states.push(state));
  try {
    await engine.loadScene(scene);
    engine.pause();
    engine.play();
    engine.reset();
    return { states, final: engine.getState() };
  } finally {
    off();
    engine.dispose();
  }
}
