/**
 * Public entry for the headless mjswan engine. See context/frontend.md.
 *
 * `createEngine(element, options?)` prepares a pure-TS, instance-scoped engine
 * (MuJoCo WASM + WebGL, no React) and exposes bytes-in / snapshot-out verbs.
 */
import * as THREE from 'three';

// Expose this bundle's single three.js instance for author custom-MDP plugins. Plugins
// are compiled to a *separate* ESM (plugins.js) whose `import 'three'` the Builder's
// esbuild resolves to a shim reading this global, so the engine and plugin bundles share
// one three instance and `instanceof THREE.Mesh`, shared scene objects and raycasting
// work. The engine module must load before any plugin module (the app guarantees this).
(globalThis as unknown as { __mjswanThree?: typeof THREE }).__mjswanThree = THREE;

export * from './types';
export { createEngine } from './createEngine';
export { createEngine as default } from './createEngine';
// Which traced graphs a config refers to, for a caller assembling inputs by hand.
export { policyGraphRefs } from '../core/onnx/graphRefs';
// MuJoCo's graphics defaults, and every tone mapping, for a host drawing its own controls.
export { DEFAULT_GRAPHICS, TONE_MAPPINGS } from '../core/scene/graphicsSettings';
