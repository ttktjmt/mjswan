// esbuild aliases `three` to this shim when bundling author custom-MDP plugins
// (see build/frontend.py's build_plugins_module), so a plugin's `import * as THREE
// from 'three'` resolves to the engine bundle's single three instance (exposed
// on globalThis by src/engine/index.ts). One shared instance across the engine
// and the separately-loaded plugin ESM makes instanceof / shared scene objects
// / raycasting work — a bundled duplicate would silently break them.
const THREE = globalThis.__mjswanThree;
if (!THREE) {
  throw new Error(
    'mjswan: the engine must be loaded before a custom-MDP plugin (three not initialized).',
  );
}
module.exports = THREE;
