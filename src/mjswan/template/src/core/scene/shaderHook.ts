import type * as THREE from 'three';

type Hook = (shader: THREE.WebGLProgramParametersWithUniforms, renderer: THREE.WebGLRenderer) => void;

const hooks = new WeakMap<THREE.Material, Array<{ key: string; hook: Hook }>>();

/**
 * Add a shader edit to `material`, after any it already has. three.js takes one
 * `onBeforeCompile` and keys the program by its source, so two materials edited
 * differently through the same function would share one program: the key says which.
 */
export function addShaderHook(material: THREE.Material, key: string, hook: Hook): void {
  let list = hooks.get(material);
  if (!list) {
    const own: Array<{ key: string; hook: Hook }> = [];
    hooks.set(material, own);
    material.onBeforeCompile = (shader, renderer) => {
      for (const each of own) each.hook(shader, renderer);
    };
    material.customProgramCacheKey = () => own.map((each) => each.key).join('|');
    list = own;
  }
  list.push({ key, hook });
  material.needsUpdate = true;
}
