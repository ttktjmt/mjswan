import type * as THREE from 'three';
import { addShaderHook } from './shaderHook';

const cubes = new WeakMap<THREE.Material, { value: THREE.CubeTexture }>();

/**
 * Has `material` sample `cube`, in MuJoCo's face order, at the geom-local point over `size`,
 * as MuJoCo does. The geometry is in three's frame, where MuJoCo's (x, y, z) is (x, -z, y).
 */
export function sampleCubeTexture(
  material: THREE.Material,
  cube: THREE.CubeTexture,
  size: readonly number[],
): void {
  const uniform = { value: cube };
  cubes.set(material, uniform);
  addShaderHook(material, 'mjCube', (shader) => {
    shader.uniforms.mjCube = uniform;
    shader.uniforms.mjCubeSize = { value: size };
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nuniform vec3 mjCubeSize;\nvarying vec3 vMjCubeDir;')
      .replace(
        '#include <begin_vertex>',
        '#include <begin_vertex>\nvMjCubeDir = vec3(position.x, -position.z, position.y) / mjCubeSize;',
      );
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform samplerCube mjCube;\nvarying vec3 vMjCubeDir;')
      .replace('#include <map_fragment>', 'diffuseColor *= textureCube(mjCube, vMjCubeDir);');
  });
  material.addEventListener('dispose', () => cube.dispose());
}

/** The cube texture `material` samples, which a pass may swap for another. */
export function sampledCube(material: THREE.Material): { value: THREE.CubeTexture } | undefined {
  return cubes.get(material);
}
