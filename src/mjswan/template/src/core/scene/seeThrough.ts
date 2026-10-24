/**
 * Meshes drawn see-through, over whatever is in front of them and casting no shadow,
 * until put back: how the Xray switch shows the robot a skin hides through the skin.
 */
import * as THREE from 'three';

const OPACITY = 0.4;

function seeThrough(material: THREE.Material): THREE.Material {
  const out = material.clone();
  out.transparent = true;
  out.opacity = OPACITY;
  out.depthTest = false;
  out.depthWrite = false;
  return out;
}

export class SeeThrough {
  private readonly own = new Map<THREE.Mesh, { material: THREE.Material | THREE.Material[]; castShadow: boolean }>();

  /** Every mesh under `objects` that is not see-through already. */
  apply(objects: Iterable<THREE.Object3D | undefined>): void {
    for (const object of objects) {
      object?.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh || this.own.has(mesh)) return;
        this.own.set(mesh, { material: mesh.material, castShadow: mesh.castShadow });
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map(seeThrough) : seeThrough(mesh.material);
        mesh.castShadow = false;
      });
    }
  }

  /** Every mesh back on its own materials. */
  restore(): void {
    for (const [mesh, own] of this.own) {
      for (const material of [mesh.material].flat()) material.dispose();
      mesh.material = own.material;
      mesh.castShadow = own.castShadow;
    }
    this.own.clear();
  }
}
