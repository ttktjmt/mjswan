/**
 * Meshes drawn see-through, over whatever is in front of them and casting no shadow,
 * until put back: how the Xray switch shows the robot a skin hides through the skin.
 */
import * as THREE from 'three';

const OPACITY = 0.4;
// Drawn without a depth test, so it must come last: three sorts transparent meshes by
// their origins' depth, and a splat is one mesh for all its points.
const LAST = Number.MAX_SAFE_INTEGER;

function seeThrough(material: THREE.Material): THREE.Material {
  const out = material.clone();
  out.transparent = true;
  out.opacity = OPACITY;
  out.depthTest = false;
  out.depthWrite = false;
  return out;
}

export class SeeThrough {
  private readonly own = new Map<THREE.Mesh, Pick<THREE.Mesh, 'material' | 'castShadow' | 'renderOrder'>>();

  /** Every mesh under `objects` that is not see-through already. */
  apply(objects: Iterable<THREE.Object3D | undefined>): void {
    for (const object of objects) {
      object?.traverse((child) => {
        const mesh = child as THREE.Mesh;
        if (!mesh.isMesh || this.own.has(mesh)) return;
        this.own.set(mesh, { material: mesh.material, castShadow: mesh.castShadow, renderOrder: mesh.renderOrder });
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map(seeThrough) : seeThrough(mesh.material);
        mesh.castShadow = false;
        mesh.renderOrder = LAST;
      });
    }
  }

  /** Run `draw` with every mesh as it was, as a camera sensor sees the robot, then see-through again. */
  asOwn<T>(draw: () => T): T {
    const through = [...this.own.keys()].map((mesh) => {
      const now = { material: mesh.material, castShadow: mesh.castShadow, renderOrder: mesh.renderOrder };
      Object.assign(mesh, this.own.get(mesh));
      return [mesh, now] as const;
    });
    try {
      return draw();
    } finally {
      for (const [mesh, now] of through) Object.assign(mesh, now);
    }
  }

  /** Every mesh back as it was. */
  restore(): void {
    for (const [mesh, own] of this.own) {
      for (const material of [mesh.material].flat()) material.dispose();
      Object.assign(mesh, own);
    }
    this.own.clear();
  }
}
