/**
 * A primitive geom as three.js draws it, from MuJoCo's `size` for its type: what the scene
 * draws a model's geoms with, and what `GeomPreview` draws one that is not in the model.
 * MuJoCo's z is three.js's y, so a geom's own z axis is the geometry's y.
 */
import * as THREE from 'three';

/** The primitive geom types, by MuJoCo's names (`mjtGeom` without its prefix). */
export type PrimitiveType = 'plane' | 'sphere' | 'capsule' | 'ellipsoid' | 'cylinder' | 'box';

/** A plane here is a finite one, its texture stretched over it: a model's planes are `plane.ts`'s. */
export function primitiveGeometry(type: PrimitiveType, size: ArrayLike<number>): THREE.BufferGeometry {
  switch (type) {
    case 'plane': {
      const geometry = new THREE.PlaneGeometry(size[0] * 2.0, size[1] * 2.0);
      geometry.rotateX(-Math.PI / 2);
      return geometry;
    }
    case 'sphere':
      return new THREE.SphereGeometry(size[0]);
    case 'capsule':
      return new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
    // A unit sphere, which the mesh scales to the radii (`primitiveScale`).
    case 'ellipsoid':
      return new THREE.SphereGeometry(1);
    case 'cylinder':
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
    case 'box':
      return new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
  }
}

/** The scale a primitive's mesh takes: an ellipsoid's radii, and none for the others. */
export function primitiveScale(
  type: PrimitiveType,
  size: ArrayLike<number>,
  target: THREE.Vector3
): THREE.Vector3 {
  return type === 'ellipsoid' ? target.set(size[0], size[2], size[1]) : target.set(1, 1, 1);
}
