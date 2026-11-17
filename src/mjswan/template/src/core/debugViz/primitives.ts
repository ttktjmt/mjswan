/**
 * Batched arrows and spheres for drawings with many primitives (a height scan's rays),
 * shaped and lit as viser draws mjlab's, without shadows. Points are MuJoCo world
 * coordinates.
 */

import * as THREE from 'three';

import { mjcToThreeCoordinate } from '../scene/coordinate';
import { colorAsRgba } from '../scene/graphics';

export type Vec3 = ArrayLike<number>;
export type Rgba = readonly [number, number, number, number];

const UP = new THREE.Vector3(0, 1, 0);
const SHAFT_FRACTION = 0.8;
/** trimesh's default `sections` for the cylinder and cone viser batches. */
const RADIAL_SEGMENTS = 32;

/** viser's `standard` mesh material at the opacity it was given (`MeshUtils.tsx`). */
export function viserMaterial(
  rgb: readonly number[] = [1, 1, 1],
  opacity = 1,
): THREE.MeshStandardMaterial {
  return colorAsRgba(
    new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgb[0], rgb[1], rgb[2]),
      transparent: true,
      opacity,
    }),
  );
}

/** trimesh's `cylinder(radius=1, height=1)`, base at the origin along +y. */
export function unitShaft(): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(1, 1, 1, RADIAL_SEGMENTS).translate(0, 0.5, 0);
}

/** trimesh's `cone(radius=2, height=1)`, base at the origin along +y. */
export function unitHead(): THREE.BufferGeometry {
  return new THREE.ConeGeometry(2, 1, RADIAL_SEGMENTS).translate(0, 0.5, 0);
}

/** trimesh's `icosphere(subdivisions=2)`: 320 faces, as three's detail 3. */
export function unitSphere(): THREE.BufferGeometry {
  return new THREE.IcosahedronGeometry(1, 3);
}

function instanced(
  geometry: THREE.BufferGeometry,
  material: THREE.Material,
  capacity: number,
  name: string,
): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(geometry, material, Math.max(capacity, 1));
  mesh.name = name;
  mesh.count = 0;
  // Instances span the scene; the geometry's own bounds say nothing about them.
  mesh.frustumCulled = false;
  return mesh;
}

/** A redraw-each-frame arrow list: `clear()`, `add()` each arrow, then `commit()`. */
export class ArrowBatch {
  readonly object = new THREE.Group();
  private readonly shafts: THREE.InstancedMesh;
  private readonly heads: THREE.InstancedMesh;
  private count = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly quat = new THREE.Quaternion();
  private readonly color = new THREE.Color();

  constructor(
    private capacity: number,
    name: string,
  ) {
    this.object.name = name;
    const material = viserMaterial();
    this.shafts = instanced(unitShaft(), material, capacity, `${name}-shafts`);
    this.heads = instanced(
      unitHead(),
      material,
      capacity,
      `${name}-heads`,
    );
    this.object.add(this.shafts, this.heads);
  }

  clear(): void {
    this.count = 0;
  }

  /** mjlab's `add_arrow`: skipped when shorter than 1e-6, as viser skips it. */
  add(start: Vec3, end: Vec3, rgba: Rgba, width: number): void {
    const from = mjcToThreeCoordinate(start);
    const direction = mjcToThreeCoordinate(end).sub(from);
    const length = direction.length();
    if (length < 1e-6 || this.count >= this.capacity) return;
    direction.divideScalar(length);
    this.quat.setFromUnitVectors(UP, direction);
    this.color.setRGB(rgba[0], rgba[1], rgba[2]);

    const shaft = length * SHAFT_FRACTION;
    this.matrix.compose(from, this.quat, new THREE.Vector3(width, shaft, width));
    this.shafts.setMatrixAt(this.count, this.matrix);
    this.shafts.setColorAt(this.count, this.color);
    const headStart = from.clone().addScaledVector(direction, shaft);
    this.matrix.compose(headStart, this.quat, new THREE.Vector3(width, length - shaft, width));
    this.heads.setMatrixAt(this.count, this.matrix);
    this.heads.setColorAt(this.count, this.color);
    this.count += 1;
  }

  commit(): void {
    for (const mesh of [this.shafts, this.heads]) {
      mesh.count = this.count;
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  dispose(): void {
    this.object.removeFromParent();
    for (const mesh of [this.shafts, this.heads]) mesh.geometry.dispose();
    (this.shafts.material as THREE.Material).dispose();
  }
}

/** Spheres sharing one opacity, as one viser sphere batch does. */
export class SphereBatch {
  readonly object: THREE.InstancedMesh;
  private count = 0;
  private readonly matrix = new THREE.Matrix4();
  private readonly color = new THREE.Color();
  private readonly identity = new THREE.Quaternion();

  constructor(
    private capacity: number,
    name: string,
    opacity: number,
  ) {
    this.object = instanced(
      unitSphere(),
      viserMaterial([1, 1, 1], opacity),
      capacity,
      name,
    );
  }

  clear(): void {
    this.count = 0;
  }

  add(center: Vec3, radius: number, rgba: Rgba): void {
    if (this.count >= this.capacity) return;
    this.matrix.compose(
      mjcToThreeCoordinate(center),
      this.identity,
      new THREE.Vector3(radius, radius, radius),
    );
    this.object.setMatrixAt(this.count, this.matrix);
    this.object.setColorAt(this.count, this.color.setRGB(rgba[0], rgba[1], rgba[2]));
    this.count += 1;
  }

  commit(): void {
    this.object.count = this.count;
    this.object.instanceMatrix.needsUpdate = true;
    if (this.object.instanceColor) this.object.instanceColor.needsUpdate = true;
  }

  dispose(): void {
    this.object.removeFromParent();
    this.object.geometry.dispose();
    (this.object.material as THREE.Material).dispose();
  }
}

const DEFAULT_AXIS_COLORS: readonly Rgba[] = [
  [0.9, 0, 0, 1],
  [0, 0.9, 0, 1],
  [0, 0, 0.9, 1],
];

/** mjlab's `add_frame`: one `scale`-long arrow per axis, `quat` being (w, x, y, z). */
export function addFrame(
  arrows: ArrowBatch,
  position: Vec3,
  quat: Vec3,
  scale: number,
  axisColors: readonly Rgba[] = DEFAULT_AXIS_COLORS,
  axisRadius = 0.01,
): void {
  const [w, x, y, z] = [quat[0], quat[1], quat[2], quat[3]];
  // Columns of the rotation matrix: where each body axis points in the world.
  const columns = [
    [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
    [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
    [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
  ];
  columns.forEach((axis, i) => {
    const end = [
      position[0] + axis[0] * scale,
      position[1] + axis[1] * scale,
      position[2] + axis[2] * scale,
    ];
    arrows.add(position, end, axisColors[i], axisRadius);
  });
}
