/**
 * The handles a host can put on the preview (ADR 0011): arrows that resize it and rings that
 * turn it, along and about its own axes (x red, y green, z blue). They are drawn a fixed size
 * on screen and over everything else, as an editor's gizmo is, and the math of a drag on
 * them is here too, in MuJoCo's axes.
 */
import * as THREE from 'three';

import { mjcToThreeCoordinate } from './coordinate';
import type { PrimitiveType } from './primitive';

/** One primitive geom in MuJoCo's world axes and units: what a drag on a handle changes. */
export interface PreviewGeom {
  type: PrimitiveType;
  size: number[];
  pos: number[];
  /** (w, x, y, z). */
  quat: number[];
  rgba: number[];
}

export type HandleKind = 'size' | 'turn';
export type Axis = 0 | 1 | 2;

/** The arrow along a local axis that drives one `size` entry, or the ring about an axis. */
export type Handle = { kind: 'size'; axis: Axis; index: number } | { kind: 'turn'; axis: Axis };

/** A half-size stops shrinking here, metres. */
const MIN_HALF_SIZE = 0.005;
const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];

/** The arrows a type gets: a radius along x, a half-length along z, an extent along its own axis. */
export function sizeArrows(type: PrimitiveType): { axis: Axis; index: number }[] {
  switch (type) {
    case 'box':
    case 'ellipsoid':
      return [{ axis: 0, index: 0 }, { axis: 1, index: 1 }, { axis: 2, index: 2 }];
    case 'plane':
      return [{ axis: 0, index: 0 }, { axis: 1, index: 1 }];
    case 'sphere':
      return [{ axis: 0, index: 0 }];
    case 'capsule':
    case 'cylinder':
      return [{ axis: 0, index: 0 }, { axis: 2, index: 1 }];
  }
}

/** How far the surface lies from the centre along a local axis. */
export function extent(type: PrimitiveType, size: ArrayLike<number>, axis: Axis): number {
  switch (type) {
    case 'sphere':
      return size[0];
    case 'capsule':
      return axis === 2 ? size[0] + size[1] : size[0];
    case 'cylinder':
      return axis === 2 ? size[1] : size[0];
    case 'plane':
      return axis === 2 ? 0 : size[axis];
    default:
      return size[axis];
  }
}

const quaternionOf = (quat: ArrayLike<number>) => new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]);

/** A local axis of the geom in world axes. */
export function axisOf(quat: ArrayLike<number>, axis: Axis): THREE.Vector3 {
  return AXES[axis].clone().applyQuaternion(quaternionOf(quat));
}

/**
 * The geom with the face under an arrow moved `distance` along it. The opposite face stays,
 * and so does the bottom (local -z) when the entry sets the height too, as a sphere's or a
 * capsule's radius does, so a geom resting on something keeps resting there.
 */
export function resized(geom: PreviewGeom, axis: Axis, index: number, distance: number): PreviewGeom {
  const size = [...geom.size];
  const grow = Math.max(MIN_HALF_SIZE, size[index] + distance / 2) - size[index];
  size[index] += grow;
  const shift = new THREE.Vector3().setComponent(axis, grow);
  if (axis !== 2 && (geom.type === 'sphere' || geom.type === 'capsule')) shift.z += grow;
  shift.applyQuaternion(quaternionOf(geom.quat));
  return { ...geom, size, pos: geom.pos.map((v, i) => v + shift.getComponent(i)) };
}

/** The geom turned `angle` radians about one of its own axes, through its centre. */
export function turned(geom: PreviewGeom, axis: Axis, angle: number): PreviewGeom {
  const q = quaternionOf(geom.quat).multiply(new THREE.Quaternion().setFromAxisAngle(AXES[axis], angle));
  return { ...geom, quat: [q.w, q.x, q.y, q.z] };
}

/** CSS pixels. */
const ARROW = { gap: 6, length: 56, head: 14, shaft: 1.5, headRadius: 5, hit: 10 };
const RING = { radius: 64, tube: 1.5, hit: 8 };
/** Linear RGB, which this renderer outputs as it is. */
const AXIS_COLORS = [
  [0.93, 0.26, 0.3],
  [0.38, 0.78, 0.22],
  [0.2, 0.5, 1],
];
const REST_OPACITY = 0.85;
/** An arrow this close to the line of sight is a dot, so it is hidden. */
const EDGE_ON = 0.97;
const UP = new THREE.Vector3(0, 1, 0);
const FORWARD = new THREE.Vector3(0, 0, 1);

interface Part {
  handle: Handle;
  root: THREE.Group;
  hit: THREE.Mesh;
  materials: THREE.MeshBasicMaterial[];
}

/** One set of handles for one kind and type, laid out every frame by `layout`. */
export class PreviewHandles {
  readonly group = new THREE.Group();
  private readonly parts: Part[] = [];
  private readonly geometries: THREE.BufferGeometry[] = [];
  private readonly hitMaterial = new THREE.MeshBasicMaterial({ visible: false });

  constructor(
    readonly kind: HandleKind,
    readonly type: PrimitiveType
  ) {
    if (kind === 'size') {
      const tip = ARROW.gap + ARROW.length;
      const shaft = new THREE.CylinderGeometry(ARROW.shaft, ARROW.shaft, ARROW.length - ARROW.head, 8);
      shaft.translate(0, ARROW.gap + (ARROW.length - ARROW.head) / 2, 0);
      const head = new THREE.ConeGeometry(ARROW.headRadius, ARROW.head, 16);
      head.translate(0, tip - ARROW.head / 2, 0);
      const hit = new THREE.CylinderGeometry(ARROW.hit, ARROW.hit, tip, 8);
      hit.translate(0, tip / 2, 0);
      this.geometries.push(shaft, head, hit);
      for (const { axis, index } of sizeArrows(type)) {
        this.add({ kind, axis, index }, [shaft, head], hit);
      }
    } else {
      const ring = new THREE.TorusGeometry(RING.radius, RING.tube, 6, 96);
      const hit = new THREE.TorusGeometry(RING.radius, RING.hit, 6, 48);
      this.geometries.push(ring, hit);
      for (const axis of [0, 1, 2] as const) this.add({ kind, axis }, [ring], hit);
    }
    this.group.userData.interactionGizmo = true;
  }

  /** Lays them on `geom`, at `worldPerPx` world units to a CSS pixel, seen from `eye` (three.js axes). */
  layout(geom: PreviewGeom, worldPerPx: number, eye: THREE.Vector3): void {
    const centre = mjcToThreeCoordinate(geom.pos);
    for (const part of this.parts) {
      const { root, handle } = part;
      const direction = mjcToThreeCoordinate(axisOf(geom.quat, handle.axis).toArray());
      root.scale.setScalar(worldPerPx);
      if (handle.kind === 'size') {
        root.position.copy(centre).addScaledVector(direction, extent(geom.type, geom.size, handle.axis));
        root.quaternion.setFromUnitVectors(UP, direction);
        const sight = eye.clone().sub(root.position).normalize();
        root.visible = Math.abs(sight.dot(direction)) < EDGE_ON;
      } else {
        root.position.copy(centre);
        root.quaternion.setFromUnitVectors(FORWARD, direction);
      }
    }
    this.group.updateMatrixWorld(true);
  }

  /** The nearest handle the ray meets, among those on show. */
  hit(raycaster: THREE.Raycaster): Handle | null {
    const shown = this.parts.filter((part) => part.root.visible).map((part) => part.hit);
    const [first] = raycaster.intersectObjects(shown, false);
    return first ? (first.object.userData.handle as Handle) : null;
  }

  /** The handle under the pointer, or being dragged, at full strength and lightened. */
  highlight(handle: Handle | null): void {
    for (const part of this.parts) {
      const lit = handle !== null && handle.kind === part.handle.kind && handle.axis === part.handle.axis;
      const [r, g, b] = AXIS_COLORS[part.handle.axis];
      const lift = lit ? 0.45 : 0;
      for (const material of part.materials) {
        material.color.setRGB(r + (1 - r) * lift, g + (1 - g) * lift, b + (1 - b) * lift);
        material.opacity = lit ? 1 : REST_OPACITY;
      }
    }
  }

  dispose(): void {
    this.group.removeFromParent();
    for (const geometry of this.geometries) geometry.dispose();
    for (const part of this.parts) for (const material of part.materials) material.dispose();
    this.hitMaterial.dispose();
  }

  private add(handle: Handle, shapes: THREE.BufferGeometry[], hitShape: THREE.BufferGeometry): void {
    const root = new THREE.Group();
    const [r, g, b] = AXIS_COLORS[handle.axis];
    // Its own material per mesh: one is lit at a time.
    const materials = shapes.map(
      () =>
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setRGB(r, g, b),
          transparent: true,
          opacity: REST_OPACITY,
          depthTest: false,
          depthWrite: false,
        })
    );
    shapes.forEach((shape, i) => {
      const mesh = new THREE.Mesh(shape, materials[i]);
      mesh.renderOrder = 1000;
      root.add(mesh);
    });
    const hit = new THREE.Mesh(hitShape, this.hitMaterial);
    hit.userData.handle = handle;
    root.add(hit);
    root.traverse((child) => {
      child.userData.interactionGizmo = true;
    });
    this.group.add(root);
    this.parts.push({ handle, root, hit, materials });
  }
}
