/**
 * What lies under a point of the canvas (ADR 0011), by MuJoCo's own ray: the answer the
 * simulation would give, an infinite plane and a height field included, and nothing that
 * is only drawn (the mode gizmos, the preview, a skin, a splat).
 */
import * as THREE from 'three';

import { threeToMjcCoordinate } from '../scene/coordinate';
import { geomGroupMask } from '../onnx/raycast';
import type { PrimitiveType } from '../scene/primitive';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;
type DoubleBuffer = import('mujoco').DoubleBuffer;
type IntBuffer = import('mujoco').IntBuffer;

/** The groups the scene draws (`geom_group < 3`): a ray does not land on what no one sees. */
const DRAWN_GROUPS = geomGroupMask([0, 1, 2]);

/** MuJoCo's geom types (`mjtGeom` without its prefix), in its order. */
const GEOM_TYPES = ['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box', 'mesh', 'sdf'] as const;
export type GeomType = PrimitiveType | 'hfield' | 'mesh' | 'sdf';

/** A surface hit, in MuJoCo's world axes. */
export interface PickHit {
  point: [number, number, number];
  /** The surface's normal, turned to face the camera. */
  normal: [number, number, number];
  geom: number;
  body: number;
  /** The geom hit (ADR 0012): its name, '' when it has none. */
  name: string;
  type: GeomType;
  /** MuJoCo's `size`, three numbers whatever the type. */
  size: [number, number, number];
  /** As the scene draws it: its material's colour when it has one. */
  rgba: [number, number, number, number];
  /** Where it is now: its centre, and its turn (w, x, y, z). */
  pos: [number, number, number];
  quat: [number, number, number, number];
}

/** A canvas's place on the page, as `getBoundingClientRect` gives it. */
export interface CanvasRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export class SurfacePicker {
  private readonly raycaster = new THREE.Raycaster();
  private readonly ndc = new THREE.Vector2();
  /** `geom_xmat` (row-major) as a turn: MuJoCo's axes, not swizzled. */
  private readonly rotation = new THREE.Matrix4();
  private readonly turn = new THREE.Quaternion();
  /** `mj_ray`'s outputs: the binding writes back only into its own buffers. */
  private geomId: IntBuffer | null = null;
  private normal: DoubleBuffer | null = null;

  constructor(private readonly mujoco: MainModule) {}

  /** `clientX` and `clientY` are CSS pixels, as a `PointerEvent` gives them. */
  pick(
    mjModel: MjModel,
    mjData: MjData,
    camera: THREE.Camera,
    rect: CanvasRect,
    clientX: number,
    clientY: number
  ): PickHit | null {
    if (!rect.width || !rect.height) return null;
    this.ndc.set(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(this.ndc, camera);
    const origin = threeToMjcCoordinate(this.raycaster.ray.origin);
    const direction = threeToMjcCoordinate(this.raycaster.ray.direction);

    this.geomId ??= new this.mujoco.IntBuffer(1);
    this.normal ??= new this.mujoco.DoubleBuffer(3);
    const distance = this.mujoco.mj_ray(
      mjModel,
      mjData,
      [origin.x, origin.y, origin.z],
      [direction.x, direction.y, direction.z],
      DRAWN_GROUPS as unknown as number[],
      true,
      -1,
      this.geomId,
      this.normal
    );
    if (distance < 0) return null;

    const geom = this.geomId.GetView()[0] as number;
    const n = this.normal.GetView() as ArrayLike<number>;
    const facing = n[0] * direction.x + n[1] * direction.y + n[2] * direction.z > 0 ? -1 : 1;
    const material = mjModel.geom_matid[geom];
    const rgba = material >= 0 ? mjModel.mat_rgba : mjModel.geom_rgba;
    const at = (material >= 0 ? material : geom) * 4;
    const m = mjData.geom_xmat;
    const g = geom * 9;
    const turn = this.turn.setFromRotationMatrix(
      this.rotation.set(m[g], m[g + 1], m[g + 2], 0, m[g + 3], m[g + 4], m[g + 5], 0, m[g + 6], m[g + 7], m[g + 8], 0, 0, 0, 0, 1)
    );
    return {
      point: [
        origin.x + direction.x * distance,
        origin.y + direction.y * distance,
        origin.z + direction.z * distance,
      ],
      normal: [n[0] * facing, n[1] * facing, n[2] * facing],
      geom,
      body: mjModel.geom_bodyid[geom],
      name: this.mujoco.mj_id2name(mjModel, this.mujoco.mjtObj.mjOBJ_GEOM.value, geom) ?? '',
      type: GEOM_TYPES[mjModel.geom_type[geom]],
      size: [mjModel.geom_size[geom * 3], mjModel.geom_size[geom * 3 + 1], mjModel.geom_size[geom * 3 + 2]],
      rgba: [rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]],
      pos: [mjData.geom_xpos[geom * 3], mjData.geom_xpos[geom * 3 + 1], mjData.geom_xpos[geom * 3 + 2]],
      quat: [turn.w, turn.x, turn.y, turn.z],
    };
  }

  dispose(): void {
    this.geomId?.delete();
    this.normal?.delete();
    this.geomId = null;
    this.normal = null;
  }
}
