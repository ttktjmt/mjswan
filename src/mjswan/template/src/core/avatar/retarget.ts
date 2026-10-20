/**
 * Robot pose -> VRM normalized-bone pose, by rotation (no IK).
 *
 * Every frame below is three.js's (y up). A robot body `b` has rotated by
 * `Δ_b = Q_b · Q0_b⁻¹` since the robot's zero pose (`Q0`). The avatar turns that into its
 * own frame (`A`: the avatar faces +z, the robot +x), applies it over the bone's rest
 * offset `O_b`, and takes each bone relative to its nearest mapped ancestor:
 *
 *   M_b = A⁻¹ · Δ_b · A · O_b        R_b = S⁻¹ · M_parent⁻¹ · M_b · S
 *
 * `S` is the model frame the normalized bones live in: identity for VRM 1.0, a half turn
 * about y for VRM 0.x, which faces -z. That conjugation is the sign flip on x and z every
 * VRM 0.x retargeter applies.
 */
import * as THREE from 'three';
import type { MjModel } from 'mujoco';

/** The avatar's VRM-1.0 front (+z) turned onto the robot's front (MuJoCo +x). */
export const AVATAR_TO_ROBOT_FACING = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(0, 1, 0),
  Math.PI / 2,
);

/** `S`: where a VRM file's normalized bones face relative to VRM 1.0's +z. */
export function modelFrame(metaVersion: string | undefined): THREE.Quaternion {
  return metaVersion === '0'
    ? new THREE.Quaternion(0, 1, 0, 0)
    : new THREE.Quaternion();
}

export interface RetargetBone {
  /** VRM humanoid bone name. */
  name: string;
  bodyId: number;
  /** Index into {@link RetargetRig.bones} of the nearest mapped ancestor; null for hips. */
  parent: number | null;
  /** `Q0_b⁻¹`. */
  restInverse: THREE.Quaternion;
  /** `O_b`. */
  offset: THREE.Quaternion;
}

export interface RetargetRig {
  /** Parents before children. */
  bones: RetargetBone[];
  facing: THREE.Quaternion;
  frame: THREE.Quaternion;
}

/** A MuJoCo `(w, x, y, z)` quaternion as the same rotation in three.js's frame. */
function fromMujoco(q: ArrayLike<number>, i: number, target: THREE.Quaternion): THREE.Quaternion {
  return target.set(q[i * 4 + 1], q[i * 4 + 3], -q[i * 4 + 2], q[i * 4]);
}

/**
 * Each body's world rotation at the model's zero pose: every hinge and slide at its
 * reference, every free body upright and facing +x. Read off `body_quat` alone, so no
 * `mjData` is touched.
 */
export function zeroPoseRotations(
  mjModel: Pick<
    MjModel,
    'nbody' | 'body_parentid' | 'body_quat' | 'body_jntadr' | 'body_jntnum' | 'jnt_type'
  >,
  freeJointType: number,
): THREE.Quaternion[] {
  const rotations: THREE.Quaternion[] = [];
  const local = new THREE.Quaternion();
  for (let b = 0; b < mjModel.nbody; b++) {
    const rotation = new THREE.Quaternion();
    rotations.push(rotation);
    if (b === 0) continue;
    const adr = mjModel.body_jntadr[b];
    const free = mjModel.body_jntnum[b] > 0 && mjModel.jnt_type[adr] === freeJointType;
    if (free) continue;
    rotation
      .copy(rotations[mjModel.body_parentid[b]])
      .multiply(fromMujoco(mjModel.body_quat, b, local));
  }
  return rotations;
}

/** A body's height (three.js y) at the pose {@link zeroPoseRotations} describes. */
export function zeroPoseHeight(
  mjModel: Pick<MjModel, 'body_parentid' | 'body_pos'>,
  zeroPose: ReadonlyArray<THREE.Quaternion>,
  bodyId: number,
): number {
  const chain: number[] = [];
  for (let b = bodyId; b > 0; b = mjModel.body_parentid[b]) chain.unshift(b);
  const position = new THREE.Vector3();
  const local = new THREE.Vector3();
  for (const b of chain) {
    const p = mjModel.body_pos;
    local.set(p[b * 3], p[b * 3 + 2], -p[b * 3 + 1]);
    position.add(local.applyQuaternion(zeroPose[mjModel.body_parentid[b]]));
  }
  return position.y;
}

/**
 * The rig for `bones` (VRM bone -> body id), in an order where each bone's mapped
 * ancestor comes first. `ancestors` lists a bone's VRM ancestors nearest first, the way
 * the file's own hierarchy has them.
 */
export function buildRig(
  bones: ReadonlyArray<{ name: string; bodyId: number; ancestors: string[] }>,
  zeroPose: ReadonlyArray<THREE.Quaternion>,
  offsets: Readonly<Record<string, ArrayLike<number>>>,
  frame: THREE.Quaternion,
): RetargetRig {
  const order = [...bones].sort((a, b) => a.ancestors.length - b.ancestors.length);
  const index = new Map(order.map((bone, i) => [bone.name, i]));
  return {
    facing: AVATAR_TO_ROBOT_FACING.clone(),
    frame: frame.clone(),
    bones: order.map((bone) => {
      const parent = bone.ancestors.find((name) => index.has(name));
      const offset = offsets[bone.name];
      return {
        name: bone.name,
        bodyId: bone.bodyId,
        parent: parent === undefined ? null : index.get(parent)!,
        restInverse: zeroPose[bone.bodyId].clone().invert(),
        offset: offset
          ? new THREE.Quaternion(offset[0], offset[1], offset[2], offset[3]).normalize()
          : new THREE.Quaternion(),
      };
    }),
  };
}

const _facingInverse = new THREE.Quaternion();
const _frameInverse = new THREE.Quaternion();
const _parentInverse = new THREE.Quaternion();
const _rigInverse = new THREE.Quaternion();

/**
 * `out[i]` = the normalized local rotation of `rig.bones[i]`, given each body's current
 * world rotation. `world` holds one `M_b` per bone and is overwritten.
 */
export function retarget(
  rig: RetargetRig,
  bodyRotation: (bodyId: number) => THREE.Quaternion,
  world: THREE.Quaternion[],
  out: THREE.Quaternion[],
): void {
  _facingInverse.copy(rig.facing).invert();
  _frameInverse.copy(rig.frame).invert();
  rig.bones.forEach((bone, i) => {
    world[i]
      .copy(bodyRotation(bone.bodyId))
      .multiply(bone.restInverse)
      .premultiply(_facingInverse)
      .multiply(rig.facing)
      .multiply(bone.offset);
    const local = out[i].copy(world[i]);
    if (bone.parent !== null) local.premultiply(_parentInverse.copy(world[bone.parent]).invert());
    local.premultiply(_frameInverse).multiply(rig.frame);
  });
}

/**
 * Where the hips go in the normalized rig's frame: on the robot root, its height scaled.
 * Scaling across the floor too would carry the avatar off the robot as it wanders.
 */
export function hipsPosition(
  rig: RetargetRig,
  rootPosition: THREE.Vector3,
  heightRatio: number,
  target: THREE.Vector3,
): THREE.Vector3 {
  _rigInverse.copy(rig.facing).multiply(rig.frame).invert();
  return target
    .set(rootPosition.x, rootPosition.y * heightRatio, rootPosition.z)
    .applyQuaternion(_rigInverse);
}
