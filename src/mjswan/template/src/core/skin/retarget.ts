/**
 * Robot pose -> VRM normalized-bone pose, by rotation, and the two-bone IK that bends a
 * limb onto a target (`reachLimb` and `dropToReach`, for `retarget: 'hybrid'`).
 *
 * Every frame below is three.js's (y up). A robot body `b` has rotated by
 * `Δ_b = Q_b · Q0_b⁻¹` since the robot's zero pose (`Q0`). The avatar turns that into its
 * own frame (`A`: the avatar faces +z, the robot +x), applies it over the bone's rest
 * offset `O_b`, and takes each bone relative to its nearest mapped ancestor:
 *
 *   M_b = A⁻¹ · Δ_b · A · O_b        R_b = S⁻¹ · M_parent⁻¹ · M_b · S
 *
 * `S` turns the frame the normalized bones live in, the file's, onto VRM 1.0's: identity
 * for VRM 1.0, a half turn about y for VRM 0.x, which faces -z (the sign flip on x and z
 * every VRM 0.x retargeter applies), a quarter turn for a .glb facing +x or -x.
 */
import * as THREE from 'three';
import type { MjModel } from 'mujoco';

/** The avatar's VRM-1.0 front (+z) turned onto the robot's front (MuJoCo +x). */
export const AVATAR_TO_ROBOT_FACING = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(0, 1, 0),
  Math.PI / 2,
);

/** `S` for a VRM file, by its `meta` version. */
export function modelFrame(metaVersion: string | undefined): THREE.Quaternion {
  return metaVersion === '0'
    ? new THREE.Quaternion(0, 1, 0, 0)
    : new THREE.Quaternion();
}

export interface RetargetBone {
  /** The bone's key in `bones`. */
  name: string;
  bodyId: number;
  /** Index into {@link RetargetRig.bones} of the nearest mapped ancestor; null for the root. */
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
export function fromMujoco(q: ArrayLike<number>, i: number, target: THREE.Quaternion): THREE.Quaternion {
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
 * The rig for `bones` (bone -> body id), in an order where each bone's mapped ancestor
 * comes first. `ancestors` lists the bones each hangs from, nearest first.
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
 * Where the root bone goes in the normalized rig's frame: on the robot root, its height
 * above `ground` (three.js y) scaled. Scaling across the floor too would carry the avatar
 * off the robot as it wanders, and scaling the ground's own height would lift it off a hill.
 */
export function hipsPosition(
  rig: RetargetRig,
  rootPosition: THREE.Vector3,
  heightRatio: number,
  ground: number,
  target: THREE.Vector3,
): THREE.Vector3 {
  _rigInverse.copy(rig.facing).multiply(rig.frame).invert();
  return target
    .set(rootPosition.x, ground + (rootPosition.y - ground) * heightRatio, rootPosition.z)
    .applyQuaternion(_rigInverse);
}

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _c = new THREE.Vector3();
const _ca = new THREE.Vector3();
const _ba = new THREE.Vector3();
const _ta = new THREE.Vector3();
const _ab = new THREE.Vector3();
const _cb = new THREE.Vector3();
const _axis0 = new THREE.Vector3();
const _axis1 = new THREE.Vector3();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _qc = new THREE.Quaternion();
const _qp = new THREE.Quaternion();
const _qi = new THREE.Quaternion();
const _r0 = new THREE.Quaternion();
const _r1 = new THREE.Quaternion();
const _r2 = new THREE.Quaternion();
/** How far short of straight, in metres, a limb stops reaching. */
const REACH_MARGIN = 1e-4;

/**
 * Two-bone IK: turns `upper` and `lower`, each keeping its length, so `end` lands on
 * `target`, or as near as the limb reaches, bending in the plane it already bends in;
 * `end` then takes `turn` as its world rotation, or keeps its own. `lower` is `upper`'s
 * child and `end` is `lower`'s. `bend` keeps that plane's normal for a frame the limb is
 * found straight. World matrices must be current; the limb's are left current.
 */
export function reachLimb(
  upper: THREE.Object3D,
  lower: THREE.Object3D,
  end: THREE.Object3D,
  target: THREE.Vector3,
  bend: THREE.Vector3,
  turn?: THREE.Quaternion,
): void {
  upper.getWorldPosition(_a);
  lower.getWorldPosition(_b);
  end.getWorldPosition(_c);
  upper.getWorldQuaternion(_qa);
  lower.getWorldQuaternion(_qb);
  end.getWorldQuaternion(_qc);
  const lab = _a.distanceTo(_b);
  const lcb = _b.distanceTo(_c);
  const lat = THREE.MathUtils.clamp(
    _a.distanceTo(target),
    Math.abs(lab - lcb) + REACH_MARGIN,
    lab + lcb - REACH_MARGIN,
  );
  _ca.subVectors(_c, _a);
  _ba.subVectors(_b, _a);
  _ta.subVectors(target, _a);
  const clamp = (v: number) => THREE.MathUtils.clamp(v, -1, 1);
  // Angles at the upper joint (bone to chord) and at the lower one, now and once the
  // chord is `lat` long (the law of cosines).
  const atUpper = _ca.angleTo(_ba);
  const atLower = _ab.subVectors(_a, _b).angleTo(_cb.subVectors(_c, _b));
  const atUpperWanted = Math.acos(clamp((lcb * lcb - lab * lab - lat * lat) / (-2 * lab * lat)));
  const atLowerWanted = Math.acos(clamp((lat * lat - lab * lab - lcb * lcb) / (-2 * lab * lcb)));
  _axis0.crossVectors(_ca, _ba);
  if (_axis0.lengthSq() > 1e-12) bend.copy(_axis0.normalize());
  else _axis0.copy(bend);
  _r0.setFromAxisAngle(_axis0, atUpperWanted - atUpper);
  _r1.setFromAxisAngle(_axis0, atLowerWanted - atLower);
  // Then the whole limb swung about the upper joint, its chord onto the target.
  _axis1.crossVectors(_ca, _ta);
  if (_axis1.lengthSq() > 1e-12) _r2.setFromAxisAngle(_axis1.normalize(), _ca.angleTo(_ta));
  else _r2.identity();

  // World rotations: upper r2·r0, lower r2·r1·r0 (each about a fixed world axis).
  _qa.premultiply(_r0).premultiply(_r2);
  _qb.premultiply(_r0).premultiply(_r1).premultiply(_r2);
  upper.parent!.getWorldQuaternion(_qp);
  upper.quaternion.copy(_qp.invert().multiply(_qa));
  lower.quaternion.copy(_qi.copy(_qa).invert().multiply(_qb));
  end.quaternion.copy(_qb.invert().multiply(turn ?? _qc));
  upper.updateMatrixWorld(true);
}

/**
 * How far `upper`'s joint must move straight down (three.js -y) for `reachLimb` to put
 * `end` on `target`: 0 when it already can, and when no move down brings it within reach.
 */
export function dropToReach(
  upper: THREE.Object3D,
  lower: THREE.Object3D,
  end: THREE.Object3D,
  target: THREE.Vector3,
): number {
  upper.getWorldPosition(_a);
  lower.getWorldPosition(_b);
  end.getWorldPosition(_c);
  const reach = _a.distanceTo(_b) + _b.distanceTo(_c) - REACH_MARGIN;
  _ta.subVectors(target, _a);
  const beyond = _ta.lengthSq() - reach * reach;
  // Dropped by d, the target is (below - d) down and as far across: the least d that
  // brings that within `reach`.
  const below = -_ta.y;
  const room = below * below - beyond;
  if (beyond <= 0 || room < 0) return 0;
  return Math.max(0, below - Math.sqrt(room));
}
