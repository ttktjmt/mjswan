/**
 * mjlab's "Root properties" and the root rows of its "Pose and velocity component
 * accessors" (`entity/data.py`), plus the root velocities in the body frame.
 */
import { quatApplyInv, quatMultiply } from '../../../observation/math';
import type { EntityIndex } from '../indexing';
import {
  concat,
  quatAt,
  rootField,
  rowAt,
  vec3At,
  velocityFromCvel,
  type FieldReader,
} from './util';

type MjData = import('mujoco').MjData;

const linkQuat = (root: number, mjData: MjData): Float32Array => quatAt(mjData.xquat, root);

/** `root_link_vel_w`: `cvel` about the subtree COM, moved to the body origin. */
const linkVel = (root: number, mjData: MjData): Float32Array =>
  velocityFromCvel(vec3At(mjData.xpos, root), vec3At(mjData.subtree_com, root), rowAt(mjData.cvel, root, 6));

/** `root_com_vel_w`: the same, moved to the COM. */
const comVel = (root: number, mjData: MjData): Float32Array =>
  velocityFromCvel(vec3At(mjData.xipos, root), vec3At(mjData.subtree_com, root), rowAt(mjData.cvel, root, 6));

/** `root_com_quat_w`: the body's rotation composed with its inertial frame's. */
const comQuat = (root: number, mjData: MjData, index: EntityIndex): Float32Array =>
  Float32Array.from(quatMultiply(quatAt(mjData.xquat, root), index.bodyIquat.subarray(0, 4)));

/** `cvel`'s angular part, the same about the link and about the COM. */
const angVel = (root: number, mjData: MjData): Float32Array => rowAt(mjData.cvel, root, 6).subarray(0, 3);

const inBody = (read: (root: number, mjData: MjData) => Float32Array): FieldReader =>
  rootField((root, mjData) => Float32Array.from(quatApplyInv(linkQuat(root, mjData), read(root, mjData))));

export const ROOT_READERS: Record<string, FieldReader> = {
  root_link_pose_w: rootField((root, d) => concat(vec3At(d.xpos, root), linkQuat(root, d))),
  root_link_vel_w: rootField(linkVel),
  root_com_pose_w: rootField((root, d, index) => concat(vec3At(d.xipos, root), comQuat(root, d, index))),
  root_com_vel_w: rootField(comVel),

  root_link_pos_w: rootField((root, d) => vec3At(d.xpos, root)),
  root_link_quat_w: rootField(linkQuat),
  root_link_lin_vel_w: rootField((root, d) => linkVel(root, d).subarray(0, 3)),
  root_link_ang_vel_w: rootField(angVel),
  root_com_pos_w: rootField((root, d) => vec3At(d.xipos, root)),
  root_com_quat_w: rootField(comQuat),
  root_com_lin_vel_w: rootField((root, d) => comVel(root, d).subarray(0, 3)),
  root_com_ang_vel_w: rootField(angVel),

  // Every body-frame velocity rotates by the *link* quaternion, as mjlab's do.
  root_link_lin_vel_b: inBody((root, d) => linkVel(root, d).subarray(0, 3)),
  root_link_ang_vel_b: inBody(angVel),
  root_com_lin_vel_b: inBody((root, d) => comVel(root, d).subarray(0, 3)),
  root_com_ang_vel_b: inBody(angVel),
};
