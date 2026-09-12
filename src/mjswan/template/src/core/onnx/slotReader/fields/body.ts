/** mjlab's "Body properties" and their component accessors: one row per entity body. */
import { quatMultiply } from '../../../observation/math';
import type { EntityIndex } from '../indexing';
import { columns, concat, perElement, quatAt, rowAt, vec3At, velocityFromCvel, type FieldReader } from './util';

const bodies = (index: EntityIndex): readonly number[] => index.bodyIds;

const linkPose = perElement(bodies, 7, (b, _at, d) => concat(vec3At(d.xpos, b), quatAt(d.xquat, b)));
// The *root* body's subtree COM for every body, as mjlab's `body_link_vel_w` has it.
const linkVel = perElement(bodies, 6, (b, _at, d, index) =>
  velocityFromCvel(vec3At(d.xpos, b), vec3At(d.subtree_com, index.rootBodyId), rowAt(d.cvel, b, 6)),
);
const comPose = perElement(bodies, 7, (b, at, d, index) =>
  concat(vec3At(d.xipos, b), quatMultiply(quatAt(d.xquat, b), index.bodyIquat.subarray(at * 4, at * 4 + 4))),
);
const comVel = perElement(bodies, 6, (b, _at, d, index) =>
  velocityFromCvel(vec3At(d.xipos, b), vec3At(d.subtree_com, index.rootBodyId), rowAt(d.cvel, b, 6)),
);
const wrench = perElement(bodies, 6, (b, _at, d) => rowAt(d.xfrc_applied, b, 6));
/** `cvel`'s angular part, the same about the link and about the COM. */
const angVel = perElement(bodies, 3, (b, _at, d) => rowAt(d.cvel, b, 6).subarray(0, 3));

export const BODY_READERS: Record<string, FieldReader> = {
  body_link_pose_w: linkPose,
  body_link_vel_w: linkVel,
  body_com_pose_w: comPose,
  body_com_vel_w: comVel,
  body_external_wrench: wrench,

  body_link_pos_w: perElement(bodies, 3, (b, _at, d) => vec3At(d.xpos, b)),
  body_link_quat_w: perElement(bodies, 4, (b, _at, d) => quatAt(d.xquat, b)),
  body_link_lin_vel_w: columns(linkVel, 6, 0, 3),
  body_link_ang_vel_w: angVel,
  body_com_pos_w: perElement(bodies, 3, (b, _at, d) => vec3At(d.xipos, b)),
  body_com_quat_w: columns(comPose, 7, 3, 7),
  body_com_lin_vel_w: columns(comVel, 6, 0, 3),
  body_com_ang_vel_w: angVel,
  body_external_force: columns(wrench, 6, 0, 3),
  body_external_torque: columns(wrench, 6, 3, 6),
};
