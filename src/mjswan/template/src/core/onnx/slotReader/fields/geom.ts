/** mjlab's "Geom properties" and their component accessors: one row per entity geom. */
import { quatFromMatrix } from '../../../observation/math';
import type { EntityIndex } from '../indexing';
import { columns, concat, perElement, rowAt, vec3At, velocityFromCvel, type FieldReader } from './util';

const geoms = (index: EntityIndex): readonly number[] => index.geomIds;

const pose = perElement(geoms, 7, (g, _at, d) =>
  concat(vec3At(d.geom_xpos, g), quatFromMatrix(rowAt(d.geom_xmat, g, 9))),
);
// A geom moves with its body: that body's `cvel`, moved from the root's subtree COM.
const vel = perElement(geoms, 6, (g, at, d, index) =>
  velocityFromCvel(
    vec3At(d.geom_xpos, g),
    vec3At(d.subtree_com, index.rootBodyId),
    rowAt(d.cvel, index.geomBodyIds[at], 6),
  ),
);

export const GEOM_READERS: Record<string, FieldReader> = {
  geom_pose_w: pose,
  geom_vel_w: vel,
  geom_pos_w: perElement(geoms, 3, (g, _at, d) => vec3At(d.geom_xpos, g)),
  geom_quat_w: columns(pose, 7, 3, 7),
  geom_lin_vel_w: columns(vel, 6, 0, 3),
  geom_ang_vel_w: perElement(geoms, 3, (_g, at, d, index) =>
    rowAt(d.cvel, index.geomBodyIds[at], 6).subarray(0, 3),
  ),
};
