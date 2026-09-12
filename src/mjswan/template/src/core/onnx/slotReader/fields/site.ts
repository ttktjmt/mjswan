/** mjlab's "Site properties" and their component accessors: one row per entity site. */
import { quatFromMatrix } from '../../../observation/math';
import type { EntityIndex } from '../indexing';
import { columns, concat, perElement, rowAt, vec3At, velocityFromCvel, type FieldReader } from './util';

const sites = (index: EntityIndex): readonly number[] => index.siteIds;

const pose = perElement(sites, 7, (s, _at, d) =>
  concat(vec3At(d.site_xpos, s), quatFromMatrix(rowAt(d.site_xmat, s, 9))),
);
// A site moves with its body: that body's `cvel`, moved from the root's subtree COM.
const vel = perElement(sites, 6, (s, at, d, index) =>
  velocityFromCvel(
    vec3At(d.site_xpos, s),
    vec3At(d.subtree_com, index.rootBodyId),
    rowAt(d.cvel, index.siteBodyIds[at], 6),
  ),
);

export const SITE_READERS: Record<string, FieldReader> = {
  site_pose_w: pose,
  site_vel_w: vel,
  site_pos_w: perElement(sites, 3, (s, _at, d) => vec3At(d.site_xpos, s)),
  site_quat_w: columns(pose, 7, 3, 7),
  site_lin_vel_w: columns(vel, 6, 0, 3),
  site_ang_vel_w: perElement(sites, 3, (_s, at, d, index) =>
    rowAt(d.cvel, index.siteBodyIds[at], 6).subarray(0, 3),
  ),
};
