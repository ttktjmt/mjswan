/**
 * `geom_rbound` and `geom_aabb`, recomputed from a `geom_size` written at runtime.
 *
 * MuJoCo compiles these two from the size and never revisits them — `mj_setConst` does not
 * either — so a geom that grows keeps the bound it was compiled with and stops colliding
 * at its own surface: the broadphase culls every contact until the other body reaches the
 * *old* radius, which for a grown box is somewhere inside it. Anything that writes
 * `geom_size` after load has to write these too. Matches
 * `dr.geom_size._recompute_geom_bounds`.
 */
import type { MjModel } from 'mujoco';

/** `mjtGeom` values whose bounds follow from `geom_size` — mjlab's supported set. */
export const GEOM_SPHERE = 2;
export const GEOM_CAPSULE = 3;
export const GEOM_ELLIPSOID = 4;
export const GEOM_CYLINDER = 5;
export const GEOM_BOX = 6;

/**
 * Write one geom's bounds from the size it now carries. Returns false for a type whose
 * bounds do not follow from its size, leaving them as compiled — the caller says what that
 * means in its own terms.
 *
 * `geom_aabb` is `(ngeom, 2, 3)`, centre then half-size; a primitive's centre stays at its
 * own origin, so only the half-size is touched.
 */
export function writeGeomBounds(mjModel: MjModel, geomId: number, geomType: number): boolean {
  const size = mjModel.geom_size as ArrayLike<number> | undefined;
  const rbound = mjModel.geom_rbound as unknown as { [index: number]: number } | undefined;
  const aabb = mjModel.geom_aabb as unknown as { [index: number]: number } | undefined;
  if (!size || !rbound || !aabb) return false;

  const s0 = size[geomId * 3];
  const s1 = size[geomId * 3 + 1];
  const s2 = size[geomId * 3 + 2];
  let bound: number;
  let half: [number, number, number];
  switch (geomType) {
    case GEOM_SPHERE:
      bound = s0;
      half = [s0, s0, s0];
      break;
    case GEOM_CAPSULE:
      bound = s0 + s1;
      half = [s0, s0, s0 + s1];
      break;
    case GEOM_ELLIPSOID:
      bound = Math.max(s0, s1, s2);
      half = [s0, s1, s2];
      break;
    case GEOM_CYLINDER:
      bound = Math.sqrt(s0 * s0 + s1 * s1);
      half = [s0, s0, s1];
      break;
    case GEOM_BOX:
      bound = Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2);
      half = [s0, s1, s2];
      break;
    default:
      return false;
  }
  rbound[geomId] = bound;
  aabb[geomId * 6 + 3] = half[0];
  aabb[geomId * 6 + 4] = half[1];
  aabb[geomId * 6 + 5] = half[2];
  return true;
}
