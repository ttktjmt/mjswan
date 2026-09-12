/**
 * mjlab's "Derived properties" that are not root velocities (those sit in `root.ts`),
 * with the two constants they rotate: `gravity_vec_w` and `forward_vec_b` are tensor
 * fields mjlab fills with the world's down and the body's forward, never read from the
 * model.
 */
import { quatApply, quatApplyInv } from '../../../observation/math';
import { quatAt, rootField, type FieldReader } from './util';

const GRAVITY_W = [0, 0, -1];
const FORWARD_B = [1, 0, 0];

export const DERIVED_READERS: Record<string, FieldReader> = {
  gravity_vec_w: () => Float32Array.from(GRAVITY_W),
  projected_gravity_b: rootField((root, d) =>
    Float32Array.from(quatApplyInv(quatAt(d.xquat, root), GRAVITY_W)),
  ),
  heading_w: rootField((root, d) => {
    const forward = quatApply(quatAt(d.xquat, root), FORWARD_B);
    return new Float32Array([Math.atan2(forward[1], forward[0])]);
  }),
};
