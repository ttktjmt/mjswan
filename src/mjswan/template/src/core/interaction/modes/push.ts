/**
 * Tap a body to shove it along the inward normal of the face hit, not the view ray, so a
 * poke at a torso's side pushes it sideways. The press is `shared`: a drag from the same
 * spot still orbits.
 */
import type { InteractionMode, ModeContext } from './mode';
import { toMjc } from './mode';
import { TAP_SLOP_PX, type PointerClaim, type PointerGesture } from '../pointer';

interface PendingShove {
  bodyId: number;
  /** Where it lands: world frame, MuJoCo axes. */
  point: [number, number, number];
  /** Unit: world frame, MuJoCo axes. */
  direction: [number, number, number];
}

export class PushMode implements InteractionMode {
  readonly id = 'push' as const;
  private pending: PendingShove | null = null;

  onDown(gesture: PointerGesture): PointerClaim {
    return gesture.hit.bodyId > 0 ? 'shared' : 'none';
  }

  onUp(gesture: PointerGesture, ctx: ModeContext): void {
    // That was a camera drag, and the camera has already had it.
    if (gesture.travel >= TAP_SLOP_PX) return;
    const { hit } = gesture;
    // No face to take a normal from (a line, say): fall back to the view ray.
    const inward = hit.normal ? hit.normal.clone().negate() : gesture.direction.clone();
    if (inward.lengthSq() < 1e-12) return;
    this.pending = {
      bodyId: hit.bodyId,
      point: toMjc(hit.point),
      direction: toMjc(inward.normalize()),
    };
    ctx.ring.fire(hit.point, hit.normal);
  }

  onCancel(ctx: ModeContext): void {
    this.pending = null;
    ctx.ring.hide();
  }

  preStep(_gesture: PointerGesture | null, ctx: ModeContext): void {
    const { mujoco, mjData, controlDt } = ctx.sim();
    ctx.ring.update(controlDt);
    if (!this.pending || !mjData) return;
    // The impulse over exactly one control step, so it is the same at any decimation. The
    // wrench clears the row next step.
    const newtons = ctx.params().impulse / Math.max(controlDt, 1e-4);
    const { direction } = this.pending;
    ctx.wrench.push(mujoco, mjData, this.pending.bodyId, this.pending.point, [
      direction[0] * newtons,
      direction[1] * newtons,
      direction[2] * newtons,
    ]);
    this.pending = null;
  }
}
