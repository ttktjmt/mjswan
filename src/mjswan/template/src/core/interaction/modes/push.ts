/**
 * Tap a body to shove it.
 *
 * Two decisions worth stating. The shove goes **into the surface**, along the inward
 * normal of the face that was hit rather than along the view ray, because that is the direction
 * a poke reads as: hitting the side of a torso from a three-quarter view should push it
 * sideways, not diagonally away from the camera. And the press is `shared`, not
 * `exclusive`: a tap shoves, but a drag from the same spot still orbits, because
 * `OrbitControls` cannot pick up a drag whose start it never saw.
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
    // Instanced and line geometry report no face; the view ray is the honest fallback.
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
    // Nothing steps once this mode is off, and the ring fades on the step clock.
    ctx.ring.hide();
  }

  preStep(_gesture: PointerGesture | null, ctx: ModeContext): void {
    const { mujoco, mjData, controlDt } = ctx.sim();
    ctx.ring.update(controlDt);
    if (!this.pending || !mjData) return;
    // An impulse over exactly one control step, so the shove is the same whatever a
    // scene's decimation is. The wrench clears the row on the next step.
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
