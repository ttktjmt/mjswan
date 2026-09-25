/**
 * Drag a body toward the pointer with a spring.
 *
 * The grab point is stored in the body's own frame at the press and rotated back out
 * every step, so it stays on the spot that was grabbed while the body tumbles. The force
 * itself is MuJoCo's (`interaction/perturbForce`).
 */
import type { InteractionMode, ModeContext } from './mode';
import { fromBodyFrame, toBodyFrame, toMjc } from './mode';
import type { PointerClaim, PointerGesture } from '../pointer';

export class PullMode implements InteractionMode {
  readonly id = 'pull' as const;
  private bodyId = 0;
  private grab: [number, number, number] = [0, 0, 0];

  onDown(gesture: PointerGesture, ctx: ModeContext): PointerClaim {
    const { mjData } = ctx.sim();
    if (!mjData || gesture.hit.bodyId <= 0) return 'none';
    this.bodyId = gesture.hit.bodyId;
    this.grab = toBodyFrame(mjData, this.bodyId, gesture.hit.point);
    return 'exclusive';
  }

  onUp(_gesture: PointerGesture, ctx: ModeContext): void {
    this.onCancel(ctx);
  }

  onCancel(ctx: ModeContext): void {
    this.bodyId = 0;
    ctx.arrow.hide();
  }

  preStep(gesture: PointerGesture | null, ctx: ModeContext): void {
    if (!gesture || this.bodyId <= 0) return;
    const { mujoco, mjModel, mjData } = ctx.sim();
    if (!mjModel || !mjData) return;
    const params = ctx.params();
    const saturated = ctx.wrench.pull(
      mujoco,
      mjModel,
      mjData,
      ctx.perturb(),
      {
        bodyId: this.bodyId,
        localPoint: this.grab,
        targetPoint: toMjc(gesture.ray),
        forceScale: params.gain,
      },
      params.maxForce,
    );
    ctx.arrow.show(fromBodyFrame(mjData, this.bodyId, this.grab), gesture.ray, saturated);
  }
}
