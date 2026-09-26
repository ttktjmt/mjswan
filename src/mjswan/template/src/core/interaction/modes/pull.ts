/**
 * Drag a body toward the pointer on a spring. The grab point is kept in the body's frame,
 * so it stays on the grabbed spot as the body tumbles.
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
    const force = ctx.wrench.pull(mujoco, mjModel, mjData, ctx.perturb(), {
      bodyId: this.bodyId,
      localPoint: this.grab,
      targetPoint: toMjc(gesture.ray),
      forceScale: ctx.params().spring,
    });
    ctx.arrow.show(fromBodyFrame(mjData, this.bodyId, this.grab), gesture.ray, force);
  }
}
