/**
 * Drag a body toward the pointer on a spring, as MuJoCo Studio does. The grab point is kept
 * in the body's frame, so it stays on the grabbed spot as the body tumbles.
 */
import type { InteractionMode, ModeContext } from './mode';
import { fromBodyFrame, toBodyFrame, toMjc } from './mode';
import { perturbMass } from '../perturbForce';
import type { PointerClaim, PointerGesture } from '../pointer';

export class PullMode implements InteractionMode {
  readonly id = 'pull' as const;
  private bodyId = 0;
  private grab: [number, number, number] = [0, 0, 0];
  private localmass = 1;

  onDown(gesture: PointerGesture, ctx: ModeContext): PointerClaim {
    const { mujoco, mjModel, mjData } = ctx.sim();
    if (!mjModel || !mjData || gesture.hit.bodyId <= 0) return 'none';
    this.bodyId = gesture.hit.bodyId;
    this.grab = toBodyFrame(mjData, this.bodyId, gesture.hit.point);
    // Once per grab, at the pose it was grabbed in, as Studio's `InitPerturb`.
    this.localmass = perturbMass(mujoco, mjModel, mjData, this.bodyId, toMjc(gesture.hit.point));
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
    const { mujoco, mjModel, mjData, controlDt } = ctx.sim();
    if (!mjModel || !mjData) return;
    // Held for a control step, the spring overshoots past omega * dt = 1: a slower policy
    // gets a softer cap than the slider's.
    mjModel.vis.map.stiffness = Math.min(ctx.params().stiffness, 1 / (controlDt * controlDt));
    const force = ctx.wrench.pull(mujoco, mjModel, mjData, ctx.perturb(), {
      bodyId: this.bodyId,
      localPoint: this.grab,
      targetPoint: toMjc(gesture.ray),
      localmass: this.localmass,
    });
    ctx.arrow.show(fromBodyFrame(mjData, this.bodyId, this.grab), gesture.ray, force);
  }
}
