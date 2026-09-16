/**
 * Grab a body and carry it.
 *
 * The hold is a weld to a mocap anchor the pointer drives, which is the same constraint
 * the XR hand uses (`core/grab/weldHold`) — only the trigger differs, raycast here and
 * pinch there. Two consequences worth knowing:
 *
 * - **Letting go throws.** Release only deactivates the constraint, so the body leaves
 *   with the velocity the carry gave it. There is no separate throw mode because this is
 *   already one.
 * - **A policy cannot feel it directly.** The constraint force lands in `qfrc_constraint`,
 *   and mjlab recovers `qfrc_external` from the smooth-dynamics identity, which does not
 *   include it. The robot feels a held limb through its own state, not as an applied force.
 */
import * as THREE from 'three';

import type { InteractionMode, InteractionSim, ModeContext } from './mode';
import { toMjc } from './mode';
import { POINTER_ANCHOR_BODY, POINTER_PARK_Z, POINTER_WELD } from '../grabInject';
import type { PointerClaim, PointerGesture } from '../pointer';

const IDENTITY_QUAT = [1, 0, 0, 0];
const PARKED: [number, number, number] = [0, 0, POINTER_PARK_Z];

export class WeldMode implements InteractionMode {
  readonly id = 'weld' as const;
  private held = 0;

  pickable(sim: InteractionSim): Set<number> | null {
    return sim.dynamicBodyIds;
  }

  unavailable(ctx: ModeContext): string | null {
    return ctx.weld.has(POINTER_WELD) ? null : 'This scene was built without a grab anchor.';
  }

  onDown(gesture: PointerGesture, ctx: ModeContext): PointerClaim {
    const { mujoco, mjModel, mjData } = ctx.sim();
    if (!mjModel || !mjData || gesture.hit.bodyId <= 0) return 'none';
    const anchorId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, POINTER_ANCHOR_BODY);
    if (anchorId < 0) return 'none';
    // A hand already carrying this one keeps it; two welds on one body is a fight.
    if (ctx.weld.holderOf(gesture.hit.bodyId) !== null) return 'none';

    this.moveAnchor(ctx, toMjc(gesture.hit.point));
    // The anchor's `xpos` comes from `mocap_pos` only at a forward, and it was parked 100 m
    // up until a moment ago — welding against the stale pose would fling the body there.
    mujoco.mj_forward(mjModel, mjData);
    const params = ctx.params();
    const grabbed = ctx.weld.hold(mjModel, mjData, POINTER_WELD, anchorId, gesture.hit.bodyId, {
      torqueScale: params.torqueScale,
      solrefTime: params.softness,
    });
    if (!grabbed) return 'none';
    this.held = gesture.hit.bodyId;
    return 'exclusive';
  }

  onUp(_gesture: PointerGesture, ctx: ModeContext): void {
    this.onCancel(ctx);
  }

  onCancel(ctx: ModeContext): void {
    const { mjData } = ctx.sim();
    if (!mjData) {
      this.held = 0;
      return;
    }
    ctx.weld.release(mjData, POINTER_WELD);
    this.moveAnchor(ctx, PARKED);
    this.held = 0;
    ctx.arrow.hide();
  }

  preStep(gesture: PointerGesture | null, ctx: ModeContext): void {
    const { mjData } = ctx.sim();
    if (!gesture || !this.held || !mjData) return;
    this.moveAnchor(ctx, toMjc(gesture.ray));
    // Draw from the body's own origin rather than the grab point: the weld holds a pose,
    // not a point, so the line reads as "this body is attached to the pointer".
    ctx.arrow.show(bodyPosition(ctx, this.held), gesture.ray);
  }

  /**
   * The anchor's orientation stays as it was at the grab, so turning the camera does not
   * twist what you are carrying.
   */
  private moveAnchor(ctx: ModeContext, position: readonly [number, number, number]): void {
    const { mujoco, mjModel, mjData } = ctx.sim();
    if (!mjModel || !mjData) return;
    const anchorId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, POINTER_ANCHOR_BODY);
    if (anchorId < 0) return;
    const mocapId = mjModel.body_mocapid[anchorId];
    if (mocapId < 0) return;
    for (let i = 0; i < 3; i++) mjData.mocap_pos[mocapId * 3 + i] = position[i];
    for (let i = 0; i < 4; i++) mjData.mocap_quat[mocapId * 4 + i] = IDENTITY_QUAT[i];
  }
}

function bodyPosition(ctx: ModeContext, bodyId: number): THREE.Vector3 {
  const { mjData } = ctx.sim();
  if (!mjData) return new THREE.Vector3();
  return new THREE.Vector3(
    mjData.xpos[bodyId * 3 + 0],
    mjData.xpos[bodyId * 3 + 2],
    -mjData.xpos[bodyId * 3 + 1],
  );
}
