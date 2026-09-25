/** What a pointer mode is handed, and what it has to answer. */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel, MjvPerturb } from 'mujoco';

import { quatApply, quatApplyInv } from '../../observation/math';
import { mjcToThreeCoordinate, threeToMjcCoordinate } from '../../scene/coordinate';
import type { WeldHold } from '../../grab/weldHold';
import type { DragArrow, PushRing } from '../gizmos';
import type { InteractionModeId } from '../params';
import type { PointerClaim, PointerGesture } from '../pointer';
import type { InteractionWrench } from '../wrench';

/** The simulation as a mode sees it: re-read per call, since a scene switch replaces it. */
export interface InteractionSim {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  /** Bodies a mode may act on: everything with a degree of freedom. */
  dynamicBodyIds: Set<number> | null;
  /** Seconds per control step: what an impulse is divided by. */
  controlDt: number;
}

export interface ModeContext {
  sim(): InteractionSim;
  /** The active mode's own parameters, already clamped to their ranges. */
  params(): Record<string, number>;
  wrench: InteractionWrench;
  perturb(): MjvPerturb;
  arrow: DragArrow;
  ring: PushRing;
  /** Shared with the XR hand: the slots a scene compiled in for holding things. */
  weld: WeldHold;
  /** The canvas's CSS cursor; '' hands it back. A touchscreen shows none of it. */
  setCursor(cursor: string): void;
}

export interface InteractionMode {
  readonly id: InteractionModeId;
  onDown(gesture: PointerGesture, ctx: ModeContext): PointerClaim;
  onMove?(gesture: PointerGesture, ctx: ModeContext): void;
  onUp?(gesture: PointerGesture, ctx: ModeContext): void;
  /** Pointer lost, mode switched, scene reloaded, viewer paused. */
  onCancel(ctx: ModeContext): void;
  /** Once per control step, with the live gesture or null. */
  preStep(gesture: PointerGesture | null, ctx: ModeContext): void;
  /** Why this mode cannot run in this scene, or null when it can. */
  unavailable?(ctx: ModeContext): string | null;
}

/** A three.js world point expressed in one body's own frame, MuJoCo axes. */
export function toBodyFrame(
  mjData: MjData,
  bodyId: number,
  world: THREE.Vector3,
): [number, number, number] {
  const point = threeToMjcCoordinate(world);
  const quat = [0, 1, 2, 3].map((i) => mjData.xquat[bodyId * 4 + i]);
  const local = quatApplyInv(quat, [
    point.x - mjData.xpos[bodyId * 3 + 0],
    point.y - mjData.xpos[bodyId * 3 + 1],
    point.z - mjData.xpos[bodyId * 3 + 2],
  ]);
  return [local[0], local[1], local[2]];
}

/** The inverse: where that body-frame point is in the drawn scene now. */
export function fromBodyFrame(
  mjData: MjData,
  bodyId: number,
  local: readonly [number, number, number],
): THREE.Vector3 {
  const quat = [0, 1, 2, 3].map((i) => mjData.xquat[bodyId * 4 + i]);
  const rotated = quatApply(quat, local);
  return mjcToThreeCoordinate([
    rotated[0] + mjData.xpos[bodyId * 3 + 0],
    rotated[1] + mjData.xpos[bodyId * 3 + 1],
    rotated[2] + mjData.xpos[bodyId * 3 + 2],
  ]);
}

/** A three.js world vector or point as MuJoCo's axes see it. */
export function toMjc(vector: THREE.Vector3): [number, number, number] {
  const v = threeToMjcCoordinate(vector);
  return [v.x, v.y, v.z];
}
