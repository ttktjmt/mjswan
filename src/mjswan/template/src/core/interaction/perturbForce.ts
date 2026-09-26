/**
 * Writing `xfrc_applied` rows. A row acts at the body's centre of mass (`xipos`), not its
 * frame origin (`xpos`), so every moment is taken about `xipos`. The pull is MuJoCo's own
 * `mjv_applyPerturbForce`, the drag `simulate` uses.
 */
import type { MainModule, MjData, MjModel, MjvPerturb } from 'mujoco';

/**
 * The stiffness hard-coded in `mjv_applyPerturbForce`, which applies
 * `100 * localmass * displacement` newtons. The test pins it against the WASM build.
 */
export const MJV_PERTURB_STIFFNESS = 100;

export interface DragPull {
  bodyId: number;
  /** Grab point in the body's own frame, MuJoCo axes. */
  localPoint: readonly [number, number, number];
  /** Where the pointer wants that point to be: world frame, MuJoCo axes. */
  targetPoint: readonly [number, number, number];
  /** Newtons per metre of displacement between the two. */
  forceScale: number;
}

/**
 * Write `pull` into `mjData.xfrc_applied`, replacing whatever that body's row held.
 * `perturb` is reused across calls: it is an embind handle, not a value.
 */
export function applyDragPull(
  mujoco: MainModule,
  mjModel: MjModel,
  mjData: MjData,
  perturb: MjvPerturb,
  pull: DragPull,
): void {
  perturb.select = pull.bodyId;
  perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE.value;
  // `localmass` carries the viewer's N/m, not the body's mass as in `simulate`, where a
  // heavy link drags like a light one.
  perturb.localmass = pull.forceScale / MJV_PERTURB_STIFFNESS;
  perturb.localpos.set(pull.localPoint);
  perturb.refselpos.set(pull.targetPoint);
  mujoco.mjv_applyPerturbForce(mjModel, mjData, perturb);
}

/** Row layout of `xfrc_applied`: force then torque, both in the world frame. */
const WRENCH_STRIDE = 6;
const IDENTITY_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * Write a world-frame force acting at `worldPoint` into that body's `xfrc_applied` row,
 * replacing it. `mju_transformSpatial` works in (torque, force) order and the row stores
 * (force, torque), hence the swap.
 */
export function applyPointForce(
  mujoco: MainModule,
  mjData: MjData,
  bodyId: number,
  worldPoint: readonly [number, number, number],
  force: readonly [number, number, number],
): void {
  const at = bodyId * WRENCH_STRIDE;
  const row = mjData.xfrc_applied.subarray(at, at + WRENCH_STRIDE);
  const com = [0, 1, 2].map((i) => mjData.xipos[bodyId * 3 + i]);
  mujoco.mju_transformSpatial(
    row,
    [0, 0, 0, force[0], force[1], force[2]],
    1,
    com,
    [worldPoint[0], worldPoint[1], worldPoint[2]],
    IDENTITY_ROTATION,
  );
  for (let i = 0; i < 3; i++) {
    const torque = row[i];
    row[i] = row[i + 3];
    row[i + 3] = torque;
  }
}

/** The magnitude of the force in one body's row, newtons. */
export function wrenchForce(mjData: MjData, bodyId: number): number {
  const at = bodyId * WRENCH_STRIDE;
  const row = mjData.xfrc_applied;
  return Math.hypot(row[at], row[at + 1], row[at + 2]);
}

/** Zero one body's row, for a mode that has stopped writing it. */
export function clearWrench(mjData: MjData, bodyId: number): void {
  const at = bodyId * WRENCH_STRIDE;
  if (at + WRENCH_STRIDE > mjData.xfrc_applied.length) return;
  for (let i = 0; i < WRENCH_STRIDE; i++) mjData.xfrc_applied[at + i] = 0;
}
