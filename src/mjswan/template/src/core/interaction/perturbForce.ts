/**
 * Pulling a body toward a point, handed to MuJoCo's own mouse perturbation: the one
 * `simulate` uses for its drag.
 *
 * Reusing it is not only less code. `xfrc_applied` acts at the body's **centre of mass**,
 * so a wrench assembled by hand has to take its moment about `xipos`; taking it about
 * `xpos` is the same thing only for a body whose frame sits on its COM, true of a lone
 * primitive and false of nearly every robot link. `mjv_applyPerturbForce` takes a body-frame
 * grab point and a world target and writes the whole row, so the only thing left to get
 * right is the coordinate flip on the way in.
 */
import type { MainModule, MjData, MjModel, MjvPerturb } from 'mujoco';

/**
 * The spring constant inside `mjv_applyPerturbForce`: it applies
 * `100 * localmass * displacement` newtons and exposes no knob for the 100. Measured
 * against the WASM build and pinned by this module's test, so a change upstream fails
 * there rather than silently retuning every drag.
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
  // MuJoCo scales the pull by `localmass`, so the viewer's own N/m rides in as one. Not
  // the body's real mass, which would make a heavy link drag like a light one, and that is
  // `simulate`'s feel, not this viewer's.
  perturb.localmass = pull.forceScale / MJV_PERTURB_STIFFNESS;
  perturb.localpos.set(pull.localPoint);
  perturb.refselpos.set(pull.targetPoint);
  mujoco.mjv_applyPerturbForce(mjModel, mjData, perturb);
}

/** Row layout of `xfrc_applied`: force then torque, both in the world frame. */
const WRENCH_STRIDE = 6;
const IDENTITY_ROTATION = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * Write a world-frame force acting at `worldPoint` into `mjData.xfrc_applied`, replacing
 * that body's row.
 *
 * The moment goes through `mju_transformSpatial` rather than a hand-rolled cross product,
 * so the point a wrench is referred to stays MuJoCo's business. Two conventions have to
 * be bridged: `mju_transformSpatial` reads and writes **(rotational, linear)**, while
 * `xfrc_applied` stores **(force, torque)**, hence the swap on the way out.
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

/**
 * Hold a body's wrench to `maxForce` newtons, scaling the torque with it so the line of
 * action is unchanged. Returns whether the clamp bit.
 */
export function clampWrench(mjData: MjData, bodyId: number, maxForce: number): boolean {
  const at = bodyId * WRENCH_STRIDE;
  const row = mjData.xfrc_applied;
  const magnitude = Math.hypot(row[at], row[at + 1], row[at + 2]);
  if (!(magnitude > maxForce)) return false;
  const scale = maxForce / magnitude;
  for (let i = 0; i < WRENCH_STRIDE; i++) row[at + i] *= scale;
  return true;
}

/** Zero one body's row, for a mode that has stopped writing it. */
export function clearWrench(mjData: MjData, bodyId: number): void {
  const at = bodyId * WRENCH_STRIDE;
  if (at + WRENCH_STRIDE > mjData.xfrc_applied.length) return;
  for (let i = 0; i < WRENCH_STRIDE; i++) mjData.xfrc_applied[at + i] = 0;
}
