/**
 * Pulling a body toward a point, handed to MuJoCo's own mouse perturbation — the one
 * `simulate` uses for its drag.
 *
 * Reusing it is not only less code. `xfrc_applied` acts at the body's **centre of mass**,
 * so a wrench assembled by hand has to take its moment about `xipos`; taking it about
 * `xpos` is the same thing only for a body whose frame sits on its COM — true of a lone
 * primitive, false of nearly every robot link. `mjv_applyPerturbForce` takes a body-frame
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
  // the body's real mass, which would make a heavy link drag like a light one — that is
  // `simulate`'s feel, not this viewer's.
  perturb.localmass = pull.forceScale / MJV_PERTURB_STIFFNESS;
  perturb.localpos.set(pull.localPoint);
  perturb.refselpos.set(pull.targetPoint);
  mujoco.mjv_applyPerturbForce(mjModel, mjData, perturb);
}
