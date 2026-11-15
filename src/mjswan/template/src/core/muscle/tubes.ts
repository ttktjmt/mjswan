// Ported from MyoSuite (myosuite/viz/muscle_tubes.py, blender_render.py), Copyright (c)
// MyoSuite Authors, Apache-2.0; modified for mjswan. See NOTICE.
import type { MainModule, MjModel } from 'mujoco';

/** MyoSuite's muscle colours, linear RGB. */
export const MUSCLE_RELAXED = [0.5, 0.16, 0.15] as const;
export const MUSCLE_ACTIVE = [0.62, 0.012, 0.02] as const;
export const MUSCLE_UNIFORM = [0.42, 0.11, 0.1] as const;
export const PATH_RELAXED = [0.05, 0.05, 0.05] as const;
export const PATH_ACTIVE = [0.95, 0.3, 0.3] as const;

/**
 * The muscles MyoSuite draws: muscle actuators on a spatial tendon of two or more
 * elements that starts at a site, in model order.
 */
export function muscleActuators(mujoco: MainModule, model: MjModel): number[] {
  const muscle = mujoco.mjtGain.mjGAIN_MUSCLE.value;
  const tendon = mujoco.mjtTrn.mjTRN_TENDON.value;
  const site = mujoco.mjtWrap.mjWRAP_SITE.value;
  // Each embind getter allocates a view, so read every array once.
  const gaintype = model.actuator_gaintype;
  const trntype = model.actuator_trntype;
  const trnid = model.actuator_trnid;
  const tendonNum = model.tendon_num;
  const tendonAdr = model.tendon_adr;
  const wrapType = model.wrap_type;
  const actuators: number[] = [];
  for (let a = 0; a < model.nu; a++) {
    if (gaintype[a] !== muscle || trntype[a] !== tendon) continue;
    const t = trnid[2 * a];
    if (tendonNum[t] > 1 && wrapType[tendonAdr[t]] === site) actuators.push(a);
  }
  return actuators;
}

/** MyoSuite's `_muscle_colour` into `out` at `offset`: a path brightens by the activation's fourth root. */
export function muscleColour(activation: number, paths: boolean, out: Float32Array, offset: number): void {
  const clamped = Math.min(Math.max(activation, 0), 1);
  const a = paths ? clamped ** 0.25 : clamped;
  const [relaxed, active] = paths ? [PATH_RELAXED, PATH_ACTIVE] : [MUSCLE_RELAXED, MUSCLE_ACTIVE];
  for (let c = 0; c < 3; c++) out[offset + c] = relaxed[c] + (active[c] - relaxed[c]) * a;
}
