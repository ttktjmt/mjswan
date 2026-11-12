/**
 * Action terms in two halves, as mjlab splits them (ADR 0009):
 *
 * - `processAction`, once per control step: the policy's action, or a command's value,
 *   into the term's `processed` targets. A traced term's graph does this instead
 *   (`OnnxActionProcess`).
 * - `applyAction`, once per physics substep: `processed` into `ctrl`, through the native
 *   per-substep options (a ramp, a hold, a relative target, a command delay) and the PD a
 *   motor needs. Synchronous, with no ONNX: it runs on the hottest loop.
 *
 * Free functions over `mjData`, not methods on the DOM-bound runtime, so the parity tests
 * can drive them in Node.
 */

import { CommandDelay, resolveDelays, type DelayConfigEntry, type DelayRng } from './commandDelay';
import { compileNamePattern } from '../policy/namePattern';
import { BufferBudget } from '../policy/bufferBudget';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** mjlab's `ActionTermCfg` kinds this runtime implements, named by what `applyAction` does. */
export type ControlType =
  | 'joint_position'
  | 'joint_position_reference'
  | 'joint_position_command'
  | 'joint_position_relative'
  | 'joint_velocity'
  | 'torque'
  | 'muscle_activation';

/**
 * How a muscle term maps its raw action, mirroring myosuite's `action_mode`.
 * `direct` clips to the actuator's own range instead of [0, 1]: a muscle model with
 * `ctrlrange="-1 1"` uses the negative half, and clamping it away degrades tracking.
 */
export type MuscleMode = 'sigmoid' | 'direct' | 'excitation';

/** One resolved action term: the build's descriptor, names already resolved to addresses. */
export interface ResolvedActionTerm {
  controlType: string;
  /** `mjData.ctrl` address per joint; `< 0` for a joint with no actuator. */
  ctrlAdr: number[];
  qposAdr: number[];
  qvelAdr: number[];
  /** Indices into the policy's action vector, or into `commandValue` for a command term. */
  actionIndices: number[];
  actionScale: Float32Array;
  actionOffset: Float32Array;
  defaultJointPos: Float32Array;
  /**
   * `joint_position_reference` only: this step's reference pose in policy-joint order,
   * refreshed by the runtime. `null` before a clip loads, where the default pose
   * stands in: a still robot rather than one folded into zeros.
   */
  referenceJointPos?: Float32Array | null;
  /** Command term's raw input for this step; `null` skips the term. */
  commandValue?: Float32Array | null;
  /** The command a reference or command term reads each step. */
  commandName?: string;
  encoderBias: Float32Array;
  /**
   * Per-actuator: true when `ctrl` is a target MuJoCo tracks (a position or velocity
   * actuator, or a `<dcmotor>` in position or velocity mode); false for a motor, whose
   * `ctrl` is a torque computed here.
   */
  positionActuator: boolean[];
  kp: Float32Array;
  kd: Float32Array;
  /** `muscle_activation` only: how a raw action becomes a control. */
  muscleMode: MuscleMode;
  /** `muscle_activation` with `direct`: each actuator's own `ctrlrange`, per joint. */
  muscleCtrlLo: Float32Array;
  muscleCtrlHi: Float32Array;
  /**
   * Per-target bounds on the *processed* action (`raw * scale + offset`), `±Infinity`
   * where unbounded. mjlab clamps there, before `joint_position` subtracts the encoder
   * bias: clamping the other side would move every bound by the bias.
   */
  clipLo: Float32Array;
  clipHi: Float32Array;
  /** This control step's processed action, one per target: `process` writes it, `apply` reads it. */
  processed: Float32Array;
  /**
   * False while there is nothing to apply (a command with no value, a traced term whose
   * graph has yet to run): `apply` skips the term.
   */
  active: boolean;
  /**
   * Set when the target ramps across the substeps (inhand's `interpolate_decimation`):
   * the processed action from before this control step, which the ramp starts from.
   */
  rampFrom: Float32Array | null;
  /**
   * Seconds from an episode's start during which every target is the joint's measured
   * position (luwu's held action), or `null`.
   */
  holdS: number | null;
  /** Whether this control step is inside `holdS`; `process` decides it. */
  holding: boolean;
  /** mjlab's actuator command delays on the term's joints, one per mjlab delay buffer. */
  delays: CommandDelay[];
}

/** MuJoCo's `mjNGAIN`, the stride of `actuator_gainprm`. */
const MJ_NGAIN = 10;
/** Where `mjs_setToDCMotor` keeps a `<dcmotor>`'s input mode: 0 voltage, 1 position, 2 velocity. */
const DCMOTOR_INPUT_MODE = 8;

/** The `mjtBias` values `isTargetActuator` tells apart, from the loaded MuJoCo build. */
export interface BiasTypes {
  affine: number;
  dcMotor: number;
}

/**
 * Whether `ctrl` at actuator `adr` is a target MuJoCo tracks rather than a torque this
 * runtime computes: a position or velocity actuator (`biastype=affine`), or a
 * `<dcmotor>` whose input mode is a position or a velocity. A voltage-mode `<dcmotor>`,
 * like a plain motor, takes what the term writes as is.
 */
export function isTargetActuator(
  model: Pick<MjModel, 'actuator_biastype' | 'actuator_gainprm'>,
  adr: number,
  bias: BiasTypes,
): boolean {
  if (adr < 0) return false;
  const biasType = model.actuator_biastype[adr];
  if (biasType === bias.affine) return true;
  if (biasType !== bias.dcMotor) return false;
  const mode = model.actuator_gainprm[adr * MJ_NGAIN + DCMOTOR_INPUT_MODE];
  return mode === 1 || mode === 2;
}

/** What a manifest entry asks of the substeps (ADR 0009). */
export interface SubstepConfig {
  interpolate?: boolean;
  hold_s?: number;
  delay?: DelayConfigEntry[];
}

/** A term's per-substep options from its entry, with `targetNames` in target order. */
export function substepOptions(
  entry: SubstepConfig,
  targetNames: readonly string[],
  termKey: string,
  rng: DelayRng,
  budget: BufferBudget = new BufferBudget(),
): Pick<ResolvedActionTerm, 'rampFrom' | 'holdS' | 'holding' | 'delays'> {
  return {
    rampFrom: entry.interpolate === true ? new Float32Array(targetNames.length) : null,
    holdS: typeof entry.hold_s === 'number' ? entry.hold_s : null,
    holding: false,
    delays: resolveDelays(entry.delay, targetNames, termKey).map((spec) => {
      budget.reserve(spec.maxLag + 1, spec.indices.length, `action "${termKey}" delay`);
      return new CommandDelay(spec, rng);
    }),
  };
}

/** What `processAction` reads besides the action: the episode clock a hold runs on. */
export interface ProcessContext {
  /** Control steps since the last reset: mjlab's `episode_length_buf`. */
  episodeLength: number;
  /** Seconds per control step: mjlab's `step_dt`. */
  stepDt: number;
}

/**
 * Whether a control step falls inside a term's hold, in mjlab's float32 arithmetic
 * (`episode_length_buf * step_dt < hold_s`), so the hold ends on the same step.
 */
export function inHold(holdS: number | null, context: ProcessContext): boolean {
  if (holdS === null) return false;
  const elapsed = Math.fround(Math.fround(context.episodeLength) * Math.fround(context.stepDt));
  return elapsed < Math.fround(holdS);
}

/**
 * One control step's `process`: the policy's action (a command term: its command) into
 * `term.processed`, as mjlab's built-in `process_actions` computes it.
 */
export function processAction(
  term: ResolvedActionTerm,
  policyActions: Float32Array,
  context: ProcessContext,
): void {
  term.holding = inHold(term.holdS, context);
  const actions =
    term.controlType === 'joint_position_command' ? term.commandValue : policyActions;
  term.active = Boolean(actions);
  if (!actions) return;
  // Before the overwrite: the ramp starts from the previous step's processed action.
  term.rampFrom?.set(term.processed);

  const { controlType, actionIndices, actionScale, actionOffset, clipLo, clipHi, processed } =
    term;
  const n = processed.length;

  if (
    controlType === 'joint_position' ||
    controlType === 'joint_position_reference' ||
    controlType === 'joint_position_command'
  ) {
    const reference = term.referenceJointPos ?? null;
    for (let i = 0; i < n; i++) {
      const base = reference ? (reference[actionIndices[i]] ?? 0) : term.defaultJointPos[i];
      processed[i] = clamp(
        base + actionOffset[i] + actionScale[i] * (actions[actionIndices[i]] ?? 0),
        clipLo[i],
        clipHi[i],
      );
    }
    return;
  }

  if (
    controlType === 'joint_position_relative' ||
    controlType === 'joint_velocity' ||
    controlType === 'torque'
  ) {
    for (let i = 0; i < n; i++) {
      // `+ offset`, as mjlab does: 0.0 on every reference task, so easily missed.
      processed[i] = clamp(
        actionScale[i] * (actions[actionIndices[i]] ?? 0) + actionOffset[i],
        clipLo[i],
        clipHi[i],
      );
    }
    return;
  }

  if (controlType === 'muscle_activation') {
    // Shared pre-step: raw = scale * action + offset, clipped, then per `muscleMode`:
    //   sigmoid:    MyoSuite-canonical σ(5 * (raw - 0.5)).
    //   direct:     clip to the actuator's own ctrlrange, written unchanged.
    //   excitation: clip(raw, 0, 1) for models that already output excitation.
    for (let i = 0; i < n; i++) {
      const raw = clamp(
        (actions[actionIndices[i]] ?? 0) * actionScale[i] + actionOffset[i],
        clipLo[i],
        clipHi[i],
      );
      processed[i] =
        term.muscleMode === 'sigmoid'
          ? 1 / (1 + Math.exp(-5 * (raw - 0.5)))
          : term.muscleMode === 'direct'
            ? clamp(raw, term.muscleCtrlLo[i] ?? -1, term.muscleCtrlHi[i] ?? 1)
            : clamp01(raw);
    }
    return;
  }

  // A kind this runtime does not implement: the Python cfg raises for these, so an
  // actuator at rest beats a garbage control.
  term.active = false;
}

type DataViews = Pick<MjData, 'ctrl' | 'qpos' | 'qvel'>;

/**
 * One physics substep's `apply`: every term's `processed` into `mjData.ctrl`.
 *
 * Zeroes `ctrl` first, so an actuator no term claims stays at rest rather than holding
 * the previous step's value. `substep` counts from 0 within the control step, for a
 * term that ramps across it.
 */
export function applyAction(
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  substep = 0,
  decimation = 1,
): void {
  // Read once: each embind getter allocates a new view.
  const views: DataViews = { ctrl: mjData.ctrl, qpos: mjData.qpos, qvel: mjData.qvel };
  views.ctrl.fill(0.0);
  for (const term of terms) {
    if (term.active) applyActionTerm(views, term, substep, decimation);
  }
}

/** Per-term scratch for the targets one substep computes, so `apply` allocates nothing. */
const targetScratch = new WeakMap<ResolvedActionTerm, Float32Array>();

function targetsOf(term: ResolvedActionTerm): Float32Array {
  let targets = targetScratch.get(term);
  if (!targets || targets.length !== term.processed.length) {
    targets = new Float32Array(term.processed.length);
    targetScratch.set(term, targets);
  }
  return targets;
}

function applyActionTerm(
  { ctrl, qpos, qvel }: DataViews,
  term: ResolvedActionTerm,
  substep: number,
  decimation: number,
): void {
  const { controlType, ctrlAdr, qposAdr, qvelAdr, processed, positionActuator, kp, kd } = term;
  const n = processed.length;

  if (controlType === 'muscle_activation') {
    for (let i = 0; i < n; i++) if (ctrlAdr[i] >= 0) ctrl[ctrlAdr[i]] = processed[i];
    return;
  }

  const targets = targetsOf(term);
  const position = controlType !== 'joint_velocity' && controlType !== 'torque';
  if (position) {
    const ramp = term.rampFrom;
    // mjlab's `(k + 1) / decimation`: the ramp lands on the new target at the last substep.
    const alpha = ramp ? (substep + 1) / decimation : 1;
    const relative = controlType === 'joint_position_relative';
    for (let i = 0; i < n; i++) {
      const value = ramp ? (1 - alpha) * ramp[i] + alpha * processed[i] : processed[i];
      if (term.holding) targets[i] = qpos[qposAdr[i]];
      else if (relative) targets[i] = qpos[qposAdr[i]] + value;
      // Un-bias the target: the policy was trained against a biased reading.
      else targets[i] = value - term.encoderBias[i];
    }
  } else {
    targets.set(processed);
  }
  for (const delay of term.delays) delay.apply(targets);

  for (let i = 0; i < n; i++) {
    const ctrlIndex = ctrlAdr[i];
    if (ctrlIndex < 0) continue;
    if (controlType === 'torque' || positionActuator[i]) {
      ctrl[ctrlIndex] = targets[i];
    } else if (position) {
      ctrl[ctrlIndex] = kp[i] * (targets[i] - qpos[qposAdr[i]]) + kd[i] * (0 - qvel[qvelAdr[i]]);
    } else {
      // A velocity actuator (`biastype=affine`) runs its own damping on `ctrl`.
      ctrl[ctrlIndex] = kd[i] * (targets[i] - qvel[qvelAdr[i]]);
    }
  }
}

function clamp01(value: number): number {
  return value < 0 ? 0 : value > 1 ? 1 : value;
}

function clamp(value: number, lo: number, hi: number): number {
  return value < lo ? lo : value > hi ? hi : value;
}

/**
 * Per-target bounds from a pattern-keyed `clip` config. Anchored, as mjlab's
 * `re.fullmatch` is, hence not the exact-name resolver `stiffness`/`damping` share.
 * An unmatched target stays unbounded.
 */
export function resolveActionClip(
  clip: Record<string, readonly number[]> | undefined,
  targetNames: readonly string[],
  length: number,
): { clipLo: Float32Array; clipHi: Float32Array } {
  const clipLo = new Float32Array(length).fill(-Infinity);
  const clipHi = new Float32Array(length).fill(Infinity);
  if (!clip) return { clipLo, clipHi };
  for (const [pattern, bounds] of Object.entries(clip)) {
    if (!Array.isArray(bounds) || bounds.length < 2) {
      console.warn(`[applyAction] clip "${pattern}" needs [min, max]; ignoring.`);
      continue;
    }
    let matched = 0;
    const matches = compileNamePattern(pattern);
    for (let i = 0; i < Math.min(length, targetNames.length); i++) {
      if (!matches(targetNames[i])) continue;
      clipLo[i] = bounds[0];
      clipHi[i] = bounds[1];
      matched++;
    }
    if (matched === 0) {
      console.warn(`[applyAction] clip "${pattern}" matched no target; ignoring.`);
    }
  }
  return { clipLo, clipHi };
}

/**
 * The policy's raw-action bound (`clip_actions`), or `null` for unbounded.
 *
 * From the *runner* config, bounding the policy's output symmetrically before any term
 * sees it, unlike `resolveActionClip`, which bounds `raw * scale + offset` per target.
 *
 * `0` is a legal bound (it pins every action to zero), so truthiness will not do here.
 * A negative one would invert the clamp, and is refused.
 */
export function readClipActions(clipActions: unknown): number | null {
  if (typeof clipActions !== 'number' || !Number.isFinite(clipActions)) return null;
  if (clipActions < 0) {
    console.warn(`[applyAction] Ignoring negative clip_actions: ${clipActions}`);
    return null;
  }
  return clipActions;
}

/**
 * Clamp the raw action in place to `[-bound, +bound]`.
 *
 * In-place because the caller's copy is stored as the last action: rsl-rl clamps ahead
 * of `env.step`, so a `last_action` observation sees the clamped vector, never the raw.
 */
export function clampActions(action: Float32Array, bound: number | null): void {
  if (bound === null) return;
  for (let i = 0; i < action.length; i++) {
    action[i] = Math.min(bound, Math.max(-bound, action[i]));
  }
}

/**
 * `decimation` substeps, re-applying the processed action each time: a motor's PD reads live
 * state. Returns true, skipping the substeps left, when `mj_step` reset the data itself on a
 * bad qpos, qvel or qacc. Studio's `StepControl` detects it the same way: the time went back.
 */
export function stepPhysics(
  mujoco: { mj_step(model: MjModel, data: MjData): void },
  mjModel: MjModel,
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  decimation: number,
  onSubstep?: () => void,
  /** After the step, where mjlab's `scene.update(dt=physics_dt)` reads sensors. */
  onSubstepEnd?: () => void,
): boolean {
  for (let substep = 0; substep < decimation; substep++) {
    applyAction(mjData, terms, substep, decimation);
    onSubstep?.();
    const time = mjData.time;
    mujoco.mj_step(mjModel, mjData);
    if (mjData.time < time) return true;
    onSubstepEnd?.();
  }
  return false;
}
