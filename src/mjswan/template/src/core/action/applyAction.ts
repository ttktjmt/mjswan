/**
 * The one manager with no ONNX: a closed built-in set on the hottest loop, mirroring
 * mjlab's own `ActionTermCfg` kinds rather than tracing anything.
 *
 * A free function over `mjData`, not a method on the DOM-bound runtime, so the
 * rollout-parity harness can drive it in Node.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** mjlab's `ActionTermCfg` kinds this runtime implements. */
export type ControlType =
  | 'joint_position'
  | 'joint_position_reference'
  | 'torque'
  | 'muscle_activation';

/**
 * How a muscle term maps its raw action, mirroring myosuite's `action_mode`.
 * `direct` clips to the actuator's own range instead of [0, 1] — a muscle model with
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
  /** Indices into the flat policy action vector for this term's joints. */
  actionIndices: number[];
  actionScale: Float32Array;
  actionOffset: Float32Array;
  defaultJointPos: Float32Array;
  /**
   * `joint_position_reference` only: this step's reference pose in policy-joint order,
   * refreshed by the runtime. `null` before a clip loads, where the default pose
   * stands in — a still robot rather than one folded into zeros.
   */
  referenceJointPos?: Float32Array | null;
  encoderBias: Float32Array;
  /**
   * Per-actuator: true = position (`biastype=affine`), so `ctrl` is a target and MuJoCo
   * runs the PD; false = motor, so `ctrl` is a torque and the PD is computed here.
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
   * bias — clamping the other side would move every bound by the bias.
   */
  clipLo: Float32Array;
  clipHi: Float32Array;
}

/**
 * Write `mjData.ctrl` for one control step from the policy's action vector.
 *
 * Zeroes `ctrl` first, so an actuator no term claims stays at rest rather than
 * holding the previous step's value.
 */
export function applyAction(
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  actions: Float32Array,
): void {
  const ctrl = mjData.ctrl;
  ctrl.fill(0.0);
  for (const term of terms) applyActionTerm(mjData, term, actions);
}

function applyActionTerm(
  mjData: MjData,
  term: ResolvedActionTerm,
  actions: Float32Array,
): void {
  const {
    controlType,
    ctrlAdr,
    qposAdr,
    qvelAdr,
    actionIndices,
    actionScale,
    actionOffset,
    defaultJointPos,
    encoderBias,
    positionActuator,
    kp,
    kd,
    muscleMode,
    muscleCtrlLo,
    muscleCtrlHi,
  } = term;
  const numJoints = ctrlAdr.length;
  const ctrl = mjData.ctrl;

  if (controlType === 'joint_position' || controlType === 'joint_position_reference') {
    const reference = term.referenceJointPos ?? null;
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex < 0) continue;
      const actionValue = actions[actionIndices[i]] ?? 0;
      const base = reference ? (reference[actionIndices[i]] ?? 0) : defaultJointPos[i];
      // Un-bias the target: the policy was trained against a biased reading.
      const processed = clamp(
        base + actionOffset[i] + actionScale[i] * actionValue,
        term.clipLo[i],
        term.clipHi[i],
      );
      const target = processed - encoderBias[i];

      if (positionActuator[i]) {
        ctrl[ctrlIndex] = target;
      } else {
        const qpos = mjData.qpos[qposAdr[i]];
        const qvel = mjData.qvel[qvelAdr[i]];
        ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
      }
    }
    return;
  }

  if (controlType === 'torque') {
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex >= 0) {
        // `+ offset`, as mjlab does — 0.0 on every reference task, so easily missed.
        ctrl[ctrlIndex] = clamp(
          actionScale[i] * (actions[actionIndices[i]] ?? 0) + actionOffset[i],
          term.clipLo[i],
          term.clipHi[i],
        );
      }
    }
    return;
  }

  if (controlType === 'muscle_activation') {
    // Shared pre-step: raw = scale * action + offset, then per `muscleMode`:
    //   sigmoid:    MyoSuite-canonical σ(5 * (raw - 0.5)).
    //   direct:     clip to the actuator's own ctrlrange, written unchanged.
    //   excitation: clip(raw, 0, 1) for models that already output excitation.
    for (let i = 0; i < numJoints; i++) {
      const ctrlIndex = ctrlAdr[i];
      if (ctrlIndex < 0) continue;
      // The bounds apply before the activation mapping; [0, 1] is the actuator's range.
      const raw = clamp(
        (actions[actionIndices[i]] ?? 0) * actionScale[i] + actionOffset[i],
        term.clipLo[i],
        term.clipHi[i],
      );
      ctrl[ctrlIndex] =
        muscleMode === 'sigmoid'
          ? 1 / (1 + Math.exp(-5 * (raw - 0.5)))
          : muscleMode === 'direct'
            ? clamp(raw, muscleCtrlLo[i] ?? -1, muscleCtrlHi[i] ?? 1)
            : clamp01(raw);
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
    const re = new RegExp(`^(?:${pattern})$`);
    for (let i = 0; i < Math.min(length, targetNames.length); i++) {
      if (!re.test(targetNames[i])) continue;
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
 * sees it — unlike `resolveActionClip`, which bounds `raw * scale + offset` per target.
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

/** `decimation` substeps, re-applying the action each time: a motor's PD reads live state. */
export function stepPhysics(
  mujoco: { mj_step(model: MjModel, data: MjData): void },
  mjModel: MjModel,
  mjData: MjData,
  terms: readonly ResolvedActionTerm[],
  actions: Float32Array,
  decimation: number,
  onSubstep?: () => void,
  /** After the step, where mjlab's `scene.update(dt=physics_dt)` reads sensors. */
  onSubstepEnd?: () => void,
): void {
  for (let substep = 0; substep < decimation; substep++) {
    applyAction(mjData, terms, actions);
    onSubstep?.();
    mujoco.mj_step(mjModel, mjData);
    onSubstepEnd?.();
  }
}
