/**
 * A native action term has no graph, so no traced-parity check reaches it: the
 * arithmetic here is the implementation, and it decides what force reaches the sim.
 *
 * The sharp case is `encoder_bias`: without subtracting it from the target, every joint
 * sits a fixed distance from where the policy meant.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  applyAction as applySubstep,
  clampActions,
  inHold,
  isTargetActuator,
  processAction,
  readClipActions,
  resolveActionClip,
  stepPhysics,
  type ProcessContext,
  type ResolvedActionTerm,
} from '../applyAction';
import { CommandDelay, resolveDelays, type DelaySpec } from '../commandDelay';

type MjData = import('mujoco').MjData;

/** Just the three arrays `applyAction` touches. */
function fakeData(nu: number, qpos: number[] = [], qvel: number[] = []): MjData {
  return {
    ctrl: new Float64Array(nu),
    qpos: Float64Array.from(qpos),
    qvel: Float64Array.from(qvel),
  } as unknown as MjData;
}

const CONTEXT: ProcessContext = { episodeLength: 0, stepDt: 0.02 };

/** One control step of one substep: `process` every term, then `apply`. */
function applyAction(
  data: MjData,
  terms: ResolvedActionTerm[],
  actions: Float32Array,
  context: ProcessContext = CONTEXT,
): void {
  for (const t of terms) processAction(t, actions, context);
  applySubstep(data, terms);
}

function term(over: Partial<ResolvedActionTerm> = {}): ResolvedActionTerm {
  const n = over.ctrlAdr?.length ?? 2;
  return {
    controlType: 'joint_position',
    ctrlAdr: [0, 1],
    qposAdr: [0, 1],
    qvelAdr: [0, 1],
    actionIndices: [0, 1],
    actionScale: Float32Array.from(Array(n).fill(1)),
    actionOffset: new Float32Array(n),
    defaultJointPos: new Float32Array(n),
    encoderBias: new Float32Array(n),
    positionActuator: Array(n).fill(true),
    kp: new Float32Array(n),
    kd: new Float32Array(n),
    muscleMode: 'sigmoid',
    muscleCtrlLo: new Float32Array(0),
    muscleCtrlHi: new Float32Array(0),
    // Unbounded by default: `resolveActionClip` leaves ±Infinity for an unnamed target.
    clipLo: new Float32Array(n).fill(-Infinity),
    clipHi: new Float32Array(n).fill(Infinity),
    processed: new Float32Array(n),
    active: false,
    rampFrom: null,
    holdS: null,
    holding: false,
    delays: [],
    ...over,
  };
}

describe('applyAction: joint_position', () => {
  it('is default + offset + scale * action, with the encoder bias removed', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({
          defaultJointPos: Float32Array.from([0.5, -0.2]),
          actionOffset: Float32Array.from([0.1, 0.0]),
          actionScale: Float32Array.from([2, 3]),
          encoderBias: Float32Array.from([0.05, -0.01]),
        }),
      ],
      Float32Array.from([0.25, 1.0]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.5 + 0.1 + 2 * 0.25 - 0.05, 6);
    expect(data.ctrl[1]).toBeCloseTo(-0.2 + 0.0 + 3 * 1.0 + 0.01, 6);
  });

  it('offsets from the live reference, not the default pose, when told to', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({
          controlType: 'joint_position_reference',
          defaultJointPos: Float32Array.from([9, 9]),
          referenceJointPos: Float32Array.from([0.4, -0.3]),
          actionScale: Float32Array.from([2, 2]),
          encoderBias: Float32Array.from([0.05, 0.0]),
        }),
      ],
      Float32Array.from([0.5, 1.0]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.4 + 2 * 0.5 - 0.05, 6);
    expect(data.ctrl[1]).toBeCloseTo(-0.3 + 2 * 1.0, 6);
  });

  it('holds the default pose while no clip is loaded', () => {
    // `referenceJointPos` is null until a motion arrives; zeros would fold the robot up.
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({
          controlType: 'joint_position_reference',
          defaultJointPos: Float32Array.from([0.4, -0.3]),
          referenceJointPos: null,
        }),
      ],
      new Float32Array(2),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.4, 6);
    expect(data.ctrl[1]).toBeCloseTo(-0.3, 6);
  });

  it('drives a command term from its command, beside the policy, taking none of its output', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({ ctrlAdr: [0], qposAdr: [0], qvelAdr: [0], actionIndices: [0] }),
        term({
          controlType: 'joint_position_command',
          ctrlAdr: [1],
          qposAdr: [1],
          qvelAdr: [1],
          actionIndices: [0],
          commandValue: Float32Array.from([0.3]),
          actionScale: Float32Array.from([2]),
          actionOffset: Float32Array.from([0.1]),
        }),
      ],
      Float32Array.from([0.5]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.5, 6);
    expect(data.ctrl[1]).toBeCloseTo(0.1 + 2 * 0.3, 6);
  });

  it('writes nothing for a command term whose command has no value', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [term({ controlType: 'joint_position_command', commandValue: null })],
      Float32Array.from([0.5, 0.5]),
    );
    expect(Array.from(data.ctrl)).toEqual([0, 0]);
  });

  it('runs the PD itself for a motor actuator', () => {
    // `biastype=none`: `ctrl` is a torque, so the browser owes the PD itself.
    const data = fakeData(1, [0.2], [0.5]);
    applyAction(
      data,
      [
        term({
          ctrlAdr: [0],
          qposAdr: [0],
          qvelAdr: [0],
          actionIndices: [0],
          actionScale: Float32Array.from([1]),
          actionOffset: new Float32Array(1),
          defaultJointPos: Float32Array.from([0.4]),
          encoderBias: new Float32Array(1),
          positionActuator: [false],
          kp: Float32Array.from([10]),
          kd: Float32Array.from([2]),
        }),
      ],
      Float32Array.from([0.1]),
    );
    // target = 0.4 + 0.1 = 0.5; ctrl = 10 * (0.5 - 0.2) + 2 * (0 - 0.5)
    expect(data.ctrl[0]).toBeCloseTo(10 * 0.3 - 2 * 0.5, 6);
  });

  it('skips a joint with no actuator instead of writing ctrl[-1]', () => {
    const data = fakeData(1);
    applyAction(
      data,
      [term({ ctrlAdr: [-1, 0], actionScale: Float32Array.from([1, 1]) })],
      Float32Array.from([9, 0.3]),
    );
    // Only the second joint has an actuator; a negative address must not wrap.
    expect(data.ctrl[0]).toBeCloseTo(0.3, 6);
  });
});

describe('applyAction: joint_velocity', () => {
  it('writes offset + scale * action as the target of a velocity actuator', () => {
    const data = fakeData(2, [0, 0], [3, -3]);
    applyAction(
      data,
      [
        term({
          controlType: 'joint_velocity',
          actionScale: Float32Array.from([100, 100]),
          actionOffset: Float32Array.from([0.5, 0]),
          // A default pose must not leak in: velocity targets have no pose offset.
          defaultJointPos: Float32Array.from([9, 9]),
        }),
      ],
      Float32Array.from([0.02, -0.01]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.5 + 100 * 0.02, 5);
    expect(data.ctrl[1]).toBeCloseTo(100 * -0.01, 5);
  });

  it('runs a damping law for a motor actuator', () => {
    const data = fakeData(1, [0.7], [1.5]);
    applyAction(
      data,
      [
        term({
          controlType: 'joint_velocity',
          ctrlAdr: [0],
          qposAdr: [0],
          qvelAdr: [0],
          actionIndices: [0],
          actionScale: Float32Array.from([2]),
          positionActuator: [false],
          kp: Float32Array.from([50]),
          kd: Float32Array.from([0.4]),
        }),
      ],
      Float32Array.from([1]),
    );
    // target = 2, so ctrl = 0.4 * (2 - 1.5); kp is unused.
    expect(data.ctrl[0]).toBeCloseTo(0.4 * 0.5, 6);
  });

  it('clips the target, and reads its slice of a shared action vector', () => {
    const data = fakeData(3);
    applyAction(
      data,
      [
        term({ ctrlAdr: [0], qposAdr: [0], qvelAdr: [0], actionIndices: [0] }),
        term({
          controlType: 'joint_velocity',
          ctrlAdr: [1, 2],
          qposAdr: [1, 2],
          qvelAdr: [1, 2],
          actionIndices: [1, 2],
          actionScale: Float32Array.from([10, 10]),
          clipLo: Float32Array.from([-5, -5]),
          clipHi: Float32Array.from([5, 5]),
        }),
      ],
      Float32Array.from([0.3, 0.2, 0.9]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.3, 6);
    expect(data.ctrl[1]).toBeCloseTo(2, 5);
    expect(data.ctrl[2]).toBeCloseTo(5, 6);
  });
});

describe('applyAction: other kinds', () => {
  it('scales straight to torque', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [term({ controlType: 'torque', actionScale: Float32Array.from([5, -2]) })],
      Float32Array.from([0.4, 0.5]),
    );
    // `actionScale` is a Float32Array, so compare at float32 resolution.
    expect(data.ctrl[0]).toBeCloseTo(2, 6);
    expect(data.ctrl[1]).toBeCloseTo(-1, 6);
  });

  it('applies the MyoSuite sigmoid when normalizing, and clips when not', () => {
    const raw = fakeData(2);
    applyAction(
      raw,
      [term({ controlType: 'muscle_activation', muscleMode: 'sigmoid' })],
      Float32Array.from([0.5, 10]),
    );
    // σ(5 * (0.5 - 0.5)) = 0.5, and a large excitation saturates toward 1.
    expect(raw.ctrl[0]).toBeCloseTo(0.5, 6);
    expect(raw.ctrl[1]).toBeGreaterThan(0.99);

    const clipped = fakeData(2);
    applyAction(
      clipped,
      [term({ controlType: 'muscle_activation', muscleMode: 'excitation' })],
      Float32Array.from([-3, 10]),
    );
    expect(Array.from(clipped.ctrl)).toEqual([0, 1]);
  });

  it('direct mode keeps the negative half of the actuator range', () => {
    // A myosuite muscle model declares ctrlrange="-1 1" and a checkpoint trained
    // against the raw control uses the negative half; `excitation` clamps it to zero.
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({
          controlType: 'muscle_activation',
          muscleMode: 'direct',
          muscleCtrlLo: Float32Array.from([-1, -1]),
          muscleCtrlHi: Float32Array.from([1, 1]),
        }),
      ],
      Float32Array.from([-0.4, 7]),
    );
    expect(data.ctrl[0]).toBeCloseTo(-0.4, 6);
    expect(data.ctrl[1]).toBeCloseTo(1, 6);
  });

  it('leaves ctrl at rest for a kind it does not implement', () => {
    // The Python cfg raises for these, so a still actuator beats a garbage torque.
    const data = fakeData(2);
    applyAction(data, [term({ controlType: 'tendon_length' })], Float32Array.from([1, 1]));
    expect(Array.from(data.ctrl)).toEqual([0, 0]);
  });

  it('zeroes actuators no term claims', () => {
    const data = fakeData(3);
    data.ctrl[2] = 7; // left over from a previous step
    applyAction(
      data,
      [term({ controlType: 'torque', actionScale: Float32Array.from([1, 1]) })],
      Float32Array.from([0.1, 0.2]),
    );
    expect(data.ctrl[2]).toBe(0);
  });
});

describe('stepPhysics', () => {
  it('re-applies the action every substep, because the PD reads live state', () => {
    const data = fakeData(1, [0], [0]);
    const seen: number[] = [];
    let stepped = 0;
    const mujoco = {
      mj_step: () => {
        stepped++;
        // Pretend the integration moved the joint, so a stale ctrl would show.
        data.qpos[0] += 0.1;
      },
    };
    const motor = term({
      ctrlAdr: [0],
      qposAdr: [0],
      qvelAdr: [0],
      actionIndices: [0],
      actionScale: Float32Array.from([0]),
      actionOffset: new Float32Array(1),
      defaultJointPos: Float32Array.from([1]),
      encoderBias: new Float32Array(1),
      positionActuator: [false],
      kp: Float32Array.from([1]),
      kd: new Float32Array(1),
    });
    processAction(motor, Float32Array.from([0]), CONTEXT);
    stepPhysics(mujoco as never, {} as never, data, [motor], 3, () => {
      seen.push(data.ctrl[0]);
    });

    expect(stepped).toBe(3);
    // ctrl = 1 * (1 - qpos); qpos advances 0.1 per substep, so the values must move.
    expect(seen.map(v => Number(v.toFixed(3)))).toEqual([1, 0.9, 0.8]);
  });

  it("stops at MuJoCo's own reset, which it reads as the time going back", () => {
    const data = { ...fakeData(0), time: 1 } as unknown as MjData & { time: number };
    let stepped = 0;
    const after: number[] = [];
    const mujoco = {
      mj_step: () => {
        stepped++;
        // The second step diverges: mj_step zeroes the data, time included.
        data.time = stepped === 2 ? 0 : data.time + 0.002;
      },
    };
    const reset = stepPhysics(mujoco as never, {} as never, data, [], 4, undefined, () => after.push(stepped));

    expect(reset).toBe(true);
    expect(stepped).toBe(2);
    expect(after).toEqual([1]);
    expect(stepPhysics(mujoco as never, {} as never, data, [], 2)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// `clip` (mjlab's `BaseActionCfg.clip`), which no reference task sets.
// ---------------------------------------------------------------------------

describe('applyAction: clip', () => {
  it('bounds the processed action *before* the encoder bias is removed', () => {
    // The distinguishing case: mjlab clamps before `apply_actions` subtracts the bias, so
    // the final ctrl sits outside the bound by it. Clamping the target would pin it to 0.5.
    const data = fakeData(1);
    applyAction(
      data,
      [
        term({
          ctrlAdr: [0],
          qposAdr: [0],
          qvelAdr: [0],
          actionIndices: [0],
          actionScale: Float32Array.from([1]),
          actionOffset: new Float32Array(1),
          defaultJointPos: new Float32Array(1),
          encoderBias: Float32Array.from([0.1]),
          positionActuator: [true],
          clipLo: Float32Array.from([-0.5]),
          clipHi: Float32Array.from([0.5]),
        }),
      ],
      Float32Array.from([10]),
    );
    expect(data.ctrl[0]).toBeCloseTo(0.5 - 0.1, 6);
  });

  it('bounds a torque, and adds the offset mjlab adds', () => {
    const data = fakeData(2);
    applyAction(
      data,
      [
        term({
          controlType: 'torque',
          actionScale: Float32Array.from([1, 1]),
          actionOffset: Float32Array.from([0.25, 0]),
          clipLo: Float32Array.from([-1, -1]),
          clipHi: Float32Array.from([1, 1]),
        }),
      ],
      Float32Array.from([0.5, 5]),
    );
    // 0.5 * 1 + 0.25 = 0.75, inside the bound; the second saturates at 1.
    expect(data.ctrl[0]).toBeCloseTo(0.75, 6);
    expect(data.ctrl[1]).toBeCloseTo(1, 6);
  });

  it('bounds a muscle excitation before the activation mapping', () => {
    const data = fakeData(1);
    applyAction(
      data,
      [
        term({
          controlType: 'muscle_activation',
          muscleMode: 'excitation',
          ctrlAdr: [0],
          actionScale: Float32Array.from([1]),
          actionOffset: new Float32Array(1),
          clipLo: Float32Array.from([0]),
          clipHi: Float32Array.from([0.4]),
        }),
      ],
      Float32Array.from([5]),
    );
    // Clipped to 0.4 first, so the [0, 1] squeeze leaves it there rather than at 1.
    expect(data.ctrl[0]).toBeCloseTo(0.4, 6);
  });
});

describe('resolveActionClip', () => {
  const NAMES = ['hip_left', 'hip_right', 'knee_left'];

  it('leaves every target unbounded without a config', () => {
    const { clipLo, clipHi } = resolveActionClip(undefined, NAMES, 3);
    expect(Array.from(clipLo)).toEqual([-Infinity, -Infinity, -Infinity]);
    expect(Array.from(clipHi)).toEqual([Infinity, Infinity, Infinity]);
  });

  it('resolves a pattern by fullmatch, as mjlab does', () => {
    // Anchored: `hip_.*` takes both hips, while `hip` alone matches nothing.
    const { clipLo, clipHi } = resolveActionClip({ 'hip_.*': [-2, 2] }, NAMES, 3);
    expect(Array.from(clipLo)).toEqual([-2, -2, -Infinity]);
    expect(Array.from(clipHi)).toEqual([2, 2, Infinity]);

    const partial = resolveActionClip({ hip: [-2, 2] }, NAMES, 3);
    expect(Array.from(partial.clipLo)).toEqual([-Infinity, -Infinity, -Infinity]);
  });

  it('lets a later pattern override an earlier one on the same target', () => {
    const { clipLo } = resolveActionClip(
      { '.*': [-5, 5], knee_left: [-1, 1] },
      NAMES,
      3,
    );
    expect(Array.from(clipLo)).toEqual([-5, -5, -1]);
  });

  it('warns rather than throwing on a pattern that matches nothing', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { clipLo } = resolveActionClip({ 'ankle_.*': [-1, 1] }, NAMES, 3);
    expect(Array.from(clipLo)).toEqual([-Infinity, -Infinity, -Infinity]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

/**
 * `clip_actions` is the *other* clip: symmetric, from the runner config, on the raw
 * policy output. The pair below is what keeps it from collapsing into `resolveActionClip`.
 */
describe('readClipActions / clampActions', () => {
  it('clamps symmetrically in place', () => {
    const action = Float32Array.from([-3, -0.5, 0.5, 3]);
    clampActions(action, 1);
    expect(Array.from(action)).toEqual([-1, -0.5, 0.5, 1]);
  });

  it('leaves the action untouched when unbounded', () => {
    const action = Float32Array.from([-3, 3]);
    clampActions(action, null);
    expect(Array.from(action)).toEqual([-3, 3]);
  });

  it('treats 0 as a real bound, not as absent', () => {
    // The truthiness bug this guards: `0` pins every action to zero, and reading it as
    // "no bound" would silently run the policy unclamped instead.
    expect(readClipActions(0)).toBe(0);
    const action = Float32Array.from([-3, 3]);
    clampActions(action, readClipActions(0));
    // `=== 0` rather than a deep-equal: clamping a negative to a `0` bound yields `-0`,
    // which is numerically zero everywhere it is used but not `toEqual([0, 0])`.
    expect(action.every((v) => v === 0)).toBe(true);
  });

  it('reads an absent or non-numeric bound as unbounded', () => {
    expect(readClipActions(undefined)).toBeNull();
    expect(readClipActions(null)).toBeNull();
    expect(readClipActions('1.0')).toBeNull();
    expect(readClipActions(NaN)).toBeNull();
    expect(readClipActions(Infinity)).toBeNull();
  });

  it('refuses a negative bound rather than inverting the clamp', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // `Math.min(-1, Math.max(1, x))` would pin everything to -1; unbounded is the safer read.
    expect(readClipActions(-1)).toBeNull();
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Native per-substep options (ADR 0009): mjlab's relative target and command delay, and
// the hold and ramp task terms add.
// ---------------------------------------------------------------------------

/** A one-joint position term on a position actuator, `override` on top. */
function oneJoint(over: Partial<ResolvedActionTerm> = {}): ResolvedActionTerm {
  return term({
    ctrlAdr: [0],
    qposAdr: [0],
    qvelAdr: [0],
    actionIndices: [0],
    actionScale: Float32Array.from([1]),
    actionOffset: new Float32Array(1),
    defaultJointPos: new Float32Array(1),
    encoderBias: new Float32Array(1),
    positionActuator: [true],
    ...over,
  });
}

describe('applyAction: joint_position_relative', () => {
  it('targets the measured position plus the scaled action, read live each substep', () => {
    const data = fakeData(1, [0.3], [0]);
    const t = oneJoint({
      controlType: 'joint_position_relative',
      actionScale: Float32Array.from([0.5]),
      // mjlab's relative target ignores the encoder bias.
      encoderBias: Float32Array.from([0.05]),
    });
    processAction(t, Float32Array.from([0.2]), CONTEXT);
    applySubstep(data, [t]);
    expect(data.ctrl[0]).toBeCloseTo(0.3 + 0.1, 6);
    data.qpos[0] = 0.35;
    applySubstep(data, [t], 1, 2);
    expect(data.ctrl[0]).toBeCloseTo(0.35 + 0.1, 6);
  });

  it('clips the delta, not the target', () => {
    const data = fakeData(1, [1.0], [0]);
    const t = oneJoint({
      controlType: 'joint_position_relative',
      clipLo: Float32Array.from([-0.1]),
      clipHi: Float32Array.from([0.1]),
    });
    applyAction(data, [t], Float32Array.from([5]));
    expect(data.ctrl[0]).toBeCloseTo(1.1, 6);
  });
});

describe('applyAction: hold_s', () => {
  it('targets the measured position while the hold lasts, then the processed action', () => {
    const data = fakeData(1, [0.7], [0]);
    const t = oneJoint({
      holdS: 1.0,
      defaultJointPos: Float32Array.from([0.2]),
      encoderBias: Float32Array.from([0.01]),
    });
    applyAction(data, [t], Float32Array.from([0.5]), { episodeLength: 49, stepDt: 0.02 });
    expect(t.holding).toBe(true);
    expect(data.ctrl[0]).toBeCloseTo(0.7, 6);
    applyAction(data, [t], Float32Array.from([0.5]), { episodeLength: 50, stepDt: 0.02 });
    expect(t.holding).toBe(false);
    expect(data.ctrl[0]).toBeCloseTo(0.2 + 0.5 - 0.01, 6);
  });

  it('ends on the step mjlab ends it, in float32', () => {
    // 50 * 0.02 is 1.0 in float32, so a 1 s hold at 50 Hz is 50 steps, not 51.
    expect(inHold(1.0, { episodeLength: 49, stepDt: 0.02 })).toBe(true);
    expect(inHold(1.0, { episodeLength: 50, stepDt: 0.02 })).toBe(false);
    expect(inHold(null, { episodeLength: 0, stepDt: 0.02 })).toBe(false);
  });

  it('feeds the PD the measured position, so a motor holds without pushing', () => {
    const data = fakeData(1, [0.4], [0.3]);
    const t = oneJoint({
      holdS: 0.5,
      positionActuator: [false],
      kp: Float32Array.from([20]),
      kd: Float32Array.from([1]),
    });
    applyAction(data, [t], Float32Array.from([1]));
    // Only damping remains: kp * (q - q) + kd * (0 - qvel).
    expect(data.ctrl[0]).toBeCloseTo(-0.3, 6);
  });
});

describe('applyAction: interpolate', () => {
  it('ramps from the previous processed action, landing on the new one at the last substep', () => {
    const data = fakeData(1);
    const t = oneJoint({ rampFrom: new Float32Array(1), encoderBias: Float32Array.from([0.01]) });
    processAction(t, Float32Array.from([0.3]), CONTEXT);
    processAction(t, Float32Array.from([0.4]), CONTEXT);
    expect(t.rampFrom?.[0]).toBeCloseTo(0.3, 6);
    const seen: number[] = [];
    for (let k = 0; k < 4; k++) {
      applySubstep(data, [t], k, 4);
      seen.push(data.ctrl[0] + 0.01);
    }
    // inhand's alpha = (k + 1) / decimation.
    expect(seen.map((v) => Number(v.toFixed(4)))).toEqual([0.325, 0.35, 0.375, 0.4]);
  });

  it('applies the processed action at once without a ramp', () => {
    const data = fakeData(1);
    const t = oneJoint();
    processAction(t, Float32Array.from([0.3]), CONTEXT);
    processAction(t, Float32Array.from([0.4]), CONTEXT);
    applySubstep(data, [t], 0, 4);
    expect(data.ctrl[0]).toBeCloseTo(0.4, 6);
  });
});

/** A stand-in for `SeededRng`: `next()` cycles through `values`. */
function scriptedRng(values: number[]) {
  let i = 0;
  return { next: () => values[i++ % values.length] };
}

function delaySpec(over: Partial<DelaySpec> = {}): DelaySpec {
  return {
    indices: [0],
    minLag: 2,
    maxLag: 2,
    holdProb: 0,
    updatePeriod: 0,
    perEnvPhase: false,
    ...over,
  };
}

describe('CommandDelay', () => {
  it('serves the target `lag` steps old, and the reset\'s zeros before that', () => {
    // mjlab writes the zeroed targets once at a reset, so they fill the history.
    const delay = new CommandDelay(delaySpec(), scriptedRng([0.5]));
    const out: number[] = [];
    for (let t = 1; t <= 5; t++) {
      const values = Float32Array.from([t]);
      delay.apply(values);
      out.push(values[0]);
    }
    expect(out).toEqual([0, 0, 1, 2, 3]);
  });

  it('starts over at a reset', () => {
    const delay = new CommandDelay(delaySpec(), scriptedRng([0.5]));
    for (let t = 1; t <= 4; t++) delay.apply(Float32Array.from([t]));
    delay.reset();
    const out: number[] = [];
    for (const t of [10, 11, 12]) {
      const values = Float32Array.from([t]);
      delay.apply(values);
      out.push(values[0]);
    }
    expect(out).toEqual([0, 0, 10]);
  });

  it('draws the lag from [min, max] each step, and keeps it with hold_prob', () => {
    // randint(0, 3) at next() = 0.99 is 3; with hold_prob 1 the draw is ignored.
    const held = new CommandDelay(delaySpec({ minLag: 0, maxLag: 3, holdProb: 1 }), scriptedRng([0.99]));
    const out: number[] = [];
    for (let t = 0; t < 5; t++) {
      const values = Float32Array.from([t]);
      held.apply(values);
      out.push(values[0]);
    }
    // The lag starts at 0 and is never redrawn.
    expect(out).toEqual([0, 1, 2, 3, 4]);
  });

  it('touches only the joints it covers', () => {
    const delay = new CommandDelay(delaySpec({ indices: [1], minLag: 1, maxLag: 1 }), scriptedRng([0]));
    const first = Float32Array.from([5, 6]);
    delay.apply(first);
    const second = Float32Array.from([7, 8]);
    delay.apply(second);
    expect(Array.from(second)).toEqual([7, 6]);
  });

  it('delays the target before the PD, which reads live state', () => {
    const data = fakeData(1, [0], [0]);
    const t = oneJoint({
      positionActuator: [false],
      kp: Float32Array.from([10]),
      delays: [new CommandDelay(delaySpec({ minLag: 1, maxLag: 1 }), scriptedRng([0]))],
    });
    processAction(t, Float32Array.from([0.1]), CONTEXT);
    applySubstep(data, [t], 0, 2);
    processAction(t, Float32Array.from([0.5]), CONTEXT);
    applySubstep(data, [t], 0, 2);
    // The second step still acts on 0.1.
    expect(data.ctrl[0]).toBeCloseTo(10 * 0.1, 6);
  });
});

describe('resolveDelays', () => {
  it('maps joints to target positions and drops what the term does not drive', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const specs = resolveDelays(
      [
        { joints: ['robot/b', 'robot/x'], min_lag: 1, max_lag: 3, hold_prob: 0.2 },
        { joints: ['robot/y'], min_lag: 2 },
      ],
      ['robot/a', 'robot/b'],
      'joint_pos',
    );
    expect(specs).toEqual([
      { indices: [1], minLag: 1, maxLag: 3, holdProb: 0.2, updatePeriod: 0, perEnvPhase: true },
    ]);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});

describe('isTargetActuator', () => {
  const bias = { affine: 1, dcMotor: 3 };
  /** One actuator per entry: its bias type and its `gainprm[8]`. */
  function model(entries: Array<[number, number]>) {
    const gainprm = new Float64Array(entries.length * 10);
    entries.forEach(([, mode], i) => (gainprm[i * 10 + 8] = mode));
    return {
      actuator_biastype: Int32Array.from(entries.map(([b]) => b)),
      actuator_gainprm: gainprm,
    } as never;
  }

  it('takes a position or velocity actuator, and a dcmotor in position or velocity mode', () => {
    const m = model([[1, 0], [0, 0], [3, 1], [3, 2], [3, 0]]);
    expect([0, 1, 2, 3, 4].map((adr) => isTargetActuator(m, adr, bias))).toEqual([
      true,
      false,
      true,
      true,
      false,
    ]);
    expect(isTargetActuator(m, -1, bias)).toBe(false);
  });
});
