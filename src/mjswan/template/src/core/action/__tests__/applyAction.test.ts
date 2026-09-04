/**
 * Action is the one manager with no graph, so no traced-parity check reaches it — the
 * arithmetic here is the implementation, and it decides what force reaches the sim.
 *
 * The sharp case is `encoder_bias`: without subtracting it from the target, every joint
 * sits a fixed distance from where the policy meant.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  applyAction,
  clampActions,
  readClipActions,
  resolveActionClip,
  stepPhysics,
  type ResolvedActionTerm,
} from '../applyAction';

type MjData = import('mujoco').MjData;

/** Just the three arrays `applyAction` touches. */
function fakeData(nu: number, qpos: number[] = [], qvel: number[] = []): MjData {
  return {
    ctrl: new Float64Array(nu),
    qpos: Float64Array.from(qpos),
    qvel: Float64Array.from(qvel),
  } as unknown as MjData;
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
    ...over,
  };
}

describe('applyAction — joint_position', () => {
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

describe('applyAction — other kinds', () => {
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
    // against the raw control really does use the negative half; `excitation` would
    // clamp it to zero and the policy tracks measurably worse.
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
    stepPhysics(mujoco as never, {} as never, data, [motor], Float32Array.from([0]), 3, () => {
      seen.push(data.ctrl[0]);
    });

    expect(stepped).toBe(3);
    // ctrl = 1 * (1 - qpos); qpos advances 0.1 per substep, so the values must move.
    expect(seen.map(v => Number(v.toFixed(3)))).toEqual([1, 0.9, 0.8]);
  });
});

// ---------------------------------------------------------------------------
// `clip` (mjlab's `BaseActionCfg.clip`), which no reference task sets.
// ---------------------------------------------------------------------------

describe('applyAction — clip', () => {
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
