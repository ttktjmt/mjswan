/**
 * Action parity: task-defined action terms against mjlab, substep by substep.
 * The Builder's own entry and graphs run through the real `OnnxActionProcess` over ORT,
 * then the real `applyAction` with the entry's ramp, hold or command delay; mjlab's
 * `ctrl` at each substep is the answer.
 *
 * The cases: in-hand rotation's delta term with its substep ramp, luwu's held action,
 * and an EMA term on mjlab's own apply behind a two-step command delay, each with an
 * episode reset partway through.
 *
 * **States are replayed, not co-simulated**, as in `rolloutParity.test.ts`: each
 * substep's joint positions come from the fixture.
 *
 * Regenerate: `MUJOCO_GL=disable .venv/bin/python tests/dump_action_fixture.py`
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import fixture from './fixtures/action/action.json';
import { createOnnxSession, type OnnxInputSlot, type OnnxSession } from '../../onnx/session';
import { SeededRng } from '../../rng';
import { ActionManager } from '../ActionManager';
import { applyAction, substepOptions, type ResolvedActionTerm, type SubstepConfig } from '../applyAction';
import {
  OnnxActionProcess,
  type OnnxActionGraphConfig,
  type OnnxActionProcessConfig,
} from '../OnnxActionProcess';

const FIXTURES = join(__dirname, 'fixtures/action');

interface Substep {
  /** The joint positions `apply_actions` read. */
  qpos: number[];
  /** What mjlab wrote, after any delay. */
  ctrl: number[];
}

interface Step {
  raw: number[];
  episode_length: number;
  /** Set when the episode restarted before this step: the joints the reset left. */
  reset_joint_pos?: number[];
  substeps: Substep[];
}

interface Case {
  entry: SubstepConfig & {
    type: string;
    process: OnnxActionProcessConfig;
    reset?: OnnxActionGraphConfig;
  };
  joints: string[];
  decimation: number;
  step_dt: number;
  steps: Step[];
}

const CASES = fixture as unknown as Record<string, Case>;

async function sessionFor(name: string, ref: string): Promise<OnnxSession> {
  const buf = readFileSync(join(FIXTURES, name, ref));
  return createOnnxSession(
    buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  );
}

/** The arm's joints, each on a position actuator at the same index. */
function armTerm(c: Case, rng: SeededRng): ResolvedActionTerm {
  const n = c.joints.length;
  const index = Array.from({ length: n }, (_, i) => i);
  return {
    controlType: c.entry.type,
    ctrlAdr: index,
    qposAdr: index,
    qvelAdr: index,
    actionIndices: index,
    actionScale: new Float32Array(n).fill(1),
    actionOffset: new Float32Array(n),
    defaultJointPos: new Float32Array(n),
    encoderBias: new Float32Array(n),
    positionActuator: new Array(n).fill(true),
    kp: new Float32Array(n),
    kd: new Float32Array(n),
    muscleMode: 'sigmoid',
    muscleCtrlLo: new Float32Array(0),
    muscleCtrlHi: new Float32Array(0),
    clipLo: new Float32Array(n).fill(-Infinity),
    clipHi: new Float32Array(n).fill(Infinity),
    processed: new Float32Array(n),
    active: false,
    ...substepOptions(c.entry, c.joints, 'joint_pos', rng),
  };
}

/** Each substep's `ctrl` as the browser writes it, replaying the fixture's states. */
async function replay(name: string): Promise<{ got: number[][]; want: number[][] }> {
  const c = CASES[name];
  const rng = new SeededRng(0);
  const live = { jointPos: [] as number[], episodeLength: 0 };
  const readSlot = (slot: OnnxInputSlot): Float32Array | null => {
    if (slot.entity === 'robot' && slot.field === 'joint_pos') return Float32Array.from(live.jointPos);
    if (slot.env === 'episode_length_buf') return Float32Array.of(live.episodeLength);
    return null;
  };
  const traced = new OnnxActionProcess('joint_pos', c.entry.process, c.entry.reset ?? null, {
    process: await sessionFor(name, c.entry.process.onnx),
    reset: c.entry.reset ? await sessionFor(name, c.entry.reset.onnx) : null,
    rng,
    readSlot,
  });
  const manager = new ActionManager([{ name: 'joint_pos', term: armTerm(c, rng), traced }]);
  const n = c.joints.length;
  const data = {
    qpos: new Float64Array(n),
    qvel: new Float64Array(n),
    ctrl: new Float64Array(n),
  } as unknown as Parameters<typeof applyAction>[0];

  const got: number[][] = [];
  const want: number[][] = [];
  for (const step of c.steps) {
    if (step.reset_joint_pos) {
      live.jointPos = step.reset_joint_pos;
      manager.resetActuators();
      await manager.reset();
    }
    live.episodeLength = step.episode_length;
    await manager.process(Float32Array.from(step.raw), {
      episodeLength: step.episode_length,
      stepDt: c.step_dt,
    });
    step.substeps.forEach((substep, k) => {
      data.qpos.set(substep.qpos);
      applyAction(data, manager.terms, k, c.decimation);
      got.push(Array.from(data.ctrl));
      want.push(substep.ctrl);
    });
  }
  return { got, want };
}

function expectClose(got: number[][], want: number[][]): void {
  expect(got).toHaveLength(want.length);
  got.forEach((row, i) =>
    row.forEach((value, j) => {
      expect(value, `substep ${i}, joint ${j}`).toBeCloseTo(want[i][j], 5);
    }),
  );
}

describe('action parity against mjlab', () => {
  it('ramps an integrated delta target across the substeps (in-hand rotation)', async () => {
    const { got, want } = await replay('ramp');
    expectClose(got, want);
  });

  it('holds the measured pose until the episode clock passes hold_s (luwu)', async () => {
    const { got, want } = await replay('hold');
    expectClose(got, want);
    // 5 * float32(0.02) rounds below float32(0.1), so step 5 still holds in mjlab.
    const step5 = CASES.hold.steps[5].substeps[0];
    expect(step5.ctrl).toEqual(step5.qpos);
  });

  it("delays mjlab's own apply by the actuator's lag, from the reset's zeros", async () => {
    const { got, want } = await replay('delay');
    expectClose(got, want);
    expect(want[0]).toEqual([0, 0, 0]);
  });
});
