/**
 * Actuator parity: mjlab actuators whose torque law mjlab computes in Python, against
 * mjlab, substep by substep. The Builder's own entries and graphs run in the real
 * `ActuatorSet` behind the real `applyAction`, the term's targets routed into the laws
 * as the runtime routes them; mjlab's `ctrl` at each substep is the answer.
 *
 * The cases: an ideal PD clipping at its effort limit, a DC motor behind a two-step
 * command delay, and a velocity term over two laws, one with Unitree-style friction,
 * each with an episode reset partway through.
 *
 * **States are replayed, not co-simulated**, as in `actionParity.test.ts`: each
 * substep's joint positions and velocities come from the fixture.
 *
 * Regenerate: `MUJOCO_GL=disable .venv/bin/python tests/dump_actuator_fixture.py`
 */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import fixture from './fixtures/actuator/actuator.json';
import { ActionManager } from '../../action/ActionManager';
import { applyAction, substepOptions, type ResolvedActionTerm, type SubstepConfig } from '../../action/applyAction';
import type { ActuatorConfigEntry } from '../../policy/types';
import { SeededRng } from '../../rng';
import { ActuatorSet, targetKind } from '../ActuatorSet';
import { ACTUATOR_INPUTS, ACTUATOR_OUTPUT, ActuatorGraph, SUPPORTED_OPS } from '../evaluator';

const FIXTURES = join(__dirname, 'fixtures/actuator');

interface Substep {
  /** The state the laws read. */
  qpos: number[];
  qvel: number[];
  /** What mjlab wrote. */
  ctrl: number[];
}

interface Step {
  /** Whether the episode restarted before this step. */
  reset: boolean;
  /** The targets the term set, before any delay. */
  processed: number[];
  substeps: Substep[];
}

interface Case {
  entry: SubstepConfig & { type: string };
  joints: string[];
  /** Per joint: its `qpos` and `qvel` addresses, and its actuator's `ctrl`. */
  addresses: Record<string, [number, number, number]>;
  actuators: Record<string, ActuatorConfigEntry>;
  decimation: number;
  steps: Step[];
}

const FIXTURE = fixture as unknown as {
  supported_ops: string[];
  inputs: string[];
  output: string;
  cases: Record<string, Case>;
};

type MjData = Parameters<typeof applyAction>[0];

function addressesOf(c: Case, joints: string[], column: 0 | 1 | 2): number[] {
  return joints.map((joint) => c.addresses[joint][column]);
}

/** The runtime's `buildActuators`, with the model's addresses from the fixture. */
function actuatorsFor(name: string, c: Case): ActuatorSet {
  return new ActuatorSet(
    Object.entries(c.actuators).map(([key, entry]) => ({
      name: key,
      graph: ActuatorGraph.fromOnnx(readFileSync(join(FIXTURES, name, entry.onnx)), entry.joints.length, key),
      qposAdr: addressesOf(c, entry.joints, 0),
      qvelAdr: addressesOf(c, entry.joints, 1),
      ctrlAdr: addressesOf(c, entry.joints, 2),
    })),
  );
}

/** The arm's term on its motors, its targets routed into the laws. */
function termFor(c: Case, actuators: ActuatorSet, rng: SeededRng): ResolvedActionTerm {
  const n = c.joints.length;
  const qposAdr = addressesOf(c, c.joints, 0);
  const kind = targetKind(c.entry.type);
  return {
    controlType: c.entry.type,
    ctrlAdr: addressesOf(c, c.joints, 2),
    qposAdr,
    qvelAdr: addressesOf(c, c.joints, 1),
    actionIndices: Array.from({ length: n }, (_, i) => i),
    actionScale: new Float32Array(n).fill(1),
    actionOffset: new Float32Array(n),
    defaultJointPos: new Float32Array(n),
    encoderBias: new Float32Array(n),
    positionActuator: new Array(n).fill(false),
    kp: new Float32Array(n),
    kd: new Float32Array(n),
    muscleMode: 'sigmoid',
    muscleCtrlLo: new Float32Array(0),
    muscleCtrlHi: new Float32Array(0),
    clipLo: new Float32Array(n).fill(-Infinity),
    clipHi: new Float32Array(n).fill(Infinity),
    processed: new Float32Array(n),
    active: false,
    ...substepOptions(c.entry, c.joints, 'joint', rng),
    actuatorTargets: qposAdr.map((adr) => actuators.targetOf(adr, kind)),
  };
}

/** Each substep's `ctrl` as the browser writes it, replaying the fixture's states. */
function replay(name: string): { got: number[][]; want: number[][]; c: Case } {
  const c = FIXTURE.cases[name];
  const actuators = actuatorsFor(name, c);
  const manager = new ActionManager([{ name: 'joint', term: termFor(c, actuators, new SeededRng(0)) }], actuators);
  const [term] = manager.terms;
  const first = c.steps[0].substeps[0];
  const data = {
    qpos: new Float64Array(first.qpos.length),
    qvel: new Float64Array(first.qvel.length),
    ctrl: new Float64Array(first.ctrl.length),
  } as unknown as MjData;

  const got: number[][] = [];
  const want: number[][] = [];
  for (const step of c.steps) {
    if (step.reset) manager.resetActuators();
    term.processed.set(step.processed);
    term.active = true;
    step.substeps.forEach((substep, k) => {
      data.qpos.set(substep.qpos);
      data.qvel.set(substep.qvel);
      applyAction(data, manager.terms, k, c.decimation, manager.actuators);
      got.push(Array.from(data.ctrl));
      want.push(substep.ctrl);
    });
  }
  return { got, want, c };
}

function expectClose(got: number[][], want: number[][]): void {
  expect(got).toHaveLength(want.length);
  got.forEach((row, i) =>
    row.forEach((value, j) => {
      expect(value, `substep ${i}, actuator ${j}`).toBeCloseTo(want[i][j], 5);
    }),
  );
}

describe('actuator parity against mjlab', () => {
  it('runs the operators, inputs and output the tracer allows', () => {
    expect([...SUPPORTED_OPS].sort()).toEqual(FIXTURE.supported_ops);
    expect([...ACTUATOR_INPUTS]).toEqual(FIXTURE.inputs);
    expect(ACTUATOR_OUTPUT).toBe(FIXTURE.output);
  });

  it('runs an ideal PD, clipped at its effort limit', () => {
    const { got, want } = replay('ideal_pd');
    expectClose(got, want);
    const torques = want.flat();
    expect(torques).toContain(6);
    expect(torques.filter((t) => Math.abs(t) < 6).length).toBeGreaterThan(torques.length / 4);
  });

  it("runs a DC motor's torque-speed curve behind its command delay, from the reset's zeros", () => {
    const { got, want, c } = replay('dc_motor');
    expectClose(got, want);
    // For its first two substeps the delayed law drives toward the reset's zero targets.
    const [start] = c.steps[0].substeps;
    expect(want[0]).toEqual(start.qpos.map((q) => Math.fround(Math.min(Math.max(-40 * q, -8), 8))));
    // Past the velocity limit, where the curve, not the effort limit, bounds the torque.
    expect(c.steps.some((s) => s.substeps.some((x) => x.qvel.some((v) => Math.abs(v) > 3)))).toBe(true);
  });

  it('routes a velocity term into two laws on one arm, one with friction', () => {
    const { got, want, c } = replay('friction_velocity');
    expectClose(got, want);
    expect(Object.keys(c.actuators)).toEqual(['FrictionPdActuator_0', 'IdealPdActuator_1']);
  });
});
