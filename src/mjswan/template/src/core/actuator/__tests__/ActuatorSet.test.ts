/**
 * Where a term's targets go and where a law reads and writes: a wrong address here
 * drives one joint with another's torque, with nothing to show for it but the motion.
 */
import { describe, expect, it } from 'vitest';

import { ActionManager } from '../../action/ActionManager';
import { applyAction, type ResolvedActionTerm } from '../../action/applyAction';
import { ActuatorSet, targetKind, type ResolvedActuator } from '../ActuatorSet';
import { ActuatorGraph } from '../evaluator';
import { onnxModel } from './onnxWriter';

type MjData = Parameters<typeof applyAction>[0];

/** `kp * (pos_target - pos) + kd * (vel_target - vel) + effort_target`. */
function pd(width: number, kp: number, kd: number): ActuatorGraph {
  const bytes = onnxModel({
    initializers: [
      { name: 'kp', dims: [], values: [kp] },
      { name: 'kd', dims: [], values: [kd] },
    ],
    nodes: [
      { op: 'Sub', inputs: ['pos_target', 'pos'], outputs: ['e'] },
      { op: 'Mul', inputs: ['kp', 'e'], outputs: ['pe'] },
      { op: 'Sub', inputs: ['vel_target', 'vel'], outputs: ['ve'] },
      { op: 'Mul', inputs: ['kd', 've'], outputs: ['de'] },
      { op: 'Add', inputs: ['pe', 'de'], outputs: ['s'] },
      { op: 'Add', inputs: ['s', 'effort_target'], outputs: ['torque'] },
    ],
  });
  return ActuatorGraph.fromOnnx(bytes, width, 'pd');
}

function law(over: Partial<ResolvedActuator> & { qposAdr: number[] }): ResolvedActuator {
  const width = over.qposAdr.length;
  return {
    name: 'pd',
    graph: pd(width, 10, 1),
    qvelAdr: over.qposAdr,
    ctrlAdr: over.qposAdr,
    ...over,
  };
}

function fakeData(nu: number, qpos: number[], qvel: number[]): MjData {
  return {
    ctrl: new Float64Array(nu),
    qpos: Float64Array.from(qpos),
    qvel: Float64Array.from(qvel),
  } as unknown as MjData;
}

describe('targetKind', () => {
  it('is the target mjlab sets for a term of each kind', () => {
    expect(targetKind('joint_velocity')).toBe('vel_target');
    expect(targetKind('torque')).toBe('effort_target');
    for (const kind of ['joint_position', 'joint_position_relative', 'joint_position_reference', 'joint_position_command']) {
      expect(targetKind(kind)).toBe('pos_target');
    }
  });
});

describe('ActuatorSet', () => {
  it("finds a joint's target in its law's inputs, at its column", () => {
    const a = law({ qposAdr: [7, 8] });
    const b = law({ qposAdr: [9] });
    const set = new ActuatorSet([a, b]);
    expect(set.targetOf(8, 'pos_target')).toEqual({ values: a.graph.inputs.pos_target, column: 1 });
    expect(set.targetOf(9, 'vel_target')).toEqual({ values: b.graph.inputs.vel_target, column: 0 });
    expect(set.targetOf(7, 'effort_target')?.values).toBe(a.graph.inputs.effort_target);
    expect(set.targetOf(0, 'pos_target')).toBeNull();
  });

  it('reads each joint at its own qpos and qvel address and writes its own ctrl', () => {
    // A floating base's: qpos 7 + i, qvel 6 + i, the actuators in another order.
    const actuator = law({ qposAdr: [7, 8], qvelAdr: [6, 7], ctrlAdr: [1, 0] });
    const set = new ActuatorSet([actuator]);
    actuator.graph.inputs.pos_target.set([1, 2]);
    const qpos = [9, 9, 9, 9, 9, 9, 9, 0.5, 1.5];
    const qvel = [9, 9, 9, 9, 9, 9, 0.25, -1];
    const ctrl = [0, 0, 7];
    set.apply(qpos, qvel, ctrl);
    expect(ctrl).toEqual([10 * (2 - 1.5) + 1 * (0 + 1), 10 * (1 - 0.5) + 1 * (0 - 0.25), 7]);
  });

  it('skips a joint with no actuator', () => {
    const set = new ActuatorSet([law({ qposAdr: [0, 1], ctrlAdr: [-1, 0] })]);
    const ctrl = [5, 5];
    set.apply([0, 0], [0, 0], ctrl);
    expect(ctrl).toEqual([0, 5]);
  });

  it('zeroes every target at a reset, so each law holds its joints toward zero', () => {
    const actuator = law({ qposAdr: [0] });
    const set = new ActuatorSet([actuator]);
    const { pos_target, vel_target, effort_target } = actuator.graph.inputs;
    pos_target[0] = 1;
    vel_target[0] = 2;
    effort_target[0] = 3;
    set.reset();
    expect([pos_target[0], vel_target[0], effort_target[0]]).toEqual([0, 0, 0]);
    const ctrl = [0];
    set.apply([0.5], [0.25], ctrl);
    expect(ctrl).toEqual([-10 * 0.5 - 0.25]);
  });
});

describe('applyAction with actuator laws', () => {
  function term(over: Partial<ResolvedActionTerm> = {}): ResolvedActionTerm {
    const n = over.qposAdr?.length ?? 2;
    return {
      controlType: 'joint_position',
      ctrlAdr: [0, 1],
      qposAdr: [0, 1],
      qvelAdr: [0, 1],
      actionIndices: [0, 1],
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
      active: true,
      rampFrom: null,
      holdS: null,
      holding: false,
      delays: [],
      ...over,
    };
  }

  function routed(set: ActuatorSet, t: ResolvedActionTerm): ResolvedActionTerm {
    t.actuatorTargets = t.qposAdr.map((adr) => set.targetOf(adr, targetKind(t.controlType)));
    return t;
  }

  it("hands a law's joints their targets and writes the others' ctrl as before", () => {
    // Joint 0 on a law; joint 1 on a position actuator, whose ctrl is the target itself.
    const set = new ActuatorSet([law({ qposAdr: [0] })]);
    const t = routed(set, term({ positionActuator: [false, true], processed: Float32Array.from([0.5, 0.75]) }));
    const data = fakeData(2, [0.25, 0], [1, 0]);
    applyAction(data, [t], 0, 1, set);
    expect(Array.from(data.ctrl)).toEqual([10 * (0.5 - 0.25) - 1, 0.75]);
  });

  it('gives a velocity term the velocity target and a torque term the effort', () => {
    const set = new ActuatorSet([law({ qposAdr: [0, 1] })]);
    const velocity = routed(set, term({ controlType: 'joint_velocity', processed: Float32Array.from([2, 0]) }));
    const effort = routed(set, term({ controlType: 'torque', processed: Float32Array.from([0, 3]) }));
    const data = fakeData(2, [0, 0], [0, 0]);
    applyAction(data, [velocity, effort], 0, 1, set);
    expect(Array.from(data.ctrl)).toEqual([1 * 2, 3]);
  });

  it("keeps a term's last targets while it is inactive, until the episode's reset", () => {
    const set = new ActuatorSet([law({ qposAdr: [0, 1] })]);
    const t = routed(set, term({ processed: Float32Array.from([1, -1]) }));
    const manager = new ActionManager([{ name: 'joint', term: t }], set);
    const data = fakeData(2, [0, 0], [0, 0]);
    applyAction(data, manager.terms, 0, 1, set);
    expect(Array.from(data.ctrl)).toEqual([10, -10]);

    t.active = false;
    applyAction(data, manager.terms, 0, 1, set);
    expect(Array.from(data.ctrl)).toEqual([10, -10]);

    manager.resetActuators();
    applyAction(data, manager.terms, 0, 1, set);
    expect(Array.from(data.ctrl)).toEqual([0, 0]);
  });

  it('leaves ctrl alone without a law', () => {
    const t = term({ positionActuator: [true, true], processed: Float32Array.from([0.5, 0.75]) });
    const data = fakeData(2, [0, 0], [0, 0]);
    applyAction(data, [t], 0, 1, null);
    expect(Array.from(data.ctrl)).toEqual([0.5, 0.75]);
  });
});
