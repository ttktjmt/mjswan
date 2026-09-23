import { describe, expect, it } from 'vitest';
import type { MainModule, MjData, MjModel } from 'mujoco';

import { PolicyStateBuilder } from '../PolicyStateBuilder';

/**
 * A fake is enough: the code under test only walks `names` by address and follows
 * `actuator_trnid` to a joint.
 */
function makeModel(
  joints: { joint: string; actuator: string }[]
): { mujoco: MainModule; mjModel: MjModel; mjData: MjData } {
  const encoder = new TextEncoder();
  const blob: number[] = [];
  const address = (text: string): number => {
    const at = blob.length;
    blob.push(...encoder.encode(text), 0);
    return at;
  };

  const name_jntadr = joints.map((j) => address(j.joint));
  const name_actuatoradr = joints.map((j) => address(j.actuator));

  const mjModel = {
    njnt: joints.length,
    nu: joints.length,
    names: new Uint8Array(blob).buffer,
    name_jntadr: Int32Array.from(name_jntadr),
    name_actuatoradr: Int32Array.from(name_actuatoradr),
    jnt_qposadr: Int32Array.from(joints.map((_, i) => i)),
    jnt_dofadr: Int32Array.from(joints.map((_, i) => i)),
    // One actuator per joint, in the same order, each a joint transmission.
    actuator_trntype: Int32Array.from(joints.map(() => 0)),
    actuator_trnid: Int32Array.from(joints.flatMap((_, i) => [i, -1])),
  } as unknown as MjModel;

  const mjData = {
    qpos: new Float32Array(joints.length),
    qvel: new Float32Array(joints.length),
    xquat: new Float32Array(8),
    cvel: new Float32Array(12),
  } as unknown as MjData;

  const mujoco = { mjtTrn: { mjTRN_JOINT: { value: 0 } } } as unknown as MainModule;
  return { mujoco, mjModel, mjData };
}

/**
 * A pattern here is a joint pattern, whatever `actuator_names` suggests: mjlab's
 * `JointPositionAction` resolves it through `Entity.find_joints_by_actuator_names`,
 * which narrows to the actuated joints and matches the patterns against their *joint*
 * names.
 */
describe('getControlMappingFor', () => {
  it('matches the joint name, as mjlab does', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'lift', actuator: 'thrust' },
      { joint: 'slide', actuator: 'push' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['lift', 'slide']);

    expect(builder.getControlMappingFor(['lift'], ['lift', 'slide'])).toMatchObject({
      actionIndices: [0],
      ctrlAdr: [0],
    });
  });

  it('does not match the actuator name, which mjlab never looks at', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'lift', actuator: 'thrust' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['lift']);

    expect(builder.getControlMappingFor(['thrust'], ['lift'])).toBeNull();
  });

  it('keeps the order it is given, which is the order the actions come out', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'hip', actuator: 'hip_act' },
      { joint: 'knee', actuator: 'knee_act' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['knee', 'hip']);

    // Action 0 drives `knee`, which is the model's actuator 1.
    expect(builder.getControlMappingFor(['.*'], ['knee', 'hip'])).toMatchObject({
      actionIndices: [0, 1],
      ctrlAdr: [1, 0],
    });
  });

  it('selects a subset of the policy, not of the model', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'hip', actuator: 'hip_act' },
      { joint: 'knee', actuator: 'knee_act' },
      { joint: 'ankle', actuator: 'ankle_act' },
    ]);
    // A policy that drives two of the model's three actuators. `.*` must mean "every
    // joint this policy drives", or a four-action policy on a twelve-actuator model
    // would be asked for twelve.
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['hip', 'knee']);

    expect(builder.getControlMappingFor(['.*'], ['hip', 'knee'])).toMatchObject({
      actionIndices: [0, 1],
    });
  });

  it('is null when nothing matches', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'hip', actuator: 'hip_act' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['hip']);

    expect(builder.getControlMappingFor(['elbow'], ['hip'])).toBeNull();
  });

  it('has nothing to match when the policy names no joints', () => {
    // Empty `policy_joint_names` leaves every pattern unmatched, so the runtime writes
    // no control and the cart never moves. Python fills the names from the task's own
    // action terms because a cartpole checkpoint carries no mjlab metadata.
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'cartpole/slider', actuator: 'cartpole/slide_act' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, []);

    expect(builder.getControlMappingFor(['cartpole/slider'], [])).toBeNull();
  });

  it('anchors each pattern, so a partial name does not match', () => {
    const { mujoco, mjModel, mjData } = makeModel([
      { joint: 'hip_pitch', actuator: 'hip_pitch_act' },
    ]);
    const builder = new PolicyStateBuilder(mujoco, mjModel, mjData, ['hip_pitch']);

    expect(builder.getControlMappingFor(['hip_'], ['hip_pitch'])).toBeNull();
  });
});
