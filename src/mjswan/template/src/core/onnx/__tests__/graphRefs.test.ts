/**
 * Which traced term graphs a config refers to.
 *
 * Getting this wrong is quiet: a ref that is missed means no session, so the manager
 * warns and drops that term, leaving a policy that runs with a hole in its observation
 * vector, or a termination that never fires.
 */
import { describe, expect, it } from 'vitest';

import { actuatorGraphRefs, policyGraphRefs } from '../graphRefs';

describe('policyGraphRefs', () => {
  it('collects refs from observations, terminations, commands and events', () => {
    // Events are one of the sections because they travel inside the MDP, and a native
    // event carries no graph to collect.
    expect(
      policyGraphRefs({
        onnx: 'policy/walk.onnx',
        observations: {
          policy: [
            { name: 'joint_pos', onnx: 'obs/joint_pos.onnx' },
            { name: 'actions', native: 'prev_action' },
          ],
          critic: [{ name: 'joint_vel', onnx: 'obs/joint_vel.onnx' }],
        },
        terminations: {
          fell_over: { name: 'fell_over', onnx: 'term/fell_over.onnx' },
          time_out: { name: 'time_out', native: 'elapsed_s >= episode_length_s' },
        },
        commands: { twist: { name: 'OnnxCommand', onnx: 'command/twist.onnx' } },
        events: [
          { name: 'push_robot', onnx: 'event/push_robot.onnx' },
          { name: 'randomize_terrain', kind: 'event', mutations: [] },
        ],
      }),
    ).toEqual([
      'command/twist.onnx',
      'event/push_robot.onnx',
      'obs/joint_pos.onnx',
      'obs/joint_vel.onnx',
      'term/fell_over.onnx',
    ]);
  });

  it('ignores the policy network, whose `onnx` is an object', () => {
    // The network arrives as PolicyInput.onnx; treating it as a term graph loads it twice.
    expect(policyGraphRefs({ onnx: { path: 'walk.onnx' } })).toEqual([]);
  });

  it('de-duplicates a graph two terms share, and sorts', () => {
    // Observation fusion points several terms at one graph.
    expect(
      policyGraphRefs({
        observations: {
          policy: [
            { name: 'b', onnx: 'obs/fused.onnx' },
            { name: 'a', onnx: 'obs/fused.onnx' },
          ],
        },
        terminations: { z: { onnx: 'term/z.onnx' } },
      }),
    ).toEqual(['obs/fused.onnx', 'term/z.onnx']);
  });

  it('is empty for a config with no traced terms', () => {
    expect(policyGraphRefs({})).toEqual([]);
    expect(policyGraphRefs({ observations: {}, terminations: {}, commands: {} })).toEqual([]);
  });
});

describe('policyGraphRefs — fused groups', () => {
  it('collects a fused group graph, named by `fused` rather than `onnx`', () => {
    // The whole group is one graph, so an `onnx`-only collector delivers no bytes.
    expect(
      policyGraphRefs({
        onnx: { path: 'walk.onnx' },
        observations: {
          policy: { fused: 'obs/policy.onnx', size: 99, layout: [] },
          critic: { fused: 'obs/critic.onnx', size: 12, layout: [] },
        },
        terminations: { fell_over: { onnx: 'term/fell_over.onnx' } },
      }),
    ).toEqual(['obs/critic.onnx', 'obs/policy.onnx', 'term/fell_over.onnx']);
  });

  it('collects a fused termination graph beside its native siblings', () => {
    // Terminations fuse too, under a reserved key: one graph, several named terms.
    expect(
      policyGraphRefs({
        terminations: {
          time_out: { native: 'elapsed_s >= episode_length_s', episode_length_s: 20 },
          __fused__: {
            fused: 'term/terminations.onnx',
            lanes: [{ name: 'anchor_pos' }, { name: 'anchor_ori' }],
          },
        },
      }),
    ).toEqual(['term/terminations.onnx']);
  });
});

describe('actuatorGraphRefs', () => {
  // The engine evaluates these itself; loaded into ONNX Runtime too, each costs a session.
  const config = {
    actions: { joint_pos: { type: 'joint_position', process: { onnx: 'mdp/m/action/delta.onnx' } } },
    actuators: {
      IdealPdActuator_0: { joints: ['robot/hip'], onnx: 'mdp/m/actuator/IdealPdActuator_0.onnx' },
      DcMotorActuator_1: { joints: ['robot/knee'], onnx: 'mdp/m/actuator/DcMotorActuator_1.onnx' },
    },
  };

  it("collects the actuator laws' graphs alone", () => {
    expect([...actuatorGraphRefs(config)].sort()).toEqual([
      'mdp/m/actuator/DcMotorActuator_1.onnx',
      'mdp/m/actuator/IdealPdActuator_0.onnx',
    ]);
    expect(actuatorGraphRefs({ actions: config.actions }).size).toBe(0);
  });

  it('leaves them among the graphs a policy fetches', () => {
    expect(policyGraphRefs(config)).toEqual([
      'mdp/m/action/delta.onnx',
      'mdp/m/actuator/DcMotorActuator_1.onnx',
      'mdp/m/actuator/IdealPdActuator_0.onnx',
    ]);
  });
});
