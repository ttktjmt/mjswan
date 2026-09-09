/**
 * `FusedObservation`: one graph per observation group (ADR 0005 §4, brief §4b).
 *
 * The graph's math is validated Python-side against
 * `ObservationManager.compute_group`. What matters here is the native half — that
 * declared slots and native inputs both reach the feeds under the names the build
 * chose, that the output is served as the group vector, and that an unreadable
 * slot holds the last vector rather than handing the policy a partly-zero input.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  FusedObservation,
  isFusedObservationConfig,
  type FusedObservationConfig,
} from '../FusedObservation';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import type { PolicyRunner } from '../../policy/PolicyRunner';

/** Narrow a feed's data union (Float32/BigInt64/Uint8) to plain numbers. */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as Float32Array);
}

class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];

  constructor(private readonly respond: (call: number) => Float32Array) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const index = this.calls.length;
    this.calls.push(feeds);
    const data = this.respond(index);
    return Promise.resolve({ obs: { data, dims: [1, data.length] } });
  }
}

const CFG: FusedObservationConfig = {
  name: 'policy',
  fused: 'obs/policy.onnx',
  size: 6,
  input_slots: [
    { entity: 'robot', field: 'joint_pos', input: 'robot__joint_pos', shape: [1, 2] },
    { entity: 'robot', field: 'site_pos_w', input: 'robot__site_pos_w', shape: [1, 2, 3] },
  ],
  native_inputs: [
    { name: 'actions', native: 'prev_action', input: 'native__actions', size: 2 },
    {
      name: 'command',
      native: 'command',
      input: 'native__command',
      size: 3,
      command_name: 'twist',
    },
  ],
  layout: [
    { name: 'joint_pos', size: 2 },
    { name: 'actions', size: 2 },
    { name: 'command', size: 2 },
  ],
};

function fakeRunner(): PolicyRunner {
  return {
    getLastActions: () => new Float32Array([0.5, -0.5]),
    getActions: () => new Float32Array([0.5, -0.5]),
    getContext: () => ({
      commandManager: {
        getCommand: () => new Float32Array([1, 2, 3]),
        // The `command` input binds at construction, so the stub must answer which names exist.
        termNames: () => ['twist'],
      },
    }),
  } as unknown as PolicyRunner;
}

describe('FusedObservation', () => {
  it('feeds declared slots and native inputs, and returns the group vector', async () => {
    const session = new FakeSession(() => Float32Array.from([1, 2, 3, 4, 5, 6]));
    const term = new FusedObservation(fakeRunner(), CFG, {
      session,
      readSlot: slot =>
        slot.field === 'joint_pos'
          ? new Float32Array([0.1, 0.2])
          : new Float32Array([1, 2, 3, 4, 5, 6]),
    });

    const out = await term.compute({} as never);
    expect(Array.from(out)).toEqual([1, 2, 3, 4, 5, 6]);
    // One run for the whole group — the point of fusing.
    expect(session.calls.length).toBe(1);
    expect(Object.keys(session.calls[0]).sort()).toEqual([
      'native__actions',
      'native__command',
      'robot__joint_pos',
      'robot__site_pos_w',
    ]);
    // The traced rank travels with the slot; site_pos_w is (batch, sites, 3).
    expect(session.calls[0]['robot__site_pos_w'].dims).toEqual([1, 2, 3]);
    expect(values(session.calls[0]['native__actions'])).toEqual([0.5, -0.5]);
    expect(values(session.calls[0]['native__command'])).toEqual([1, 2, 3]);
  });

  it('conforms a native input to the width the graph declared', async () => {
    // A browser-side command whose width drifted would make ORT reject the group.
    const session = new FakeSession(() => new Float32Array(6));
    const runner = {
      getLastActions: () => new Float32Array([1, 2, 3, 4]), // 4, graph wants 2
      getActions: () => new Float32Array([1, 2, 3, 4]),
      getContext: () => null,
    } as unknown as PolicyRunner;
    const term = new FusedObservation(runner, CFG, {
      session,
      readSlot: () => new Float32Array([0, 0]),
    });
    await term.compute({} as never);
    expect(session.calls[0]['native__actions'].data.length).toBe(2);
    // No command manager at all: the declared width, zero-filled, not a crash.
    expect(session.calls[0]['native__command'].data.length).toBe(3);
  });

  it('holds the previous vector when a slot is unreadable', async () => {
    const session = new FakeSession(() => Float32Array.from([1, 2, 3, 4, 5, 6]));
    let available = true;
    const term = new FusedObservation(fakeRunner(), CFG, {
      session,
      readSlot: () => (available ? new Float32Array([0.1, 0.2]) : null),
    });
    await term.compute({} as never);

    available = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const held = await term.compute({} as never);
    // Not run on missing state: the last good vector stands rather than a zeroed slice.
    expect(session.calls.length).toBe(1);
    expect(Array.from(held)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('feeds a named action term its own slice', async () => {
    // The group is fed the whole action vector and must narrow it, or the graph gets the
    // wrong term at the right width. Here `getLastActions` is 4 wide and `gripper` last.
    const session = new FakeSession(() => new Float32Array(6));
    const runner = {
      getLastActions: () => Float32Array.from([1, 2, 3, 9]),
      getActions: () => Float32Array.from([1, 2, 3, 9]),
      getContext: () => null,
    } as unknown as PolicyRunner;
    const config: FusedObservationConfig = {
      ...CFG,
      native_inputs: [
        {
          name: 'gripper_action',
          native: 'prev_action',
          input: 'native__gripper',
          size: 1,
          action_name: 'gripper',
          action_offset: 3,
        },
      ],
    };
    const term = new FusedObservation(runner, config, {
      session,
      readSlot: () => new Float32Array([0, 0]),
    });
    await term.compute({} as never);
    expect(values(session.calls[0]['native__gripper'])).toEqual([9]);
  });

  it('refuses to build when a native command input names no command term', () => {
    // A fused group's output *is* the policy's input vector, so an unbound command input
    // is a zero block at a known offset — as silent as the missing graph this throws for.
    const session = new FakeSession(() => new Float32Array(6));
    const config: FusedObservationConfig = {
      ...CFG,
      native_inputs: [
        { name: 'actions', native: 'prev_action', input: 'native__actions', size: 2 },
        {
          name: 'command',
          native: 'command',
          input: 'native__command',
          size: 3,
          command_name: 'heading', // the stub only holds `twist`
        },
      ],
    };
    expect(
      () =>
        new FusedObservation(fakeRunner(), config, {
          session,
          readSlot: () => new Float32Array([0, 0]),
        }),
    ).toThrow(/policy\.command.*heading.*twist/s);
  });

  it('conforms a wrong-width graph output to the declared size', async () => {
    const session = new FakeSession(() => Float32Array.from([1, 2, 3]));
    const term = new FusedObservation(fakeRunner(), CFG, {
      session,
      readSlot: () => new Float32Array([0.1, 0.2]),
    });
    expect((await term.compute({} as never)).length).toBe(6);
  });
});

describe('isFusedObservationConfig', () => {
  it('recognizes a fused group and rejects the per-term shapes', () => {
    expect(isFusedObservationConfig({ fused: 'obs/policy.onnx' })).toBe(true);
    expect(isFusedObservationConfig([{ name: 'joint_pos', onnx: 'obs/j.onnx' }])).toBe(false);
    expect(isFusedObservationConfig({ name: 'joint_pos', onnx: 'obs/j.onnx' })).toBe(false);
    expect(isFusedObservationConfig(undefined)).toBe(false);
  });
});
