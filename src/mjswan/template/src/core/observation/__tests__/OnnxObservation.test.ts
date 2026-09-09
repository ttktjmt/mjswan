/**
 * Traced-ONNX and natively-computed observation terms, plus the clip/scale pipeline they
 * share. The parity harness validates the graph's math; what is tested here is that a
 * term feeds exactly the slots it declares under the build's input names, clips before
 * scaling, holds its declared width, and degrades to the last good value.
 *
 * The ONNX session is a fake, so these run headless with no ORT.
 */
import { describe, expect, it, vi } from 'vitest';

import { NativeObservation } from '../NativeObservation';
import { OnnxObservation, type OnnxObservationConfig } from '../OnnxObservation';
import { applyObservationPipeline, conformToSize } from '../pipeline';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import type { PolicyRunner } from '../../policy/PolicyRunner';

/** Tensor data as a plain number array (the data union needs one narrowing). */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as ArrayLike<number>, Number);
}

/** Records feeds and returns a scripted output. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];

  constructor(private readonly values: number[]) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    this.calls.push(feeds);
    return Promise.resolve({
      value: { data: Float32Array.from(this.values), dims: [1, this.values.length] },
    });
  }
}

/**
 * Only the accessors the observation terms use. `termNames` is modelled alongside
 * `getCommand` because a `command` term binds its name against it at construction —
 * a stub that answered lookups but not "which names exist" would let a dangling
 * name through here while the real manager rejects it.
 */
function fakeRunner(overrides: Partial<{
  lastActions: Float32Array;
  olderActions: Float32Array[];
  commands: Record<string, Float32Array>;
}> = {}): PolicyRunner {
  const window = [overrides.lastActions ?? new Float32Array(0), ...(overrides.olderActions ?? [])];
  return {
    getLastActions: () => window[0],
    getActions: (age: number) => window[age] ?? new Float32Array(window[0].length),
    getContext: () => ({
      commandManager: {
        getCommand: (name: string) =>
          overrides.commands?.[name] ?? new Float32Array(0),
        termNames: () => Object.keys(overrides.commands ?? {}),
      },
    }),
  } as unknown as PolicyRunner;
}

const GRAVITY_CFG: OnnxObservationConfig = {
  name: 'projected_gravity',
  onnx: 'obs/projected_gravity.onnx',
  size: 3,
  input_slots: [
    { entity: 'robot', field: 'projected_gravity_b', input: 'robot__projected_gravity_b' },
  ],
};

describe('OnnxObservation', () => {
  it('reports the build-supplied size without running inference', () => {
    const session = new FakeSession([0, 0, -1]);
    const obs = new OnnxObservation(fakeRunner(), GRAVITY_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, -1]),
    });
    // PolicyRunner lays out the group synchronously, before any inference has run.
    expect(obs.size).toBe(3);
    expect(session.calls.length).toBe(0);
  });

  it('feeds each declared slot under its build-supplied input name', async () => {
    const session = new FakeSession([0, 0, -1]);
    const obs = new OnnxObservation(fakeRunner(), GRAVITY_CFG, {
      session,
      readSlot: slot =>
        slot.field === 'projected_gravity_b' ? new Float32Array([0.1, 0.2, -0.9]) : null,
    });
    const out = await obs.compute({} as never);
    expect(Object.keys(session.calls[0])).toEqual(['robot__projected_gravity_b']);
    expect(values(session.calls[0].robot__projected_gravity_b)).toEqual(
      [0.1, 0.2, -0.9].map(v => Math.fround(v)),
    );
    expect(Array.from(out)).toEqual([0, 0, -1]);
  });

  it('feeds a sensor slot, whose input name it cannot derive itself', async () => {
    const session = new FakeSession([1, 2, 3]);
    const obs = new OnnxObservation(
      fakeRunner(),
      {
        name: 'base_lin_vel',
        onnx: 'obs/base_lin_vel.onnx',
        size: 3,
        input_slots: [
          { sensor: 'robot/imu_lin_vel', input: 'sensor__robot_imu_lin_vel' },
        ],
      },
      { session, readSlot: () => new Float32Array([1, 2, 3]) },
    );
    await obs.compute({} as never);
    expect(Object.keys(session.calls[0])).toEqual(['sensor__robot_imu_lin_vel']);
  });

  it('applies clip then scale, matching mjlab ordering', async () => {
    // Raw 5 clipped to 2, then scaled by 10 → 20. Scaling first would give 50.
    const session = new FakeSession([5]);
    const obs = new OnnxObservation(
      fakeRunner(),
      { ...GRAVITY_CFG, size: 1, clip: [-2, 2], scale: 10 },
      { session, readSlot: () => new Float32Array([0]) },
    );
    expect(Array.from(await obs.compute({} as never))).toEqual([20]);
  });

  it('applies a per-element scale', async () => {
    const session = new FakeSession([1, 1, 1]);
    const obs = new OnnxObservation(
      fakeRunner(),
      { ...GRAVITY_CFG, scale: [2, 3, 4] },
      { session, readSlot: () => new Float32Array([0, 0, 0]) },
    );
    expect(Array.from(await obs.compute({} as never))).toEqual([2, 3, 4]);
  });

  it('serves the previous value when a slot is unreadable, never zeros', async () => {
    const session = new FakeSession([7, 8, 9]);
    let available = true;
    const obs = new OnnxObservation(fakeRunner(), GRAVITY_CFG, {
      session,
      readSlot: () => (available ? new Float32Array([0, 0, -1]) : null),
    });
    expect(Array.from(await obs.compute({} as never))).toEqual([7, 8, 9]);

    available = false;
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Stale, but a stale gravity vector is far less wrong than a zero one.
    expect(Array.from(await obs.compute({} as never))).toEqual([7, 8, 9]);
    expect(session.calls.length).toBe(1); // the graph was not run
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('conforms a graph output that disagrees with the declared size', async () => {
    const session = new FakeSession([1, 2, 3, 4, 5]);
    const obs = new OnnxObservation(fakeRunner(), GRAVITY_CFG, {
      session,
      readSlot: () => new Float32Array([0, 0, 0]),
    });
    // The group buffer is sized from `size`; an over-long output must not overflow it.
    expect(Array.from(await obs.compute({} as never))).toEqual([1, 2, 3]);
  });
});

describe('NativeObservation', () => {
  it('reads the policy previous action for prev_action', () => {
    const obs = new NativeObservation(
      fakeRunner({ lastActions: Float32Array.from([0.5, -0.5]) }),
      { name: 'actions', native: 'prev_action', size: 2 },
    );
    expect(obs.size).toBe(2);
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([0.5, -0.5]);
  });

  it('reads the named command term for command', () => {
    const obs = new NativeObservation(
      fakeRunner({ commands: { twist: Float32Array.from([1, 0, 0.25]) } }),
      { name: 'command', native: 'command', command_name: 'twist', size: 3 },
    );
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([1, 0, 0.25]);
  });

  it('serves the baked value for constant', () => {
    const obs = new NativeObservation(fakeRunner(), {
      name: 'impedance_cmd',
      native: 'constant',
      value: [0, 0, 0, 0],
    });
    // Size comes from the baked value when the build did not state one.
    expect(obs.size).toBe(4);
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([0, 0, 0, 0]);
  });

  it('resolves its width from the runtime when the build could not supply one', () => {
    // A browser-only command has no build-time width, so the first read fixes the layout.
    const obs = new NativeObservation(
      fakeRunner({ commands: { velocity: Float32Array.from([0, 0, 0]) } }),
      { name: 'velocity_cmd', native: 'command', command_name: 'velocity' },
    );
    expect(obs.size).toBe(3);
  });

  it('reads only the named action term\'s slice, not the vector head', () => {
    // `arm` occupies [0,3) and `gripper` [3,4), so `last_action("gripper")` is the tail.
    // Without `action_offset` it truncates to `arm`'s first number — right width, wrong term.
    const obs = new NativeObservation(
      fakeRunner({ lastActions: Float32Array.from([1, 2, 3, 9]) }),
      {
        name: 'gripper_action',
        native: 'prev_action',
        action_name: 'gripper',
        action_offset: 3,
        size: 1,
      },
    );
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([9]);
  });

  it('reads the whole vector when the term names no action term', () => {
    // The bare `mdp.last_action` every reference task uses: no offset, the whole vector.
    const obs = new NativeObservation(
      fakeRunner({ lastActions: Float32Array.from([1, 2, 3, 9]) }),
      { name: 'actions', native: 'prev_action', size: 4 },
    );
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([1, 2, 3, 9]);
  });

  it('refuses to build when command_name names no command term', () => {
    // Otherwise the miss arrives as a zero block in the policy's input vector, logged
    // nowhere. A slot the scene means to leave empty says so with `constant`.
    expect(
      () =>
        new NativeObservation(fakeRunner({ commands: { twist: new Float32Array(3) } }), {
          name: 'velocity_cmd',
          native: 'command',
          command_name: 'velocty', // typo
          size: 3,
        }),
    ).toThrow(/velocty.*does not define.*twist/s);
  });

  it('refuses to build a command term with no command_name at all', () => {
    expect(
      () =>
        new NativeObservation(fakeRunner(), { name: 'velocity_cmd', native: 'command', size: 3 }),
    ).toThrow(/no command_name/);
  });

  it('still zero-fills when the embedding runs no CommandManager', () => {
    // Not an error: a host that runs no commands is not evidence of a wrong name.
    const runner = { getContext: () => null } as unknown as PolicyRunner;
    const obs = new NativeObservation(runner, {
      name: 'velocity_cmd',
      native: 'command',
      command_name: 'twist',
      size: 3,
    });
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([0, 0, 0]);
  });

  it('does not hand out the runtime\'s own buffer to be mutated', () => {
    const lastActions = Float32Array.from([1, 2]);
    const obs = new NativeObservation(fakeRunner({ lastActions }), {
      name: 'actions',
      native: 'prev_action',
      size: 2,
      scale: 10,
    });
    expect(Array.from(obs.compute({} as never) as Float32Array)).toEqual([10, 20]);
    expect(Array.from(lastActions)).toEqual([1, 2]); // untouched
  });
});

describe('observation pipeline helpers', () => {
  it('conformToSize pads and truncates', () => {
    expect(Array.from(conformToSize(Float32Array.from([1, 2]), 4))).toEqual([1, 2, 0, 0]);
    expect(Array.from(conformToSize(Float32Array.from([1, 2, 3]), 2))).toEqual([1, 2]);
  });

  it('is a no-op without scale or clip', () => {
    const values = Float32Array.from([1, -5, 3]);
    expect(Array.from(applyObservationPipeline(values, {}))).toEqual([1, -5, 3]);
  });
});

/**
 * `age` reaches back into mjlab's action window (`prev_action`, `prev_prev_action`),
 * which a task observing action history reads instead of the newest entry.
 */
describe('NativeObservation — prev_action age', () => {
  const window = {
    lastActions: Float32Array.from([1, 2]),
    olderActions: [Float32Array.from([3, 4]), Float32Array.from([5, 6])],
  };

  it('defaults to the newest action, as last_action does', () => {
    const term = new NativeObservation(fakeRunner(window), {
      name: 'a',
      native: 'prev_action',
      size: 2,
    });
    expect(Array.from(term.compute({} as never))).toEqual([1, 2]);
  });

  it('reads one and two steps back', () => {
    for (const [age, expected] of [
      [1, [3, 4]],
      [2, [5, 6]],
    ] as const) {
      const term = new NativeObservation(fakeRunner(window), {
        name: 'a',
        native: 'prev_action',
        size: 2,
        age,
      });
      expect(Array.from(term.compute({} as never))).toEqual([...expected]);
    }
  });

  it('reads zeros past the window, as mjlab does before that many steps', () => {
    const term = new NativeObservation(fakeRunner(window), {
      name: 'a',
      native: 'prev_action',
      size: 2,
      age: 3,
    });
    expect(Array.from(term.compute({} as never))).toEqual([0, 0]);
  });
});
