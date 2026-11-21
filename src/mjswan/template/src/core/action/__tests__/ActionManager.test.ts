/**
 * A traced action term's native half: its state carried across control steps,
 * its reset graph, the slice of the action it is fed, and what an observation reads off
 * it. The graph's math is checked against mjlab in Python and in `actionParity.test.ts`;
 * the session here is a fake, so these run with no ORT.
 */
import { describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import type { OnnxSession, OnnxTensorLike } from '../../onnx/session';
import { applyAction, type ResolvedActionTerm } from '../applyAction';
import { ActionManager } from '../ActionManager';
import { OnnxActionProcess, type OnnxActionProcessConfig } from '../OnnxActionProcess';

type Outputs = Record<string, OnnxTensorLike>;

/** Records feeds as it is called and answers with `respond`. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, number[]>> = [];

  constructor(
    private readonly respond: (feeds: Record<string, OnnxTensorLike>) => Outputs | Promise<Outputs>,
    readonly inputNames?: string[],
  ) {}

  async run(feeds: Record<string, OnnxTensorLike>): Promise<Outputs> {
    this.calls.push(
      Object.fromEntries(
        Object.entries(feeds).map(([k, v]) => [k, Array.from(v.data as ArrayLike<number>, Number)]),
      ),
    );
    return this.respond(feeds);
  }
}

/** `n` microtasks: an ORT run that takes a while. */
async function ticks(n: number): Promise<void> {
  for (let i = 0; i < n; i++) await Promise.resolve();
}

function tensor(values: number[]): OnnxTensorLike {
  return { data: Float32Array.from(values), dims: [1, values.length] };
}

function read(t: OnnxTensorLike): number[] {
  return Array.from(t.data as ArrayLike<number>, Number);
}

/** inhand's delta term, as its graph computes it: target += clip(raw) / 24. */
const DELTA_CFG: OnnxActionProcessConfig = {
  onnx: 'mdp/a/action/joint_pos.onnx',
  state_fields: [{ name: '_target', shape: [1, 2], dtype: 'float32', init: [0.1, 0.2] }],
  input_slots: [{ entity: 'robot', field: 'joint_pos', input: 'robot__joint_pos', shape: [1, 2] }],
};

function deltaSession(): FakeSession {
  return new FakeSession((feeds) => {
    const prev = read(feeds.prev__target);
    const raw = read(feeds.raw);
    const next = prev.map((p, i) => p + Math.max(-1, Math.min(1, raw[i])) / 24);
    return {
      next__target: tensor(next),
      processed: tensor(next),
      processed_from: tensor(prev),
    };
  });
}

/** A reset graph that puts the target at the measured joint position. */
function resetSession(): FakeSession {
  return new FakeSession((feeds) => ({ next__target: tensor(read(feeds.robot__joint_pos)) }));
}

function tracedTerm(over: Partial<ResolvedActionTerm> = {}): ResolvedActionTerm {
  return {
    controlType: 'joint_position',
    ctrlAdr: [0, 1],
    qposAdr: [0, 1],
    qvelAdr: [0, 1],
    actionIndices: [2, 0],
    actionScale: new Float32Array(2),
    actionOffset: new Float32Array(2),
    defaultJointPos: new Float32Array(2),
    encoderBias: new Float32Array(2),
    positionActuator: [true, true],
    kp: new Float32Array(2),
    kd: new Float32Array(2),
    muscleMode: 'sigmoid',
    muscleCtrlLo: new Float32Array(0),
    muscleCtrlHi: new Float32Array(0),
    clipLo: new Float32Array(2).fill(-Infinity),
    clipHi: new Float32Array(2).fill(Infinity),
    processed: new Float32Array(2),
    active: false,
    rampFrom: new Float32Array(2),
    holdS: null,
    holding: false,
    delays: [],
    ...over,
  };
}

/** A native term on the same two joints, reading action 1 for both. */
function nativeTerm(): ResolvedActionTerm {
  return tracedTerm({ actionIndices: [1, 1], actionScale: Float32Array.from([1, 1]), rampFrom: null });
}

const JOINT_POS = [0.4, -0.3];
const readSlot = () => Float32Array.from(JOINT_POS);
const CONTEXT = { episodeLength: 0, stepDt: 0.05 };

function rounded(values: ArrayLike<number>): number[] {
  return Array.from(values, (v) => Number(v.toFixed(6)));
}

describe('OnnxActionProcess', () => {
  it('feeds its slice of the action and its state, and carries the next state over', async () => {
    const session = deltaSession();
    const processor = new OnnxActionProcess('joint_pos', DELTA_CFG, null, {
      process: session,
      rng: new SeededRng(1),
      readSlot,
    });
    const term = tracedTerm();
    const manager = new ActionManager([{ name: 'joint_pos', term, traced: processor }]);

    await manager.process(Float32Array.from([0.5, 9, 1.0]), CONTEXT);
    // `actionIndices` [2, 0] gather the term's slice in its own target order.
    expect(session.calls[0].raw).toEqual([1.0, 0.5]);
    expect(session.calls[0].prev__target.map((v) => Number(v.toFixed(6)))).toEqual([0.1, 0.2]);
    expect(term.active).toBe(true);
    expect(Array.from(term.processed).map((v) => Number(v.toFixed(6)))).toEqual([
      Number((0.1 + 1 / 24).toFixed(6)),
      Number((0.2 + 0.5 / 24).toFixed(6)),
    ]);
    // The ramp starts from where the term stood before this step.
    expect(Array.from(term.rampFrom!).map((v) => Number(v.toFixed(6)))).toEqual([0.1, 0.2]);

    await manager.process(Float32Array.from([0, 0, 0]), CONTEXT);
    expect(session.calls[1].prev__target).toEqual(Array.from(term.processed));
  });

  it('runs its reset graph at a reset, and only that one', async () => {
    const process = deltaSession();
    const reset = resetSession();
    const processor = new OnnxActionProcess(
      'joint_pos',
      DELTA_CFG,
      { onnx: 'mdp/a/action/joint_pos_reset.onnx', input_slots: DELTA_CFG.input_slots },
      { process, reset, rng: new SeededRng(1), readSlot },
    );
    const manager = new ActionManager([{ name: 'joint_pos', term: tracedTerm(), traced: processor }]);

    await manager.reset();
    expect(process.calls).toHaveLength(0);
    expect(reset.calls).toHaveLength(1);
    // An observation of the term's internals sees the reset's state.
    const target = Array.from(manager.getStateField('joint_pos', '_target') ?? []);
    expect(target.map((v) => Number(v.toFixed(6)))).toEqual(JOINT_POS);
  });

  it('keeps applying the previous targets when a slot cannot be read', async () => {
    const session = deltaSession();
    const processor = new OnnxActionProcess('joint_pos', DELTA_CFG, null, {
      process: session,
      rng: new SeededRng(1),
      readSlot: () => null,
    });
    const term = tracedTerm({ processed: Float32Array.from([0.7, 0.8]), active: true });
    const manager = new ActionManager([{ name: 'joint_pos', term, traced: processor }]);
    const warn = console.warn;
    console.warn = () => {};
    try {
      await manager.process(Float32Array.from([1, 1, 1]), CONTEXT);
    } finally {
      console.warn = warn;
    }
    expect(session.calls).toHaveLength(0);
    // Not driven to zero: `apply` skips an inactive term after zeroing `ctrl`.
    const data = { ctrl: new Float64Array(2), qpos: new Float64Array(2), qvel: new Float64Array(2) };
    applyAction(data as unknown as Parameters<typeof applyAction>[0], manager.terms);
    expect(rounded(data.ctrl)).toEqual([0.7, 0.8]);
  });

  it('runs a reset started mid-process on the state the process leaves', async () => {
    // The runtime does not await a UI reset, so one can start while a step's process is
    // still in flight; the slower process must not land on top of it.
    const process = new FakeSession(async (feeds) => {
      const next = tensor(read(feeds.prev__target).map((v) => v + 1));
      await ticks(5);
      return { next__target: next, processed: next };
    });
    const reset = new FakeSession(async (feeds) => {
      const next = tensor(read(feeds.prev__target).map((v) => v + 10));
      await ticks(1);
      return { next__target: next };
    });
    const processor = new OnnxActionProcess(
      'joint_pos',
      { onnx: 'p.onnx', state_fields: [{ name: '_target', shape: [1, 2], dtype: 'float32' }] },
      { onnx: 'r.onnx' },
      { process, reset, rng: new SeededRng(1), readSlot },
    );
    const stepping = processor.process(new Float32Array(2), new Float32Array(2), null);
    const resetting = processor.reset();
    await Promise.all([stepping, resetting]);
    expect(reset.calls[0].prev__target).toEqual([1, 1]);
    expect(Array.from(processor.getStateField('_target')!)).toEqual([11, 11]);
  });

  it('draws `rand` from the seeded generator when the graph takes one', async () => {
    const session = new FakeSession(() => ({ processed: tensor([0, 0]) }));
    const processor = new OnnxActionProcess(
      'noisy',
      { onnx: 'x.onnx', state_fields: [], rand_dim: 3, rand_ranges: [[0, 1], [0, 1], [0, 1]] },
      null,
      { process: session, rng: new SeededRng(7), readSlot },
    );
    await processor.process(new Float32Array(2), new Float32Array(2), null);
    expect(session.calls[0].rand).toHaveLength(3);
    expect(session.calls[0].rand.every((v) => v >= 0 && v < 1)).toBe(true);
  });
});

describe('ActionManager', () => {
  it('serves a native term\'s processed action as `_processed_actions`', () => {
    const term = tracedTerm({ processed: Float32Array.from([1, 2]) });
    const manager = new ActionManager([{ name: 'joint_pos', term }]);
    expect(Array.from(manager.getStateField('joint_pos', '_processed_actions')!)).toEqual([1, 2]);
    expect(manager.getStateField('joint_pos', '_target')).toBeNull();
    expect(manager.getStateField('other', '_processed_actions')).toBeNull();
  });

  it('logs a failing traced term once, and processes the terms after it', async () => {
    const failing = new OnnxActionProcess('arm', DELTA_CFG, null, {
      process: new FakeSession(() => {
        throw new Error('bad feed');
      }),
      rng: new SeededRng(1),
      readSlot,
    });
    const arm = tracedTerm({ processed: Float32Array.from([0.7, 0.8]), active: true });
    const gripper = nativeTerm();
    const manager = new ActionManager([
      { name: 'arm', term: arm, traced: failing },
      { name: 'gripper', term: gripper },
    ]);
    const error = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      await manager.process(Float32Array.from([0, 0.5, 0]), CONTEXT);
      await manager.process(Float32Array.from([0, 0.5, 0]), CONTEXT);
      expect(error).toHaveBeenCalledTimes(1);
    } finally {
      error.mockRestore();
    }
    expect(arm.active).toBe(true);
    expect(rounded(arm.processed)).toEqual([0.7, 0.8]);
    expect(gripper.active).toBe(true);
    expect(Array.from(gripper.processed)).toEqual([0.5, 0.5]);
  });

  it('holds a traced term on a step with no fresh action, and still processes a native one', async () => {
    const session = deltaSession();
    const processor = new OnnxActionProcess('arm', DELTA_CFG, null, {
      process: session,
      rng: new SeededRng(1),
      readSlot,
    });
    const arm = tracedTerm();
    const gripper = nativeTerm();
    const manager = new ActionManager([
      { name: 'arm', term: arm, traced: processor },
      { name: 'gripper', term: gripper },
    ]);
    await manager.process(Float32Array.from([1, 0.5, 1]), CONTEXT);
    const target = Array.from(arm.processed);

    // Inference stored no new action: a traced term integrating the held one again would
    // walk its target off on its own.
    await manager.process(Float32Array.from([1, 0.25, 1]), CONTEXT, { fresh: false });
    expect(session.calls).toHaveLength(1);
    expect(Array.from(arm.processed)).toEqual(target);
    // A flat ramp, not a replay of the last step's.
    expect(Array.from(arm.rampFrom!)).toEqual(target);
    expect(Array.from(gripper.processed)).toEqual([0.25, 0.25]);
  });

  it('decides a traced term\'s hold on the same clock as a native one', async () => {
    const term = tracedTerm({ holdS: 1.0 });
    const processor = new OnnxActionProcess('held', DELTA_CFG, null, {
      process: deltaSession(),
      rng: new SeededRng(1),
      readSlot,
    });
    const manager = new ActionManager([{ name: 'held', term, traced: processor }]);
    await manager.process(new Float32Array(3), { episodeLength: 19, stepDt: 0.05 });
    expect(term.holding).toBe(true);
    await manager.process(new Float32Array(3), { episodeLength: 20, stepDt: 0.05 });
    expect(term.holding).toBe(false);
  });
});
