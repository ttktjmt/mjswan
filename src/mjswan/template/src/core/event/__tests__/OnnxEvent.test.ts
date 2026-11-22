/**
 * `OnnxEvent`: the generic ONNX-backed event handler.
 *
 * The graph's math is validated Python-side; what needs testing here is the
 * native half: seeded `rand`, the `entity_write` hand-off, and the async
 * in-flight guard (an event never queues a backlog). The ONNX session is
 * injected as a fake, so these run headless with no ORT.
 */
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../../rng';
import type { OnnxTensorLike } from '../../onnx/session';
import { OnnxEvent, isOnnxEventConfig } from '../OnnxEvent';
import type { OnnxEventConfig } from '../OnnxEvent';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** Tensor data as a plain number array (the data union needs one narrowing). */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as ArrayLike<number>, Number);
}

class FakeSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];
  private pending: Array<() => void> = [];

  constructor(
    private readonly respond: (feeds: Record<string, OnnxTensorLike>) => Record<string, OnnxTensorLike>,
    private readonly manual = false,
  ) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    this.calls.push(
      Object.fromEntries(
        Object.entries(feeds).map(([k, v]) => [
          k,
          { data: (v.data as Float32Array).slice(), dims: [...v.dims] } as OnnxTensorLike,
        ]),
      ),
    );
    const result = this.respond(feeds);
    if (!this.manual) return Promise.resolve(result);
    return new Promise((resolve) => this.pending.push(() => resolve(result)));
  }

  flush(): void {
    this.pending.shift()?.();
  }

  get inFlightCount(): number {
    return this.pending.length;
  }
}

function fakeModelData(nJoint = 1, jntType = 2): { mjModel: MjModel; mjData: MjData } {
  const mjModel = {
    njnt: nJoint,
    jnt_type: Int32Array.from({ length: nJoint }, () => jntType), // mjJNT_SLIDE
    jnt_qposadr: Int32Array.from({ length: nJoint }, (_v, i) => i * 7),
    jnt_dofadr: Int32Array.from({ length: nJoint }, (_v, i) => i * 6),
    names: new Uint8Array(0).buffer,
    name_jntadr: Int32Array.from({ length: nJoint }, () => 0),
  } as unknown as MjModel;
  const mjData = {
    qpos: new Float64Array(nJoint * 7),
    qvel: new Float64Array(nJoint * 6),
  } as unknown as MjData;
  return { mjModel, mjData };
}

describe('isOnnxEventConfig', () => {
  it('recognizes an onnx-backed event config', () => {
    expect(isOnnxEventConfig({ name: 'push_robot', mode: 'interval', onnx: 'x.onnx' })).toBe(true);
  });

  it('rejects a config that names no graph', () => {
    expect(isOnnxEventConfig({ name: 'randomize_terrain' })).toBe(false);
  });

  it('rejects configs missing onnx or mode', () => {
    expect(isOnnxEventConfig({ name: 'x', onnx: 'x.onnx' })).toBe(false);
    expect(isOnnxEventConfig({ name: 'x', mode: 'reset' })).toBe(false);
  });
});

describe('OnnxEvent: seeded rand', () => {
  const CFG: OnnxEventConfig = { name: 'push_robot', mode: 'interval', onnx: 'event/push_robot.onnx', rand_dim: 6 };

  it('feeds rand_dim draws from the orchestrator PRNG', async () => {
    const session = new FakeSession(() => ({}));
    const event = new OnnxEvent(CFG, { session, rng: new SeededRng(1) });
    await event.fire({ mjModel: null, mjData: null });
    expect(session.calls[0].rand.data.length).toBe(6);
  });

  it('is reproducible from a seed (replay)', async () => {
    const run = async (): Promise<number[]> => {
      const session = new FakeSession(() => ({}));
      const event = new OnnxEvent(CFG, { session, rng: new SeededRng(2026) });
      for (let i = 0; i < 3; i++) await event.fire({ mjModel: null, mjData: null });
      return session.calls.flatMap((c) => Array.from(c.rand.data as Float32Array));
    };
    expect(await run()).toEqual(await run());
  });
});

describe('OnnxEvent: dynamic input slots', () => {
  it('threads a declared runtime read into the feed', async () => {
    const session = new FakeSession(() => ({}));
    const cfg: OnnxEventConfig = {
      name: 'push_robot',
      mode: 'interval',
      onnx: 'x.onnx',
      rand_dim: 6,
      input_slots: [{ entity: 'robot', field: 'root_link_vel_w' }],
    };
    const event = new OnnxEvent(cfg, {
      session,
      rng: new SeededRng(1),
      readSlot: (slot) => (slot.field === 'root_link_vel_w' ? new Float32Array([1, 2, 3, 0, 0, 0]) : null),
    });
    await event.fire({ mjModel: null, mjData: null });
    expect(values(session.calls[0].robot__root_link_vel_w)).toEqual([1, 2, 3, 0, 0, 0]);
  });
});

describe('OnnxEvent: entity_write hand-off', () => {
  it('applies the graph-computed joint state', async () => {
    const { mjModel, mjData } = fakeModelData();
    const cfg: OnnxEventConfig = {
      name: 'reset_slider',
      mode: 'reset',
      onnx: 'x.onnx',
      rand_dim: 2,
      write_targets: [{ kind: 'joint_state', fields: ['position', 'velocity'], joint_ids: [0] }],
    };
    const session = new FakeSession(() => ({
      joint_state__position: { data: new Float32Array([0.5]), dims: [1, 1] },
      joint_state__velocity: { data: new Float32Array([0.1]), dims: [1, 1] },
    }));
    const event = new OnnxEvent(cfg, { session, rng: new SeededRng(1) });
    await event.fire({ mjModel, mjData });
    expect(mjData.qpos[0]).toBeCloseTo(0.5, 6);
    expect(mjData.qvel[0]).toBeCloseTo(0.1, 6);
  });

  it('does nothing when the context has no model/data (does not throw)', async () => {
    const cfg: OnnxEventConfig = {
      name: 'reset_slider',
      mode: 'reset',
      onnx: 'x.onnx',
      rand_dim: 2,
      write_targets: [{ kind: 'joint_state', fields: ['position', 'velocity'] }],
    };
    const session = new FakeSession(() => ({
      joint_state__position: { data: new Float32Array([0.5]), dims: [1, 1] },
    }));
    const event = new OnnxEvent(cfg, { session, rng: new SeededRng(1) });
    await expect(event.fire({ mjModel: null, mjData: null })).resolves.toBeUndefined();
  });
});

describe('OnnxEvent: async in-flight guard', () => {
  it('a second fire() while one is in flight is a no-op (never queues)', async () => {
    const session = new FakeSession(() => ({}), true);
    const event = new OnnxEvent(
      { name: 'push_robot', mode: 'interval', onnx: 'x.onnx', rand_dim: 6 },
      { session, rng: new SeededRng(1) },
    );
    const first = event.fire({ mjModel: null, mjData: null });
    expect(event.busy).toBe(true);
    await event.fire({ mjModel: null, mjData: null }); // returns immediately, no-op
    expect(session.calls.length).toBe(1);

    session.flush();
    await first;
    expect(event.busy).toBe(false);
  });
});
