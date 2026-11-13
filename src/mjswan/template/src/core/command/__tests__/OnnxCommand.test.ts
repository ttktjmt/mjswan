/**
 * The parity harness validates the graph's math Python-side; what is tested here is the
 * native half — the resample timer, state threading, seeded `rand`, the UI override,
 * the `entity_write` hand-off, and the never-block-never-queue boundary.
 *
 * The ONNX session is a fake, so these run headless with no ORT.
 */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { SeededRng } from '../../rng';
import { OnnxCommand } from '../OnnxCommand';
import type { OnnxCommandConfig, OnnxSession, OnnxTensorLike } from '../OnnxCommand';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** Drain the microtask queue, so the in-flight flag clears between fake `update()`s. */
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

/** Tensor data as a plain number array (the data union needs one narrowing). */
function values(tensor: OnnxTensorLike): number[] {
  return Array.from(tensor.data as ArrayLike<number>, Number);
}

/** Records feeds and returns scripted outputs. */
class FakeSession implements OnnxSession {
  readonly calls: Array<Record<string, OnnxTensorLike>> = [];
  private pending: Array<() => void> = [];

  constructor(
    private readonly respond: (
      feeds: Record<string, OnnxTensorLike>,
      call: number,
    ) => Record<string, OnnxTensorLike>,
    private readonly manual = false,
  ) {}

  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const index = this.calls.length;
    // Snapshot feed data — the handler reuses tensors across frames.
    this.calls.push(
      Object.fromEntries(
        Object.entries(feeds).map(([k, v]) => [
          k,
          { data: (v.data as Float32Array).slice(), dims: [...v.dims] } as OnnxTensorLike,
        ]),
      ),
    );
    const result = this.respond(feeds, index);
    if (!this.manual) return Promise.resolve(result);
    return new Promise(resolve => {
      this.pending.push(() => resolve(result));
    });
  }

  /** Resolve one in-flight call (manual mode). */
  flush(): void {
    const next = this.pending.shift();
    next?.();
  }

  get inFlightCount(): number {
    return this.pending.length;
  }
}

const VELOCITY_CFG: OnnxCommandConfig = {
  name: 'OnnxCommand',
  onnx: 'command/twist.onnx',
  command_field: 'vel_command_b',
  rand_dim: 6,
  state_fields: [
    { name: 'vel_command_b', shape: [1, 3], dtype: 'float32' },
    { name: 'is_standing_env', shape: [1], dtype: 'bool' },
  ],
  input_slots: [{ entity: 'robot', field: 'heading_w' }],
  resampling_time_range: [3.0, 8.0],
};

function velocityOutputs(vx: number, vy: number, wz: number): Record<string, OnnxTensorLike> {
  return {
    next_vel_command_b: { data: new Float32Array([vx, vy, wz]), dims: [1, 3] },
    next_is_standing_env: { data: new Uint8Array([0]), dims: [1] },
  };
}

describe('OnnxCommand: state threading and command output', () => {
  it('feeds prev_<field> and reads back next_<field>', async () => {
    const session = new FakeSession((_f, i) => velocityOutputs(i + 1, 0, 0));
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });

    await cmd.step(true);
    expect(Array.from(cmd.getCommand())).toEqual([1, 0, 0]);

    await cmd.step(false);
    // Second call must have been fed the first call's output as prev state.
    expect(values(session.calls[1].prev_vel_command_b)).toEqual([1, 0, 0]);
    expect(Array.from(cmd.getCommand())).toEqual([2, 0, 0]);
  });

  it('seeds state to zeros from the declared shape/dtype when no init is given', async () => {
    // An older bundle carries no `init`, and zero is right for every reference task.
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });
    await cmd.step(true);
    expect(values(session.calls[0].prev_vel_command_b)).toEqual([0, 0, 0]);
    expect(session.calls[0].prev_vel_command_b.dims).toEqual([1, 3]);
  });

  it('seeds state from the declared init (ADR 0005 §3)', async () => {
    // A term whose first resample skips a field starts where the build found it.
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      {
        ...VELOCITY_CFG,
        state_fields: [
          { name: 'vel_command_b', shape: [1, 3], dtype: 'float32', init: [0.4, -0.2, 1.5] },
          { name: 'is_standing_env', shape: [1], dtype: 'bool', init: [1] },
        ],
      },
      null,
      { session, rng: new SeededRng(1) },
    );
    await cmd.step(true);
    expect(values(session.calls[0].prev_vel_command_b)).toEqual([
      0.4000000059604645, -0.20000000298023224, 1.5,
    ]);
    expect(values(session.calls[0].prev_is_standing_env)).toEqual([1]);
  });

  it('ignores an init that disagrees with the declared width', async () => {
    // Config and graph disagreeing is a build bug: drop the extra, keep the zeros.
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      {
        ...VELOCITY_CFG,
        state_fields: [
          { name: 'vel_command_b', shape: [1, 3], dtype: 'float32', init: [1, 2, 3, 4, 5] },
          { name: 'is_standing_env', shape: [1], dtype: 'bool', init: [] },
        ],
      },
      null,
      { session, rng: new SeededRng(1) },
    );
    await cmd.step(true);
    expect(values(session.calls[0].prev_vel_command_b)).toEqual([1, 2, 3]);
    expect(values(session.calls[0].prev_is_standing_env)).toEqual([0]);
  });

  it('threads a declared dynamic input slot from the slot reader', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
      readSlot: slot => (slot.field === 'heading_w' ? new Float32Array([1.25]) : null),
    });
    await cmd.step(true);
    expect(values(session.calls[0].robot__heading_w)).toEqual([1.25]);
  });

  it('feeds a sensor slot under its build-supplied input name', async () => {
    // A sensor slot has no `field`, so only the build-supplied `input` names its input.
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      {
        ...VELOCITY_CFG,
        input_slots: [
          { sensor: 'robot/imu_ang_vel', input: 'sensor__robot_imu_ang_vel' },
        ],
      },
      null,
      {
        session,
        rng: new SeededRng(1),
        readSlot: slot =>
          slot.sensor === 'robot/imu_ang_vel' ? new Float32Array([0.1, 0.2, 0.3]) : null,
      },
    );
    await cmd.step(true);
    expect(values(session.calls[0].sensor__robot_imu_ang_vel)).toEqual(
      [0.1, 0.2, 0.3].map(v => Math.fround(v)),
    );
  });
});

describe('OnnxCommand: resample timer (scalar, ADR §5)', () => {
  it('resamples on the first frame, then not until the timer expires', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, resampling_time_range: [1.0, 1.0] },
      null,
      { session, rng: new SeededRng(1) },
    );
    cmd.update(0.1);
    await settle();
    expect(session.calls[0].resample_mask.data[0]).toBe(1); // reset semantics

    cmd.update(0.1);
    await settle();
    expect(session.calls[1].resample_mask.data[0]).toBe(0);
  });

  it('sets resample_mask when the interval elapses', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, resampling_time_range: [1.0, 1.0] },
      null,
      { session, rng: new SeededRng(1) },
    );
    for (let i = 0; i < 12; i++) {
      cmd.update(0.1);
      await settle();
    }
    const resampleFrames = session.calls.filter(c => c.resample_mask.data[0] === 1).length;
    // First frame (reset) + one when the 1.0s timer expired.
    expect(resampleFrames).toBe(2);
  });

  it('reset() resamples immediately; the frame\'s update only refreshes', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, resampling_time_range: [10.0, 10.0] },
      null,
      { session, rng: new SeededRng(1) },
    );
    cmd.update(0.1);
    await settle();
    cmd.update(0.1);
    await settle();
    expect(session.calls[1].resample_mask.data[0]).toBe(0);

    await cmd.reset();
    expect(session.calls.length).toBe(3);
    expect(session.calls[2].resample_mask.data[0]).toBe(1);
    // Only an episode reset runs what a term's `reset` adds to the resample.
    expect(session.calls.map(call => call.reset_mask.data[0])).toEqual([0, 0, 1]);

    // A later step's `update()` is `_update_command` alone.
    cmd.update(0.1);
    await settle();
    expect(session.calls[3].resample_mask.data[0]).toBe(0);
  });

  it('after an auto-reset, the same step\'s update leaves the state alone', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, resampling_time_range: [10.0, 10.0] },
      null,
      { session, rng: new SeededRng(1) },
    );
    await cmd.reset();
    cmd.update(0.1, true);
    await settle();
    expect(session.calls.length).toBe(1);

    cmd.update(0.1);
    await settle();
    expect(session.calls.length).toBe(2);
    expect(session.calls[1].resample_mask.data[0]).toBe(0);
  });

  it('without resampling_time_range, resamples only on reset', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cfg = { ...VELOCITY_CFG };
    delete cfg.resampling_time_range;
    const cmd = new OnnxCommand('lift', cfg, null, { session, rng: new SeededRng(1) });
    for (let i = 0; i < 5; i++) {
      cmd.update(1.0);
      await settle();
    }
    const resampleFrames = session.calls.filter(c => c.resample_mask.data[0] === 1).length;
    expect(resampleFrames).toBe(1);
  });
});

describe('OnnxCommand: seeded rand (ADR §2)', () => {
  it('feeds rand_dim draws from the orchestrator PRNG', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });
    await cmd.step(true);
    expect(session.calls[0].rand.data.length).toBe(6);
  });

  it('is reproducible from a seed (bit-for-bit replay)', async () => {
    const run = async (): Promise<number[]> => {
      const session = new FakeSession(() => velocityOutputs(0, 0, 0));
      const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
        session,
        rng: new SeededRng(2026),
      });
      for (let i = 0; i < 4; i++) await cmd.step(true);
      return session.calls.flatMap(c => Array.from(c.rand.data as Float32Array));
    };
    expect(await run()).toEqual(await run());
  });

  it('applies rand_ranges per element', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, rand_dim: 2, rand_ranges: [[-1, -0.5], [10, 11]] },
      null,
      { session, rng: new SeededRng(3) },
    );
    await cmd.step(true);
    const rand = session.calls[0].rand.data as Float32Array;
    expect(rand[0]).toBeGreaterThanOrEqual(-1);
    expect(rand[0]).toBeLessThan(-0.5);
    expect(rand[1]).toBeGreaterThanOrEqual(10);
    expect(rand[1]).toBeLessThan(11);
  });

  it('feeds only what the graph declares', async () => {
    // A term that draws nothing exports no `rand`; ORT rejects it as a feed.
    const session = new FakeSession(() => velocityOutputs(0, 0, 0));
    Object.assign(session, { inputNames: ['prev_vel_command_b', 'resample_mask'] });
    const cmd = new OnnxCommand(
      'clock',
      { ...VELOCITY_CFG, rand_dim: 0, rand_ranges: [] },
      null,
      { session, rng: new SeededRng(4) },
    );
    await cmd.step(true);
    expect(Object.keys(session.calls[0]).sort()).toEqual([
      'prev_vel_command_b',
      'resample_mask',
    ]);
  });
});

describe('OnnxCommand: UI override (mjlab play parity, §3a)', () => {
  const UI_CFG: OnnxCommandConfig = {
    ...VELOCITY_CFG,
    ui: {
      inputs: [
        { type: 'checkbox', name: 'enabled', label: 'Joystick', default: false },
        { type: 'slider', name: 'lin_vel_x', label: 'X', min: -1, max: 1, step: 0.01, default: 0 },
        { type: 'slider', name: 'lin_vel_y', label: 'Y', min: -1, max: 1, step: 0.01, default: 0 },
        { type: 'slider', name: 'ang_vel_z', label: 'Yaw', min: -1, max: 1, step: 0.01, default: 0 },
        { type: 'button', name: 'zero', label: 'Zero' },
      ],
    },
  };

  it('returns the autonomous value while the checkbox is off', async () => {
    const session = new FakeSession(() => velocityOutputs(0.4, 0.1, -0.2));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    await cmd.step(true);
    cmd.setValue('lin_vel_x', 0.9);
    expect(Array.from(cmd.getCommand())).toEqual([0.4, 0.1, -0.2].map(v => Math.fround(v)));
  });

  it('overwrites per axis once enabled', async () => {
    const session = new FakeSession(() => velocityOutputs(0.4, 0.1, -0.2));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    await cmd.step(true);
    cmd.setValue('enabled', 1);
    cmd.setValue('lin_vel_x', 0.9);
    cmd.setValue('ang_vel_z', 0.5);
    const out = Array.from(cmd.getCommand());
    expect(out[0]).toBeCloseTo(0.9, 6);
    expect(out[1]).toBeCloseTo(0, 6); // slider default, not the graph's 0.1
    expect(out[2]).toBeCloseTo(0.5, 6);
  });

  it('serves other graphs the command with the override, by `command` and by its field', async () => {
    const session = new FakeSession(() => velocityOutputs(0.4, 0.1, -0.2));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    await cmd.step(true);
    cmd.setValue('enabled', 1);
    cmd.setValue('lin_vel_x', 0.9);
    const command = Array.from(cmd.getCommand());
    expect(command[0]).toBeCloseTo(0.9, 6);
    expect(Array.from(cmd.getStateField('command')!)).toEqual(command);
    expect(Array.from(cmd.getStateField('vel_command_b')!)).toEqual(command);
    expect(Array.from(cmd.getStateField('is_standing_env')!)).toEqual([0]);
  });

  it('never skips the autonomous computation while enabled (mjlab parity)', async () => {
    const session = new FakeSession((_f, i) => velocityOutputs(i, 0, 0));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    cmd.setValue('enabled', 1);
    await cmd.step(true);
    await cmd.step(false);
    // The graph ran on both frames even though the UI is overriding the output.
    expect(session.calls.length).toBe(2);
  });

  it('the zero button zeroes the sliders', async () => {
    const session = new FakeSession(() => velocityOutputs(0.4, 0.1, -0.2));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    await cmd.step(true);
    cmd.setValue('enabled', 1);
    cmd.setValue('lin_vel_x', 0.9);
    expect(cmd.triggerButton('zero')).toBe(true);
    expect(Array.from(cmd.getCommand())).toEqual([0, 0, 0]);
  });

  it('says so for a button it has no action for', () => {
    // mjlab's command GUIs are free to declare a button this class has no action for.
    const session = new FakeSession(() => velocityOutputs(0.4, 0.1, -0.2));
    const cmd = new OnnxCommand('twist', UI_CFG, null, { session, rng: new SeededRng(1) });
    expect(cmd.triggerButton('start_here')).toBe(false);
  });
});

describe('OnnxCommand: entity_write hand-off (§3b)', () => {
  function fakeModelData(): { mjModel: MjModel; mjData: MjData } {
    const mjModel = {
      njnt: 1,
      jnt_type: Int32Array.from([0]), // free joint
      jnt_qposadr: Int32Array.from([0]),
      jnt_dofadr: Int32Array.from([0]),
      names: new Uint8Array(0).buffer,
      name_jntadr: Int32Array.from([0]),
    } as unknown as MjModel;
    const mjData = {
      qpos: new Float64Array(7),
      qvel: new Float64Array(6),
    } as unknown as MjData;
    return { mjModel, mjData };
  }

  const LIFT_CFG: OnnxCommandConfig = {
    name: 'OnnxCommand',
    onnx: 'command/lift_height.onnx',
    command_field: 'target_pos',
    rand_dim: 7,
    state_fields: [{ name: 'target_pos', shape: [1, 3], dtype: 'float32' }],
    write_targets: [
      { kind: 'root_pose', entity: 'cube', fields: ['pose'] },
      { kind: 'root_velocity', entity: 'cube', fields: ['velocity'] },
    ],
  };

  it('applies the graph-computed cube pose/velocity to mjData', async () => {
    const { mjModel, mjData } = fakeModelData();
    const session = new FakeSession(() => ({
      next_target_pos: { data: new Float32Array([0.4, 0, 0.3]), dims: [1, 3] },
      root_pose__pose: { data: new Float32Array([0.1, 0.2, 0.05, 1, 0, 0, 0]), dims: [1, 7] },
      root_velocity__velocity: { data: new Float32Array(6), dims: [1, 6] },
    }));
    const cmd = new OnnxCommand(
      'lift_height',
      LIFT_CFG,
      { mjModel, mjData } as unknown as import('../types').CommandTermContext,
      { session, rng: new SeededRng(1) },
    );
    await cmd.step(true);
    expect(mjData.qpos[0]).toBeCloseTo(0.1, 6);
    expect(mjData.qpos[1]).toBeCloseTo(0.2, 6);
    expect(mjData.qpos[3]).toBeCloseTo(1, 6);
    expect(Array.from(cmd.getCommand())).toEqual([0.4, 0, 0.3].map(v => Math.fround(v)));
  });

  it('leaves the entity alone between resamples', async () => {
    // mjlab writes the cube only from `_resample_command`. The graph still emits a
    // freshly drawn pose every frame, so applying it unconditionally teleported the
    // cube on every step instead of once per resampling interval.
    const { mjModel, mjData } = fakeModelData();
    const session = new FakeSession(() => ({
      next_target_pos: { data: new Float32Array([0.4, 0, 0.3]), dims: [1, 3] },
      root_pose__pose: { data: new Float32Array([0.1, 0.2, 0.05, 1, 0, 0, 0]), dims: [1, 7] },
      root_velocity__velocity: { data: new Float32Array(6), dims: [1, 6] },
    }));
    const cmd = new OnnxCommand(
      'lift_height',
      LIFT_CFG,
      { mjModel, mjData } as unknown as import('../types').CommandTermContext,
      { session, rng: new SeededRng(1) },
    );

    await cmd.step(false);

    expect(mjData.qpos[0]).toBe(0);
    expect(mjData.qpos[3]).toBe(0);
  });

  it('applies a gated write only when the graph sets its gate', async () => {
    // mjlab's `init_velocity_prob`: the reset starts an env moving only if a draw says so.
    const { mjModel, mjData } = fakeModelData();
    let gate = 0;
    const session = new FakeSession(() => ({
      next_target_pos: { data: new Float32Array([0, 0, 0]), dims: [1, 3] },
      robot__root_velocity_b__velocity: {
        data: new Float32Array([0.5, 0, 0, 0, 0, 0]),
        dims: [1, 6],
      },
      robot__root_velocity_b__gate: { data: new Float32Array([gate]), dims: [1] },
    }));
    const cmd = new OnnxCommand(
      'twist',
      {
        ...LIFT_CFG,
        write_targets: [
          {
            kind: 'root_velocity_b',
            entity: 'robot',
            fields: ['velocity'],
            outputs: ['robot__root_velocity_b__velocity'],
            gate: 'robot__root_velocity_b__gate',
          },
        ],
      },
      { mjModel, mjData } as unknown as import('../types').CommandTermContext,
      { session, rng: new SeededRng(1) },
    );
    mjData.qpos[3] = 1; // identity orientation

    await cmd.step(true, true);
    expect(mjData.qvel[0]).toBe(0);

    gate = 1;
    await cmd.step(true, true);
    expect(mjData.qvel[0]).toBeCloseTo(0.5, 6);
  });

  it('does not treat next_<state> outputs as writes', async () => {
    const { mjModel, mjData } = fakeModelData();
    const session = new FakeSession(() => ({
      next_target_pos: { data: new Float32Array([9, 9, 9]), dims: [1, 3] },
    }));
    const cmd = new OnnxCommand(
      'lift_height',
      LIFT_CFG,
      { mjModel, mjData } as unknown as import('../types').CommandTermContext,
      { session, rng: new SeededRng(1) },
    );
    await cmd.step(true);
    expect(mjData.qpos[0]).toBe(0); // untouched — no write value supplied
  });
});

describe('OnnxCommand: debug-vis marker (generic — replaces LiftingCommand.ts)', () => {
  const LIFT_VIZ_CFG: OnnxCommandConfig = {
    name: 'OnnxCommand',
    onnx: 'command/lift_height.onnx',
    command_field: 'target_pos',
    rand_dim: 7,
    state_fields: [{ name: 'target_pos', shape: [1, 3], dtype: 'float32' }],
    debug_vis: true,
    viz: [
      { shape: 'sphere', radius: 0.03, color: [1, 0.5, 0, 0.3], origin: { state: 'target_pos' } },
    ],
  };

  function fakeContext(): import('../types').CommandTermContext {
    return { scene: new THREE.Scene() } as unknown as import('../types').CommandTermContext;
  }

  it('adds a hidden marker to the scene at construction', () => {
    const context = fakeContext();
    new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session: new FakeSession(() => ({})),
      rng: new SeededRng(1),
    });
    expect(context.scene.children.length).toBe(1);
    expect(context.scene.children[0].visible).toBe(false);
  });

  it('shows and positions the marker at the viz field once switched on', async () => {
    const context = fakeContext();
    const session = new FakeSession(() => ({
      next_target_pos: { data: new Float32Array([0.4, 0.1, 0.3]), dims: [1, 3] },
    }));
    const cmd = new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session,
      rng: new SeededRng(1),
    });
    await cmd.step(true);
    cmd.setDebugVisEnabled(true);
    cmd.updateDebugVisuals();
    const marker = context.scene.children[0];
    expect(marker.visible).toBe(true);
    // mjcToThreeCoordinate: (x, z, -y).
    expect(marker.position.x).toBeCloseTo(0.4, 6);
    expect(marker.position.y).toBeCloseTo(0.3, 6);
    expect(marker.position.z).toBeCloseTo(-0.1, 6);
  });

  it('hides the marker when debug_vis is off', async () => {
    const context = fakeContext();
    const cfg = { ...LIFT_VIZ_CFG, debug_vis: false };
    const cmd = new OnnxCommand('lift_height', cfg, context, {
      session: new FakeSession(() => ({
        next_target_pos: { data: new Float32Array([0.4, 0.1, 0.3]), dims: [1, 3] },
      })),
      rng: new SeededRng(1),
    });
    await cmd.step(true);
    cmd.updateDebugVisuals();
    expect(context.scene.children[0].visible).toBe(false);
  });

  it('serves an entity-sourced primitive from the same slot reader its graph uses', async () => {
    // The velocity arrows read `root_link_*`: without `readSlot` reaching the drawing,
    // they stay hidden while the graph itself keeps running.
    const context = fakeContext();
    const cfg: OnnxCommandConfig = {
      ...VELOCITY_CFG,
      debug_vis: true,
      viz: [
        {
          shape: 'arrow',
          color: [0, 0.6, 1, 0.7],
          width: 0.015,
          origin: { const: [0, 0, 0] },
          vector: { entity: 'robot', field: 'root_link_lin_vel_b', scale: 1 },
        },
      ],
    };
    const cmd = new OnnxCommand('twist', cfg, context, {
      session: new FakeSession(() => velocityOutputs(0, 0, 0)),
      rng: new SeededRng(1),
      readSlot: slot =>
        slot.field === 'root_link_lin_vel_b' ? new Float32Array([1, 0, 0]) : null,
    });
    await cmd.step(true);
    cmd.setDebugVisEnabled(true);
    cmd.updateDebugVisuals();
    expect(context.scene.children[0].visible).toBe(true);
  });

  it('starts switched on, as mjlab\'s viewers do', async () => {
    const context = fakeContext();
    const cmd = new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session: new FakeSession(() => ({
        next_target_pos: { data: new Float32Array([0.4, 0.1, 0.3]), dims: [1, 3] },
      })),
      rng: new SeededRng(1),
    });
    await cmd.step(true);
    cmd.updateDebugVisuals();
    expect(cmd.debugVisEnabled()).toBe(true);
    expect(context.scene.children[0].visible).toBe(true);
  });

  it('hides the drawing while the viewer toggle is off, and brings it back', async () => {
    const context = fakeContext();
    const cmd = new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session: new FakeSession(() => ({
        next_target_pos: { data: new Float32Array([0.4, 0.1, 0.3]), dims: [1, 3] },
      })),
      rng: new SeededRng(1),
    });
    await cmd.step(true);

    cmd.setDebugVisEnabled(false);
    cmd.updateDebugVisuals();
    expect(context.scene.children[0].visible).toBe(false);
    expect(cmd.debugVisEnabled()).toBe(false);

    cmd.setDebugVisEnabled(true);
    cmd.updateDebugVisuals();
    expect(context.scene.children[0].visible).toBe(true);
  });

  it('offers no toggle for a term with nothing to draw', () => {
    // The panel lists on `debugVisEnabled()`, so "nothing to draw" must not read as false.
    const context = fakeContext();
    const deps = { session: new FakeSession(() => ({})), rng: new SeededRng(1) };
    expect(new OnnxCommand('twist', VELOCITY_CFG, context, deps).debugVisEnabled()).toBeNull();
    expect(
      new OnnxCommand(
        'lift_height',
        { ...LIFT_VIZ_CFG, debug_vis: false },
        context,
        deps,
      ).debugVisEnabled(),
    ).toBeNull();
  });

  it('attaches under the model root, not the bare scene', () => {
    // The root carries whatever transform the scene applies to the model; a marker
    // beside it would drift away from what it marks.
    const scene = new THREE.Scene();
    const mujocoRoot = new THREE.Group();
    scene.add(mujocoRoot);
    const context = { scene, mujocoRoot } as unknown as import('../types').CommandTermContext;
    new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session: new FakeSession(() => ({})),
      rng: new SeededRng(1),
    });
    expect(mujocoRoot.children.length).toBe(1);
    expect(scene.children.length).toBe(1); // the root itself, nothing beside it
  });

  it('does not create a marker without a viz descriptor', () => {
    const context = fakeContext();
    new OnnxCommand('twist', VELOCITY_CFG, context, {
      session: new FakeSession(() => velocityOutputs(0, 0, 0)),
      rng: new SeededRng(1),
    });
    expect(context.scene.children.length).toBe(0);
  });

  it('dispose() removes the marker from the scene', () => {
    const context = fakeContext();
    const cmd = new OnnxCommand('lift_height', LIFT_VIZ_CFG, context, {
      session: new FakeSession(() => ({})),
      rng: new SeededRng(1),
    });
    expect(context.scene.children.length).toBe(1);
    cmd.dispose?.();
    expect(context.scene.children.length).toBe(0);
  });
});

describe('OnnxCommand: sync/async boundary', () => {
  it('update() does not block and serves the last completed value', () => {
    const session = new FakeSession(() => velocityOutputs(1, 2, 3), true);
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });
    cmd.update(0.02);
    // Inference is still pending: the command is the previous (zero) value.
    expect(Array.from(cmd.getCommand())).toEqual([0, 0, 0]);
    expect(session.inFlightCount).toBe(1);
  });

  it('settled() resolves once the update in flight has landed', async () => {
    const session = new FakeSession(() => velocityOutputs(1, 2, 3), true);
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });
    cmd.update(0.02);
    let landed = false;
    const settled = cmd.settled().then(() => (landed = true));
    await settle();
    expect(landed).toBe(false);

    session.flush();
    await settled;
    expect(Array.from(cmd.getCommand())).toEqual([1, 2, 3]);
  });

  it('skips frames while inference is in flight (never queues a backlog)', async () => {
    const session = new FakeSession(() => velocityOutputs(1, 2, 3), true);
    const cmd = new OnnxCommand('twist', VELOCITY_CFG, null, {
      session,
      rng: new SeededRng(1),
    });
    for (let i = 0; i < 10; i++) cmd.update(0.02);
    expect(session.calls.length).toBe(1); // 9 frames skipped, not queued

    session.flush();
    await settle();
    expect(Array.from(cmd.getCommand())).toEqual([1, 2, 3]);
  });

  it('a resample that lands during an in-flight frame is not lost', async () => {
    const session = new FakeSession(() => velocityOutputs(0, 0, 0), true);
    const cmd = new OnnxCommand(
      'twist',
      { ...VELOCITY_CFG, resampling_time_range: [1.0, 1.0] },
      null,
      { session, rng: new SeededRng(1) },
    );
    cmd.update(0.1); // first frame: resample (reset), stays in flight
    cmd.update(2.0); // timer expires while in flight → pending resample
    session.flush();
    await settle();
    cmd.update(0.1); // next admitted frame must carry the pending resample
    await settle();
    expect(session.calls[1].resample_mask.data[0]).toBe(1);
  });
});
