/**
 * What a traced graph's declared inputs resolve to against a live `mjModel`/`mjData` —
 * the browser-side half of the Python parity harness, which proves the graph matches
 * mjlab but not the numbers fed into it.
 *
 * Each field is checked against mjlab's own definition, computed by hand from a fake
 * model whose addresses are deliberately not the identity.
 */
import { describe, expect, it, vi } from 'vitest';

import { createSlotReader, isReadableEntityField, type SlotReaderContext } from '../slotReader';
import { slotDims } from '../session';

type Mutable = Record<string, unknown>;

/**
 * A two-entity scene, mjlab-style: names prefixed, `robot` attached first with a free
 * joint (excluded from `joint_pos`), a ball and two hinges, so its qpos and qvel
 * addresses differ. `cube` follows, shifting `robot` off a naive 0-based guess.
 */
function fakeScene() {
  const jointNames = ['robot/floating_base', 'robot/shoulder', 'robot/elbow', 'robot/wrist', 'cube/free'];
  const bodyNames = ['world', 'robot/pelvis', 'robot/arm', 'cube/body'];
  const siteNames = ['robot/imu', 'robot/grasp', 'cube/center'];
  const sensorNames = ['robot/imu_lin_vel', 'robot/imu_ang_vel'];

  // One shared NUL-separated name table, as MuJoCo has; each adr array indexes it.
  const encoder = new TextEncoder();
  const all: number[] = [];
  const adrOf = (names: string[]): number[] =>
    names.map(name => {
      const at = all.length;
      all.push(...encoder.encode(name), 0);
      return at;
    });
  const name_jntadr = adrOf(jointNames);
  const name_bodyadr = adrOf(bodyNames);
  const name_siteadr = adrOf(siteNames);
  const name_sensoradr = adrOf(sensorNames);

  const mjModel = {
    names: Uint8Array.from(all).buffer,
    njnt: jointNames.length,
    nbody: bodyNames.length,
    nsite: siteNames.length,
    nsensor: sensorNames.length,
    name_jntadr,
    name_bodyadr,
    name_siteadr,
    name_sensoradr,
    //              free  ball  hinge hinge  free
    jnt_type: [0, 1, 3, 3, 0],
    // free 7 | ball 4 | hinge 1 | hinge 1 | free 7  -> 20 qpos
    jnt_qposadr: [0, 7, 11, 12, 13],
    // free 6 | ball 3 | hinge 1 | hinge 1 | free 6  -> 17 dofs
    jnt_dofadr: [0, 6, 9, 10, 11],
    sensor_adr: [0, 3],
    sensor_dim: [3, 3],
  } as unknown as Mutable;

  const qpos = new Float64Array(20);
  const qvel = new Float64Array(17);
  // robot's non-free joints: ball at qpos 7..10, hinges at 11 and 12.
  qpos.set([0.5, 0.5, 0.5, 0.5], 7);
  qpos[11] = 1.25;
  qpos[12] = -0.75;
  // ...and their dofs: ball at 6..8, hinges at 9 and 10.
  qvel.set([0.1, 0.2, 0.3], 6);
  qvel[9] = -1.5;
  qvel[10] = 2.5;

  const xpos = new Float64Array(bodyNames.length * 3);
  const xquat = new Float64Array(bodyNames.length * 4);
  const cvel = new Float64Array(bodyNames.length * 6);
  const subtree_com = new Float64Array(bodyNames.length * 3);
  const site_xpos = new Float64Array(siteNames.length * 3);
  const sensordata = new Float64Array(6);

  // robot/pelvis (body 1) is the entity root.
  xpos.set([1, 2, 3], 3);
  // 90 deg about +z: heading = pi/2, and gravity projects onto -y in body frame.
  const s = Math.SQRT1_2;
  xquat.set([s, 0, 0, s], 4);
  // angular (0.1, 0.2, 0.3), linear-about-COM (1, 0, 0).
  cvel.set([0.1, 0.2, 0.3, 1, 0, 0], 6);
  subtree_com.set([1, 2, 4], 3); // 1 m above the body origin in z
  site_xpos.set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9, 9, 9]);
  sensordata.set([1, 2, 3, 4, 5, 6]);

  const mjData = {
    qpos,
    qvel,
    xpos,
    xquat,
    cvel,
    subtree_com,
    site_xpos,
    sensordata,
  } as unknown as Mutable;

  return { mjModel, mjData };
}

function context(overrides: Partial<SlotReaderContext> = {}): SlotReaderContext {
  const { mjModel, mjData } = fakeScene();
  return { mjModel, mjData, ...overrides } as SlotReaderContext;
}

function close(actual: Float32Array | null, expected: number[], tol = 1e-6): void {
  expect(actual).not.toBeNull();
  expect(actual!.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual![i]).toBeCloseTo(expected[i], Math.round(-Math.log10(tol)));
  }
}

describe('createSlotReader — entity data fields', () => {
  const read = createSlotReader(() => context());

  it('joint_pos skips the free joint and follows spec order', () => {
    // ball (4) + hinge + hinge = 6 values, from qpos 7..12 — never qpos[0..].
    close(read({ entity: 'robot', field: 'joint_pos' }), [0.5, 0.5, 0.5, 0.5, 1.25, -0.75]);
  });

  it('joint_vel uses dof addresses, not qpos addresses', () => {
    // A ball joint is 4 qpos but 3 dofs, so these lists genuinely differ.
    close(read({ entity: 'robot', field: 'joint_vel' }), [0.1, 0.2, 0.3, -1.5, 2.5]);
  });

  it('joint_pos_biased adds the per-joint encoder bias', () => {
    const biased = createSlotReader(() => context(), {
      jointBias: name => (name === 'elbow' ? 0.25 : 0),
    });
    // By *unprefixed* name, and to that joint only: `elbow` is the hinge at index 4.
    close(biased({ entity: 'robot', field: 'joint_pos_biased' }), [
      0.5, 0.5, 0.5, 0.5, 1.25 + 0.25, -0.75,
    ]);
  });

  it('joint_pos_biased matches joint_pos with no bias supplied', () => {
    close(read({ entity: 'robot', field: 'joint_pos_biased' }), [0.5, 0.5, 0.5, 0.5, 1.25, -0.75]);
  });

  it('root pose comes from the entity root body, not body 0', () => {
    close(read({ entity: 'robot', field: 'root_link_pos_w' }), [1, 2, 3]);
    close(read({ entity: 'robot', field: 'root_link_quat_w' }), [Math.SQRT1_2, 0, 0, Math.SQRT1_2]);
    close(read({ entity: 'robot', field: 'root_link_pose_w' }), [
      1, 2, 3, Math.SQRT1_2, 0, 0, Math.SQRT1_2,
    ]);
  });

  it('root_link_lin_vel_w removes the subtree-COM offset from cvel', () => {
    // cvel's linear part is about the subtree COM, 1 m above the body origin:
    //   lin_w = lin_c - ang x (com - pos), ang = (.1,.2,.3), offset = (0,0,1)
    //         = (1,0,0) - (0.2*1 - 0, 0.3*0 - 0.1*1, 0) = (0.8, 0.1, 0)
    close(read({ entity: 'robot', field: 'root_link_lin_vel_w' }), [0.8, 0.1, 0]);
    close(read({ entity: 'robot', field: 'root_link_ang_vel_w' }), [0.1, 0.2, 0.3]);
    close(read({ entity: 'robot', field: 'root_link_vel_w' }), [0.8, 0.1, 0, 0.1, 0.2, 0.3]);
  });

  it('body-frame velocities inverse-rotate the world-frame ones', () => {
    // Yaw +90 deg maps world x -> body -y and world y -> body x.
    close(read({ entity: 'robot', field: 'root_link_lin_vel_b' }), [0.1, -0.8, 0]);
    close(read({ entity: 'robot', field: 'root_link_ang_vel_b' }), [0.2, -0.1, 0.3]);
  });

  it('gravity_vec_w is mjlab\'s constant down direction', () => {
    // mjlab fills it with (0, 0, -1) rather than reading mjModel.opt.gravity.
    close(read({ entity: 'robot', field: 'gravity_vec_w' }), [0, 0, -1]);
  });

  it('projected_gravity_b is world -z in the body frame', () => {
    // Yaw-only rotation leaves gravity along -z.
    close(read({ entity: 'robot', field: 'projected_gravity_b' }), [0, 0, -1]);
  });

  it('heading_w is a single value: the yaw of the body x-axis', () => {
    const heading = read({ entity: 'robot', field: 'heading_w' });
    close(heading, [Math.PI / 2]);
    // Rank matters — mjlab's heading_w is (num_envs,), not (num_envs, 1).
    expect(slotDims({ shape: [1], input: 'robot__heading_w' }, heading!.length)).toEqual([1]);
  });

  it('site_pos_w returns every site of the entity, flattened', () => {
    // Both robot sites, not just the one a term reads, and not cube/center.
    close(read({ entity: 'robot', field: 'site_pos_w' }), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  });

  it('scopes to the named entity', () => {
    // cube: one free joint (excluded) -> no joint values at all; root body 3.
    close(read({ entity: 'cube', field: 'joint_pos' }), []);
    close(read({ entity: 'cube', field: 'site_pos_w' }), [9, 9, 9]);
  });

  it('falls back to the whole model only when the model carries no prefixes', () => {
    // The set_trace_env path: a plain-MJCF model has no attach prefix and is
    // single-entity, so any recorded entity name resolves to everything.
    const plain = context();
    const model = plain.mjModel as unknown as Mutable;
    const encoder = new TextEncoder();
    const bytes: number[] = [];
    const adrOf = (names: string[]): number[] =>
      names.map(name => {
        const at = bytes.length;
        bytes.push(...encoder.encode(name), 0);
        return at;
      });
    model.name_jntadr = adrOf(['floating_base', 'shoulder', 'elbow', 'wrist', 'free']);
    model.name_bodyadr = adrOf(['world', 'pelvis', 'arm', 'body']);
    model.name_siteadr = adrOf(['imu', 'grasp', 'center']);
    model.name_sensoradr = adrOf(['imu_lin_vel', 'imu_ang_vel']);
    model.names = Uint8Array.from(bytes).buffer;

    const plainRead = createSlotReader(() => plain);
    close(plainRead({ entity: 'robot', field: 'joint_pos' }), [0.5, 0.5, 0.5, 0.5, 1.25, -0.75]);
    // Every site, since there is no prefix to scope by.
    close(plainRead({ entity: 'robot', field: 'site_pos_w' }), [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9, 9, 9,
    ]);
  });

  it('does not fall back in a prefixed model, even for an unknown entity', () => {
    // mjlab attaches terrain with `prefix=""`, indistinguishable from a misspelled
    // name; answering either with the robot's joints is the failure to prevent.
    close(read({ entity: 'terrain', field: 'joint_pos' }), []);
    expect(read({ entity: 'terrain', field: 'root_link_pos_w' })).toBeNull();
  });

  it('returns null for a field it does not implement', () => {
    // Loud (the caller warns and holds) rather than an approximation.
    expect(read({ entity: 'robot', field: 'joint_acc' })).toBeNull();
    expect(isReadableEntityField('joint_acc')).toBe(false);
    expect(isReadableEntityField('joint_pos')).toBe(true);
  });

  it('returns null before the simulation exists', () => {
    expect(createSlotReader(() => null)({ entity: 'robot', field: 'joint_pos' })).toBeNull();
    const empty = createSlotReader(() => ({ mjModel: null, mjData: null }));
    expect(empty({ entity: 'robot', field: 'joint_pos' })).toBeNull();
  });

  it('hands back a fresh array per read', () => {
    const first = read({ entity: 'robot', field: 'joint_pos' })!;
    first[0] = 99;
    close(read({ entity: 'robot', field: 'joint_pos' }), [0.5, 0.5, 0.5, 0.5, 1.25, -0.75]);
  });

  it('re-resolves addresses when the model is replaced', () => {
    let current = context();
    const reader = createSlotReader(() => current);
    close(reader({ entity: 'robot', field: 'joint_pos' }), [0.5, 0.5, 0.5, 0.5, 1.25, -0.75]);

    // A scene reload swaps mjModel; the cached per-entity index must not survive.
    const next = context();
    (next.mjModel as unknown as Mutable).jnt_qposadr = [0, 13, 17, 18, 7];
    (next.mjData as unknown as Mutable).qpos = Float64Array.from(
      { length: 20 },
      (_, i) => i / 100,
    );
    current = next;
    close(reader({ entity: 'robot', field: 'joint_pos' }), [0.13, 0.14, 0.15, 0.16, 0.17, 0.18]);
  });
});

describe('createSlotReader — sensor slots', () => {
  const read = createSlotReader(() => context());

  it('serves a sensor’s own window of sensordata', () => {
    close(read({ sensor: 'robot/imu_lin_vel' }), [1, 2, 3]);
    close(read({ sensor: 'robot/imu_ang_vel' }), [4, 5, 6]);
  });

  it('matches an unprefixed model sensor against the prefixed build name', () => {
    const { mjModel, mjData } = fakeScene();
    // Plain MJCF names it `imu_lin_vel`; the build recorded `robot/imu_lin_vel`.
    const bare = createSlotReader(() => {
      const encoder = new TextEncoder();
      const bytes: number[] = [];
      const adrs = ['imu_lin_vel', 'imu_ang_vel'].map(name => {
        const at = bytes.length;
        bytes.push(...encoder.encode(name), 0);
        return at;
      });
      return {
        mjModel: {
          ...(mjModel as object),
          names: Uint8Array.from(bytes).buffer,
          name_sensoradr: adrs,
        },
        mjData,
      } as unknown as SlotReaderContext;
    });
    close(bare({ sensor: 'robot/imu_lin_vel' }), [1, 2, 3]);
  });

  it('returns null for a sensor the model does not have', () => {
    expect(read({ sensor: 'robot/absent' })).toBeNull();
  });
});

describe('createSlotReader — command state slots', () => {
  it('reads a live command term’s state field', () => {
    const term = { getStateField: (f: string) => (f === 'target_pos' ? new Float32Array([1, 2, 3]) : null) };
    const read = createSlotReader(() => ({
      ...context(),
      commandManager: { getTerm: (name: string) => (name === 'lift_height' ? term : undefined) },
    }));
    close(read({ command: 'lift_height', field: 'target_pos' }), [1, 2, 3]);
    expect(read({ command: 'lift_height', field: 'absent' })).toBeNull();
    expect(read({ command: 'missing', field: 'target_pos' })).toBeNull();
  });

  it('returns null for a command term that exposes no state fields', () => {
    // A native command (e.g. a UI-driven one) has no traced state to read.
    const read = createSlotReader(() => ({
      ...context(),
      commandManager: { getTerm: () => ({ getCommand: () => new Float32Array(3) }) },
    }));
    expect(read({ command: 'velocity', field: 'target_pos' })).toBeNull();
  });
});

describe('createSlotReader — sim slots', () => {
  it('serves a whole raw mjData array as float32', () => {
    const { mjModel, mjData } = fakeScene();
    mjData.act = Float64Array.from([0.25, 0.5, 0.75]);
    const read = createSlotReader(() => ({ mjModel, mjData }) as unknown as SlotReaderContext);
    close(read({ sim: 'act' }), [0.25, 0.5, 0.75]);
  });

  it('wraps a scalar field such as time in a one-element array', () => {
    const { mjModel, mjData } = fakeScene();
    mjData.time = 1.25;
    const read = createSlotReader(() => ({ mjModel, mjData }) as unknown as SlotReaderContext);
    close(read({ sim: 'time' }), [1.25]);
  });

  it('returns null for a field mjData does not carry', () => {
    const read = createSlotReader(() => context());
    expect(read({ sim: 'no_such_field' })).toBeNull();
  });
});

describe('slotDims', () => {
  it('rebuilds the traced rank with batch pinned to 1', () => {
    // site_pos_w is (batch, num_sites, 3) — feeding (1, 6) would be rejected.
    expect(slotDims({ shape: [1, 2, 3], input: 'robot__site_pos_w' }, 6)).toEqual([1, 2, 3]);
    expect(slotDims({ shape: [1], input: 'robot__heading_w' }, 1)).toEqual([1]);
    expect(slotDims({ shape: [1, 12], input: 'robot__joint_pos' }, 12)).toEqual([1, 12]);
  });

  it('falls back to (1, n) without a declared shape', () => {
    expect(slotDims({ input: 'robot__joint_pos' }, 12)).toEqual([1, 12]);
  });

  it('falls back to (1, n) when the model no longer matches the declared shape', () => {
    // Better a shape ORT rejects loudly than one that lies about the data.
    expect(slotDims({ shape: [1, 2, 3], input: 'robot__site_pos_w' }, 9)).toEqual([1, 9]);
  });
});

describe('createSlotReader — structured sensor fields', () => {
  const SCAN = {
    sensor: 'terrain_scan',
    field: 'distances',
    input: 'sensor__terrain_scan_distances',
  };

  it('says so by name when the build emitted no descriptor', () => {
    // No `sensordata` window, so falling through would land on another sensor.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const read = createSlotReader(() => ({ ...context(), mujoco: {} as never }));
    expect(read(SCAN)).toBeNull();
    expect(warn.mock.calls.flat().join(' ')).toContain('terrain_scan');
    warn.mockRestore();
  });

  it('reports unavailable before the WASM module exists', () => {
    // `mj_ray` lives on the module; no module, nothing to cast with.
    const read = createSlotReader(() => context(), {
      raycastSensors: () => ({ terrain_scan: {} as never }),
    });
    expect(read(SCAN)).toBeNull();
  });

  it('re-resolves a raycast sensor when the scene swaps the model', () => {
    // Both scenes call it `terrain_scan`, on their own robot's body. Held across the
    // switch, the first scene's caster looked for a body the new model does not have.
    const modelWith = (body: string) => {
      const names = new TextEncoder().encode(`world\0${body}\0`);
      return {
        model: { names: names.buffer, nbody: 2, name_bodyadr: [0, 6] } as never,
        data: { xpos: new Float64Array([0, 0, 0, 1, 2, 3]), xmat: new Float64Array(18) } as never,
      };
    };
    const scan = (body: string) => ({
      frames: [{ type: 'body' as const, name: body }],
      local_offsets: [[0, 0, 0]],
      local_directions: [[0, 0, -1]],
      ray_alignment: 'yaw' as const,
      max_distance: 5,
    });

    let scene = { ...modelWith('robot/pelvis'), descriptor: scan('robot/pelvis') };
    const read = createSlotReader(
      () => ({ mujoco: {} as never, mjModel: scene.model, mjData: scene.data }),
      { raycastSensors: () => ({ terrain_scan: scene.descriptor as never }) },
    );
    const slot = { sensor: 'terrain_scan', field: 'frame_pos_w', input: 'x' };
    close(read(slot), [1, 2, 3]);

    scene = { ...modelWith('robot/trunk'), descriptor: scan('robot/trunk') };
    close(read(slot), [1, 2, 3]);
  });

  it('still serves a builtin sensor, which carries no field', () => {
    const read = createSlotReader(() => context());
    close(read({ sensor: 'robot/imu_lin_vel' }), [1, 2, 3]);
  });
});
