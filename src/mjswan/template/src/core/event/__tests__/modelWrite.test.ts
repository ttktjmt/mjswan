/**
 * A traced event's `mjModel` reads and writes, against the real MuJoCo WASM.
 *
 * The graph is checked against mjlab in Python; what can go wrong here is where its
 * numbers come from and land: a row resolved to the wrong element, a dof addressed off
 * its joint, a write the MDP switch cannot undo. All of them change how the robot moves
 * without an error, hence a real model with MuJoCo's own names, strides and addresses.
 */
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import type { OnnxTensorLike } from '../../onnx/session';
import { readModelSlot } from '../../onnx/slotReader/model';
import { OnnxEvent, type OnnxEventConfig } from '../OnnxEvent';
import {
  applyModelWrite,
  ModelFieldDefaults,
  setConstKeepingQpos,
  type ModelWriteTarget,
} from '../modelWrite';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

// Unprefixed, as a plain scene's browser model is; the build names elements `robot/…`.
const SCENE = `<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="torso" pos="0 0 1">
      <joint name="root" type="free"/>
      <geom name="torso_collision" type="box" size="0.1 0.1 0.1" mass="2"/>
      <body name="arm" pos="0 0 -0.3">
        <joint name="shoulder" type="ball"/>
        <geom name="arm_collision" type="capsule" size="0.05 0.1" mass="0.3"/>
        <body name="hand" pos="0 0 -0.3">
          <joint name="wrist" type="hinge" axis="0 1 0" damping="0.5"/>
          <geom name="hand_collision" type="sphere" size="0.04" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>`;

const GEOMS = ['floor', 'torso_collision', 'arm_collision', 'hand_collision'];
const HAND_BODY = 3;
// nv: free 6, ball 3, hinge 1, so the wrist is dof 9 and the shoulder's dofs are 6..8.
const WRIST_DOF = 9;

let mujoco: MainModule;
let mjModel: MjModel;
let mjData: MjData;

function friction(geom: string): number[] {
  const i = GEOMS.indexOf(geom);
  return Array.from(mjModel.geom_friction.slice(i * 3, i * 3 + 3) as ArrayLike<number>);
}

function writeTarget(over: Partial<ModelWriteTarget> = {}): ModelWriteTarget {
  return {
    kind: 'model',
    field: 'geom_friction',
    element: 'geom',
    names: ['robot/hand_collision'],
    cells: [[0, 1]],
    outputs: ['model__geom_friction__0'],
    ...over,
  };
}

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  mujoco = await load();
});

beforeEach(() => {
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
  mjData = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(mjModel);
  mujoco.mj_forward(mjModel, mjData);
});

describe('applyModelWrite on an index array', () => {
  // MuJoCo uses jnt_qposadr as a memory offset: a document could move the qpos store.
  it('refuses an integer field and leaves it unchanged', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = Array.from(mjModel.jnt_qposadr as ArrayLike<number>);
    const target = writeTarget({ field: 'jnt_qposadr', element: 'joint', names: ['wrist'], cells: [[0, 0]], outputs: ['o'] });

    expect(applyModelWrite(mjModel, target, { o: new Float32Array([1e9]) })).toBe(false);
    expect(Array.from(mjModel.jnt_qposadr as ArrayLike<number>)).toEqual(before);
    warn.mockRestore();
  });
});

describe('readModelSlot', () => {
  it('serves the named rows in slot order, finding a prefixed name by its bare one', () => {
    const slot = {
      model: 'geom_friction',
      element: 'geom',
      names: ['robot/hand_collision', 'robot/torso_collision'],
      shape: [1, 2, 3],
    };
    const out = readModelSlot(mjModel, slot);
    expect(Array.from(out ?? [])).toEqual(
      [...friction('hand_collision'), ...friction('torso_collision')].map(Math.fround),
    );
  });

  it('addresses a dof by its joint and the offset within it', () => {
    const damping = mjModel.dof_damping as unknown as { [index: number]: number };
    damping[7] = 0.25;
    const slot = {
      model: 'dof_damping',
      element: 'dof',
      names: ['robot/shoulder', 'robot/wrist'],
      offsets: [1, 0],
      shape: [1, 2],
    };
    expect(Array.from(readModelSlot(mjModel, slot) ?? [])).toEqual([0.25, 0.5]);
  });

  it('addresses qpos the same way, by the joint’s qpos block', () => {
    // The ball joint's quaternion starts after the free joint's 7.
    const slot = {
      model: 'qpos0',
      element: 'qpos',
      names: ['shoulder', 'shoulder'],
      offsets: [0, 1],
      shape: [1, 2],
    };
    expect(Array.from(readModelSlot(mjModel, slot) ?? [])).toEqual([1, 0]);
  });

  it('serves the compiled value for a default slot after the live one moved', () => {
    const defaults = new ModelFieldDefaults(mjModel);
    const slot = {
      model: 'geom_friction',
      element: 'geom',
      names: ['hand_collision'],
      default: true,
      shape: [1, 1, 3],
    };
    const compiled = Array.from(readModelSlot(mjModel, slot, defaults) ?? []);
    applyModelWrite(mjModel, writeTarget({ cells: [[0, 0]] }), { model__geom_friction__0: [9] }, defaults);
    expect(Array.from(readModelSlot(mjModel, slot, defaults) ?? [])).toEqual(compiled);
    expect(compiled[0]).not.toBe(9);
  });

  it('serves nothing it cannot place', () => {
    const slot = { model: 'geom_friction', element: 'geom', names: ['hand_collision'], shape: [1, 1, 3] };
    expect(readModelSlot(mjModel, { ...slot, names: ['gripper'] })).toBeNull();
    expect(readModelSlot(mjModel, { ...slot, model: 'geom_nothing' })).toBeNull();
    // A default read with no snapshot store must not quietly serve the live value.
    expect(readModelSlot(mjModel, { ...slot, default: true })).toBeNull();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(readModelSlot(mjModel, { ...slot, shape: [1, 1, 2] })).toBeNull();
    expect(warn).toHaveBeenCalledOnce();
    warn.mockRestore();
  });
});

describe('applyModelWrite', () => {
  it('writes only the cells the target names', () => {
    const before = GEOMS.map(friction);
    expect(applyModelWrite(mjModel, writeTarget(), { model__geom_friction__0: [0.125] })).toBe(true);
    const after = GEOMS.map(friction);
    expect(after[3]).toEqual([before[3][0], 0.125, before[3][2]]);
    expect(after.slice(0, 3)).toEqual(before.slice(0, 3));
  });

  it('records the compiled field first, so restore() undoes the write', () => {
    const compiled = friction('hand_collision');
    const defaults = new ModelFieldDefaults(mjModel);
    applyModelWrite(mjModel, writeTarget(), { model__geom_friction__0: [0.125] }, defaults);
    expect(defaults.restore()).toBe(true);
    expect(friction('hand_collision')).toEqual(compiled);
  });

  it('writes nothing when an element is missing from this model', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = GEOMS.map(friction);
    const target = writeTarget({ names: ['hand_collision', 'gripper'], cells: [[0, 0], [1, 0]] });
    expect(applyModelWrite(mjModel, target, { model__geom_friction__0: [7, 7] })).toBe(false);
    expect(GEOMS.map(friction)).toEqual(before);
    warn.mockRestore();
  });

  it('writes nothing when a cell is past the element’s row', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = GEOMS.map(friction);
    const target = writeTarget({ cells: [[0, 0], [0, 3]] });
    expect(applyModelWrite(mjModel, target, { model__geom_friction__0: [7, 7] })).toBe(false);
    expect(GEOMS.map(friction)).toEqual(before);
    warn.mockRestore();
  });
});

describe('setConstKeepingQpos', () => {
  it('recomputes the derived constants and leaves the state where it was', () => {
    const qpos = mjData.qpos as unknown as { [index: number]: number };
    qpos[0] = 0.5;
    qpos[11] = 0.3;
    const kept = Array.from(mjData.qpos as ArrayLike<number>);
    const mass = mjModel.body_mass as unknown as { [index: number]: number };
    const subtree = (): number => (mjModel.body_subtreemass as ArrayLike<number>)[1];
    const before = subtree();
    mass[HAND_BODY] += 1;

    setConstKeepingQpos(mujoco, mjModel, mjData);

    expect(subtree()).toBeCloseTo(before + 1, 6);
    expect(Array.from(mjData.qpos as ArrayLike<number>)).toEqual(kept);
  });
});

describe('OnnxEvent: model writes', () => {
  const CONFIG: OnnxEventConfig = {
    name: 'hand_mass',
    mode: 'startup',
    onnx: 'event/hand_mass.onnx',
    rand_dim: 0,
    write_targets: [
      {
        kind: 'model',
        field: 'body_mass',
        element: 'body',
        names: ['robot/hand'],
        cells: [[0, 0]],
        outputs: ['model__body_mass__0'],
      },
      {
        kind: 'model',
        field: 'dof_damping',
        element: 'dof',
        names: ['robot/wrist'],
        offsets: [0],
        cells: [[0, 0]],
        outputs: ['model__dof_damping__1'],
      },
    ],
  };

  function session(): { run(): Promise<Record<string, OnnxTensorLike>> } {
    return {
      run: () =>
        Promise.resolve({
          model__body_mass__0: { data: Float32Array.of(0.75), dims: [1, 1] },
          model__dof_damping__1: { data: Float32Array.of(2), dims: [1, 1] },
        }),
    };
  }

  function context(setConst: (m: MjModel, d: MjData) => void) {
    return {
      mjModel,
      mjData,
      mujoco: { mj_setConst: setConst } as unknown as MainModule,
      modelDefaults: new ModelFieldDefaults(mjModel),
    };
  }

  it('applies each output to its field and element', async () => {
    const event = new OnnxEvent(CONFIG, { session: session(), rng: new SeededRng(1) });
    await event.fire(context(vi.fn()));
    expect((mjModel.body_mass as ArrayLike<number>)[HAND_BODY]).toBe(0.75);
    expect((mjModel.dof_damping as ArrayLike<number>)[WRIST_DOF]).toBe(2);
  });

  it('runs mj_setConst, keeping qpos, only when the build says the writes owe it', async () => {
    const setConst = vi.fn((m: MjModel, d: MjData) => mujoco.mj_setConst(m, d));
    const qpos = mjData.qpos as unknown as { [index: number]: number };
    qpos[0] = 0.5;

    await new OnnxEvent(CONFIG, { session: session(), rng: new SeededRng(1) }).fire(
      context(setConst),
    );
    expect(setConst).not.toHaveBeenCalled();

    await new OnnxEvent(
      { ...CONFIG, set_const: true },
      { session: session(), rng: new SeededRng(1) },
    ).fire(context(setConst));
    expect(setConst).toHaveBeenCalledOnce();
    expect(qpos[0]).toBe(0.5);
  });
});
