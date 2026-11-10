/**
 * Startup model-field randomization, against the real MuJoCo WASM.
 *
 * These events perturb `mjModel` rather than `mjData`, so nothing about them shows
 * up in a graph and nothing in the Python parity harness reaches them. They are
 * also the quietest kind of wrong: a friction coefficient written to the wrong
 * geom, or an axis clobbered that another event owns, changes how the robot walks
 * without any error. Hence a real model, with names and strides that are MuJoCo's
 * rather than a fixture's.
 */
import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';

import { applyModelFieldDr, isModelFieldDrConfig, type ModelFieldDrConfig } from '../modelFieldDr';
import { ModelFieldDefaults } from '../modelWrite';
import { SeededRng } from '../../rng';

type MainModule = import('mujoco').MainModule;

const SCENE = `<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot/torso" pos="0 0 1">
      <joint type="free"/>
      <geom name="robot/torso_collision" type="box" size="0.1 0.1 0.1" mass="2"/>
      <body name="robot/foot" pos="0 0 -0.3">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="robot/foot_collision" type="sphere" size="0.05" mass="0.3"/>
      </body>
    </body>
  </worldbody>
</mujoco>`;

let mujoco: MainModule;
let mjModel: import('mujoco').MjModel;
let mjData: import('mujoco').MjData;

function freshModel(): void {
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
}

function friction(name: string): [number, number, number] {
  const names = ['floor', 'robot/torso_collision', 'robot/foot_collision'];
  const i = names.indexOf(name);
  const f = mjModel.geom_friction;
  return [f[i * 3], f[i * 3 + 1], f[i * 3 + 2]];
}

function config(over: Partial<ModelFieldDrConfig> = {}): ModelFieldDrConfig {
  const operation = over.operation ?? 'abs';
  return {
    name: 'foot_friction',
    kind: 'model_field',
    field: 'geom_friction',
    entity_type: 'geom',
    entity_names: ['robot/foot_collision'],
    axis_ranges: { '0': [0.7, 0.7] }, // degenerate range: one exact value
    operation,
    distribution: 'uniform',
    shared_random: false,
    // As the build emits: `abs` overwrites, the others use the compiled default.
    uses_defaults: operation === 'add' || operation === 'scale',
    set_const: false,
    ...over,
  };
}

/** Apply one event, with a fresh defaults snapshot unless the test shares one. */
function apply(
  cfg: ModelFieldDrConfig,
  rng: SeededRng,
  defaults = new ModelFieldDefaults(mjModel),
): boolean {
  return applyModelFieldDr(mujoco, mjModel, mjData, cfg, rng, defaults);
}

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  mujoco = await load();
});

beforeEach(() => {
  // A fresh model per test: these write it, and the writes must not leak sideways.
  freshModel();
  mjData = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(mjModel);
  mujoco.mj_forward(mjModel, mjData);
});

describe('applyModelFieldDr', () => {
  it('refuses an integer field and leaves it unchanged', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = Array.from(mjModel.body_jntadr as ArrayLike<number>);
    const cfg = config({ field: 'body_jntadr', entity_type: 'body', entity_names: ['robot/foot'], axis_ranges: { '0': [1e9, 1e9] } });

    expect(apply(cfg, new SeededRng(1))).toBe(false);
    expect(Array.from(mjModel.body_jntadr as ArrayLike<number>)).toEqual(before);
    warn.mockRestore();
  });

  it('writes the named geom and leaves every other one alone', () => {
    const before = friction('floor');
    expect(apply(config(), new SeededRng(1))).toBe(true);
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(0.7, 6);
    // The scoping is the point: mjlab targets fingertip or foot geoms, not the world.
    expect(friction('floor')).toEqual(before);
    expect(friction('robot/torso_collision')[0]).not.toBeCloseTo(0.7, 6);
  });

  it('writes only the targeted axis, so sibling events compose', () => {
    // Lift randomizes friction axes as three events; writing the triple would erase two.
    const before = friction('robot/foot_collision');
    apply(config({ axis_ranges: { '1': [0.05, 0.05] } }), new SeededRng(1));
    const after = friction('robot/foot_collision');
    expect(after[1]).toBeCloseTo(0.05, 6);
    expect(after[0]).toBeCloseTo(before[0], 6);
    expect(after[2]).toBeCloseTo(before[2], 6);
  });

  it('applies each operation against the compiled base', () => {
    const base = friction('robot/foot_collision')[0];
    apply(config({ operation: 'scale', axis_ranges: { '0': [2, 2] } }), new SeededRng(1));
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(base * 2, 6);

    freshModel();
    apply(config({ operation: 'add', axis_ranges: { '0': [0.25, 0.25] } }), new SeededRng(1));
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(base + 0.25, 6);
  });

  it('does not let two add events on one axis accumulate', () => {
    // `add`/`scale` read the compile-time default, so a second event on one axis offsets
    // the same base. Reading the live field would drift by however many events hit it.
    const base = friction('robot/foot_collision')[0];
    const defaults = new ModelFieldDefaults(mjModel);
    const rng = new SeededRng(1);
    apply(config({ operation: 'add', axis_ranges: { '0': [0.25, 0.25] } }), rng, defaults);
    apply(config({ operation: 'add', axis_ranges: { '0': [0.1, 0.1] } }), rng, defaults);
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(base + 0.1, 6);
  });

  it('gives every geom the same draw under shared_random', () => {
    // mjlab's foot friction shares one coefficient across both feet.
    const names = ['robot/foot_collision', 'robot/torso_collision'];
    apply(
      config({ entity_names: names, axis_ranges: { '0': [0.2, 1.8] }, shared_random: true }),
      new SeededRng(7),
    );
    expect(friction(names[0])[0]).toBeCloseTo(friction(names[1])[0], 6);
  });

  it('gives them different draws without it', () => {
    const names = ['robot/foot_collision', 'robot/torso_collision'];
    apply(config({ entity_names: names, axis_ranges: { '0': [0.2, 1.8] } }), new SeededRng(7));
    expect(friction(names[0])[0]).not.toBeCloseTo(friction(names[1])[0], 3);
  });

  it('is reproducible from the seed', () => {
    // A recorded session replays, so the draws can only come from the seeded stream.
    const cfg = config({ axis_ranges: { '0': [0.2, 1.8] } });
    apply(cfg, new SeededRng(42));
    const first = friction('robot/foot_collision')[0];

    freshModel();
    apply(cfg, new SeededRng(42));
    expect(friction('robot/foot_collision')[0]).toBe(first);
  });

  it('draws log-uniformly when asked', () => {
    // A [0.01, 1] log range clusters low; a uniform draw would average near 0.5.
    const rng = new SeededRng(3);
    const samples: number[] = [];
    for (let i = 0; i < 40; i++) {
      freshModel();
      apply(config({ distribution: 'log_uniform', axis_ranges: { '0': [0.01, 1] } }), rng);
      samples.push(friction('robot/foot_collision')[0]);
    }
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(0.01);
      expect(s).toBeLessThanOrEqual(1);
    }
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(mean).toBeLessThan(0.4);
  });

  it('calls mj_setConst for an inertial field', () => {
    // `body_ipos` feeds precomputed constants, so skipping this half-applies it.
    const setConst = vi.spyOn(mujoco, 'mj_setConst');
    apply(
      config({
        name: 'base_com',
        field: 'body_ipos',
        entity_type: 'body',
        entity_names: ['robot/torso'],
        axis_ranges: { '0': [0.01, 0.01] },
        operation: 'add',
        set_const: true,
      }),
      new SeededRng(1),
    );
    expect(setConst).toHaveBeenCalledWith(mjModel, mjData);
    setConst.mockRestore();
  });

  it('skips and says so when a name or field is not in this model', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    expect(apply(config({ entity_names: ['robot/absent'] }), new SeededRng(1))).toBe(false);
    expect(apply(config({ field: 'no_such_field' }), new SeededRng(1))).toBe(false);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });

  it('skips an axis past the field width instead of writing a neighbour', () => {
    // `geom_friction` is 3 wide; axis 5 would land in the next geom's slot.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = friction('robot/torso_collision');
    apply(config({ axis_ranges: { '5': [9, 9] } }), new SeededRng(1));
    expect(friction('robot/torso_collision')).toEqual(before);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe('geom_size and the bounds that follow it', () => {
  /** `[size, rbound, aabbHalf]` for one geom, by name. */
  function bounds(name: string): [number[], number, number[]] {
    const names = ['floor', 'robot/torso_collision', 'robot/foot_collision'];
    const i = names.indexOf(name);
    const size = mjModel.geom_size as ArrayLike<number>;
    const rbound = mjModel.geom_rbound as ArrayLike<number>;
    const aabb = mjModel.geom_aabb as ArrayLike<number>;
    return [
      [size[i * 3], size[i * 3 + 1], size[i * 3 + 2]],
      rbound[i],
      [aabb[i * 6 + 3], aabb[i * 6 + 4], aabb[i * 6 + 5]],
    ];
  }

  const sizeConfig = (over: Partial<ModelFieldDrConfig> = {}): ModelFieldDrConfig =>
    config({
      name: 'randomize_ball_size',
      field: 'geom_size',
      entity_names: ['robot/foot_collision'],
      axis_ranges: { '0': [0.125, 0.125] },
      operation: 'abs',
      uses_defaults: false,
      recompute_bounds: true,
      ...over,
    });

  it("writes a sphere's radius and the broadphase bound with it", () => {
    // Without the recompute the broadphase keeps 0.05 and the ball stops colliding
    // at its own surface.
    expect(apply(sizeConfig(), new SeededRng(1))).toBe(true);
    const [size, rbound, aabbHalf] = bounds('robot/foot_collision');
    expect(size[0]).toBeCloseTo(0.125, 6);
    expect(rbound).toBeCloseTo(0.125, 6);
    expect(aabbHalf).toEqual([0.125, 0.125, 0.125]);
  });

  it("leaves an untargeted geom's bounds as compiled", () => {
    const before = bounds('robot/torso_collision');
    apply(sizeConfig(), new SeededRng(1));
    expect(bounds('robot/torso_collision')).toEqual(before);
  });

  it("uses the box's own bound formula, not the sphere's", () => {
    apply(
      sizeConfig({
        entity_names: ['robot/torso_collision'],
        axis_ranges: { '0': [2, 2], '1': [2, 2], '2': [2, 2] },
        operation: 'scale',
        uses_defaults: true,
      }),
      new SeededRng(1),
    );
    const [size, rbound, aabbHalf] = bounds('robot/torso_collision');
    // 0.1 half-extents doubled; a box's bounding sphere is its diagonal.
    expect(size).toEqual([0.2, 0.2, 0.2]);
    expect(rbound).toBeCloseTo(Math.sqrt(3) * 0.2, 6);
    expect(aabbHalf).toEqual([0.2, 0.2, 0.2]);
  });

  it('leaves the bounds alone for a geom whose size does not define them', () => {
    // The build refuses a plane, so the runtime only warns rather than inventing one.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const before = bounds('floor');
    apply(sizeConfig({ entity_names: ['floor'] }), new SeededRng(1));
    expect(bounds('floor')[1]).toBe(before[1]);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });

  it('does not touch the bounds when the build did not ask', () => {
    const before = bounds('robot/foot_collision');
    apply(sizeConfig({ recompute_bounds: false }), new SeededRng(1));
    expect(bounds('robot/foot_collision')[0][0]).toBeCloseTo(0.125, 6);
    // Stale on purpose: this is what mjlab's own `geom_friction` path leaves behind.
    expect(bounds('robot/foot_collision')[1]).toBeCloseTo(before[1], 6);
  });
});

describe('isModelFieldDrConfig', () => {
  it('separates it from a traced event', () => {
    expect(isModelFieldDrConfig(config())).toBe(true);
    expect(isModelFieldDrConfig({ name: 'push_robot', mode: 'interval', onnx: 'x.onnx' })).toBe(
      false,
    );
    expect(isModelFieldDrConfig({ name: 'x', native: true, reason: 'nope' })).toBe(false);
  });
});

describe('ModelFieldDefaults across an MDP switch (ADR 0006 §9)', () => {
  it('restores every touched field to its compiled value', () => {
    const compiled = friction('robot/foot_collision')[0];
    const defaults = new ModelFieldDefaults(mjModel);
    apply(config({ operation: 'scale', axis_ranges: { '0': [3, 3] } }), new SeededRng(1), defaults);
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(compiled * 3, 6);

    expect(defaults.restore()).toBe(true);
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(compiled, 6);
  });

  it('has nothing to restore before anything was touched', () => {
    expect(new ModelFieldDefaults(mjModel).restore()).toBe(false);
  });

  it('lets a second startup pass compose against the compiled value, not the first pass', () => {
    // The switch sequence: restore, then the incoming MDP's startup over the compiled
    // model. Without the restore the second scale would land on 3x, giving 9x.
    const compiled = friction('robot/foot_collision')[0];
    const defaults = new ModelFieldDefaults(mjModel);
    const scale = config({ operation: 'scale', axis_ranges: { '0': [3, 3] } });
    apply(scale, new SeededRng(1), defaults);
    defaults.restore();
    apply(scale, new SeededRng(1), defaults);
    expect(friction('robot/foot_collision')[0]).toBeCloseTo(compiled * 3, 6);
  });

  it('restores the geom bounds a size randomization recomputed', () => {
    const names = ['floor', 'robot/torso_collision', 'robot/foot_collision'];
    const i = names.indexOf('robot/foot_collision');
    const rbound = mjModel.geom_rbound as unknown as { [index: number]: number };
    const compiledBound = rbound[i];
    const defaults = new ModelFieldDefaults(mjModel);
    apply(
      config({
        name: 'foot_size',
        field: 'geom_size',
        axis_ranges: { '0': [2, 2] },
        operation: 'scale',
        recompute_bounds: true,
      }),
      new SeededRng(1),
      defaults,
    );
    expect(rbound[i]).toBeCloseTo(compiledBound * 2, 6);
    defaults.restore();
    expect(rbound[i]).toBeCloseTo(compiledBound, 6);
  });
});
