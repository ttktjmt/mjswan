/**
 * The throwable boxes, against the real WASM.
 *
 * Two things here would break quietly rather than loudly. Injecting bodies changes what
 * `buildEntityIndex` counts as an entity's — but only on a model that is not
 * entity-prefixed, which is exactly the gate the runtime applies. And re-massing a box
 * needs `mj_setConst`, which puts `qpos` back to `qpos0`: forget to carry the state across
 * it and every throw teleports the scene.
 */
import { beforeAll, describe, expect, it } from 'vitest';

import { DEFAULT_SPAWN_POOL, SpawnPool, injectSpawnPoolXml } from '../spawnPool';
import { buildEntityIndex } from '../../onnx/slotReader/indexing';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** `prefix` mirrors what mjlab's `attach` does to every name it brings in. */
const scene = (prefixed: boolean): string => {
  const p = prefixed ? 'robot/' : '';
  return `<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 .1"/>
    <body name="${p}torso" pos="0 0 0.6">
      <freejoint name="${p}root"/>
      <geom name="${p}torso_g" type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.06" mass="5"/>
      <body name="${p}arm" pos="0 0 0.2">
        <joint name="${p}shoulder" type="hinge" axis="0 1 0"/>
        <geom name="${p}arm_g" type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>`;
};

const POOL = 4;

describe('SpawnPool against the real WASM', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  const compile = (xml: string): MjModel =>
    (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(xml);

  const dataFor = (mjModel: MjModel): MjData => {
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(mjModel);
    mujoco.mj_forward(mjModel, mjData);
    return mjData;
  };

  const elements = (mjModel: MjModel) => {
    const index = buildEntityIndex(mjModel, 'robot');
    return { bodyIds: index.bodyIds, geomIds: index.geomIds, qposAdr: index.qposAdr };
  };

  const jointAddresses = (mjModel: MjModel): number[] =>
    Array.from({ length: mjModel.njnt }, (_, j) => mjModel.jnt_qposadr[j]);

  it('is invisible to an entity-prefixed model, and visible to a plain one', () => {
    const prefixed = compile(scene(true));
    const prefixedWithPool = compile(injectSpawnPoolXml(scene(true), POOL));
    expect(elements(prefixedWithPool)).toEqual(elements(prefixed));

    // The negative half is the point: this is why the runtime declines to inject into an
    // unprefixed model that arrived with a policy. A traced graph's input is a fixed
    // width, so an entity that grew four bodies no longer fits it.
    const plain = compile(scene(false));
    const plainWithPool = compile(injectSpawnPoolXml(scene(false), POOL));
    expect(elements(plainWithPool).bodyIds.length).toBe(elements(plain).bodyIds.length + POOL);
  });

  it('leaves every existing joint at the address it had', () => {
    for (const prefixed of [true, false]) {
      const before = compile(scene(prefixed));
      const after = compile(injectSpawnPoolXml(scene(prefixed), POOL));
      const addresses = jointAddresses(before);
      expect(jointAddresses(after).slice(0, addresses.length), `prefixed=${prefixed}`).toEqual(
        addresses,
      );
      expect(after.nq).toBe(before.nq + POOL * 7);
    }
  });

  it('throws a box at the size and density it was given', () => {
    const mjModel = compile(injectSpawnPoolXml(scene(true), POOL));
    const mjData = dataFor(mjModel);
    const pool = new SpawnPool();
    pool.bind(mujoco, mjModel);
    expect(pool.size).toBe(POOL);

    const size = 0.2;
    const density = 600;
    const bodyId = pool.materialize(
      mujoco,
      mjModel,
      mjData,
      { size, density },
      { position: [1, 0, 3], velocity: [0, 0, 0] },
    );
    expect(bodyId).toBeGreaterThan(0);
    expect(mjModel.body_mass[bodyId]).toBeCloseTo(density * 8 * size ** 3, 6);
    expect(mjModel.geom_size[mjModel.body_geomadr[bodyId] * 3]).toBeCloseTo(size, 9);

    // Mass is inert until `mj_setConst`, so check the dynamics rather than the field: a
    // net 1g upward force on top of gravity should accelerate it at exactly g.
    const mass = mjModel.body_mass[bodyId];
    mjData.xfrc_applied[bodyId * 6 + 2] = 2 * mass * 9.81;
    mujoco.mj_forward(mjModel, mjData);
    const dof = mjModel.jnt_dofadr[mjModel.body_jntadr[bodyId]];
    expect(mjData.qacc[dof + 2]).toBeCloseTo(9.81, 2);
  });

  it('does not disturb the rest of the sim when it re-masses a box', () => {
    const mjModel = compile(injectSpawnPoolXml(scene(true), POOL));
    const mjData = dataFor(mjModel);
    const pool = new SpawnPool();
    pool.bind(mujoco, mjModel);
    pool.park(mjData);
    for (let i = 0; i < 200; i++) mujoco.mj_step(mjModel, mjData);

    const torso = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, 'robot/torso');
    const before = [0, 1, 2].map((i) => mjData.qpos[i]);
    const velocity = [0, 1, 2].map((i) => mjData.qvel[i]);
    expect(mjData.xpos[torso * 3 + 2]).toBeGreaterThan(0);

    pool.materialize(
      mujoco,
      mjModel,
      mjData,
      { size: 0.3, density: 900 },
      { position: [2, 0, 2], velocity: [0, 0, 0] },
    );

    // `mj_setConst` would have put the robot back at qpos0 — 0.6 m up and level.
    for (let i = 0; i < 3; i++) {
      expect(mjData.qpos[i], `qpos[${i}]`).toBeCloseTo(before[i], 12);
      expect(mjData.qvel[i], `qvel[${i}]`).toBeCloseTo(velocity[i], 12);
    }
  });

  it('keeps the boxes it has not thrown parked, and parks them all on reset', () => {
    const mjModel = compile(injectSpawnPoolXml(scene(true), POOL));
    const mjData = dataFor(mjModel);
    const pool = new SpawnPool();
    pool.bind(mujoco, mjModel);
    pool.park(mjData);

    const thrown = pool.materialize(
      mujoco,
      mjModel,
      mjData,
      { size: 0.06, density: 400 },
      { position: [0, 0, 1], velocity: [0, 0, 0] },
    );
    // Two seconds of gravity. A free joint left alone is a free joint falling.
    for (let i = 0; i < 1000; i++) {
      pool.holdIdle(mjData);
      mujoco.mj_step(mjModel, mjData);
    }
    for (const bodyId of pool.bodyIds()) {
      if (bodyId === thrown) continue;
      expect(mjData.xpos[bodyId * 3 + 2], `parked body ${bodyId}`).toBeCloseTo(100, 6);
    }
    expect(mjData.xpos[thrown * 3 + 2], 'thrown').toBeLessThan(1);

    pool.park(mjData);
    mujoco.mj_forward(mjModel, mjData);
    for (const bodyId of pool.bodyIds()) {
      expect(mjData.xpos[bodyId * 3 + 2], `reset body ${bodyId}`).toBeCloseTo(100, 9);
    }
  });

  it('carries no boxes when the scene declared none', () => {
    const mjModel = compile(injectSpawnPoolXml(scene(true), 0));
    const pool = new SpawnPool();
    pool.bind(mujoco, mjModel);
    expect(pool.size).toBe(0);
    expect(
      pool.materialize(
        mujoco,
        mjModel,
        dataFor(mjModel),
        { size: 0.06, density: 400 },
        { position: [0, 0, 1], velocity: [0, 0, 0] },
      ),
    ).toBe(0);
    expect(DEFAULT_SPAWN_POOL).toBeGreaterThan(0);
  });
});
