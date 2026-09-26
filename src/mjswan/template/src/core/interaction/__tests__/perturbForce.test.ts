/**
 * applyDragPull against the real WASM: it must move a body as `mj_applyFT` does for the
 * same force at the same grab point.
 */
import { beforeAll, describe, expect, it } from 'vitest';

import { MJV_PERTURB_STIFFNESS, applyDragPull } from '../perturbForce';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MjvPerturb = import('mujoco').MjvPerturb;

/**
 * A free body whose COM sits 0.4 m off its frame's origin, so a moment taken about `xpos`
 * instead of `xipos` is off by `(xipos - xpos) x F`.
 */
const OFFSET_COM = `<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <freejoint/>
      <geom type="box" pos="0.4 0 0" size="0.05 0.05 0.05" mass="2"/>
    </body>
  </worldbody>
</mujoco>`;

/** The grab, in the body's own frame: a corner of the geom, not the COM and not the origin. */
const GRAB: readonly [number, number, number] = [0.45, 0.05, 0];
const FORCE_SCALE = 250;
/** Pointer target relative to the grab point's rest position, metres. */
const PULL: readonly [number, number, number] = [0.02, -0.03, 0.06];

describe('applyDragPull against the real WASM', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  function load(): { mjModel: MjModel; mjData: MjData; bodyId: number } {
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(OFFSET_COM);
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(
      mjModel,
    );
    // Rotated, or an implementation that never rotates the grab point into the world would pass.
    const half = Math.SQRT1_2;
    mjData.qpos[3] = half;
    mjData.qpos[6] = half; // 90 deg about z
    mujoco.mj_forward(mjModel, mjData);
    const bodyId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, 'link');
    return { mjModel, mjData, bodyId };
  }

  /** The grab point in world coordinates, as MuJoCo itself resolves a body-frame point. */
  function grabInWorld(mjData: MjData, bodyId: number): number[] {
    const out = [0, 0, 0];
    for (let row = 0; row < 3; row++) {
      let v = mjData.xpos[bodyId * 3 + row];
      for (let col = 0; col < 3; col++) v += mjData.xmat[bodyId * 9 + row * 3 + col] * GRAB[col];
      out[row] = v;
    }
    return out;
  }

  function accelerations(mjModel: MjModel, mjData: MjData): number[] {
    mujoco.mj_forward(mjModel, mjData);
    return Array.from(mjData.qacc.slice(0, 6));
  }

  it('moves the body exactly as mj_applyFT at the same point does', () => {
    const pulled = load();
    const perturb = new (mujoco as unknown as { MjvPerturb: new () => MjvPerturb }).MjvPerturb();
    const point = grabInWorld(pulled.mjData, pulled.bodyId);
    applyDragPull(mujoco, pulled.mjModel, pulled.mjData, perturb, {
      bodyId: pulled.bodyId,
      localPoint: GRAB,
      targetPoint: [point[0] + PULL[0], point[1] + PULL[1], point[2] + PULL[2]],
      forceScale: FORCE_SCALE,
    });
    const viaPerturb = accelerations(pulled.mjModel, pulled.mjData);
    perturb.delete();

    const reference = load();
    const refPoint = grabInWorld(reference.mjData, reference.bodyId);
    mujoco.mj_applyFT(
      reference.mjModel,
      reference.mjData,
      PULL.map((d) => d * FORCE_SCALE),
      [0, 0, 0],
      refPoint,
      reference.bodyId,
      reference.mjData.qfrc_applied,
    );
    const viaApplyFT = accelerations(reference.mjModel, reference.mjData);

    // Not `toEqual`: one path routes the wrench through xfrc_applied and the other through
    // qfrc_applied, so they agree to solver precision rather than bit for bit.
    expect(viaPerturb).toHaveLength(6);
    for (let i = 0; i < 6; i++) {
      expect(viaPerturb[i], `qacc[${i}]`).toBeCloseTo(viaApplyFT[i], 6);
    }
    // And the body really did accelerate, so a pair of zeros cannot pass.
    expect(Math.hypot(...viaPerturb)).toBeGreaterThan(1);
  });

  it('pulls with forceScale newtons per metre', () => {
    const { mjModel, mjData, bodyId } = load();
    const perturb = new (mujoco as unknown as { MjvPerturb: new () => MjvPerturb }).MjvPerturb();
    const point = grabInWorld(mjData, bodyId);
    applyDragPull(mujoco, mjModel, mjData, perturb, {
      bodyId,
      localPoint: GRAB,
      targetPoint: [point[0] + PULL[0], point[1] + PULL[1], point[2] + PULL[2]],
      forceScale: FORCE_SCALE,
    });
    const force = [0, 1, 2].map((i) => mjData.xfrc_applied[bodyId * 6 + i]);
    perturb.delete();

    for (let i = 0; i < 3; i++) {
      expect(force[i], `force[${i}]`).toBeCloseTo(PULL[i] * FORCE_SCALE, 6);
    }
    expect(MJV_PERTURB_STIFFNESS).toBe(100);
  });
});
