/**
 * The single writer of `xfrc_applied`, against the real WASM: a shove has to deliver the
 * impulse it says it does, and it has to stop delivering it on the next step.
 */
import { beforeAll, describe, expect, it } from 'vitest';

import { InteractionWrench } from '../wrench';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MjvPerturb = import('mujoco').MjvPerturb;

const MASS = 2.5;
/** No gravity and no floor: whatever momentum the body ends with, the shove put there. */
const FREE_BODY = `<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="crate" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.08 0.08 0.08" mass="${MASS}"/>
    </body>
  </worldbody>
</mujoco>`;

const DECIMATION = 20;
const CONTROL_DT = 0.02;
const IMPULSE = 6;
/** Hits a top corner, so a wrong moment arm shows up as spin rather than cancelling out. */
const CONTACT_POINT: [number, number, number] = [0.08, 0.08, 1.08];
const DIRECTION: [number, number, number] = [0, 0, -1];

describe('InteractionWrench against the real WASM', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  function load(): { mjModel: MjModel; mjData: MjData; bodyId: number } {
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(FREE_BODY);
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(mjModel);
    mujoco.mj_forward(mjModel, mjData);
    return {
      mjModel,
      mjData,
      bodyId: mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, 'crate'),
    };
  }

  function step(mjModel: MjModel, mjData: MjData): void {
    for (let i = 0; i < DECIMATION; i++) mujoco.mj_step(mjModel, mjData);
  }

  it('delivers the impulse it was asked for, over exactly one control step', () => {
    const { mjModel, mjData, bodyId } = load();
    const wrench = new InteractionWrench();
    const newtons = IMPULSE / CONTROL_DT;

    wrench.begin(mjData);
    wrench.push(mujoco, mjData, bodyId, CONTACT_POINT, [
      DIRECTION[0] * newtons,
      DIRECTION[1] * newtons,
      DIRECTION[2] * newtons,
    ]);
    step(mjModel, mjData);

    // A free joint's first three dofs are the body's linear velocity in world axes.
    const momentum = [0, 1, 2].map((i) => mjData.qvel[i] * MASS);
    expect(momentum[2], 'along the shove').toBeCloseTo(-IMPULSE, 2);
    expect(Math.hypot(momentum[0], momentum[1]), 'across it').toBeLessThan(IMPULSE * 0.02);
    // Off-centre, so it has to be spinning too.
    expect(Math.hypot(mjData.qvel[3], mjData.qvel[4], mjData.qvel[5])).toBeGreaterThan(0.1);

    // The next step takes the force away: a shove is one step, not a held push.
    const spun: number[] = Array.from(mjData.qvel.slice(0, 6) as ArrayLike<number>);
    wrench.begin(mjData);
    expect(Array.from(mjData.xfrc_applied.slice(bodyId * 6, bodyId * 6 + 6))).toEqual([
      0, 0, 0, 0, 0, 0,
    ]);
    step(mjModel, mjData);
    for (let i = 0; i < 6; i++) {
      expect(mjData.qvel[i], `qvel[${i}] coasts`).toBeCloseTo(spun[i], 6);
    }
  });

  // The arrow is drawn from this number, so it has to be the force the body is under.
  it('reports the force a pull applied, after the clamp', () => {
    const { mjModel, mjData, bodyId } = load();
    const wrench = new InteractionWrench();
    const perturb = new (mujoco as unknown as { MjvPerturb: new () => MjvPerturb }).MjvPerturb();
    const pull = (target: [number, number, number], maxForce: number) => {
      wrench.begin(mjData);
      return wrench.pull(
        mujoco,
        mjModel,
        mjData,
        perturb,
        { bodyId, localPoint: [0, 0, 0], targetPoint: target, forceScale: 100 },
        maxForce,
      );
    };
    const applied = () =>
      Math.hypot(...Array.from(mjData.xfrc_applied.slice(bodyId * 6, bodyId * 6 + 3) as ArrayLike<number>));

    // 0.3 m at 100 N/m: under the clamp, so the spring's own 30 N.
    const light = pull([0.3, 0, 1], 500);
    expect(light.saturated).toBe(false);
    expect(light.force).toBeCloseTo(30, 3);
    expect(light.force).toBeCloseTo(applied(), 9);

    // 5 m would be 500 N; the clamp holds it to 40.
    const heavy = pull([5, 0, 1], 40);
    expect(heavy.saturated).toBe(true);
    expect(heavy.force).toBeCloseTo(40, 6);
    expect(heavy.force).toBeCloseTo(applied(), 9);
    perturb.delete();
  });

  it('leaves rows it never wrote alone', () => {
    const { mjModel, mjData, bodyId } = load();
    const wrench = new InteractionWrench();
    // Something outside the interaction layer (a custom plugin, say) parks a wrench on
    // the world body. The drag this replaced zeroed the entire array every step.
    mjData.xfrc_applied[2] = 1.5;
    wrench.begin(mjData);
    wrench.push(mujoco, mjData, bodyId, CONTACT_POINT, [0, 0, -1]);
    wrench.begin(mjData);
    expect(mjData.xfrc_applied[2]).toBe(1.5);
    expect(mjData.xfrc_applied[bodyId * 6 + 2]).toBe(0);
    expect(mjModel.nbody).toBe(2);
  });
});
