/**
 * The shared hold, against the real WASM: it has to carry a body, and it has to hand the
 * model back exactly as it compiled — `mj_resetData` restores `eq_active` but not the
 * retargeting, so anything this leaves behind outlives the reset.
 */
import { beforeAll, describe, expect, it } from 'vitest';

import { WeldHold } from '../weldHold';
import { POINTER_ANCHOR_BODY, POINTER_WELD, injectPointerGrabXml } from '../../interaction/grabInject';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

const CUBE_HALF = 0.05;
const SUBSTEPS = 10;

const SCENE = `<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="cube" pos="0 0 ${CUBE_HALF}">
      <freejoint/>
      <geom type="box" size="${CUBE_HALF} ${CUBE_HALF} ${CUBE_HALF}" mass="1"/>
    </body>
  </worldbody>
</mujoco>`;

describe('WeldHold against the real WASM', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  function load() {
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(injectPointerGrabXml(SCENE));
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(mjModel);
    mujoco.mj_forward(mjModel, mjData);
    const id = (name: string) => mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, name);
    const weldId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_EQUALITY.value, POINTER_WELD);
    return { mjModel, mjData, cube: id('cube'), anchor: id(POINTER_ANCHOR_BODY), weldId };
  }

  /** Everything `hold` is allowed to touch on the model, as one comparable value. */
  function modelState(mjModel: MjModel, weldId: number) {
    const row = (array: ArrayLike<number>, stride: number) =>
      Array.from({ length: stride }, (_, i) => array[weldId * stride + i]);
    return {
      obj1: mjModel.eq_obj1id[weldId],
      obj2: mjModel.eq_obj2id[weldId],
      data: row(mjModel.eq_data, mujoco.mjNEQDATA),
      solref: row(mjModel.eq_solref, mujoco.mjNREF),
    };
  }

  function moveAnchor(mjModel: MjModel, mjData: MjData, anchor: number, to: number[]): void {
    const mocapId = mjModel.body_mocapid[anchor];
    for (let i = 0; i < 3; i++) mjData.mocap_pos[mocapId * 3 + i] = to[i];
    mjData.mocap_quat[mocapId * 4] = 1;
    for (let i = 1; i < 4; i++) mjData.mocap_quat[mocapId * 4 + i] = 0;
  }

  it('carries a body and gives it back on release', () => {
    const { mjModel, mjData, cube, anchor } = load();
    const weld = new WeldHold();
    weld.bind(mujoco, mjModel, [POINTER_WELD]);
    expect(weld.has(POINTER_WELD)).toBe(true);

    moveAnchor(mjModel, mjData, anchor, [0, 0, CUBE_HALF]);
    mujoco.mj_forward(mjModel, mjData);
    expect(weld.hold(mjModel, mjData, POINTER_WELD, anchor, cube)).toBe(true);
    expect(weld.holderOf(cube)).toBe(POINTER_WELD);

    // Lift over half a second. The floor is right there, so a weld that did not take
    // would leave the cube resting rather than following.
    for (let k = 1; k <= 25; k++) {
      moveAnchor(mjModel, mjData, anchor, [0, 0, CUBE_HALF + 0.4 * (k / 25)]);
      for (let s = 0; s < SUBSTEPS; s++) mujoco.mj_step(mjModel, mjData);
    }
    expect(mjData.xpos[cube * 3 + 2], 'carried').toBeGreaterThan(0.3);

    weld.release(mjData, POINTER_WELD);
    expect(weld.holderOf(cube)).toBeNull();
    for (let s = 0; s < 40 * SUBSTEPS; s++) mujoco.mj_step(mjModel, mjData);
    expect(mjData.xpos[cube * 3 + 2], 'dropped').toBeLessThan(0.1);
  });

  it('leaves the model exactly as it compiled', () => {
    const { mjModel, mjData, cube, anchor, weldId } = load();
    const weld = new WeldHold();
    weld.bind(mujoco, mjModel, [POINTER_WELD]);
    const compiled = modelState(mjModel, weldId);

    moveAnchor(mjModel, mjData, anchor, [0.2, 0, 0.4]);
    mujoco.mj_forward(mjModel, mjData);
    weld.hold(mjModel, mjData, POINTER_WELD, anchor, cube, { torqueScale: 0.25, solrefTime: 0.05 });
    // A hold that changed nothing on the model would make the restore check meaningless.
    expect(modelState(mjModel, weldId)).not.toEqual(compiled);
    expect(mjData.eq_active[weldId]).toBe(1);

    weld.restore(mjModel, mjData);
    expect(modelState(mjModel, weldId)).toEqual(compiled);
    expect(mjData.eq_active[weldId]).toBe(0);
    expect(weld.holderOf(cube)).toBeNull();
  });

  it('refuses a body another slot is already holding', () => {
    const { mjModel, mjData, cube, anchor } = load();
    const weld = new WeldHold();
    weld.bind(mujoco, mjModel, [POINTER_WELD]);
    mujoco.mj_forward(mjModel, mjData);
    expect(weld.hold(mjModel, mjData, POINTER_WELD, anchor, cube)).toBe(true);
    // The same slot may re-aim at what it already has; another one may not take it.
    expect(weld.hold(mjModel, mjData, POINTER_WELD, anchor, cube)).toBe(true);
    expect(weld.hold(mjModel, mjData, 'mjswan_xr0_grab', anchor, cube)).toBe(false);
  });

  it('has no slots at all in a scene that was never injected', () => {
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(SCENE);
    const weld = new WeldHold();
    weld.bind(mujoco, mjModel, [POINTER_WELD]);
    expect(weld.has(POINTER_WELD)).toBe(false);
  });
});
