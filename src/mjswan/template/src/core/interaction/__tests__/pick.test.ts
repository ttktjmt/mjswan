/** What lies under a point of the canvas, cast by MuJoCo itself through a camera looking down. */
import * as THREE from 'three';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { SurfacePicker } from '../pick';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

let mujoco: MainModule;
let model: MjModel;
let data: MjData;
let picker: SurfacePicker;

// An infinite floor in a material, a crate on it turned a quarter about z, and a slab over
// the crate in group 3, which the scene does not draw: the ray must pass through it.
const XML = `<mujoco>
  <asset><material name="tile" rgba="0.2 0.3 0.4 1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" material="tile" rgba="0.9 0.9 0.9 1"/>
    <geom name="slab" type="box" size="0.5 0.5 0.05" pos="0 0 2" group="3"/>
    <body name="crate" pos="0 0 0.5" euler="0 0 90">
      <geom name="crate" type="box" size="0.2 0.2 0.2" rgba="0.8 0.4 0.3 1"/>
    </body>
  </worldbody>
</mujoco>`;

const RECT = { left: 10, top: 20, width: 100, height: 100 };
const CENTRE = [RECT.left + 50, RECT.top + 50] as const;

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
  model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(XML);
  data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
  mujoco.mj_forward(model, data);
  picker = new SurfacePicker(mujoco);
});

afterAll(() => picker.dispose());

/** A camera at MuJoCo height `z` over the origin, looking straight down (or up). */
function camera(z: number, down = true): THREE.PerspectiveCamera {
  const cam = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  cam.position.set(0, z, 0);
  cam.up.set(0, 0, -1);
  cam.lookAt(0, down ? z - 1 : z + 1, 0);
  cam.updateMatrixWorld(true);
  return cam;
}

const geom = (name: string) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name);
const round = (v: number[]) => v.map((x) => Number(x.toFixed(4)) + 0);

describe('SurfacePicker', () => {
  it("lands on the crate's top under the centre, through a slab the scene does not draw", () => {
    const hit = picker.pick(model, data, camera(5), RECT, ...CENTRE);
    expect(hit).not.toBeNull();
    expect(round(hit!.point)).toEqual([0, 0, 0.7]);
    expect(round(hit!.normal)).toEqual([0, 0, 1]);
    expect(hit!.geom).toBe(geom('crate'));
    expect(hit!.body).toBe(model.geom_bodyid[geom('crate')]);
  });

  it('lands on the infinite floor beside it, in MuJoCo axes', () => {
    // Up and left on the canvas, with the camera's up along MuJoCo's y: +y and -x.
    const hit = picker.pick(model, data, camera(5), RECT, RECT.left + 10, RECT.top + 10);
    expect(hit).not.toBeNull();
    expect(hit!.geom).toBe(geom('floor'));
    expect(hit!.body).toBe(0);
    expect(round([hit!.point[2]])).toEqual([0]);
    expect(hit!.point[0]).toBeLessThan(-1);
    expect(hit!.point[1]).toBeGreaterThan(1);
    expect(round(hit!.normal)).toEqual([0, 0, 1]);
  });

  it('says what it hit: its name, type, size, colour and where it is now', () => {
    const hit = picker.pick(model, data, camera(5), RECT, ...CENTRE)!;
    expect(hit.name).toBe('crate');
    expect(hit.type).toBe('box');
    expect(round(hit.size)).toEqual([0.2, 0.2, 0.2]);
    expect(round(hit.rgba)).toEqual([0.8, 0.4, 0.3, 1]);
    expect(round(hit.pos)).toEqual([0, 0, 0.5]);
    expect(round(hit.quat)).toEqual(round([Math.SQRT1_2, 0, 0, Math.SQRT1_2]));

    // The scene draws a geom in its material's colour.
    const floor = picker.pick(model, data, camera(5), RECT, RECT.left + 10, RECT.top + 10)!;
    expect(floor.type).toBe('plane');
    expect(round(floor.rgba)).toEqual([0.2, 0.3, 0.4, 1]);
  });

  it('answers null over empty sky, and for a canvas with no size', () => {
    expect(picker.pick(model, data, camera(5, false), RECT, ...CENTRE)).toBeNull();
    expect(picker.pick(model, data, camera(5), { ...RECT, width: 0 }, ...CENTRE)).toBeNull();
  });
});
