/**
 * The tubes against MyoSuite's own numbers: `fixtures/tubes.json` is what MyoSuite's
 * `muscle_tubes.py` draws for an arm with a cylinder wrap, two pulleys and a shared
 * measured volume (regenerate with `tests/dump_muscle_fixture.py`).
 */
import { beforeAll, describe, expect, it } from 'vitest';

import fixture from './fixtures/tubes.json';
import { bellyLengths, branchCount, muscleActuators, muscleShapes, muscleTube } from '../tubes';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

let mujoco: MainModule;

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
});

describe('muscleTube', () => {
  it("draws MyoSuite's tubes, pulley branches included, with the shape fixed at qpos0", () => {
    const model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
      .MjModel.from_xml_string(fixture.xml);
    const data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
    mujoco.mj_forward(model, data);

    const actuators = muscleActuators(mujoco, model);
    expect(actuators).toEqual(fixture.actuators);
    const tendons = Int32Array.from(actuators, (a) => model.actuator_trnid[2 * a]);
    const lengths = bellyLengths(data.wrap_xpos, data.wrap_obj, data.ten_wrapadr, data.ten_wrapnum, tendons);
    const shapes = muscleShapes(model, actuators, lengths);
    shapes.forEach((shape, i) => expect(Math.abs(shape.volume / fixture.volume[i] - 1)).toBeLessThan(1e-12));

    data.qpos.set(fixture.qpos);
    mujoco.mj_forward(model, data);
    const [wrapXpos, wrapObj, wrapadr, wrapnum] = [data.wrap_xpos, data.wrap_obj, data.ten_wrapadr, data.ten_wrapnum];
    const positions = new Float64Array(fixture.points.length);
    let vertex = 0;
    tendons.forEach((t, i) => {
      const branches = branchCount(mujoco, model, t);
      vertex = muscleTube(wrapXpos, wrapObj, wrapadr[t], wrapnum[t], branches, shapes[i], fixture.scale, 12, positions, null, vertex);
    });
    expect(3 * vertex).toBe(fixture.points.length);
    let error = 0;
    positions.forEach((v, i) => (error = Math.max(error, Math.abs(v - fixture.points[i]))));
    expect(error).toBeLessThan(1e-9);
    model.delete();
    data.delete();
  });
});
