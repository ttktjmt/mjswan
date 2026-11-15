import * as THREE from 'three';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { createTendonMeshes, createTendonState, snapshotTendons, updateTendonRendering } from '../tendons';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** The only tendon, over a cylinder: 4 path points from 3 wrap objects. */
const WRAP = `<mujoco><worldbody><body pos="0 0 1">
  <joint type="slide" axis="0 0 1"/>
  <geom name="c" type="cylinder" size="0.02 0.05" euler="90 0 0"/>
  <site name="a" pos="-0.1 0 0"/><site name="b" pos="0.1 0 0"/><site name="side" pos="0 0 0.08"/>
</body></worldbody>
<tendon><spatial width="0.005"><site site="a"/><geom geom="c" sidesite="side"/><site site="b"/></spatial></tendon></mujoco>`;

const PULLEY = `<mujoco><worldbody>
  <site name="o"/><site name="a" pos="0 0 0.3"/><site name="b" pos="0.2 0 0.3"/><site name="c" pos="0.2 0 0.6"/>
</worldbody>
<tendon><spatial width="0.005">
  <site site="o"/><site site="a"/><pulley divisor="2"/><site site="b"/><site site="c"/>
</spatial></tendon></mujoco>`;

let mujoco: MainModule;
const held: { delete(): void }[] = [];

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
});

afterEach(() => held.splice(0).forEach((o) => o.delete()));

function load(xml: string) {
  const model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(xml);
  const data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
  held.push(data, model);
  mujoco.mj_forward(model, data);
  return { model, data, meshes: createTendonMeshes(new THREE.Group(), model), state: createTendonState() };
}

/** Tendon `t`'s path as three.js places it. */
function path(data: MjData, t: number): number[][] {
  const x = data.wrap_xpos;
  return Array.from({ length: data.ten_wrapnum[t] }, (_, i) => {
    const w = data.ten_wrapadr[t] + i;
    return [x[3 * w], x[3 * w + 2], -x[3 * w + 1]];
  });
}

const middles = (points: number[][]) => points.slice(1).map((q, i) => q.map((v, k) => (v + points[i][k]) / 2));

function expectAt(mesh: THREE.InstancedMesh, expected: number[][]): void {
  expect(mesh.count).toBe(expected.length);
  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  expected.forEach((point, i) => {
    mesh.getMatrixAt(i, matrix);
    position.setFromMatrixPosition(matrix).toArray().forEach((v, k) => expect(v).toBeCloseTo(point[k], 6));
  });
}

describe('tendons', () => {
  it('draws every point of a path a wrapped geom adds to', () => {
    const { model, data, meshes, state } = load(WRAP);
    expect([model.nwrap, data.ten_wrapnum[0]]).toEqual([3, 4]);
    snapshotTendons(data, state);
    updateTendonRendering(model, meshes, state);
    expectAt(meshes.spheres, path(data, 0));
    expectAt(meshes.cylinders, middles(path(data, 0)));
  });

  it('splits a path at a pulley, and draws a point at the world origin', () => {
    const { model, data, meshes, state } = load(PULLEY);
    snapshotTendons(data, state);
    updateTendonRendering(model, meshes, state);
    expectAt(meshes.spheres, [[0, 0, 0], [0, 0.3, 0], [0.2, 0.3, 0], [0.2, 0.6, 0]]);
    expectAt(meshes.cylinders, [[0, 0.15, 0], [0.2, 0.45, 0]]);
  });

  it('draws the last snapshot once, not the data as it is now', () => {
    const { model, data, meshes, state } = load(WRAP);
    snapshotTendons(data, state);
    updateTendonRendering(model, meshes, state);
    const drawn = path(data, 0);
    const version = meshes.spheres.instanceMatrix.version;

    data.qpos[0] = 0.5;
    mujoco.mj_forward(model, data);
    updateTendonRendering(model, meshes, state);
    expect(meshes.spheres.instanceMatrix.version).toBe(version);
    expectAt(meshes.spheres, drawn);

    snapshotTendons(data, state);
    updateTendonRendering(model, meshes, state);
    expectAt(meshes.spheres, path(data, 0));
    expect(path(data, 0)[0][1]).toBeCloseTo(drawn[0][1] + 0.5, 6);
  });
});
