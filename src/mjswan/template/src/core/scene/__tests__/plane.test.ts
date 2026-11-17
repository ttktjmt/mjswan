/**
 * A plane as MuJoCo draws it: an infinite side wider than its far plane, following the
 * viewer by whole texture periods, and the texture placed by `GL_OBJECT_PLANE` (`S = 0.5
 * scale x - 0.5`, `T = -0.5 scale y - 0.5`, the scale `texrepeat` over the size unless
 * `texuniform`).
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { modelZfar, planeGeometry, recentrePlane } from '../plane';

type MainModule = import('mujoco').MainModule;

const SCENE = `<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8"/>
    <material name="uniform" texture="grid" texrepeat="5 4" texuniform="true"/>
    <material name="stretched" texture="grid" texrepeat="2 2"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.05" material="uniform"/>
    <geom name="tile" type="plane" size="2 1 0.1" pos="0 0 1" material="stretched"/>
  </worldbody>
</mujoco>`;

let mjModel: import('mujoco').MjModel;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  const mujoco: MainModule = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } }).MjModel.from_xml_string(SCENE);
});

/** Each vertex's MuJoCo (x, y) and its (u, v). */
function vertices(geometry: THREE.BufferGeometry): Array<[number, number, number, number]> {
  const position = geometry.getAttribute('position');
  const uv = geometry.getAttribute('uv');
  return Array.from({ length: position.count }, (_, i) => [position.getX(i), -position.getZ(i), uv.getX(i), uv.getY(i)]);
}

describe('planeGeometry', () => {
  it('reaches past the far plane on an infinite side, a texture period to each square', () => {
    const zfar = modelZfar(mjModel);
    const { geometry, step } = planeGeometry(mjModel, 0, zfar);
    expect(step).toEqual([2 * (2 / 5), 2 * (2 / 4)]);
    const xs = vertices(geometry).map(([x]) => x);
    expect(Math.max(...xs)).toBeCloseTo(1.05 * zfar + 2 / 5, 5);
    for (const [x, y, u, v] of vertices(geometry)) {
      expect(u).toBeCloseTo(0.5 * 5 * x - 0.5, 4);
      expect(v).toBeCloseTo(-0.5 * 4 * y - 0.5, 4);
    }
  });

  it('spreads texrepeat over a finite plane', () => {
    const { geometry, step } = planeGeometry(mjModel, 1, modelZfar(mjModel));
    expect(step).toEqual([0, 0]);
    for (const [x, y, u, v] of vertices(geometry)) {
      expect([Math.abs(x), Math.abs(y)]).toEqual([2, 1]);
      expect(u).toBeCloseTo(0.5 * (2 / 2) * x - 0.5, 6);
      expect(v).toBeCloseTo(-0.5 * (2 / 1) * y - 0.5, 6);
    }
  });
});

describe('recentrePlane', () => {
  it('follows the viewer along its infinite sides, a whole stride at a time', () => {
    const body = new THREE.Group();
    const plane = new THREE.Mesh();
    body.add(plane);
    plane.userData.recentre = { origin: new THREE.Vector3(0, 0, 0), step: [0.8, 0] };
    recentrePlane(plane, new THREE.Vector3(0.8 * 2.4, 5, 3));
    expect(plane.position.toArray()).toEqual([1.6, 0, 0]);
    // MuJoCo's y is three's -z: a turned plane strides along its own axes.
    plane.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
    plane.userData.recentre.step = [0, 1];
    recentrePlane(plane, new THREE.Vector3(3.2, 0, 0));
    expect(plane.position.x).toBeCloseTo(3, 6);
    expect(plane.position.z).toBeCloseTo(0, 6);
  });
});
