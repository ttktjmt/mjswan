/**
 * Which geoms a camera sensor can see. A sensor camera draws its enabled groups' layers
 * and the light layer only, so a geom off its group's layer is invisible to the policy
 * however the viewer draws it: an all-black image, with nothing failing.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { SENSOR_LIGHT_LAYER, geomGroupLayer } from '../layers';
import { addSensorGeomMeshes, loadSceneFromURL } from '../scene';

type MainModule = import('mujoco').MainModule;

const SCENE = `<mujoco>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <body name="block">
      <geom name="seen" type="box" size="0.1 0.1 0.1" group="0"/>
      <geom name="visual" type="sphere" size="0.1" group="2"/>
      <geom name="collision" type="box" size="0.1 0.1 0.1" group="3"/>
    </body>
  </worldbody>
</mujoco>`;

let mujoco: MainModule;
let mjModel: import('mujoco').MjModel;
let bodies: Record<number, THREE.Group>;
let lights: THREE.Light[];

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  mujoco = await load();
  try {
    mujoco.FS.mkdir('/working');
  } catch {
    // Already there.
  }
  mujoco.FS.writeFile('/working/geom_layers.xml', SCENE);
  [mjModel, , bodies, lights] = await loadSceneFromURL(mujoco, 'geom_layers.xml', {
    scene: new THREE.Scene(),
  });
});

/** Each body is also under the world body, so a set: one entry per mesh. */
function meshesOf(group: number): THREE.Mesh[] {
  const found = new Set<THREE.Mesh>();
  for (const body of Object.values(bodies)) {
    body.traverse((child) => {
      const geomId = child.userData.geomId as number | undefined;
      if ((child as THREE.Mesh).isMesh && geomId !== undefined) {
        if (mjModel.geom_group[geomId] === group) found.add(child as THREE.Mesh);
      }
    });
  }
  return [...found];
}

const onlyLayer = (layer: number) => {
  const layers = new THREE.Layers();
  layers.set(layer);
  return layers.mask;
};

describe('geom layers', () => {
  it('puts a geom the viewer draws on layer 0 and on its group\'s layer', () => {
    for (const group of [0, 2]) {
      const [mesh] = meshesOf(group);
      expect(mesh.layers.isEnabled(0)).toBe(true);
      expect(mesh.layers.isEnabled(geomGroupLayer(group))).toBe(true);
    }
  });

  it('puts every light on the sensor light layer', () => {
    expect(lights.length).toBeGreaterThan(0);
    for (const light of lights) expect(light.layers.isEnabled(SENSOR_LIGHT_LAYER)).toBe(true);
  });

  it('draws a group the viewer skips only when a sensor asks, on its layer alone', () => {
    expect(meshesOf(3)).toHaveLength(0);

    addSensorGeomMeshes(mujoco, mjModel, { bodies }, [0, 3]);
    addSensorGeomMeshes(mujoco, mjModel, { bodies }, [3]);

    const collision = meshesOf(3);
    expect(collision).toHaveLength(1);
    expect(collision[0].layers.mask).toBe(onlyLayer(geomGroupLayer(3)));
    // Group 0 was already drawn; asking again adds nothing.
    expect(meshesOf(0)).toHaveLength(1);
  });
});
