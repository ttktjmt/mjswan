/**
 * A geom's colour, as MuJoCo's `setMaterial` resolves it: the material's, unless the geom's own
 * rgba is set away from the default; a result at alpha 0 is not drawn at all.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { geomRgba } from '../geomRgba';
import { loadSceneFromURL } from '../scene';

type MainModule = import('mujoco').MainModule;

const DEFAULT = [0.5, 0.5, 0.5, 1];
const TILE = [0.2, 0.3, 0.4, 1];

describe('geomRgba', () => {
  it('takes a geom without a material at its own rgba', () => {
    const model = { geom_rgba: [0.8, 0.4, 0.3, 1], geom_matid: [-1], mat_rgba: [] };
    expect(geomRgba(model, 0)).toEqual([0.8, 0.4, 0.3, 1]);
  });

  it("takes the material's colour while the geom's rgba is the default", () => {
    const model = { geom_rgba: DEFAULT, geom_matid: [0], mat_rgba: TILE };
    expect(geomRgba(model, 0)).toEqual(TILE);
  });

  it("lets a geom's own rgba win over its material's, alpha included", () => {
    const model = {
      geom_rgba: [...DEFAULT, 0.8, 0.9, 0.8, 0, 0.5, 0.5, 0.5, 0.25],
      geom_matid: [0, 0, 0],
      mat_rgba: TILE,
    };
    expect(geomRgba(model, 1)).toEqual([0.8, 0.9, 0.8, 0]);
    expect(geomRgba(model, 2)).toEqual([0.5, 0.5, 0.5, 0.25]);
  });
});

// A textured plane hidden at alpha 0 inside an opaque floor. Drawn opaque, the two
// flickered in and out of each other.
const SCENE = `<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8" rgb1="0 0 0" rgb2="1 1 1"/>
    <material name="grid" texture="grid"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="0 0 0.1" rgba="0.75 0.75 0.75 1"/>
    <geom name="hidden" type="plane" size="0 0 0.1" material="grid" rgba="0.8 0.9 0.8 0"/>
    <geom name="tinted" type="box" size="0.1 0.1 0.1" pos="0 0 0.5" material="grid" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>`;

describe('the scene', () => {
  let mujoco: MainModule;
  let root: THREE.Group;

  beforeAll(async () => {
    const load = (await import('mujoco')).default;
    mujoco = await load();
    try {
      mujoco.FS.mkdir('/working');
    } catch {
      // Already there.
    }
    mujoco.FS.writeFile('/working/geomRgba.xml', SCENE);
    const parent: { scene: THREE.Scene; mujocoRoot?: THREE.Group } = { scene: new THREE.Scene() };
    await loadSceneFromURL(mujoco, 'geomRgba.xml', parent);
    root = parent.mujocoRoot!;
  });

  function drawn(geom: number): THREE.Mesh[] {
    const meshes: THREE.Mesh[] = [];
    root.traverse((object) => {
      if (object.userData.geomId === geom && (object as THREE.Mesh).isMesh) meshes.push(object as THREE.Mesh);
    });
    return meshes;
  }

  it('draws no geom whose colour is fully transparent, as MuJoCo does', () => {
    expect(drawn(0).length).toBeGreaterThan(0);
    expect(drawn(1)).toEqual([]);
  });

  it("draws a geom's own rgba over its material's, keeping the material's texture", () => {
    const material = drawn(2)[0].material as THREE.MeshPhysicalMaterial;
    expect(material.userData.rgba).toEqual([1, 0, 0]);
    expect(material.opacity).toBe(1);
    expect(material.map).not.toBeNull();
  });
});
