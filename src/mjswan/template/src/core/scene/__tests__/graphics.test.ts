/**
 * A scene's graphics settings: MuJoCo's names, what the renderer is set to, and the
 * model's pieces (the reflection, the camera sensor's own pass) against the real MuJoCo
 * WASM.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import {
  DEFAULT_GRAPHICS,
  SceneGraphics,
  applyToRenderer,
  mergeGraphics,
  toneMappingNamed,
  type GraphicsSettings,
} from '../graphics';
import { loadSceneFromURL } from '../scene';

type MainModule = import('mujoco').MainModule;

const FILAMENT: GraphicsSettings = { ...DEFAULT_GRAPHICS, renderer: 'filament' };

describe('graphics settings', () => {
  it("reads MuJoCo's names as the operators they mean", () => {
    expect(toneMappingNamed('pbr_neutral')).toBe('neutral');
    expect(toneMappingNamed('aces_legacy')).toBe('aces_filmic');
    expect(toneMappingNamed('agx')).toBe('agx');
    expect(toneMappingNamed('toString')).toBeNull();
  });

  it('keeps a setting it cannot draw', () => {
    const merged = mergeGraphics(DEFAULT_GRAPHICS, {
      renderer: 'filament',
      toneMapping: 'bogus' as never,
      exposure: Number.NaN,
      haze: false,
    });
    expect(merged).toEqual({ ...DEFAULT_GRAPHICS, renderer: 'filament', haze: false });
  });

  it('tone maps and encodes under Filament alone, its exposure in stops', () => {
    const renderer = {} as THREE.WebGLRenderer;
    applyToRenderer(renderer, { ...DEFAULT_GRAPHICS, toneMapping: 'agx', exposure: 1 });
    expect([renderer.toneMapping, renderer.outputColorSpace]).toEqual([THREE.NoToneMapping, THREE.LinearSRGBColorSpace]);
    applyToRenderer(renderer, { ...FILAMENT, toneMapping: 'agx', exposure: 1 });
    expect([renderer.toneMapping, renderer.toneMappingExposure, renderer.outputColorSpace]).toEqual([
      THREE.AgXToneMapping,
      2,
      THREE.SRGBColorSpace,
    ]);
    // MuJoCo's ACES is three's without the 1/0.6 three brightens it by.
    applyToRenderer(renderer, { ...FILAMENT, toneMapping: 'aces', outputColorSpace: 'linear' });
    expect([renderer.toneMappingExposure, renderer.outputColorSpace]).toEqual([0.6, THREE.LinearSRGBColorSpace]);
    expect(THREE.ShaderChunk.tonemapping_pars_fragment).toContain('2.51 * color + 0.03');
  });
});

const SCENE = `<mujoco>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="8" height="8" rgb1="0.2 0.3 0.4"
      rgb2="0.5 0.5 0.5" colorspace="sRGB"/>
    <material name="glass" rgba="1 1 1 0.5" reflectance="0.5"/>
    <material name="floor" texture="grid" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" castshadow="true"/>
    <geom name="glass" type="box" size="1 1 0.1" pos="0 0 -1" material="glass"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"/>
    <body name="block" pos="0 0 0.5">
      <geom name="block" type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
    </body>
  </worldbody>
</mujoco>`;

let mujoco: MainModule;
let mjModel: import('mujoco').MjModel;
let root: THREE.Group;
let lights: THREE.Light[];
const scene = new THREE.Scene();

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  mujoco = await load();
  try {
    mujoco.FS.mkdir('/working');
  } catch {
    // Already there.
  }
  mujoco.FS.writeFile('/working/graphics.xml', SCENE);
  const parent: { scene: THREE.Scene; mujocoRoot?: THREE.Group } = { scene };
  [mjModel, , , lights] = await loadSceneFromURL(mujoco, 'graphics.xml', parent);
  root = parent.mujocoRoot!;
});

function material(name: string): THREE.MeshPhysicalMaterial {
  const g = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_GEOM.value, name);
  let found: THREE.MeshPhysicalMaterial | undefined;
  root.traverse((object) => {
    if (!found && object.userData.geomId === g) found = (object as THREE.Mesh).material as THREE.MeshPhysicalMaterial;
  });
  return found!;
}

describe('SceneGraphics', () => {
  it('mirrors in the first opaque plane or box with a reflectance, as MuJoCo does', () => {
    new SceneGraphics(mujoco, mjModel, root, scene, lights);
    expect(material('floor').customProgramCacheKey()).toContain('mjMirror');
    expect(material('glass').customProgramCacheKey()).not.toContain('mjMirror');
  });

  it("draws a camera sensor's image from the model as stored, and the view back as set", () => {
    const graphics = new SceneGraphics(mujoco, mjModel, root, scene, lights);
    graphics.apply({ ...FILAMENT, shadows: true });
    const block = material('block');
    const floor = material('floor');
    const viewed = floor.map!;
    expect(viewed.colorSpace).toBe(THREE.SRGBColorSpace);
    expect(block.color.r).toBeCloseTo(new THREE.Color().setRGB(0.5, 0.5, 0.5, THREE.SRGBColorSpace).r, 6);
    expect(lights.find((light) => light.castShadow)).toBeDefined();

    graphics.asSensor(false, () => {
      expect(block.color.r).toBe(0.5);
      expect(floor.map).not.toBe(viewed);
      expect(floor.map!.colorSpace).toBe(THREE.NoColorSpace);
      expect(floor.map!.source).toBe(viewed.source);
      expect(lights.some((light) => light.castShadow)).toBe(false);
    });
    expect(floor.map).toBe(viewed);
    expect(block.color.r).toBeLessThan(0.5);
    expect(lights.some((light) => light.castShadow)).toBe(true);

    graphics.apply(DEFAULT_GRAPHICS);
    expect(block.color.r).toBe(0.5);
    graphics.apply({ ...DEFAULT_GRAPHICS, shadows: false });
    expect(lights.some((light) => light.castShadow)).toBe(false);
  });

  it('shows a camera sensor what only the viewer hides, and hides it again', () => {
    const graphics = new SceneGraphics(mujoco, mjModel, root, scene, lights);
    graphics.apply(DEFAULT_GRAPHICS);
    const block = material('block');
    let floor: THREE.Object3D | undefined;
    root.traverse((object) => (floor ??= object.userData.isGroundPlane ? object : undefined));
    // A skin hides the robot by material; an AR session hides the floor by object.
    block.visible = false;
    floor!.visible = false;
    // A colour a pass left decoded, as a see-through clone's original can be.
    block.color.setRGB(0.1, 0.1, 0.1);
    graphics.asSensor(false, () => {
      expect([block.visible, floor!.visible]).toEqual([true, true]);
      expect(block.color.r).toBe(0.5);
    });
    expect([block.visible, floor!.visible]).toEqual([false, false]);
    block.visible = true;
    floor!.visible = true;
  });
});

describe("the model's extent", () => {
  const load = async (name: string, xml: string) => {
    mujoco.FS.writeFile(`/working/${name}`, xml);
    const [model] = await loadSceneFromURL(mujoco, name, { scene: new THREE.Scene() });
    return model.stat.extent;
  };
  const body = '<body pos="0.5 0 0.2"><geom size="0.1"/></body>';
  // The pointer's anchor, as `injectPointerGrabXml` adds it: 100 m above the scene.
  const anchor = '<body name="mjswan_ptr_anchor" mocap="true" pos="0 0 100"/>';

  it("is the shipped model's, whatever the viewer parks above it", async () => {
    const shipped = await load('shipped.xml', `<mujoco><worldbody>${body}</worldbody></mujoco>`);
    expect(await load('injected.xml', `<mujoco><worldbody>${body}${anchor}</worldbody></mujoco>`)).toBeCloseTo(shipped, 9);
    expect(shipped).toBeLessThan(1);
  });

  it("is MuJoCo's for a model with no body of its own", async () => {
    const geom = '<geom size="0.1"/>';
    const shipped = await load('static.xml', `<mujoco><worldbody>${geom}</worldbody></mujoco>`);
    expect(await load('static_injected.xml', `<mujoco><worldbody>${geom}${anchor}</worldbody></mujoco>`)).toBeCloseTo(shipped, 9);
  });

  it("is the author's when the model states it", async () => {
    const stated = `<mujoco><statistic extent="7"/><worldbody>${body}${anchor}</worldbody></mujoco>`;
    expect(await load('stated.xml', stated)).toBe(7);
  });
});
