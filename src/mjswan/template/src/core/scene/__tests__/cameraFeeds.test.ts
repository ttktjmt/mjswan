/**
 * mjlab's Camera Feeds as the engine keeps them: the size `ViserCameraViewer` shows an
 * image at, viser's frustum, and the images a host's canvas receives, across policy loads.
 */
import * as THREE from 'three';
import type { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { beforeAll, describe, expect, it } from 'vitest';

import { CameraFeeds, feedScale, frustumSegments, type CameraFeedSensor } from '../cameraFeeds';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

const SCENE = `<mujoco>
  <worldbody>
    <camera name="down" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <camera name="wrist" pos="0.3 0 0.2" xyaxes="0 1 0 0 0 1" fovy="61"/>
  </worldbody>
</mujoco>`;

let mjModel: MjModel;
let mjData: MjData;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  const mujoco = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
  mjData = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(mjModel);
  mujoco.mj_forward(mjModel, mjData);
});

const WRIST: CameraFeedSensor = { sensor: 'wrist_cam', camera: 'wrist', cameraId: 1, width: 2, height: 2 };

/** A canvas that keeps what is put into it. */
function canvas(): HTMLCanvasElement & { drawn: Uint8ClampedArray[] } {
  const drawn: Uint8ClampedArray[] = [];
  const context = {
    createImageData: (w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (image: { data: Uint8ClampedArray }) => drawn.push(image.data.slice()),
  };
  return { width: 0, height: 0, drawn, getContext: () => context } as never;
}

/** Two rows of two RGBA pixels, bottom row first as WebGL reads them, alpha 0. */
const PIXELS = new Uint8Array([1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 4, 4, 4, 0]);

describe('feedScale', () => {
  it('enlarges by whole steps toward 128 px, as ViserCameraViewer does', () => {
    expect(feedScale(32, 32)).toBe(4);
    expect(feedScale(64, 64)).toBe(2);
    expect(feedScale(100, 100)).toBe(1);
    expect(feedScale(320, 240)).toBe(1);
    expect(feedScale(16, 12)).toBe(8);
  });
});

describe('frustumSegments', () => {
  const points = (fovy: number, aspect: number) => {
    const flat = frustumSegments(fovy, aspect);
    return Array.from({ length: flat.length / 3 }, (_, i) => flat.slice(i * 3, i * 3 + 3));
  };

  it('is viser\'s nine segments: the rectangle, four to the camera, and the up tick', () => {
    expect(points(61, 1)).toHaveLength(18);
  });

  it('sizes the pyramid to viser\'s volume, then 0.15', () => {
    const y = Math.tan(THREE.MathUtils.degToRad(61) / 2);
    const volume = Math.cbrt((y * y) / 3);
    const [corner] = points(61, 1);
    expect(corner[0]).toBeCloseTo((-y / volume) * 0.15);
    expect(corner[2]).toBeCloseTo((-1 / volume) * 0.15);
    // A wider image is as deep at the same volume only if it is shallower.
    expect(Math.abs(points(61, 2)[0][2])).toBeLessThan(Math.abs(corner[2]));
  });

  it('looks down -z and puts the tick above the rectangle, where up is', () => {
    const all = points(61, 4 / 3);
    const top = Math.max(...all.slice(0, 8).map((p) => p[1]));
    const [tickStart, tickEnd] = all.slice(16);
    expect(all[0][2]).toBeLessThan(0);
    expect(tickStart[1]).toBeGreaterThan(top);
    expect(tickEnd[1]).toBeLessThan(tickStart[1]);
    expect(tickEnd[1]).toBeGreaterThan(top * 0.85);
  });
});

describe('CameraFeeds', () => {
  it('lists each sensor as mjlab labels it, its frustum on', () => {
    const feeds = new CameraFeeds();
    feeds.load([WRIST], mjModel, new THREE.Group());
    expect(feeds.list()).toEqual([
      { id: 'wrist_cam', label: 'wrist_rgb', width: 2, height: 2, scale: 64, frustum: true },
    ]);
  });

  it('draws each image top row first and opaque into every attached canvas', () => {
    const feeds = new CameraFeeds();
    feeds.load([WRIST], mjModel, new THREE.Group());
    const target = canvas();
    feeds.attach('wrist_cam', target);
    feeds.update(1, PIXELS, 2, 2);
    expect([target.width, target.height]).toEqual([2, 2]);
    expect([...target.drawn[target.drawn.length - 1]]).toEqual([3, 3, 3, 255, 4, 4, 4, 255, 1, 1, 1, 255, 2, 2, 2, 255]);
  });

  it('ignores another camera\'s image, and one of another size', () => {
    const feeds = new CameraFeeds();
    feeds.load([WRIST], mjModel, new THREE.Group());
    const target = canvas();
    feeds.attach('wrist_cam', target);
    feeds.update(0, PIXELS, 2, 2);
    feeds.update(1, PIXELS, 4, 1);
    expect(target.drawn).toHaveLength(0);
  });

  it('starts a canvas attached late from the last image, and stops after the detach', () => {
    const feeds = new CameraFeeds();
    feeds.load([WRIST], mjModel, new THREE.Group());
    feeds.update(1, PIXELS, 2, 2);
    const target = canvas();
    const detach = feeds.attach('wrist_cam', target);
    expect(target.drawn).toHaveLength(1);
    detach();
    feeds.update(1, PIXELS, 2, 2);
    expect(target.drawn).toHaveLength(1);
  });

  it('keeps a canvas attached across a policy reload of the same sensor', () => {
    const feeds = new CameraFeeds();
    const root = new THREE.Group();
    feeds.load([WRIST], mjModel, root);
    const target = canvas();
    feeds.attach('wrist_cam', target);
    feeds.clear();
    expect(root.children).toHaveLength(0);
    feeds.load([WRIST], mjModel, root);
    feeds.update(1, PIXELS, 2, 2);
    expect(target.drawn).toHaveLength(1);
  });

  it('draws the frustum as viser does, light grey and 2 px wide', () => {
    const feeds = new CameraFeeds();
    const root = new THREE.Group();
    feeds.load([WRIST], mjModel, root);
    const material = (root.children[0] as THREE.Mesh).material as LineMaterial;
    expect(material.linewidth).toBe(2);
    expect(material.worldUnits).toBe(false);
    expect(material.color.getHex()).toBe(0xc8c8c8);
  });

  it('shows and hides the frustum, which follows its camera', () => {
    const feeds = new CameraFeeds();
    const root = new THREE.Group();
    feeds.load([WRIST], mjModel, root);
    const [frustum] = root.children;
    feeds.pose(mjData);
    root.updateMatrixWorld(true);
    // The wrist camera stands at MuJoCo (0.3, 0, 0.2), three's (0.3, 0.2, 0), and looks down
    // MuJoCo's -x: the frustum's +z, which points away from where it looks, is +x.
    expect(frustum.getWorldPosition(new THREE.Vector3()).toArray().map((v) => +v.toFixed(6)))
      .toEqual([0.3, 0.2, 0]);
    expect(frustum.getWorldDirection(new THREE.Vector3()).x).toBeCloseTo(1);
    feeds.setFrustum('wrist_cam', false);
    expect(frustum.visible).toBe(false);
    expect(feeds.list()[0].frustum).toBe(false);
  });
});
