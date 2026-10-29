/** The geom a host previews: drawn in MuJoCo's axes, never in the way of a ray, cheap to move, and edited by its handles. */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { GeomPreview, type PreviewGeom } from '../preview';

const shown = (scene: THREE.Scene) =>
  scene.children.filter((child): child is THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial> =>
    child instanceof THREE.Mesh
  );

const WIDTH = 800;
const HEIGHT = 600;

/** A canvas and its window as far as the preview uses them; Node has no DOM. */
function stage(camera = sideCamera()) {
  const window = new EventTarget();
  const canvas = Object.assign(new EventTarget(), {
    ownerDocument: { defaultView: window },
    clientHeight: HEIGHT,
    style: { cursor: '' },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: WIDTH, height: HEIGHT }),
  });
  const scene = new THREE.Scene();
  const preview = new GeomPreview(scene, canvas as unknown as HTMLElement, camera);
  return { window, canvas, scene, camera, preview };
}

/** At MuJoCo (3, 0, 0.5), looking back at the origin's side. */
function sideCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(45, WIDTH / HEIGHT, 0.01, 100);
  camera.position.set(3, 0.5, 0);
  camera.lookAt(0, 0.1, 0);
  camera.updateMatrixWorld(true);
  return camera;
}

/** Straight down from MuJoCo z = 3: x to the right on screen, y up. */
function topCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(45, WIDTH / HEIGHT, 0.01, 100);
  camera.position.set(0, 3, 0);
  camera.up.set(0, 0, -1);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld(true);
  return camera;
}

/** A MuJoCo point on screen, CSS pixels. */
function screenOf(camera: THREE.Camera, [x, y, z]: number[]): [number, number] {
  const ndc = new THREE.Vector3(x, z, -y).project(camera);
  return [((ndc.x + 1) / 2) * WIDTH, ((1 - ndc.y) / 2) * HEIGHT];
}

function pointer(type: string, [x, y]: number[], options: { target?: EventTarget; shiftKey?: boolean } = {}): Event {
  const event = new Event(type, { cancelable: true });
  const fields: Record<string, unknown> = { clientX: x, clientY: y, pointerId: 1, button: 0, pointerType: 'mouse' };
  fields.shiftKey = options.shiftKey ?? false;
  if (options.target) fields.target = options.target;
  for (const [key, value] of Object.entries(fields)) Object.defineProperty(event, key, { value });
  return event;
}

const round = (v: number[]) => v.map((x) => Number(x.toFixed(3)) + 0);

describe('GeomPreview', () => {
  it("draws one geom at MuJoCo's position and turn, where the pointer's raycast skips it", () => {
    const { scene, preview } = stage();
    // A quarter turn about MuJoCo's z, which is three.js's y.
    const half = Math.SQRT1_2;
    preview.show({ type: 'box', size: [0.1, 0.2, 0.3], pos: [1, 2, 3], quat: [half, 0, 0, half] });
    const [mesh] = shown(scene);
    expect(shown(scene)).toHaveLength(1);
    expect(mesh.position.toArray()).toEqual([1, 3, -2]);
    const x = new THREE.Vector3(1, 0, 0).applyQuaternion(mesh.quaternion);
    // MuJoCo's x turned to its y, which is three.js's -z.
    expect(x.toArray().map((v) => Number(v.toFixed(6)))).toEqual([0, 0, -1]);
    expect(mesh.userData.interactionGizmo).toBe(true);
    expect(mesh.castShadow).toBe(true);
  });

  it("is MuJoCo's grey unless coloured, and see-through only below full opacity", () => {
    const { scene, preview } = stage();
    preview.show({ type: 'sphere', size: [0.1], pos: [0, 0, 0] });
    const [mesh] = shown(scene);
    expect(mesh.material.color.toArray()).toEqual([0.5, 0.5, 0.5]);
    expect(mesh.material.transparent).toBe(false);
    preview.show({ type: 'sphere', size: [0.1], pos: [0, 0, 0], rgba: [0.85, 0.4, 0.3, 0.6] });
    expect(mesh.material.color.toArray().map((v) => Number(v.toFixed(6)))).toEqual([0.85, 0.4, 0.3]);
    expect(mesh.material.opacity).toBe(0.6);
    expect(mesh.material.transparent).toBe(true);
  });

  it('moves without building anything, and rebuilds only for a new shape', () => {
    const { scene, preview } = stage();
    preview.show({ type: 'capsule', size: [0.05, 0.2], pos: [0, 0, 0] });
    const [mesh] = shown(scene);
    const first = mesh.geometry;
    preview.show({ type: 'capsule', size: [0.05, 0.2], pos: [0.5, 0, 0] });
    expect(mesh.geometry).toBe(first);
    preview.show({ type: 'capsule', size: [0.05, 0.4], pos: [0.5, 0, 0] });
    expect(mesh.geometry).not.toBe(first);
    expect(shown(scene)).toEqual([mesh]);
  });

  it('scales an ellipsoid to its radii', () => {
    const { scene, preview } = stage();
    preview.show({ type: 'ellipsoid', size: [0.1, 0.2, 0.3], pos: [0, 0, 0] });
    expect(shown(scene)[0].scale.toArray()).toEqual([0.1, 0.3, 0.2]);
  });

  it('leaves the scene as it found it when hidden, handles and all, and can show again', () => {
    const { scene, preview } = stage();
    preview.hide();
    preview.show({ type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0], handles: 'size' });
    preview.hide();
    expect(scene.children).toEqual([]);
    preview.show({ type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0] });
    expect(shown(scene)).toHaveLength(1);
  });

  it('knows when a point of the canvas is on it', () => {
    const { camera, preview } = stage(topCamera());
    expect(preview.hit(WIDTH / 2, HEIGHT / 2)).toBe(false);
    preview.show({ type: 'box', size: [0.1, 0.1, 0.1], pos: [0, 0, 0.1] });
    expect(preview.hit(WIDTH / 2, HEIGHT / 2)).toBe(true);
    expect(preview.hit(...screenOf(camera, [0.5, 0.5, 0]))).toBe(false);
  });
});

describe('GeomPreview handles', () => {
  const BOX = { type: 'box' as const, size: [0.1, 0.1, 0.1], pos: [0, 0, 0.1] };

  it('holds a press on an arrow and resizes the geom as it is dragged, the bottom staying put', () => {
    const { window, canvas, camera, preview } = stage();
    const changes: [PreviewGeom, boolean][] = [];
    preview.onChange((geom, done) => changes.push([geom, done]));
    let below = 0;
    window.addEventListener('pointerdown', () => below++, true);
    preview.show({ ...BOX, handles: 'size' });

    // On the z arrow, a little above the top face.
    window.dispatchEvent(pointer('pointerdown', screenOf(camera, [0, 0, 0.3]), { target: canvas }));
    expect(below).toBe(0);
    window.dispatchEvent(pointer('pointermove', screenOf(camera, [0, 0, 0.4])));
    window.dispatchEvent(pointer('pointerup', screenOf(camera, [0, 0, 0.4])));

    const [last, done] = changes[changes.length - 1];
    expect(done).toBe(true);
    expect(round(last.size)).toEqual([0.1, 0.1, 0.15]);
    expect(round(last.pos)).toEqual([0, 0, 0.15]);
    expect(changes.slice(0, -1).every(([, d]) => !d)).toBe(true);
  });

  it('leaves a press anywhere else to the pointer modes and the camera', () => {
    const { window, canvas, camera, preview } = stage();
    const changes: unknown[] = [];
    preview.onChange((geom) => changes.push(geom));
    let below = 0;
    window.addEventListener('pointerdown', () => below++, true);
    preview.show({ ...BOX, handles: 'size' });
    window.dispatchEvent(pointer('pointerdown', screenOf(camera, [0, 0.6, 0]), { target: canvas }));
    window.dispatchEvent(pointer('pointermove', screenOf(camera, [0, 0.8, 0])));
    expect(below).toBe(1);
    expect(changes).toEqual([]);
  });

  it('keeps its handles the same size on screen however far away the geom is', () => {
    // Scaled to world units per CSS pixel at the geom's depth, below a camera at height 3.
    const perPixel = [0.1, 1.1].map((z) => {
      const { scene, preview } = stage(topCamera());
      preview.show({ ...BOX, pos: [0, 0, z], handles: 'size' });
      const handles = scene.children.find((c) => c instanceof THREE.Group)!;
      return (handles.children[0].scale.x / (3 - z)) * HEIGHT;
    });
    expect(perPixel[0]).toBeCloseTo(2 * Math.tan(Math.PI / 8), 9);
    expect(perPixel[1]).toBeCloseTo(perPixel[0], 9);
  });

  it('turns the geom about its own axis as a ring is swept round, in 15 degree steps with Shift', () => {
    const { window, canvas, camera, preview } = stage(topCamera());
    let turn: number[] = [];
    preview.onChange((geom) => (turn = geom.quat));
    preview.show({ ...BOX, handles: 'turn' });
    const [cx, cy] = screenOf(camera, BOX.pos);
    const on = (degrees: number) => {
      const a = (degrees * Math.PI) / 180;
      return [cx + 64 * Math.cos(a), cy - 64 * Math.sin(a)];
    };

    // On the z ring, clear of where the x and y rings cross it, then a quarter turn anticlockwise.
    window.dispatchEvent(pointer('pointerdown', on(45), { target: canvas }));
    window.dispatchEvent(pointer('pointermove', on(90)));
    window.dispatchEvent(pointer('pointermove', on(135)));
    expect(round(turn)).toEqual(round([Math.SQRT1_2, 0, 0, Math.SQRT1_2]));

    window.dispatchEvent(pointer('pointermove', on(146), { shiftKey: true }));
    const half = (105 * Math.PI) / 360;
    expect(round(turn)).toEqual(round([Math.cos(half), 0, 0, Math.sin(half)]));
    window.dispatchEvent(pointer('pointerup', on(146)));
  });

  it('stands in for a geom of the model, hiding it until it goes', () => {
    const { scene, preview } = stage();
    const model = [3, 4].map((geomId) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
      mesh.userData.geomId = geomId;
      scene.add(mesh);
      return mesh;
    });
    preview.show({ ...BOX, replaces: 3 });
    expect(model.map((m) => m.visible)).toEqual([false, true]);
    preview.show({ ...BOX, replaces: 4 });
    expect(model.map((m) => m.visible)).toEqual([true, false]);
    preview.hide();
    expect(model.map((m) => m.visible)).toEqual([true, true]);

    // After a load the id names another model's geom: nothing of the new one is touched.
    preview.show({ ...BOX, replaces: 3 });
    preview.onSceneLoaded();
    model[0].visible = true;
    preview.show({ ...BOX });
    expect(model.map((m) => m.visible)).toEqual([true, true]);
  });
});
