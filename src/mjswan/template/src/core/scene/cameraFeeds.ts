/**
 * mjlab's "Camera Feeds" (`mjlab/viewer/viser/camera_viewer.py`): each camera sensor's
 * image as the policy last read it, drawn into whatever canvases a host attaches, and the
 * camera's frustum in the scene. The image is the observation's own render, not a second
 * one, so it shows exactly what the policy got.
 */

import * as THREE from 'three';
import type { MjData, MjModel } from 'mujoco';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';

import { cameraFovy, poseCamera } from './sensorCamera';

/** `ViserCameraViewer`'s `min_display_size`: images are enlarged by whole steps toward it. */
const MIN_DISPLAY_SIZE = 128;
/**
 * viser's `add_camera_frustum` as mjlab calls it: `scale=0.15, color=(200, 200, 200)`, and
 * its default `line_width=2.0`, in CSS pixels as `LineMaterial` takes it.
 */
const FRUSTUM_SCALE = 0.15;
const FRUSTUM_COLOR = 0xc8c8c8;
const FRUSTUM_LINE_WIDTH = 2;

/** A camera sensor the loaded policy reads, resolved to the model's camera. */
export interface CameraFeedSensor {
  /** The sensor's name, unique where camera names need not be. */
  sensor: string;
  /** The model's camera, as mjlab names it (`robot/wrist_cam`). */
  camera: string;
  cameraId: number;
  width: number;
  height: number;
}

export interface CameraFeedInfo {
  id: string;
  label: string;
  width: number;
  height: number;
  scale: number;
  frustum: boolean;
}

/** The factor `ViserCameraViewer` enlarges an image by: 128 over its longer side, at least 1. */
export function feedScale(width: number, height: number): number {
  return Math.max(1, Math.floor(MIN_DISPLAY_SIZE / Math.max(width, height)));
}

/**
 * viser's wireframe frustum in the camera's frame, as line segment end points: the image
 * rectangle, a line from each corner to the camera, and a tick above the rectangle marking
 * up. viser sizes it to a fixed volume before scaling, so a narrow camera's is deeper.
 */
export function frustumSegments(fovyDeg: number, aspect: number): number[] {
  let y = Math.tan(THREE.MathUtils.degToRad(fovyDeg) / 2);
  let x = y * aspect;
  let z = 1;
  const volume = Math.cbrt((x * y * z) / 3);
  x *= FRUSTUM_SCALE / volume;
  y *= FRUSTUM_SCALE / volume;
  z *= FRUSTUM_SCALE / volume;
  // viser's points are OpenCV's (+z ahead, +y down); MuJoCo's camera looks down -z, y up.
  const at = (u: number, v: number, w: number) => [u * x, -v * y, -w * z];
  const camera = [0, 0, 0];
  return [
    ...at(-1, -1, 1), ...at(1, -1, 1),
    ...at(1, -1, 1), ...at(1, 1, 1),
    ...at(1, 1, 1), ...at(-1, 1, 1),
    ...at(-1, 1, 1), ...at(-1, -1, 1),
    ...at(-1, -1, 1), ...camera,
    ...camera, ...at(1, -1, 1),
    ...at(-1, 1, 1), ...camera,
    ...camera, ...at(1, 1, 1),
    ...at(0, -1.2, 1), ...at(0, -0.9, 1),
  ];
}

/** RGBA rows top first and opaque, from WebGL's bottom-first readback. */
function flipped(pixels: Uint8Array, width: number, height: number): Uint8ClampedArray {
  const image = new Uint8ClampedArray(width * height * 4);
  const rowBytes = width * 4;
  for (let row = 0; row < height; row++) {
    image.set(pixels.subarray((height - 1 - row) * rowBytes, (height - row) * rowBytes), row * rowBytes);
  }
  for (let alpha = 3; alpha < image.length; alpha += 4) image[alpha] = 255;
  return image;
}

class Feed {
  /** The last image, RGBA rows top first: what a canvas attached later starts from. */
  last: Uint8ClampedArray | null = null;
  /** Fat lines: WebGL draws its own one pixel wide whatever the width asked. */
  readonly frustum: LineSegments2;

  constructor(readonly sensor: CameraFeedSensor, fovyDeg: number) {
    const geometry = new LineSegmentsGeometry().setPositions(
      frustumSegments(fovyDeg, sensor.width / sensor.height),
    );
    this.frustum = new LineSegments2(
      geometry,
      new LineMaterial({ color: FRUSTUM_COLOR, linewidth: FRUSTUM_LINE_WIDTH }),
    );
    this.frustum.name = `${sensor.camera}_frustum`;
    this.frustum.matrixAutoUpdate = false;
  }

  draw(canvas: HTMLCanvasElement): void {
    const { width, height } = this.sensor;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = this.last && canvas.getContext('2d');
    if (!context || !this.last) return;
    const image = context.createImageData(width, height);
    image.data.set(this.last);
    context.putImageData(image, 0, 0);
  }

  dispose(): void {
    this.frustum.removeFromParent();
    this.frustum.geometry.dispose();
    this.frustum.material.dispose();
  }
}

export class CameraFeeds {
  private readonly feeds = new Map<string, Feed>();
  /** By sensor, apart from the feeds: a canvas stays attached across policy loads. */
  private readonly canvases = new Map<string, Set<HTMLCanvasElement>>();

  /** The policy's camera sensors, replacing the previous policy's, their frusta under `root`. */
  load(sensors: CameraFeedSensor[], mjModel: MjModel, root: THREE.Object3D): void {
    this.clear();
    for (const sensor of sensors) {
      const feed = new Feed(sensor, cameraFovy(mjModel, sensor.cameraId));
      root.add(feed.frustum);
      this.feeds.set(sensor.sensor, feed);
      for (const canvas of this.canvases.get(sensor.sensor) ?? []) feed.draw(canvas);
    }
  }

  list(): CameraFeedInfo[] {
    return [...this.feeds].map(([id, { sensor, frustum }]) => ({
      id,
      label: `${sensor.camera}_rgb`,
      width: sensor.width,
      height: sensor.height,
      scale: feedScale(sensor.width, sensor.height),
      frustum: frustum.visible,
    }));
  }

  /** Draw a feed's images into `canvas` from now on, starting with the last. Returns the detach. */
  attach(id: string, canvas: HTMLCanvasElement): () => void {
    let canvases = this.canvases.get(id);
    if (!canvases) {
      canvases = new Set();
      this.canvases.set(id, canvases);
    }
    canvases.add(canvas);
    this.feeds.get(id)?.draw(canvas);
    return () => {
      canvases.delete(canvas);
      if (canvases.size === 0 && this.canvases.get(id) === canvases) this.canvases.delete(id);
    };
  }

  setFrustum(id: string, shown: boolean): void {
    const feed = this.feeds.get(id);
    if (feed) feed.frustum.visible = shown;
  }

  /** A camera's image just rendered: RGBA rows bottom first, as WebGL reads them. */
  update(cameraId: number, pixels: Uint8Array, width: number, height: number): void {
    // One copy a read, shared by every feed of this camera at this size: they only read it.
    let image: Uint8ClampedArray | null = null;
    for (const [id, feed] of this.feeds) {
      const { sensor } = feed;
      if (sensor.cameraId !== cameraId || sensor.width !== width || sensor.height !== height) {
        continue;
      }
      image ??= flipped(pixels, width, height);
      feed.last = image;
      for (const canvas of this.canvases.get(id) ?? []) feed.draw(canvas);
    }
  }

  /** Each shown frustum follows its camera. */
  pose(mjData: MjData): void {
    for (const { sensor, frustum } of this.feeds.values()) {
      if (frustum.visible) poseCamera(frustum, mjData, sensor.cameraId);
    }
  }

  clear(): void {
    for (const feed of this.feeds.values()) feed.dispose();
    this.feeds.clear();
  }
}
