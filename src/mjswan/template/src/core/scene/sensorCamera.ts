/**
 * Renders a MuJoCo camera into an image the size a camera sensor asks for, with the
 * scene the viewer draws, so a traced observation can read it as mjlab reads its
 * `CameraSensor`. It is three.js drawing, not mujoco_warp's ray tracer: shading differs,
 * so a policy keyed to it may act differently than in mjlab.
 */

import * as THREE from 'three';
import type { MjData, MjModel } from 'mujoco';
import { mjcToThreeCoordinate } from './coordinate';
import { SENSOR_LIGHT_LAYER, geomGroupLayer } from './layers';
import { updateHeadlightFromCamera } from './lights';

/**
 * mjlab ray-traces its cameras, so nothing near is clipped; MuJoCo's `znear * extent`
 * reaches metres in a scene with terrain, past a wrist camera's whole view.
 */
const NEAR_PLANE_M = 0.001;

export type SensorCameraView = {
  /** Model camera id. */
  cameraId: number;
  width: number;
  height: number;
  enabledGeomGroups: readonly number[];
  useShadows: boolean;
};

/** MuJoCo's vertical field of view, in degrees: `fovy`, or from the intrinsics a camera
 * with a sensor size is given instead. */
export function cameraFovy(mjModel: MjModel, cameraId: number): number {
  const sensorHeight = mjModel.cam_sensorsize?.[cameraId * 2 + 1] ?? 0;
  const focalY = mjModel.cam_intrinsic?.[cameraId * 4 + 1] ?? 0;
  if (sensorHeight > 0 && focalY > 0) {
    return THREE.MathUtils.radToDeg(2 * Math.atan(sensorHeight / 2 / focalY));
  }
  return mjModel.cam_fovy[cameraId];
}

/**
 * Pose `camera` as MuJoCo's camera `cameraId` is now. Both look down their local -z with y
 * up, so the camera's world axes carry over with only the world's change of basis.
 */
export function poseCamera(
  camera: THREE.Camera,
  mjData: MjData,
  cameraId: number,
): void {
  const m: ArrayLike<number> = mjData.cam_xmat.slice(cameraId * 9, cameraId * 9 + 9);
  const column = (i: number) => mjcToThreeCoordinate([m[i], m[3 + i], m[6 + i]]);
  const position = mjcToThreeCoordinate(mjData.cam_xpos.slice(cameraId * 3, cameraId * 3 + 3));
  camera.matrix.makeBasis(column(0), column(1), column(2)).setPosition(position);
  camera.matrix.decompose(camera.position, camera.quaternion, camera.scale);
  camera.updateMatrixWorld(true);
}

/** RGBA rows as WebGL reads them, bottom first, to RGB rows top first as mjlab's image. */
export function rgbFromPixels(pixels: Uint8Array, width: number, height: number): Float32Array {
  const rgb = new Float32Array(width * height * 3);
  for (let row = 0; row < height; row++) {
    const src = (height - 1 - row) * width * 4;
    const dst = row * width * 3;
    for (let col = 0; col < width; col++) {
      rgb[dst + col * 3] = pixels[src + col * 4];
      rgb[dst + col * 3 + 1] = pixels[src + col * 4 + 1];
      rgb[dst + col * 3 + 2] = pixels[src + col * 4 + 2];
    }
  }
  return rgb;
}

export class SensorCameraRenderer {
  private readonly camera = new THREE.PerspectiveCamera();
  private readonly targets = new Map<string, THREE.WebGLRenderTarget>();
  /** The far plane per model: each read of `vis` makes a handle. */
  private clip: { model: MjModel; far: number } | null = null;

  constructor(
    private readonly renderer: THREE.WebGLRenderer,
    private readonly scene: THREE.Scene,
  ) {
    this.camera.matrixAutoUpdate = false;
  }

  /** `width * height * 4` RGBA bytes of the camera's view, bottom row first. */
  /** `lights` takes the headlight to this camera, as MuJoCo's renderer puts it there. */
  render(
    view: SensorCameraView,
    mjModel: MjModel,
    mjData: MjData,
    lights: THREE.Light[] = [],
  ): Uint8Array {
    const { width, height } = view;
    const target = this.target(width, height);
    const camera = this.camera;
    camera.fov = cameraFovy(mjModel, view.cameraId);
    camera.aspect = width / height;
    camera.near = NEAR_PLANE_M;
    camera.far = this.farPlane(mjModel);
    camera.updateProjectionMatrix();
    poseCamera(camera, mjData, view.cameraId);
    camera.layers.disableAll();
    camera.layers.enable(SENSOR_LIGHT_LAYER);
    for (const group of view.enabledGeomGroups) camera.layers.enable(geomGroupLayer(group));
    // The viewer's next frame takes it back to the viewer's camera.
    updateHeadlightFromCamera(camera, lights);

    const renderer = this.renderer;
    const previousTarget = renderer.getRenderTarget();
    const xrEnabled = renderer.xr.enabled;
    const shadows = renderer.shadowMap.enabled;
    // In an XR session the renderer would draw the headset's view instead.
    renderer.xr.enabled = false;
    renderer.shadowMap.enabled = view.useShadows && shadows;
    try {
      renderer.setRenderTarget(target);
      renderer.render(this.scene, camera);
      const pixels = new Uint8Array(width * height * 4);
      renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
      return pixels;
    } finally {
      renderer.setRenderTarget(previousTarget);
      renderer.xr.enabled = xrEnabled;
      renderer.shadowMap.enabled = shadows;
    }
  }

  dispose(): void {
    for (const target of this.targets.values()) target.dispose();
    this.targets.clear();
  }

  private farPlane(mjModel: MjModel): number {
    if (this.clip?.model !== mjModel) {
      this.clip = { model: mjModel, far: mjModel.vis.map.zfar * mjModel.stat.extent };
    }
    return this.clip.far;
  }

  private target(width: number, height: number): THREE.WebGLRenderTarget {
    const key = `${width}x${height}`;
    let target = this.targets.get(key);
    if (!target) {
      target = new THREE.WebGLRenderTarget(width, height);
      this.targets.set(key, target);
    }
    return target;
  }
}
