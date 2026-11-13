/**
 * The parts of a camera sensor's render that decide what the policy sees before any
 * pixel is drawn: where the camera stands and looks, how wide it sees, and which pixel
 * lands where in mjlab's image. Against the real MuJoCo WASM, for the model's own pose
 * and intrinsics.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { cameraFovy, poseCamera, rgbFromPixels } from '../sensorCamera';

type MainModule = import('mujoco').MainModule;

const SCENE = `<mujoco>
  <worldbody>
    <camera name="down" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <camera name="back" pos="2 0 0.5" xyaxes="0 1 0 0 0 1"/>
    <camera name="d405" pos="0 0 1" sensorsize="0.01 0.0075" focal="0.0075 0.0075"
      resolution="320 240"/>
  </worldbody>
</mujoco>`;

let mjModel: import('mujoco').MjModel;
let mjData: import('mujoco').MjData;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  const mujoco: MainModule = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
  mjData = new (mujoco as unknown as { MjData: new (m: unknown) => never }).MjData(mjModel);
  mujoco.mj_forward(mjModel, mjData);
});

function posed(cameraId: number): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera();
  camera.matrixAutoUpdate = false;
  poseCamera(camera, mjData, cameraId);
  return camera;
}

describe('poseCamera', () => {
  it('looks down the MuJoCo camera\'s -z, which is down here', () => {
    const camera = posed(0);
    const direction = camera.getWorldDirection(new THREE.Vector3());
    // MuJoCo's -z is three's -y; the camera stands 1 m up, three's y.
    expect(direction.x).toBeCloseTo(0);
    expect(direction.y).toBeCloseTo(-1);
    expect(direction.z).toBeCloseTo(0);
    expect(camera.position.toArray()).toEqual([0, 1, -0]);
  });

  it('keeps the camera\'s up: its y axis is MuJoCo\'s z, three\'s y', () => {
    const camera = posed(1);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion);
    expect(up.y).toBeCloseTo(1);
    // Looking down -(x cross y) = MuJoCo's -x, from x = 2 back at the origin.
    const direction = camera.getWorldDirection(new THREE.Vector3());
    expect(direction.x).toBeCloseTo(-1);
  });
});

describe('cameraFovy', () => {
  it('reads fovy for a camera without a sensor size', () => {
    expect(cameraFovy(mjModel, 0)).toBeCloseTo(60);
  });

  it('derives it from the focal length when the camera has a sensor size', () => {
    // 2 atan(0.00375 / 0.0075)
    expect(cameraFovy(mjModel, 2)).toBeCloseTo(53.13, 1);
  });
});

describe('rgbFromPixels', () => {
  it('drops alpha and puts the top row first, as WebGL reads it last', () => {
    // 2 x 2 RGBA, bottom row first: bottom (1,2,3)(4,5,6), top (7,8,9)(10,11,12).
    const pixels = Uint8Array.from([1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255]);
    expect(Array.from(rgbFromPixels(pixels, 2, 2))).toEqual([
      7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6,
    ]);
  });
});
