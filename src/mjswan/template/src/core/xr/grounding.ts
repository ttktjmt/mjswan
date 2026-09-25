/**
 * Stands the XR viewer on the ground under it, rather than on the plane z = 0, which is a
 * terrain generator's base plane rather than its surface.
 *
 * One `mj_ray` cast straight down under the head places the rig's floor. How far above it
 * the eyes end up is the headset's to report: the rig carries the viewer's floor, not
 * their height.
 */
import * as THREE from 'three';

import { threeToMjcCoordinate } from '../scene/coordinate';
import { geomGroupMask } from '../onnx/raycast';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

/**
 * Group 0 only, the mask mjlab's height scan uses to miss the robot's own legs. A scene
 * that leaves everything there can have a ray land on the robot; the rate limit bounds it.
 */
const TERRAIN_GROUP = geomGroupMask([0]);

/** Straight down in MuJoCo's frame, where z is up. */
const DOWN: number[] = [0, 0, -1];

/** The ray starts this far above the head, so it never begins inside the ground. */
const RAY_START_ABOVE = 20;

/** Metres a second the floor may travel. A step edge would otherwise be a teleport. */
const CLIMB_SPEED = 3;

/** The head is never allowed closer to the ground than this. */
const MIN_HEAD_CLEARANCE = 0.15;

const head = new THREE.Vector3();
const origin: number[] = [0, 0, 0];
/** `mj_ray` writes the geom it hit here; unused, but the binding wants the slot. */
const geomId = new Int32Array(1);

/** Once per rendered XR frame, after locomotion and tracking have moved the rig. */
export function updateRigGrounding(
  rig: THREE.Object3D,
  camera: THREE.Camera,
  mujoco: MainModule,
  mjModel: MjModel | null,
  mjData: MjData | null,
  seconds: number
): void {
  if (!mjModel || !mjData) {
    return;
  }
  camera.getWorldPosition(head);
  // Read before the rig moves below, which would otherwise take this with it.
  const headAboveRig = head.y - rig.position.y;

  const ground = sampleGround(mujoco, mjModel, mjData);
  if (ground === null) {
    return; // nothing under the viewer: keep the height it already had
  }

  const limit = CLIMB_SPEED * seconds;
  rig.position.y += THREE.MathUtils.clamp(ground - rig.position.y, -limit, limit);

  // Being inside the ground is worse than a jump, so this one ignores the rate limit.
  const floor = ground + MIN_HEAD_CLEARANCE - headAboveRig;
  if (rig.position.y < floor) {
    rig.position.y = floor;
  }
}

/** World height of the ground under the head, or null where the ray hit nothing. */
function sampleGround(mujoco: MainModule, mjModel: MjModel, mjData: MjData): number | null {
  const mjc = threeToMjcCoordinate(head);
  origin[0] = mjc.x;
  origin[1] = mjc.y;
  origin[2] = mjc.z + RAY_START_ABOVE;
  const distance = mujoco.mj_ray(
    mjModel,
    mjData,
    origin,
    DOWN,
    TERRAIN_GROUP as unknown as number[],
    true,
    -1,
    geomId,
    null
  );
  if (distance < 0) {
    return null;
  }
  // MuJoCo's z is three.js's y, so the hit height needs no conversion of its own.
  return origin[2] - distance;
}
