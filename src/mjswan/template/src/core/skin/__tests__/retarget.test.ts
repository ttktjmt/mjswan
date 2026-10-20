import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  AVATAR_TO_ROBOT_FACING,
  buildRig,
  hipsPosition,
  modelFrame,
  retarget,
  zeroPoseHeight,
  zeroPoseRotations,
} from '../retarget';

const Y = new THREE.Vector3(0, 1, 0);
const FREE = 0;
const HINGE = 3;

/** world -> pelvis (free) -> thigh (hinge, tilted 10° by body_quat) -> shin (hinge). */
const model = {
  nbody: 4,
  body_parentid: Int32Array.from([0, 0, 1, 2]),
  body_pos: Float64Array.from([0, 0, 0, 0, 0, 0.8, 0, 0.1, -0.1, 0, 0, -0.3]),
  body_quat: Float64Array.from([
    1, 0, 0, 0,
    1, 0, 0, 0,
    Math.cos(Math.PI / 36), 0, Math.sin(Math.PI / 36), 0,
    1, 0, 0, 0,
  ]),
  body_jntadr: Int32Array.from([-1, 0, 1, 2]),
  body_jntnum: Int32Array.from([0, 1, 1, 1]),
  jnt_type: Int32Array.from([FREE, HINGE, HINGE]),
};
const zeroPose = zeroPoseRotations(model, FREE);

const BONES = [
  { name: 'hips', bodyId: 1, ancestors: [] },
  { name: 'leftLowerLeg', bodyId: 3, ancestors: ['leftUpperLeg', 'hips'] },
  { name: 'leftUpperLeg', bodyId: 2, ancestors: ['hips'] },
];

function rig(version: string, offsets: Record<string, number[]> = {}) {
  return buildRig(BONES, zeroPose, offsets, modelFrame(version));
}

function pose(r: ReturnType<typeof rig>, rotations: Record<number, THREE.Quaternion>) {
  const world = r.bones.map(() => new THREE.Quaternion());
  const local = r.bones.map(() => new THREE.Quaternion());
  retarget(r, (b) => rotations[b], world, local);
  return Object.fromEntries(r.bones.map((bone, i) => [bone.name, local[i]]));
}

/** Each body at its zero pose, then turned by `turn` in the world. */
function turned(turn: THREE.Quaternion): Record<number, THREE.Quaternion> {
  return Object.fromEntries(zeroPose.map((q, b) => [b, turn.clone().multiply(q)]));
}

const close = (a: THREE.Quaternion, b: THREE.Quaternion) =>
  expect(Math.abs(a.dot(b))).toBeCloseTo(1, 9);

describe('zero pose', () => {
  it('takes a free body as upright and accumulates body_quat below it', () => {
    close(zeroPose[1], new THREE.Quaternion());
    // MuJoCo +y (left) is three.js -z, so a pitch about MuJoCo y turns about -z.
    close(zeroPose[3], new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, -1), Math.PI / 18));
  });

  it('stacks body_pos for the root height', () => {
    expect(zeroPoseHeight(model, zeroPose, 1)).toBeCloseTo(0.8, 12);
  });

  it('orders parents first and links each bone to its nearest mapped ancestor', () => {
    const r = rig('1');
    expect(r.bones.map((b) => b.name)).toEqual(['hips', 'leftUpperLeg', 'leftLowerLeg']);
    expect(r.bones.map((b) => b.parent)).toEqual([null, 0, 1]);
  });

  it.each(['0', '1'])('poses every bone at rest, VRM %s', (version) => {
    for (const q of Object.values(pose(rig(version), turned(new THREE.Quaternion())))) {
      close(q, new THREE.Quaternion());
    }
  });
});

describe('root yaw', () => {
  it.each(['0', '1'])('turns only the hips, without drift over many turns, VRM %s', (version) => {
    const r = rig(version);
    for (let step = 0; step <= 400; step++) {
      const angle = step * 0.37;
      const yaw = new THREE.Quaternion().setFromAxisAngle(Y, angle);
      const local = pose(r, turned(yaw));
      close(local.hips, yaw);
      close(local.leftUpperLeg, new THREE.Quaternion());
      close(local.leftLowerLeg, new THREE.Quaternion());
    }
  });
});

describe('knee', () => {
  it('bends the lower leg about the avatar left axis, as the robot bends about its own', () => {
    const r = rig('1');
    // MuJoCo pitch about +y (left) at the knee; three.js -z; the avatar's left is +x.
    const bend = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, -1), 1.2);
    const rotations = turned(new THREE.Quaternion());
    rotations[3] = bend.clone().multiply(zeroPose[3]);
    const local = pose(r, rotations);
    close(local.leftLowerLeg, new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), 1.2));
  });
});

describe('VRM 0.x and 1.0', () => {
  it('flip x and z of every local rotation and agree in the world', () => {
    const rotations: Record<number, THREE.Quaternion> = {};
    for (let b = 0; b < 4; b++) {
      rotations[b] = new THREE.Quaternion()
        .setFromEuler(new THREE.Euler(0.3 * b, -0.7 + b, 0.2 * b))
        .multiply(zeroPose[b]);
    }
    const offsets = { leftUpperLeg: [0, 0, Math.SQRT1_2, Math.SQRT1_2] };
    const v0 = pose(rig('0', offsets), rotations);
    const v1 = pose(rig('1', offsets), rotations);
    for (const name of Object.keys(v1)) {
      const flipped = new THREE.Quaternion(-v1[name].x, v1[name].y, -v1[name].z, v1[name].w);
      close(v0[name], flipped);
    }
  });
});

describe('rest offsets', () => {
  it('hold a bone at its offset while the robot rests, relative to its parent', () => {
    const lowered = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), -Math.PI / 2);
    const local = pose(
      rig('1', { leftUpperLeg: lowered.toArray(), leftLowerLeg: lowered.toArray() }),
      turned(new THREE.Quaternion()),
    );
    close(local.leftUpperLeg, lowered);
    close(local.leftLowerLeg, new THREE.Quaternion());
  });
});

describe('hips position', () => {
  it.each([
    ['1', 1],
    ['0', -1],
  ])('maps the robot forward to the avatar front and scales only the height, VRM %s', (version, sign) => {
    // MuJoCo (1, 0, 0.8) is three.js (1, 0.8, 0).
    const target = hipsPosition(rig(version), new THREE.Vector3(1, 0.8, 0), 1.1, new THREE.Vector3());
    expect(target.x).toBeCloseTo(0, 12);
    expect(target.y).toBeCloseTo(0.88, 12);
    expect(target.z).toBeCloseTo(sign, 12);
  });

  it('agrees with the container turn', () => {
    const r = rig('0');
    const root = new THREE.Vector3(0.3, 0.7, -0.4);
    const local = hipsPosition(r, root, 1, new THREE.Vector3());
    const container = AVATAR_TO_ROBOT_FACING.clone().multiply(r.frame);
    const back = local.applyQuaternion(container);
    expect(back.distanceTo(root)).toBeCloseTo(0, 12);
  });
});
