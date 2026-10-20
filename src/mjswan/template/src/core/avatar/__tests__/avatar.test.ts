/**
 * Real three-vrm on two hand-built files, one VRM 0.x and one VRM 1.0, the same T-pose
 * skeleton mirrored the way each version faces. Driven by the same robot pose, the two
 * must put every bone in the same place.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { MainModule, MjModel } from 'mujoco';
import { Avatar, type AvatarConfig } from '../avatar';

/** Bone, parent, offset from the parent in VRM 1.0's frame (facing +z, left at +x). */
const SKELETON: Array<[string, string | null, [number, number, number]]> = [
  ['hips', null, [0, 0.9, 0]],
  ['spine', 'hips', [0, 0.1, 0]],
  ['chest', 'spine', [0, 0.15, 0]],
  ['neck', 'chest', [0, 0.2, 0]],
  ['head', 'neck', [0, 0.1, 0]],
  ...(['left', 'right'] as const).flatMap((side) => {
    const x = side === 'left' ? 1 : -1;
    return [
      [`${side}UpperArm`, 'chest', [0.2 * x, 0.15, 0]],
      [`${side}LowerArm`, `${side}UpperArm`, [0.25 * x, 0, 0]],
      [`${side}Hand`, `${side}LowerArm`, [0.25 * x, 0, 0]],
      [`${side}UpperLeg`, 'hips', [0.1 * x, -0.05, 0]],
      [`${side}LowerLeg`, `${side}UpperLeg`, [0, -0.4, 0]],
      [`${side}Foot`, `${side}LowerLeg`, [0, -0.4, 0]],
    ] as Array<[string, string, [number, number, number]]>;
  }),
];

function glb(json: object): ArrayBuffer {
  let text = JSON.stringify(json);
  while (text.length % 4) text += ' ';
  const chunk = new TextEncoder().encode(text);
  const out = new DataView(new ArrayBuffer(12 + 8 + chunk.length));
  out.setUint32(0, 0x46546c67, true);
  out.setUint32(4, 2, true);
  out.setUint32(8, out.byteLength, true);
  out.setUint32(12, chunk.length, true);
  out.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out.buffer, 20).set(chunk);
  return out.buffer;
}

function vrmFile(version: '0' | '1'): ArrayBuffer {
  // VRM 0.x faces -z, so its left side is at -x.
  const mirror = version === '0' ? -1 : 1;
  const index = new Map(SKELETON.map(([name], i) => [name, i]));
  const nodes = SKELETON.map(([name, , [x, y, z]]) => ({
    name,
    translation: [x * mirror, y, z * mirror],
    children: SKELETON.filter(([, parent]) => parent === name).map(([child]) => index.get(child)!),
  }));
  const vrm =
    version === '1'
      ? {
          extensionsUsed: ['VRMC_vrm'],
          extensions: {
            VRMC_vrm: {
              specVersion: '1.0',
              meta: { name: 't', authors: ['mjswan'], licenseUrl: 'https://vrm.dev/licenses/1.0/' },
              humanoid: {
                humanBones: Object.fromEntries(SKELETON.map(([name], i) => [name, { node: i }])),
              },
            },
          },
        }
      : {
          extensionsUsed: ['VRM'],
          extensions: {
            VRM: {
              specVersion: '0.0',
              meta: { title: 't', author: 'mjswan', licenseName: 'CC0' },
              humanoid: { humanBones: SKELETON.map(([bone], node) => ({ bone, node })) },
            },
          },
        };
  return glb({ asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes, ...vrm });
}

/** A G1-like robot: pelvis (free), torso, left arm hanging down with the forearm forward. */
const BODY_NAMES = ['world', 'robot/pelvis', 'robot/torso', 'robot/l_upper', 'robot/l_fore', 'robot/l_hand'];
const robot = {
  nbody: 6,
  body_parentid: Int32Array.from([0, 0, 1, 2, 3, 4]),
  body_pos: Float64Array.from([0, 0, 0, 0, 0, 0.9, 0, 0, 0.3, 0, 0.2, 0, 0, 0, -0.25, 0.25, 0, 0]),
  body_quat: Float64Array.from(Array.from({ length: 6 }, () => [1, 0, 0, 0]).flat()),
  body_jntadr: Int32Array.from([-1, 0, 1, 2, 3, 4]),
  body_jntnum: Int32Array.from([0, 1, 1, 1, 1, 1]),
  jnt_type: Int32Array.from([0, 3, 3, 3, 3]),
  body: (b: number) => ({ name: BODY_NAMES[b] }),
} as unknown as MjModel;
const mujoco = { mjtJoint: { mjJNT_FREE: { value: 0 } } } as unknown as MainModule;

const CONFIG: AvatarConfig = {
  bones: {
    hips: 'pelvis',
    spine: 'torso',
    leftUpperArm: 'l_upper',
    leftLowerArm: 'l_fore',
    leftHand: 'l_hand',
  },
  rest_offsets: {
    leftUpperArm: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI / 2, 'ZYX')).toArray(),
    leftLowerArm: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, -Math.PI / 2, 'ZYX')).toArray(),
    leftHand: new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, -Math.PI / 2, 'ZYX')).toArray(),
  },
  height_ratio: 1,
  offset: [0, 0, 0],
};

/** Body groups as the runtime draws them: pelvis at 0.9 m, turned by `yaw` and `shoulder`. */
function bodies(yaw: number, shoulder: number): Record<number, THREE.Object3D> {
  const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
  // A shoulder pitch forward: about MuJoCo +y, three.js -z.
  const raise = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, -1), shoulder);
  const out: Record<number, THREE.Object3D> = {};
  for (let b = 0; b < 6; b++) {
    const body = new THREE.Object3D();
    body.position.set(0, 0.9, 0);
    body.quaternion.copy(turn);
    if (b >= 3) body.quaternion.multiply(raise);
    out[b] = body;
  }
  return out;
}

async function handPosition(version: '0' | '1', yaw: number, shoulder: number) {
  const avatar = await Avatar.load(vrmFile(version), CONFIG);
  expect(avatar.metaVersion).toBe(version);
  avatar.bind(mujoco, robot);
  avatar.update(bodies(yaw, shoulder), 0);
  avatar.root.updateMatrixWorld(true);
  const hand = avatar.root.getObjectByName('leftHand')!;
  return hand.getWorldPosition(new THREE.Vector3());
}

describe('Avatar', () => {
  it.each(['0', '1'] as const)('loads VRM %s and hangs the arm as the robot does at rest', async (version) => {
    // Robot +x is three.js +x and its left is -z: shoulder (0.2 left, 1.3 up), the upper
    // arm 0.25 down, the forearm 0.25 forward.
    const hand = await handPosition(version, 0, 0);
    expect(hand.x).toBeCloseTo(0.25, 6);
    expect(hand.y).toBeCloseTo(1.05, 6);
    expect(hand.z).toBeCloseTo(-0.2, 6);
  });

  it('puts the hand in the same place for both versions under any pose', async () => {
    for (const [yaw, shoulder] of [
      [0.4, -1.1],
      [2.5, 0.6],
      [-3.0, -2.0],
    ]) {
      const v0 = await handPosition('0', yaw, shoulder);
      const v1 = await handPosition('1', yaw, shoulder);
      expect(v0.distanceTo(v1)).toBeLessThan(1e-6);
    }
  });

  it('skips a bone the file lacks and a body the model lacks', async () => {
    const avatar = await Avatar.load(vrmFile('1'), {
      ...CONFIG,
      bones: { ...CONFIG.bones, leftToes: 'l_hand', jaw: 'no_such_body' },
    });
    avatar.bind(mujoco, robot);
    expect(() => avatar.update(bodies(0, 0), 0)).not.toThrow();
  });
});
