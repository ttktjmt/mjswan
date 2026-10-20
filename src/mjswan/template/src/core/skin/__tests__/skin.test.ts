/**
 * Real three-vrm on two hand-built files, one VRM 0.x and one VRM 1.0, the same T-pose
 * skeleton mirrored the way each version faces. Driven by the same robot pose, the two
 * must put every bone in the same place.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import type { MainModule, MjModel } from 'mujoco';
import { Skin, type SkinConfig } from '../skin';

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

function glb(json: object, bin?: Uint8Array): ArrayBuffer {
  let text = JSON.stringify(json);
  while (text.length % 4) text += ' ';
  const chunk = new TextEncoder().encode(text);
  const binLength = bin ? 8 + bin.length : 0;
  const out = new DataView(new ArrayBuffer(12 + 8 + chunk.length + binLength));
  out.setUint32(0, 0x46546c67, true);
  out.setUint32(4, 2, true);
  out.setUint32(8, out.byteLength, true);
  out.setUint32(12, chunk.length, true);
  out.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out.buffer, 20).set(chunk);
  if (bin) {
    out.setUint32(20 + chunk.length, bin.length, true);
    out.setUint32(24 + chunk.length, 0x004e4942, true);
    new Uint8Array(out.buffer, 28 + chunk.length).set(bin);
  }
  return out.buffer;
}

/** The VRM 1.0 file with one triangle drawn in MToon: lit (1, 0.8, 0.6), shade (0.5, 0.4, 0.3). */
function toonFile(): ArrayBuffer {
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(vrmFile('1'), 20)).trim());
  const triangle = new Float32Array([0, 0, 0, 0.1, 0, 0, 0, 0.1, 0]);
  json.nodes.push({ name: 'body', mesh: 0 });
  json.scenes[0].nodes.push(json.nodes.length - 1);
  json.buffers = [{ byteLength: triangle.byteLength }];
  json.bufferViews = [{ buffer: 0, byteLength: triangle.byteLength }];
  json.accessors = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [0.1, 0.1, 0] },
  ];
  json.meshes = [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }];
  json.materials = [
    {
      pbrMetallicRoughness: { baseColorFactor: [1, 0.8, 0.6, 1] },
      extensions: { VRMC_materials_mtoon: { specVersion: '1.0', shadeColorFactor: [0.5, 0.4, 0.3] } },
    },
  ];
  json.extensionsUsed.push('VRMC_materials_mtoon');
  return glb(json, new Uint8Array(triangle.buffer));
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

/**
 * A G1-like robot: pelvis (free), torso, left arm hanging down with the forearm forward;
 * and a box beside it, a tree of its own.
 */
const BODY_NAMES = ['world', 'robot/pelvis', 'robot/torso', 'robot/l_upper', 'robot/l_fore', 'robot/l_hand', 'box'];
const robot = {
  nbody: 7,
  body_parentid: Int32Array.from([0, 0, 1, 2, 3, 4, 0]),
  body_rootid: Int32Array.from([0, 1, 1, 1, 1, 1, 6]),
  body_pos: Float64Array.from([0, 0, 0, 0, 0, 0.9, 0, 0, 0.3, 0, 0.2, 0, 0, 0, -0.25, 0.25, 0, 0, 1, 0, 0.1]),
  body_quat: Float64Array.from(Array.from({ length: 7 }, () => [1, 0, 0, 0]).flat()),
  body_jntadr: Int32Array.from([-1, 0, 1, 2, 3, 4, -1]),
  body_jntnum: Int32Array.from([0, 1, 1, 1, 1, 1, 0]),
  jnt_type: Int32Array.from([0, 3, 3, 3, 3]),
  body: (b: number) => ({ name: BODY_NAMES[b] }),
} as unknown as MjModel;
const mujoco = { mjtJoint: { mjJNT_FREE: { value: 0 } } } as unknown as MainModule;

const CONFIG: SkinConfig = {
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
  const skin = await Skin.load(vrmFile(version), CONFIG);
  expect(skin.metaVersion).toBe(version);
  skin.bind(mujoco, robot);
  skin.update(bodies(yaw, shoulder), 0);
  skin.root.updateMatrixWorld(true);
  const hand = skin.root.getObjectByName('leftHand')!;
  return hand.getWorldPosition(new THREE.Vector3());
}

describe('Skin', () => {
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

  it('hides the robot it follows, and nothing else, only with hide_robot', async () => {
    const beside = await Skin.load(vrmFile('1'), CONFIG);
    beside.bind(mujoco, robot);
    expect(beside.hiddenBodies).toEqual([]);
    const skin = await Skin.load(vrmFile('1'), { ...CONFIG, hide_robot: true });
    skin.bind(mujoco, robot);
    expect(skin.hiddenBodies).toEqual([1, 2, 3, 4, 5]);
  });

  it('is not posed while hidden, and takes up the pose when shown again', async () => {
    const skin = await Skin.load(vrmFile('1'), CONFIG);
    skin.bind(mujoco, robot);
    const hand = () => {
      skin.root.updateMatrixWorld(true);
      return skin.root.getObjectByName('leftHand')!.getWorldPosition(new THREE.Vector3());
    };
    skin.update(bodies(0, 0), 0);
    const atRest = hand();

    skin.setShown(false);
    skin.update(bodies(0, -1.2), 16);
    expect(skin.root.visible).toBe(false);
    expect(hand().distanceTo(atRest)).toBeLessThan(1e-9);

    skin.setShown(true);
    skin.update(bodies(0, -1.2), 32);
    expect(hand().distanceTo(atRest)).toBeGreaterThan(0.1);
  });

  it("divides its toon colours by the scene's light, never brightening them", async () => {
    // Two lights of 0.7 and a 0.6 headlight with 0.3 ambient: 2.3, as mjlab's scenes.
    const lit = (diffuse: number, brightness?: number) => ({
      model: {
        ...robot,
        nlight: 1,
        light_active: Uint8Array.from([1]),
        light_diffuse: Float32Array.from([diffuse, diffuse, diffuse]),
        light_ambient: Float32Array.from([0, 0, 0]),
        vis: { headlight: { active: true, diffuse: [0.6, 0.6, 0.6], ambient: [0.3, 0.3, 0.3] } },
      } as unknown as MjModel,
      brightness,
    });
    const colours = async ({ model, brightness }: ReturnType<typeof lit>) => {
      const skin = await Skin.load(toonFile(), { ...CONFIG, brightness });
      skin.bind(mujoco, model);
      let material: { color: THREE.Color; shadeColorFactor: THREE.Color } | undefined;
      skin.root.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) material = (o as THREE.Mesh).material as never;
      });
      return [material!.color.toArray(), material!.shadeColorFactor.toArray()];
    };
    const near = (a: number[], b: number[]) => a.forEach((x, i) => expect(x).toBeCloseTo(b[i], 5));

    const [color, shade] = await colours(lit(1.4));
    near(color, [1 / 2.3, 0.8 / 2.3, 0.6 / 2.3]);
    near(shade, [0.5 / 2.3, 0.4 / 2.3, 0.3 / 2.3]);
    near((await colours(lit(1.4, 1.5)))[0], [1.5 / 2.3, 1.2 / 2.3, 0.9 / 2.3]);
    // 0.1 + 0.6 + 0.3 = 1: nothing to take away.
    near((await colours(lit(0.1)))[0], [1, 0.8, 0.6]);
  });

  it('skips a bone the file lacks and a body the model lacks', async () => {
    const skin = await Skin.load(vrmFile('1'), {
      ...CONFIG,
      bones: { ...CONFIG.bones, leftToes: 'l_hand', jaw: 'no_such_body' },
    });
    skin.bind(mujoco, robot);
    expect(() => skin.update(bodies(0, 0), 0)).not.toThrow();
  });
});
