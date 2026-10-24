/**
 * Real three-vrm on hand-built files: a VRM 0.x and a VRM 1.0, the same T-pose skeleton
 * mirrored the way each version faces, and the same skeleton as plain .glb files, scaled,
 * turned or nested otherwise. Driven by the same robot pose, all must pose alike.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { MToonMaterial } from '@pixiv/three-vrm';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { Skin, geomBottom, type SkinConfig } from '../skin';
import { asPainted } from '../skinModel';

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

/** Linear `rgb` as a skin's colour is stored: sRGB, as painted. */
const painted = (...rgb: [number, number, number]) => new THREE.Color(...rgb).convertLinearToSRGB().toArray();

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

/** The VRM 1.0 file with a triangle on the left hand, its lowest corner 0.02 m below it. */
function gloveFile(): ArrayBuffer {
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(vrmFile('1'), 20)).trim());
  const triangle = new Float32Array([0, -0.02, 0, 0.05, 0, 0, 0, 0, 0.05]);
  json.nodes.push({ name: 'glove', mesh: 0 });
  json.nodes.find((node: { name: string }) => node.name === 'leftHand').children.push(json.nodes.length - 1);
  json.buffers = [{ byteLength: triangle.byteLength }];
  json.bufferViews = [{ buffer: 0, byteLength: triangle.byteLength }];
  json.accessors = [
    { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, -0.02, 0], max: [0.05, 0, 0.05] },
  ];
  json.meshes = [{ primitives: [{ attributes: { POSITION: 0 } }] }];
  return glb(json, new Uint8Array(triangle.buffer));
}

/** A model's front -> the turn about y from VRM 1.0's (+z) onto it. */
const FACING = { '+z': 0, '+x': Math.PI / 2, '-z': Math.PI, '-x': -Math.PI / 2 } as const;

/**
 * The skeleton as a plain .glb, no VRM extension: `units` of the file to the metre, turned
 * to face `facing`, its nodes named `<bone>_node`. `armOnHips` hangs the left arm from the
 * hips rather than the chest, as VRM never nests it; `handOnHips` the left hand, where it
 * rests, as an IK control holds a foot.
 */
function gltfFile(
  options: { units?: number; facing?: keyof typeof FACING; armOnHips?: boolean; handOnHips?: boolean } = {},
): { data: ArrayBuffer; nodes: Record<string, number> } {
  const { units = 1, facing = '+z', armOnHips = false, handOnHips = false } = options;
  const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), FACING[facing]);
  const skeleton = SKELETON.map(([name, parent, offset]): [string, string | null, [number, number, number]] => {
    if (armOnHips && name === 'leftUpperArm') return [name, 'hips', [0.2, 0.4, 0]];
    if (handOnHips && name === 'leftHand') return [name, 'hips', [0.7, 0.4, 0]];
    return [name, parent, offset];
  });
  const index = new Map(skeleton.map(([name], i) => [name, i]));
  const nodes = skeleton.map(([name, , offset]) => ({
    name: `${name}_node`,
    translation: new THREE.Vector3(...offset).applyQuaternion(turn).multiplyScalar(units).toArray(),
    children: skeleton.filter(([, parent]) => parent === name).map(([child]) => index.get(child)!),
  }));
  return {
    data: glb({ asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes }),
    nodes: Object.fromEntries(index),
  };
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
const [SPHERE, BOX] = [2, 6];
const SHAPES = {
  mjtGeom: Object.fromEntries(
    ['PLANE', 'HFIELD', 'SPHERE', 'CAPSULE', 'ELLIPSOID', 'CYLINDER', 'BOX', 'MESH'].map((shape, value) => [
      `mjGEOM_${shape}`,
      { value },
    ]),
  ),
};

const CONFIG: SkinConfig = {
  format: 'vrm',
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
  limbs: [['leftUpperArm', 'leftLowerArm', 'leftHand']],
  height_ratio: 1,
  offset: [0, 0, 0],
  retarget: 'rotation',
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
  return posedHand(await Skin.load(vrmFile(version), CONFIG), 'leftHand', bodies(yaw, shoulder));
}

function posedHand(skin: Skin, name: string, pose: Record<number, THREE.Object3D>): THREE.Vector3 {
  skin.bind(mujoco, robot);
  skin.update(pose, 0);
  skin.root.updateMatrixWorld(true);
  return skin.root.getObjectByName(name)!.getWorldPosition(new THREE.Vector3());
}

/** `skin` with `file` as a .glb keyed by VRM bone name, as a humanoid's manifest entry is. */
async function gltfSkin(file: ReturnType<typeof gltfFile>, config: Partial<SkinConfig> = {}): Promise<Skin> {
  const nodes = Object.fromEntries(Object.keys(CONFIG.bones).map((bone) => [bone, file.nodes[bone]]));
  return Skin.load(file.data, { ...CONFIG, format: 'gltf', nodes, ...config });
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

  it('poses a .glb as it does the VRM, whatever its units and facing', async () => {
    for (const [yaw, shoulder] of [
      [0, 0],
      [0.4, -1.1],
      [-3.0, -2.0],
    ]) {
      const vrm = await handPosition('1', yaw, shoulder);
      for (const options of [{}, { units: 100 }, { facing: '+x' as const }, { units: 2.54, facing: '-z' as const }]) {
        const units = options.units ?? 1;
        const skin = await gltfSkin(gltfFile(options), { scale: 1 / units, facing: options.facing });
        expect(posedHand(skin, 'leftHand_node', bodies(yaw, shoulder)).distanceTo(vrm)).toBeLessThan(1e-6);
      }
    }
  });

  it("turns each node of a .glb as its body does, however the file nests them", async () => {
    // The torso turned against the pelvis, which VRM would carry the arm with.
    const pose = bodies(0.3, -0.8);
    pose[2].quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.9));
    const turns = (skin: Skin, names: string[]) => {
      posedHand(skin, names[0], pose);
      return names.map((name) => skin.root.getObjectByName(name)!.getWorldQuaternion(new THREE.Quaternion()));
    };
    const bones = Object.keys(CONFIG.bones);
    const vrm = turns(await Skin.load(vrmFile('1'), CONFIG), bones);
    const file = gltfFile({ armOnHips: true });
    // Keyed by node name, as a node map is.
    const nodes = Object.fromEntries(bones.map((bone) => [`${bone}_node`, file.nodes[bone]]));
    const skin = await Skin.load(file.data, {
      ...CONFIG,
      format: 'gltf',
      nodes,
      bones: Object.fromEntries(Object.entries(CONFIG.bones).map(([bone, body]) => [`${bone}_node`, body])),
      rest_offsets: Object.fromEntries(Object.entries(CONFIG.rest_offsets!).map(([bone, q]) => [`${bone}_node`, q])),
    });
    turns(skin, Object.keys(nodes)).forEach((turn, i) => expect(turn.angleTo(vrm[i])).toBeLessThan(1e-6));
  });

  it('carries a hand hung from the hips once reparent hangs it from its forearm', async () => {
    const vrm = await handPosition('1', 0.4, -1.1);
    const file = gltfFile({ handOnHips: true });
    const left = await gltfSkin(file);
    expect(posedHand(left, 'leftHand_node', bodies(0.4, -1.1)).distanceTo(vrm)).toBeGreaterThan(0.1);
    const carried = await gltfSkin(file, { reparent: [[file.nodes.leftHand, file.nodes.leftLowerArm]] });
    expect(posedHand(carried, 'leftHand_node', bodies(0.4, -1.1)).distanceTo(vrm)).toBeLessThan(1e-6);
  });

  it('hides the robot it follows, and nothing else, unless hide_robot is false', async () => {
    const skin = await Skin.load(vrmFile('1'), CONFIG);
    skin.bind(mujoco, robot);
    expect(skin.hiddenBodies).toEqual([1, 2, 3, 4, 5]);
    const beside = await Skin.load(vrmFile('1'), { ...CONFIG, hide_robot: false });
    beside.bind(mujoco, robot);
    expect(beside.hiddenBodies).toEqual([]);
  });

  it('draws a .glb in MToon when asked, its shade a share of its colour', async () => {
    const file = gltfFile();
    const json = JSON.parse(new TextDecoder().decode(new Uint8Array(file.data, 20)).trim());
    const triangle = new Float32Array([0, 0, 0, 0.1, 0, 0, 0, 0.1, 0]);
    json.nodes.push({ name: 'body', mesh: 0 });
    json.scenes[0].nodes.push(json.nodes.length - 1);
    json.buffers = [{ byteLength: triangle.byteLength }];
    json.bufferViews = [{ buffer: 0, byteLength: triangle.byteLength }];
    json.accessors = [
      { bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [0.1, 0.1, 0] },
    ];
    json.meshes = [{ primitives: [{ attributes: { POSITION: 0 }, material: 0 }] }];
    json.materials = [{ pbrMetallicRoughness: { baseColorFactor: [1, 0.8, 0.6, 1] } }];
    const material = async (toon?: { shade: number; toony: number; shift: number }) => {
      const skin = await gltfSkin({ ...file, data: glb(json, new Uint8Array(triangle.buffer)) }, toon && { material: { toon } });
      skin.bind(mujoco, robot);
      return (skin.root.getObjectByName('body') as THREE.Mesh).material as THREE.Material & {
        isMToonMaterial?: boolean;
        color: THREE.Color;
        shadeColorFactor: THREE.Color;
        shadingToonyFactor: number;
      };
    };
    expect((await material()).isMToonMaterial).toBeFalsy();
    const toon = await material({ shade: 0.5, toony: 0.7, shift: 0 });
    expect(toon.isMToonMaterial).toBe(true);
    // The test robot has no lights: nothing to divide by.
    expect(toon.color.toArray()).toEqual(painted(1, 0.8, 0.6).map((c) => expect.closeTo(c, 6)));
    expect(toon.shadeColorFactor.toArray()).toEqual(painted(1, 0.8, 0.6).map((c) => expect.closeTo(c / 2, 6)));
    expect(toon.shadingToonyFactor).toBeCloseTo(0.7, 6);
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

    const [front, back] = [painted(1, 0.8, 0.6), painted(0.5, 0.4, 0.3)];
    const [color, shade] = await colours(lit(1.4));
    near(color, front.map((c) => c / 2.3));
    near(shade, back.map((c) => c / 2.3));
    near((await colours(lit(1.4, 1.5)))[0], front.map((c) => (c * 1.5) / 2.3));
    // 0.1 + 0.6 + 0.3 = 1: nothing to take away.
    near((await colours(lit(0.1)))[0], front);
  });

  it('stores a skin as painted: its sRGB textures read raw, its linear colours encoded', () => {
    const texture = () => Object.assign(new THREE.Texture(), { colorSpace: THREE.SRGBColorSpace });
    const grey = () => new THREE.Color(0.2, 0.2, 0.2);
    const toon = new MToonMaterial({ map: texture(), color: grey(), shadeColorFactor: grey() });
    const standard = new THREE.MeshStandardMaterial({ map: texture(), color: grey() });
    const coloured = new THREE.BufferGeometry();
    coloured.setAttribute('color', new THREE.Float32BufferAttribute([0.2, 0.2, 0.2], 3));
    // One material on two meshes, and twice on one, and one geometry on two: encoded once.
    const scene = new THREE.Group().add(
      new THREE.Mesh(undefined, toon),
      new THREE.Mesh(coloured, standard),
      new THREE.Mesh(coloured, [standard, standard]),
    );
    asPainted(scene);
    const [encoded] = painted(0.2, 0.2, 0.2);
    for (const material of [toon, standard]) {
      expect(material.map!.colorSpace).toBe(THREE.NoColorSpace);
      expect(material.color.r).toBeCloseTo(encoded, 6);
    }
    expect(toon.shadeColorFactor.r).toBeCloseTo(encoded, 6);
    expect(coloured.getAttribute('color').getX(0)).toBeCloseTo(encoded, 6);
  });

  it('sits the hips at hips_offset in the body frame, turning with the robot', async () => {
    const hipsAt = async (yaw: number) => {
      // 0.2 m behind the pelvis, as a quadruped's hips sit behind its mid-trunk root.
      const skin = await Skin.load(vrmFile('1'), { ...CONFIG, hips_offset: [-0.2, 0, 0] });
      skin.bind(mujoco, robot);
      skin.update(bodies(yaw, 0), 0);
      skin.root.updateMatrixWorld(true);
      return skin.root.getObjectByName('hips')!.getWorldPosition(new THREE.Vector3());
    };
    const ahead = await hipsAt(0);
    expect(ahead.x).toBeCloseTo(-0.2, 6);
    expect(ahead.y).toBeCloseTo(0.9, 6);
    expect(ahead.z).toBeCloseTo(0, 6);
    // A quarter turn left: the robot's back is now three.js +z.
    const turned = await hipsAt(Math.PI / 2);
    expect(turned.x).toBeCloseTo(0, 6);
    expect(turned.z).toBeCloseTo(0.2, 6);
  });

  it('stands the hips at their rest height whenever the root is at root_height', async () => {
    const hipsHeight = async (config: SkinConfig) => {
      const skin = await Skin.load(vrmFile('1'), config);
      skin.bind(mujoco, robot);
      skin.update(bodies(0, 0), 0);
      skin.root.updateMatrixWorld(true);
      return skin.root.getObjectByName('hips')!.getWorldPosition(new THREE.Vector3()).y;
    };
    const auto = { ...CONFIG, height_ratio: undefined };
    // The model puts the pelvis at 0.9 m, as high as the avatar's hips: the pelvis drawn
    // at 0.9 m puts them there.
    expect(await hipsHeight(auto)).toBeCloseTo(0.9, 6);
    // Resting with the pelvis at 0.6 m, the hips go half as high again as the pelvis.
    expect(await hipsHeight({ ...auto, root_height: 0.6 })).toBeCloseTo(1.35, 6);
  });

  it('stands the hips on the ground under the robot, past its own geoms', async () => {
    // Straight down from the pelvis: first the robot's own hand (body 5), then the box's
    // top (body 6) at 1 m, a step the robot stands on.
    const hits: Array<[number, number]> = [
      [5, 1.4],
      [6, 1],
    ];
    const rays: number[] = [];
    const withRay = {
      ...mujoco,
      mj_ray: (_m: MjModel, _d: MjData, origin: number[], _v: number[], _g: number[], _s: boolean, _b: number, geom: Int32Array) => {
        rays.push(origin[2]);
        const [body, height] = hits.find(([, h]) => h < origin[2]) ?? [-1, 0];
        if (body < 0) return -1;
        geom[0] = body;
        return origin[2] - height;
      },
    } as unknown as MainModule;
    const model = { ...robot, geom_bodyid: Int32Array.from([0, 1, 2, 3, 4, 5, 6]) } as unknown as MjModel;
    const skin = await Skin.load(vrmFile('1'), { ...CONFIG, height_ratio: 2 });
    skin.bind(withRay, model);
    const raised = bodies(0, 0);
    for (const body of Object.values(raised)) body.position.y = 1.9;
    skin.update(raised, 0, model, {} as MjData);
    skin.root.updateMatrixWorld(true);
    // 0.9 m above the step, twice that above it: not twice the pelvis's 1.9 m.
    expect(skin.root.getObjectByName('hips')!.getWorldPosition(new THREE.Vector3()).y).toBeCloseTo(1 + 1.8, 6);
    expect(rays).toEqual([1.9, expect.closeTo(1.4, 5)]);
  });

  describe("retarget: 'position'", () => {
    // The zero pose as MuJoCo's kinematics give it: this robot's bodies and geoms are not turned.
    type Kinematics = { xpos: Float64Array; xquat: Float64Array; geom_xpos: Float64Array; geom_xmat: Float64Array };
    const kinematics = {
      ...mujoco,
      ...SHAPES,
      MjData: class {
        xpos: Float64Array;
        xquat: Float64Array;
        geom_xpos: Float64Array;
        geom_xmat: Float64Array;
        constructor(model: MjModel) {
          this.xpos = new Float64Array(model.nbody * 3);
          this.xquat = new Float64Array(model.nbody * 4);
          this.geom_xpos = new Float64Array((model.ngeom ?? 0) * 3);
          this.geom_xmat = new Float64Array((model.ngeom ?? 0) * 9);
        }
        delete() {}
      },
      mj_resetDataKeyframe: () => {},
      mj_kinematics: (model: MjModel, data: Kinematics) => {
        for (let b = 0; b < model.nbody; b++) {
          data.xquat[b * 4] = 1;
          const p = model.body_parentid[b];
          for (let k = 0; k < 3 && b > 0; k++) data.xpos[b * 3 + k] = data.xpos[p * 3 + k] + model.body_pos[b * 3 + k];
        }
        for (let g = 0; g < (model.ngeom ?? 0); g++) {
          for (let k = 0; k < 3; k++) data.geom_xpos[g * 3 + k] = data.xpos[model.geom_bodyid[g] * 3 + k];
          data.geom_xmat.set([1, 0, 0, 0, 1, 0, 0, 0, 1], g * 9);
        }
      },
    } as unknown as MainModule;
    // Each body where the zero pose puts it, in three.js axes, `lift` metres higher.
    const zeroPose = (lift = 0): Record<number, THREE.Object3D> => {
      const at = [[0, 0, 0]];
      for (let b = 1; b < 6; b++) {
        const p = at[robot.body_parentid[b]];
        at.push([0, 1, 2].map((k) => p[k] + robot.body_pos[b * 3 + k]));
      }
      const out: Record<number, THREE.Object3D> = {};
      at.forEach(([x, y, z], b) => {
        const body = new THREE.Object3D();
        body.position.set(x, z + lift, -y);
        out[b] = body;
      });
      return out;
    };
    const boneAt = (skin: Skin, name: string) => skin.root.getObjectByName(name)!.getWorldPosition(new THREE.Vector3());

    it('lifts the hips exactly as far as the robot, the height unscaled', async () => {
      const skin = await Skin.load(vrmFile('1'), { ...CONFIG, height_ratio: 2, retarget: 'position' });
      skin.bind(kinematics, robot);
      skin.update(zeroPose(), 0);
      skin.root.updateMatrixWorld(true);
      const before = boneAt(skin, 'hips');
      skin.update(zeroPose(0.5), 16);
      skin.root.updateMatrixWorld(true);
      // By rotation, a height ratio of 2 would lift them 1 m.
      expect(boneAt(skin, 'hips').y - before.y).toBeCloseTo(0.5, 6);
    });

    it('carries each bone with its body, as a rigid part', async () => {
      const skin = await Skin.load(vrmFile('1'), { ...CONFIG, retarget: 'position' });
      skin.bind(kinematics, robot);
      const rest = zeroPose();
      skin.update(rest, 0);
      skin.root.updateMatrixWorld(true);
      const hand = boneAt(skin, 'leftHand');
      const spine = boneAt(skin, 'spine');
      // The arm swung a quarter turn about the vertical through the shoulder.
      const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2);
      const pivot = rest[3].position.clone();
      const swung = zeroPose();
      for (const b of [3, 4, 5]) {
        swung[b].position.sub(pivot).applyQuaternion(turn).add(pivot);
        swung[b].quaternion.copy(turn);
      }
      skin.update(swung, 16);
      skin.root.updateMatrixWorld(true);
      const expected = hand.clone().sub(pivot).applyQuaternion(turn).add(pivot);
      expect(boneAt(skin, 'leftHand').distanceTo(expected)).toBeCloseTo(0, 6);
      expect(boneAt(skin, 'spine').distanceTo(spine)).toBeCloseTo(0, 6);
    });

    it("hybrid puts the hand where 'position' does, the arm keeping its lengths", async () => {
      // The arm raised a quarter turn forward about the robot's shoulder, which sits
      // 0.1 m below the avatar's: turned by rotation alone, the hand ends elsewhere.
      const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), Math.PI / 2);
      const raised = () => {
        const out = zeroPose();
        const pivot = out[3].position.clone();
        for (const b of [3, 4, 5]) {
          out[b].position.sub(pivot).applyQuaternion(turn).add(pivot);
          out[b].quaternion.copy(turn);
        }
        return out;
      };
      const hand = async (retarget: SkinConfig['retarget']) => {
        const skin = await Skin.load(vrmFile('1'), { ...CONFIG, retarget });
        skin.bind(kinematics, robot);
        skin.update(zeroPose(), 0);
        skin.update(raised(), 16);
        skin.root.updateMatrixWorld(true);
        return { skin, at: boneAt(skin, 'leftHand') };
      };
      const fixed = await hand('position');
      const turned = await hand('rotation');
      const both = await hand('hybrid');
      expect(turned.at.distanceTo(fixed.at)).toBeGreaterThan(0.05);
      expect(both.at.distanceTo(fixed.at)).toBeCloseTo(0, 5);
      // Unset, it is hybrid.
      expect((await hand(undefined)).at.distanceTo(both.at)).toBeCloseTo(0, 9);
      const turnOf = (skin: Skin) => skin.root.getObjectByName('leftHand')!.getWorldQuaternion(new THREE.Quaternion());
      expect(turnOf(both.skin).angleTo(turnOf(fixed.skin))).toBeCloseTo(0, 5);
      expect(boneAt(both.skin, 'leftUpperArm').distanceTo(boneAt(both.skin, 'leftLowerArm'))).toBeCloseTo(0.25, 6);
      expect(boneAt(both.skin, 'leftLowerArm').distanceTo(both.at)).toBeCloseTo(0.25, 6);
    });

    it('hybrid lowers the avatar until the hand it stands on reaches, the arm unstretched', async () => {
      // The hand 0.3 m lower: 0.55 m below the avatar's shoulder and 0.25 m ahead, past
      // its arm's 0.5 m.
      const lowered = () => {
        const out = zeroPose();
        for (const b of [4, 5]) out[b].position.y -= 0.3;
        return out;
      };
      const posed = async (retarget: SkinConfig['retarget']) => {
        const skin = await Skin.load(vrmFile('1'), { ...CONFIG, retarget });
        skin.bind(kinematics, robot);
        skin.update(zeroPose(), 0);
        skin.update(lowered(), 16);
        skin.root.updateMatrixWorld(true);
        return skin;
      };
      const fixed = await posed('position');
      const both = await posed('hybrid');
      expect(boneAt(both, 'leftHand').distanceTo(boneAt(fixed, 'leftHand'))).toBeCloseTo(0, 5);
      expect(boneAt(both, 'leftUpperArm').distanceTo(boneAt(both, 'leftLowerArm'))).toBeCloseTo(0.25, 6);
      expect(boneAt(both, 'leftLowerArm').distanceTo(boneAt(both, 'leftHand'))).toBeCloseTo(0.25, 6);
      // No further than it takes: the arm ends straight.
      expect(boneAt(both, 'hips').y).toBeCloseTo(0.9 - (0.55 - Math.sqrt(0.5 ** 2 - 0.25 ** 2)), 3);
    });

    it('stands the avatar on the robot it is fixed to, lowest point on lowest point', async () => {
      // A colliding sphere 0.05 m across on the hand, at 0.95 m; lower down, a sphere
      // drawn only, on the pelvis, and a box on the robot's neighbour.
      const model = {
        ...robot,
        ngeom: 3,
        geom_bodyid: Int32Array.from([5, 1, 6]),
        geom_type: Int32Array.from([SPHERE, SPHERE, BOX]),
        geom_size: Float64Array.from([0.05, 0, 0, 0.3, 0, 0, 0.1, 0.1, 0.1]),
        geom_contype: Int32Array.from([1, 0, 1]),
        geom_conaffinity: Int32Array.from([1, 0, 1]),
      } as unknown as MjModel;
      const skin = await Skin.load(gloveFile(), { ...CONFIG, retarget: 'position' });
      skin.bind(kinematics, model);
      skin.update(zeroPose(), 0);
      skin.root.updateMatrixWorld(true);
      const glove = skin.root.getObjectByName('glove') as THREE.Mesh;
      const corners = [0, 1, 2].map((i) => glove.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(glove.matrixWorld));
      expect(Math.min(...corners.map((corner) => corner.y))).toBeCloseTo(0.9, 6);
    });
  });

  it('finds how low each shape of geom reaches, as it is turned', () => {
    const bottom = (type: number, size: number[], turn = new THREE.Matrix4(), aabb = [0, 0, 0, 0, 0, 0]) => {
      const model = { geom_type: [type], geom_size: size, geom_aabb: aabb } as unknown as MjModel;
      // MuJoCo's rotation matrices are row-major, three.js's column-major.
      const xmat = new THREE.Matrix3().setFromMatrix4(turn).transpose().elements;
      return geomBottom(SHAPES as unknown as MainModule, model, { geom_xpos: [0, 0, 1], geom_xmat: xmat } as unknown as MjData, 0);
    };
    const lying = new THREE.Matrix4().makeRotationY(Math.PI / 2);
    const [plane, , sphere, capsule, ellipsoid, cylinder, box, mesh] = [0, 1, 2, 3, 4, 5, 6, 7];
    expect(bottom(sphere, [0.1, 0, 0])).toBeCloseTo(0.9, 9);
    expect(bottom(capsule, [0.05, 0.2, 0])).toBeCloseTo(0.75, 9);
    expect(bottom(capsule, [0.05, 0.2, 0], lying)).toBeCloseTo(0.95, 9);
    expect(bottom(cylinder, [0.1, 0.2, 0])).toBeCloseTo(0.8, 9);
    expect(bottom(cylinder, [0.1, 0.2, 0], lying)).toBeCloseTo(0.9, 9);
    expect(bottom(ellipsoid, [0.1, 0.2, 0.3], new THREE.Matrix4().makeRotationX(Math.PI / 2))).toBeCloseTo(0.8, 9);
    expect(bottom(box, [0.1, 0.2, 0.3], new THREE.Matrix4().makeRotationX(Math.PI / 4))).toBeCloseTo(1 - 0.5 * Math.SQRT1_2, 9);
    expect(bottom(mesh, [0, 0, 0], undefined, [0, 0, -0.1, 0.1, 0.1, 0.05])).toBeCloseTo(0.85, 9);
    expect(bottom(plane, [0, 0, 0])).toBe(Infinity);
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

describe('a model rebuilt under the skin', () => {
  let real: MainModule;
  beforeAll(async () => {
    real = await (await import('mujoco')).default();
  });

  it('finds the ground in the model drawn now, the one it was bound to freed', async () => {
    // A step 0.5 m high under a free pelvis 0.9 m up.
    const xml = `<mujoco><worldbody>
      <geom type="box" pos="0 0 0.25" size="1 1 0.25"/>
      <body name="robot/pelvis" pos="0 0 0.9"><freejoint/><geom type="sphere" size="0.05" group="2"/></body>
    </worldbody></mujoco>`;
    const compile = () =>
      (real as unknown as { MjModel: { from_xml_string(xml: string): MjModel } }).MjModel.from_xml_string(xml);
    const skin = await Skin.load(vrmFile('1'), { ...CONFIG, bones: { hips: 'pelvis' }, height_ratio: 2 });
    const bound = compile();
    skin.bind(real, bound);
    // As the runtime rebuilds a scene for XR hands: the old model freed, a new one drawn.
    bound.delete();
    const model = compile();
    const data = new real.MjData(model);
    real.mj_forward(model, data);
    const pelvis = new THREE.Object3D();
    pelvis.position.set(0, 0.9, 0);
    skin.update({ 1: pelvis }, 0, model, data);
    skin.root.updateMatrixWorld(true);
    // 0.4 m above the step, twice that above it.
    expect(skin.root.getObjectByName('hips')!.getWorldPosition(new THREE.Vector3()).y).toBeCloseTo(1.3, 6);
    data.delete();
    model.delete();
    skin.dispose();
  });
});
