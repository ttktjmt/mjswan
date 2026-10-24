/**
 * The spring model against the real WASM and real three-vrm, on a hand-built VRM 1.0: a
 * strand of three bones hanging back from the head, and a sphere collider on the head.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { MainModule, MjModel } from 'mujoco';
import type { VRM } from '@pixiv/three-vrm';
import { SpringPhysics } from '../springs';
import { Skin } from '../skin';

type Joint = { stiffness: number; gravityPower: number; dragForce?: number };

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

/** Bone, parent, offset from the parent (VRM 1.0: facing +z, left at +x). */
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
/** From 0.1 m above and behind the head centre, three 0.1 m bones straight back. */
const STRAND: Array<[string, string, [number, number, number]]> = [
  ['hair0', 'head', [0, 0.1, -0.1]],
  ['hair1', 'hair0', [0, 0, -0.1]],
  ['hair2', 'hair1', [0, 0, -0.1]],
];

/** The skeleton and the strand; `collider` puts a sphere on the head at that offset. */
function vrmFile(joint: Joint, collider?: { offset: [number, number, number]; radius: number }): ArrayBuffer {
  const all = [...SKELETON, ...STRAND];
  const index = new Map(all.map(([name], i) => [name, i]));
  const nodes = all.map(([name, , translation]) => ({
    name,
    translation,
    children: all.filter(([, parent]) => parent === name).map(([child]) => index.get(child)!),
  }));
  const head = index.get('head')!;
  const settings = { hitRadius: 0.02, gravityDir: [0, -1, 0], dragForce: 0.4, ...joint };
  return glb({
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes,
    extensionsUsed: ['VRMC_vrm', 'VRMC_springBone'],
    extensions: {
      VRMC_vrm: {
        specVersion: '1.0',
        meta: { name: 't', authors: ['mjswan'], licenseUrl: 'https://vrm.dev/licenses/1.0/' },
        humanoid: { humanBones: Object.fromEntries(SKELETON.map(([bone], node) => [bone, { node }])) },
      },
      VRMC_springBone: {
        specVersion: '1.0',
        colliders: collider ? [{ node: head, shape: { sphere: collider } }] : [],
        colliderGroups: collider ? [{ colliders: [0] }] : [],
        springs: [
          {
            joints: STRAND.map(([name]) => ({ node: index.get(name)!, ...settings })),
            colliderGroups: collider ? [0] : [],
          },
        ],
      },
    },
  });
}

/** The skeleton and the strand as a plain .glb, its springs left to its manifest entry. */
function gltfFile(): { data: ArrayBuffer; index: Map<string, number> } {
  const all = [...SKELETON, ...STRAND];
  const index = new Map(all.map(([name], i) => [name, i]));
  const nodes = all.map(([name, , translation]) => ({
    name,
    translation,
    children: all.filter(([, parent]) => parent === name).map(([child]) => index.get(child)!),
  }));
  return { data: glb({ asset: { version: '2.0' }, scene: 0, scenes: [{ nodes: [0] }], nodes }), index };
}

async function loadVrm(data: ArrayBuffer): Promise<VRM> {
  const { VRMLoaderPlugin } = await import('@pixiv/three-vrm');
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  return (await loader.parseAsync(data, '')).userData.vrm as VRM;
}

describe('SpringPhysics', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  async function rig(joint: Joint, collider?: { offset: [number, number, number]; radius: number }) {
    const vrm = await loadVrm(vrmFile(joint, collider));
    const physics = SpringPhysics.build(mujoco, vrm.springBoneManager!, vrm.scene)!;
    const head = vrm.scene.getObjectByName('head')!;
    const bone = (name: string) => vrm.scene.getObjectByName(name)!;
    const world = (name: string) => {
      vrm.scene.updateMatrixWorld(true);
      return bone(name).getWorldPosition(new THREE.Vector3());
    };
    /** `seconds` of 60 fps frames; `move(t)` places the head before each. */
    const run = (seconds: number, move?: (t: number) => void, each?: (t: number) => void) => {
      for (let frame = 1; frame <= Math.round(seconds * 60); frame++) {
        move?.(frame / 60);
        vrm.scene.updateMatrixWorld(true);
        physics.step(1 / 60);
        each?.(frame / 60);
      }
    };
    vrm.scene.updateMatrixWorld(true);
    physics.reset();
    return { vrm, physics, head, bone, world, run };
  }

  it('makes a ball joint of each spring bone and anchors the strand at the head', async () => {
    const vrm = await loadVrm(vrmFile({ stiffness: 1, gravityPower: 0 }, { offset: [0, 0, 0], radius: 0.05 }));
    const { xml, anchors, links } = SpringPhysics.describe(vrm.springBoneManager!, vrm.scene);
    expect(links.map((l) => l.bone.name)).toEqual(['hair0', 'hair1']);
    expect(anchors.map((a) => a.node.name)).toEqual(['head']);
    expect(xml.match(/type="ball"/g)).toHaveLength(2);
    expect(SpringPhysics.build(mujoco, vrm.springBoneManager!, vrm.scene)!.jointCount).toBe(2);
  });

  it('holds the rest pose with no gravity and a still head', async () => {
    const { run, world } = await rig({ stiffness: 1, gravityPower: 0 });
    const tip = world('hair2');
    run(1);
    expect(world('hair2').distanceTo(tip)).toBeLessThan(1e-4);
  });

  it('hangs under gravity with no stiffness, and holds level against it when stiff', async () => {
    const limp = await rig({ stiffness: 0, gravityPower: 0.3 });
    const root = limp.world('hair0');
    limp.run(3);
    const hang = limp.world('hair2').sub(root);
    // Two 0.1 m bones straight down.
    expect(hang.y).toBeLessThan(-0.19);

    const stiff = await rig({ stiffness: 4, gravityPower: 0.05 });
    stiff.run(3);
    const held = stiff.world('hair2').sub(stiff.world('hair0'));
    expect(Math.abs(held.y)).toBeLessThan(0.05);
  });

  it('lags behind the head as it turns, and settles back once it stops', async () => {
    const { run, head, world, vrm } = await rig({ stiffness: 1, gravityPower: 0 });
    const up = new THREE.Vector3(0, 1, 0);
    // The strand's direction in the head's own frame: rest is straight back (-z).
    const relative = () => {
      const d = world('hair2').sub(world('hair0'));
      const q = head.getWorldQuaternion(new THREE.Quaternion()).invert();
      return d.applyQuaternion(q).normalize();
    };
    let lag = 0;
    // A quarter turn about the vertical in 0.15 s, then held.
    run(
      1,
      (t) => head.quaternion.setFromAxisAngle(up, (Math.PI / 2) * Math.min(t / 0.15, 1)),
      () => (lag = Math.max(lag, relative().angleTo(new THREE.Vector3(0, 0, -1)))),
    );
    expect(lag).toBeGreaterThan((5 * Math.PI) / 180);
    expect(relative().angleTo(new THREE.Vector3(0, 0, -1))).toBeLessThan((1 * Math.PI) / 180);
    void vrm;
  });

  it('keeps the strand out of a collider it falls onto', async () => {
    // A sphere under the strand's middle; limp and heavy, it would otherwise hang through.
    const radius = 0.05;
    const offset: [number, number, number] = [0, 0, -0.2];
    const { run, world, head } = await rig({ stiffness: 0, gravityPower: 0.3 }, { offset, radius });
    run(3);
    head.updateMatrixWorld(true);
    const centre = head.localToWorld(new THREE.Vector3(...offset));
    for (const name of ['hair1', 'hair2']) {
      expect(world(name).distanceTo(centre)).toBeGreaterThan(radius + 0.02 - 0.01);
    }
  });

  it('starts over at the pose on reset', async () => {
    const { run, physics, world, head } = await rig({ stiffness: 0, gravityPower: 0.3 });
    run(1);
    head.position.x += 5;
    head.updateMatrixWorld(true);
    physics.reset();
    const root = world('hair0');
    // At rest again: straight back from the root.
    expect(world('hair2').sub(root).z).toBeCloseTo(-0.2, 4);
  });

  describe('in a Skin', () => {
    /** A robot that is a pelvis on a free joint, standing at 0.9 m. */
    const robot = {
      nbody: 2,
      body_parentid: Int32Array.from([0, 0]),
      body_rootid: Int32Array.from([0, 1]),
      body_pos: Float64Array.from([0, 0, 0, 0, 0, 0.9]),
      body_quat: Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0]),
      body_jntadr: Int32Array.from([-1, 0]),
      body_jntnum: Int32Array.from([0, 1]),
      jnt_type: Int32Array.from([0]),
      body: (b: number) => ({ name: ['world', 'pelvis'][b] }),
    } as unknown as MjModel;
    const pelvis = new THREE.Object3D();
    pelvis.position.set(0, 0.9, 0);

    /** The strand as a .glb's manifest entry gives it, as the VRM's own settings have it. */
    async function gltfSkin(config: { mujoco_springs?: boolean; tip?: number } = {}): Promise<Skin> {
      const { data, index } = gltfFile();
      const joints = STRAND.map(([name]) => index.get(name)!);
      return Skin.load(data, {
        format: 'gltf',
        bones: { hips: 'pelvis' },
        nodes: { hips: 0 },
        mujoco_springs: config.mujoco_springs,
        springs: [
          {
            joints,
            stiffness: 0,
            drag: 0.4,
            gravity: 0.3,
            gravity_dir: [0, -1, 0],
            hit_radius: 0.02,
            ...(config.tip === undefined ? {} : { tip: config.tip }),
            colliders: [],
          },
        ],
      });
    }

    async function strandDrop(mujocoSprings: boolean | undefined, load?: () => Promise<Skin>) {
      const vrm = () =>
        Skin.load(vrmFile({ stiffness: 0, gravityPower: 0.3 }), {
          format: 'vrm',
          bones: { hips: 'pelvis' },
          offset: [0, 0, 0],
          mujoco_springs: mujocoSprings,
        });
      const skin = await (load ?? vrm)();
      skin.bind(mujoco, robot);
      const tip = () => {
        skin.root.updateMatrixWorld(true);
        return skin.root.getObjectByName('hair2')!.getWorldPosition(new THREE.Vector3()).y;
      };
      skin.update({ 1: pelvis }, 0);
      const before = tip();
      for (let frame = 1; frame <= 120; frame++) skin.update({ 1: pelvis }, (frame * 1000) / 60);
      return { joints: skin.simulatedSpringJoints, drop: before - tip() };
    }

    it('starts the springs over after a stall rather than whipping them through it', async () => {
      for (const mujoco_springs of [true, false]) {
        const skin = await Skin.load(vrmFile({ stiffness: 0, gravityPower: 0.3 }), {
          format: 'vrm',
          bones: { hips: 'pelvis' },
          offset: [0, 0, 0],
          mujoco_springs,
        });
        skin.bind(mujoco, robot);
        const strand = () => {
          skin.root.updateMatrixWorld(true);
          const at = (name: string) => skin.root.getObjectByName(name)!.getWorldPosition(new THREE.Vector3());
          return at('hair2').sub(at('hair0'));
        };
        for (let frame = 0; frame <= 120; frame++) skin.update({ 1: pelvis }, (frame * 1000) / 60);
        expect(strand().y).toBeLessThan(-0.15);
        skin.update({ 1: pelvis }, 2000 + 1000);
        // Level and full length, as at rest: MuJoCo's reset lands on it, three-vrm's steps one frame from it.
        expect(strand().y).toBeCloseTo(0, mujoco_springs ? 4 : 1);
        expect(strand().length()).toBeCloseTo(0.2, mujoco_springs ? 4 : 2);
      }
    });

    it("swings a .glb's chains as the VRM's own springs, its last node only an end unless tipped", async () => {
      for (const mujocoSprings of [true, false]) {
        const vrm = await strandDrop(mujocoSprings);
        const gltf = await strandDrop(mujocoSprings, () => gltfSkin({ mujoco_springs: mujocoSprings }));
        expect(gltf.joints).toBe(vrm.joints);
        expect(gltf.drop).toBeCloseTo(vrm.drop, 6);
      }
      const tipped = await strandDrop(true, () => gltfSkin({ mujoco_springs: true, tip: 0.1 }));
      expect(tipped.joints).toBe(3);
      expect(tipped.drop).toBeGreaterThan(0.15);
    });

    it('runs the springs in three-vrm, or in MuJoCo when asked to', async () => {
      const vrmRun = await strandDrop(undefined);
      expect(vrmRun.joints).toBe(0);
      expect(vrmRun.drop).toBeGreaterThan(0.15);
      const mujocoRun = await strandDrop(true);
      expect(mujocoRun.joints).toBe(2);
      expect(mujocoRun.drop).toBeGreaterThan(0.15);
    });
  });
});
