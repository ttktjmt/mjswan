/**
 * A skin's file, loaded: its scene, the rig the retarget poses, its spring bones and the
 * way it faces. A VRM brings all of them itself; a .glb is given them by its manifest
 * entry, the file left as it is.
 */
import * as THREE from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import {
  MToonMaterial,
  VRMLoaderPlugin,
  VRMSpringBoneCollider,
  VRMSpringBoneColliderShapeCapsule,
  VRMSpringBoneColliderShapeSphere,
  VRMSpringBoneJoint,
  VRMSpringBoneManager,
  VRMUtils,
  type VRM,
} from '@pixiv/three-vrm';
import { modelFrame } from './retarget';
import { NodeRig, humanoidRig, type SkinRig } from './rig';
import type { SkinConfig, SpringChainConfig, ToonConfig } from './config';

export interface SkinModel {
  /** The file's scene in metres, with its rig's normalized nodes; at rest until posed. */
  readonly scene: THREE.Object3D;
  readonly rig: SkinRig;
  readonly springBoneManager: VRMSpringBoneManager | null;
  /** `S` in retarget.ts: the turn from the file's frame onto VRM 1.0's, which faces +z. */
  readonly frame: THREE.Quaternion;
  /** Once posed: the rig onto the file's nodes, then what else the file animates. */
  update(dt: number): void;
  dispose(): void;
}

/** A model's front -> the turn about y that takes VRM 1.0's (+z) onto it. */
const FACING_TURNS = { '+z': 0, '+x': Math.PI / 2, '-z': Math.PI, '-x': -Math.PI / 2 } as const;
const UP = new THREE.Vector3(0, 1, 0);

export async function loadSkinModel(data: ArrayBuffer, config: SkinConfig): Promise<SkinModel> {
  const loader = new GLTFLoader();
  if (config.format === 'vrm') loader.register((parser) => new VRMLoaderPlugin(parser));
  const gltf = await loader.parseAsync(data, '');
  asPainted(gltf.scene);
  return config.format === 'vrm' ? fromVrm(gltf) : fromGltf(gltf, config);
}

/**
 * mjswan draws to a linear target unencoded, as MuJoCo does, so a colour reaches the
 * screen as stored. glTF stores its colours linear and three.js decodes sRGB textures:
 * left so, a skin is drawn darker than it was painted.
 */
export function asPainted(scene: THREE.Object3D): void {
  const materials = new Set<THREE.Material>();
  const geometries = new Set<THREE.BufferGeometry>();
  scene.traverse((object) => {
    const { material, geometry } = object as THREE.Mesh;
    for (const each of material ? [material].flat() : []) materials.add(each);
    if (geometry) geometries.add(geometry);
  });
  for (const material of materials) {
    // MToon keeps its colours and textures in uniforms, behind getters.
    const { uniforms } = material as THREE.ShaderMaterial;
    const values: unknown[] = uniforms ? Object.values(uniforms).map((uniform) => uniform.value) : Object.values(material);
    for (const value of values) {
      const texture = value as THREE.Texture | null;
      const colour = value as THREE.Color | null;
      if (texture?.isTexture && texture.colorSpace === THREE.SRGBColorSpace) {
        texture.colorSpace = THREE.NoColorSpace;
        texture.needsUpdate = true;
      } else if (colour?.isColor) {
        colour.convertLinearToSRGB();
      }
    }
  }
  // Vertex colours are stored linear too.
  const colour = new THREE.Color();
  for (const geometry of geometries) {
    const colours = geometry.getAttribute('color');
    if (!colours) continue;
    for (let i = 0; i < colours.count; i++) {
      colour.setRGB(colours.getX(i), colours.getY(i), colours.getZ(i)).convertLinearToSRGB();
      colours.setXYZ(i, colour.r, colour.g, colour.b);
    }
    colours.needsUpdate = true;
  }
}

function fromVrm(gltf: GLTF): SkinModel {
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) throw new Error('not a VRM file (no VRM or VRMC_vrm extension)');
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  return {
    scene: vrm.scene,
    rig: humanoidRig(vrm.humanoid),
    springBoneManager: vrm.springBoneManager ?? null,
    frame: modelFrame(vrm.meta?.metaVersion),
    // `vrm.update` but for its springs, which the skin runs itself.
    update(dt) {
      vrm.humanoid.update();
      vrm.lookAt?.update(dt);
      vrm.expressionManager?.update();
      vrm.nodeConstraintManager?.update();
      for (const material of vrm.materials ?? []) (material as { update?(dt: number): void }).update?.(dt);
    },
    dispose: () => VRMUtils.deepDispose(vrm.scene),
  };
}

async function fromGltf(gltf: GLTF, config: SkinConfig): Promise<SkinModel> {
  const node = (index: number) => gltf.parser.getDependency('node', index) as Promise<THREE.Object3D>;
  // The file is scaled to metres inside a holder that is not, so the rig's normalized
  // nodes, measured in metres, sit in the holder unscaled.
  gltf.scene.scale.setScalar(config.scale ?? 1);
  const scene = new THREE.Group();
  scene.add(gltf.scene);
  scene.updateMatrixWorld(true);
  // Before the rig reads the nesting: a foot an IK control carried then hangs from its leg.
  for (const [index, parent] of config.reparent ?? []) (await node(parent)).attach(await node(index));

  const nodes: Record<string, THREE.Object3D> = {};
  for (const [bone, index] of Object.entries(config.nodes ?? {})) nodes[bone] = await node(index);
  const rig = new NodeRig(nodes);
  scene.add(rig.normalizedRoot);
  const springBoneManager = config.springs?.length ? await springs(config.springs, node) : null;
  if (config.material?.toon) toon(gltf.scene, config.material.toon);
  return {
    scene,
    rig,
    springBoneManager,
    frame: new THREE.Quaternion().setFromAxisAngle(UP, -FACING_TURNS[config.facing ?? '+z']),
    update: () => rig.update(),
    dispose: () => VRMUtils.deepDispose(scene),
  };
}

/** Metres to `object`'s own units. */
function unitsOf(object: THREE.Object3D): number {
  return 1 / object.getWorldScale(new THREE.Vector3()).x;
}

/** Each chain as three-vrm springs, its lengths taken from metres to the nodes' units. */
async function springs(
  chains: SpringChainConfig[],
  node: (index: number) => Promise<THREE.Object3D>,
): Promise<VRMSpringBoneManager> {
  const manager = new VRMSpringBoneManager();
  const v = (xyz: [number, number, number], units: number) => new THREE.Vector3(...xyz).multiplyScalar(units);
  for (const chain of chains) {
    const colliders = await Promise.all(
      chain.colliders.map(async ({ node: index, radius, offset, tail }) => {
        const at = await node(index);
        const units = unitsOf(at);
        // three-vrm reads a collider's radius in metres, its offset and tail in the node's units.
        const shape = tail
          ? new VRMSpringBoneColliderShapeCapsule({ radius, offset: v(offset, units), tail: v(tail, units) })
          : new VRMSpringBoneColliderShapeSphere({ radius, offset: v(offset, units) });
        const collider = new VRMSpringBoneCollider(shape);
        at.add(collider);
        return collider;
      }),
    );
    const settings = {
      hitRadius: chain.hit_radius,
      stiffness: chain.stiffness,
      dragForce: chain.drag,
      gravityPower: chain.gravity,
      gravityDir: new THREE.Vector3(...chain.gravity_dir),
    };
    const groups = [{ colliders }];
    const nodes = await Promise.all(chain.joints.map(node));
    if (chain.tip !== undefined) nodes.push(tip(nodes[nodes.length - 1], chain.tip));
    // As in a VRM spring, each node swings toward the next; the last only marks an end.
    for (let i = 0; i + 1 < nodes.length; i++) {
      manager.addJoint(new VRMSpringBoneJoint(nodes[i], nodes[i + 1], settings, groups));
    }
  }
  manager.setInitState();
  return manager;
}

/**
 * A child `length` metres on from `bone`, the way three-vrm points a tail it is not
 * given: along the bone's offset from its parent, read in its own frame.
 */
function tip(bone: THREE.Object3D, length: number): THREE.Object3D {
  const end = new THREE.Object3D();
  end.position.copy(bone.position).setLength(length * unitsOf(bone));
  bone.add(end);
  return end;
}

/** Each PBR material as MToon: lit its base colour and map, shade a share of them. */
function toon(scene: THREE.Object3D, style: ToonConfig): void {
  const swapped = new Map<THREE.Material, THREE.Material>();
  const swap = (material: THREE.Material): THREE.Material => {
    const pbr = material as THREE.MeshStandardMaterial;
    if (!pbr.isMeshStandardMaterial) return material;
    let out = swapped.get(pbr);
    if (!out) {
      out = new MToonMaterial({
        color: pbr.color.clone(),
        map: pbr.map ?? undefined,
        shadeColorFactor: pbr.color.clone().multiplyScalar(style.shade),
        shadeMultiplyTexture: pbr.map ?? undefined,
        shadingToonyFactor: style.toony,
        shadingShiftFactor: style.shift,
        transparent: pbr.transparent,
        alphaTest: pbr.alphaTest,
        side: pbr.side,
        vertexColors: pbr.vertexColors,
      });
      swapped.set(pbr, out);
      pbr.dispose();
    }
    return out;
  };
  scene.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(swap) : swap(mesh.material);
  });
}
