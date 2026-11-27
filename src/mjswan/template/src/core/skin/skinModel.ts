/**
 * A skin's file, loaded: its scene, the rig the retarget poses, its spring bones and the
 * way it faces. A VRM brings all of them itself; a .glb is given them by its manifest
 * entry, the file left as it is.
 */
import * as THREE from 'three';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
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
import { parseSealedGltf } from '../scene/gltf';
import { BoundedSpringBones, checkConstraints, checkSpringChains, checkSpringUpdates } from './budget';
import { modelFrame } from './retarget';
import { NodeRig, humanoidRig, type SkinRig } from './rig';
import type { SkinConfig, SolidConfig, SpringChainConfig, ToonConfig } from './config';

export interface SkinModel {
  /** The file's scene in metres, with its rig's normalized nodes; at rest until posed. */
  readonly scene: THREE.Object3D;
  readonly rig: SkinRig;
  /**
   * A .glb's skin joints, each at its bind pose in `scene`'s frame: where it stands for
   * its meshes to be drawn as the file models them. Empty for a VRM.
   */
  readonly binds: ReadonlyMap<THREE.Object3D, THREE.Matrix4>;
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
  const gltf = await parseSealedGltf(
    data,
    config.format === 'vrm'
      ? (parser) => new VRMLoaderPlugin(parser, { springBonePlugin: new BoundedSpringBones(parser) })
      : undefined,
  );
  return config.format === 'vrm' ? fromVrm(gltf) : fromGltf(gltf, config);
}

/**
 * The switch between a skin drawn as painted and as its file says. The classic renderer
 * draws to a linear target unencoded, as MuJoCo does, so a colour reaches the screen as
 * stored; glTF stores its colours linear and three.js decodes sRGB textures, so left so a
 * skin is drawn darker than it was painted: painted, its sRGB textures read raw and its
 * colours encoded. Filament encodes the output, so it draws the file's own. A material
 * coloured from an `rgba` (`SolidShading`) is the scene's graphics settings' to read.
 */
export function paintable(scene: THREE.Object3D): (painted: boolean) => void {
  const materials = new Set<THREE.Material>();
  const geometries = new Set<THREE.BufferGeometry>();
  scene.traverse((object) => {
    const { material, geometry } = object as THREE.Mesh;
    for (const each of material ? [material].flat() : []) {
      if (!each.userData.rgba) materials.add(each);
    }
    if (geometry) geometries.add(geometry);
  });
  const textures = new Set<THREE.Texture>();
  // Each colour, and what the file has it as.
  const colours = new Map<THREE.Color, THREE.Color>();
  for (const material of materials) {
    // MToon keeps its colours and textures in uniforms, behind getters.
    const { uniforms } = material as THREE.ShaderMaterial;
    const values: unknown[] = uniforms ? Object.values(uniforms).map((uniform) => uniform.value) : Object.values(material);
    for (const value of values) {
      const texture = value as THREE.Texture | null;
      const colour = value as THREE.Color | null;
      if (texture?.isTexture && texture.colorSpace === THREE.SRGBColorSpace) textures.add(texture);
      else if (colour?.isColor) colours.set(colour, colour.clone());
    }
  }
  // Vertex colours are stored linear too.
  const vertexColours = new Map<THREE.BufferAttribute | THREE.InterleavedBufferAttribute, number[]>();
  for (const geometry of geometries) {
    const attribute = geometry.getAttribute('color');
    if (!attribute) continue;
    const file: number[] = [];
    for (let i = 0; i < attribute.count; i++) file.push(attribute.getX(i), attribute.getY(i), attribute.getZ(i));
    vertexColours.set(attribute, file);
  }
  const colour = new THREE.Color();
  return (painted) => {
    const space = painted ? THREE.NoColorSpace : THREE.SRGBColorSpace;
    for (const texture of textures) {
      if (texture.colorSpace === space) continue;
      texture.colorSpace = space;
      texture.needsUpdate = true;
    }
    // Every time, so a colour scaled since (a toon's, by the light) starts again from the file.
    for (const [each, file] of colours) {
      each.copy(file);
      if (painted) each.convertLinearToSRGB();
    }
    for (const [attribute, file] of vertexColours) {
      for (let i = 0; i < attribute.count; i++) {
        colour.setRGB(file[i * 3], file[i * 3 + 1], file[i * 3 + 2]);
        if (painted) colour.convertLinearToSRGB();
        attribute.setXYZ(i, colour.r, colour.g, colour.b);
      }
      (attribute as THREE.BufferAttribute).needsUpdate = true;
    }
  };
}

function fromVrm(gltf: GLTF): SkinModel {
  const vrm = gltf.userData.vrm as VRM | undefined;
  if (!vrm) throw new Error('not a VRM file (no VRM or VRMC_vrm extension)');
  checkConstraints(vrm);
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  // Settled once: mjswan sets no expression weight, and each update costs binds x primitives.
  vrm.expressionManager?.update();
  return {
    scene: vrm.scene,
    rig: humanoidRig(vrm.humanoid),
    binds: new Map(),
    springBoneManager: vrm.springBoneManager ?? null,
    frame: modelFrame(vrm.meta?.metaVersion),
    // `vrm.update` but for its springs, which the skin runs itself, and its expressions, settled above.
    update(dt) {
      vrm.humanoid.update();
      vrm.lookAt?.update(dt);
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
  if (config.material?.solid) solid(gltf.scene, config.material.solid);
  return {
    scene,
    rig,
    binds: bindPoses(gltf.scene),
    springBoneManager,
    frame: new THREE.Quaternion().setFromAxisAngle(UP, -FACING_TURNS[config.facing ?? '+z']),
    update: () => rig.update(),
    dispose: () => VRMUtils.deepDispose(scene),
  };
}

/** Each skin joint's bind pose (the inverse of its inverse bind matrix) in `file`'s parent's frame. */
function bindPoses(file: THREE.Object3D): Map<THREE.Object3D, THREE.Matrix4> {
  const binds = new Map<THREE.Object3D, THREE.Matrix4>();
  file.traverse((object) => {
    const { skeleton } = object as THREE.SkinnedMesh;
    skeleton?.bones.forEach((bone, i) => {
      if (!binds.has(bone)) binds.set(bone, file.matrix.clone().multiply(skeleton.boneInverses[i].clone().invert()));
    });
  });
  return binds;
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
  checkSpringChains(chains);
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
  // checkSpringChains counts joints, not the nodes each re-matrixes: a chain may name one node over and over.
  checkSpringUpdates([...manager.joints].map((joint) => joint.bone));
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
        // A share of the colour as painted, so the shade is as dark under either renderer.
        shadeColorFactor: pbr.color.clone().convertLinearToSRGB().multiplyScalar(style.shade).convertSRGBToLinear(),
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

/**
 * Every mesh in one colour, not metallic. A mesh with no normals, which three.js draws
 * faceted, is given smooth ones unless `smooth` is off.
 */
function solid(scene: THREE.Object3D, { color, opacity, smooth, roughness, clearcoat, clearcoat_roughness }: SolidConfig): void {
  const material = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(...color),
    opacity,
    transparent: opacity < 1,
    roughness,
    clearcoat,
    clearcoatRoughness: clearcoat_roughness,
  });
  // Stored as MuJoCo's `rgba` is, so read as the scene reads a geom's (`setBaseColor`).
  material.userData.rgba = [...color];
  const faceted = smooth ? material : Object.assign(material.clone(), { flatShading: true });
  const replaced = new Set<THREE.Material>();
  scene.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const each of [mesh.material].flat()) replaced.add(each);
    if (smooth && !mesh.geometry.getAttribute('normal')) mesh.geometry.computeVertexNormals();
    mesh.material = mesh.geometry.getAttribute('normal') ? material : faceted;
  });
  for (const each of replaced) each.dispose();
}
