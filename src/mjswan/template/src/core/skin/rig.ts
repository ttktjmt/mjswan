/**
 * The skeleton a skin is posed through, whatever its file: a VRM's humanoid, or a .glb's
 * nodes as its manifest entry names them. Either way each bone has a normalized twin, its
 * rest turn taken out, that the retarget turns; `update` carries the pose onto the file.
 */
import * as THREE from 'three';
import { VRMHumanBoneParentMap, type VRMHumanBoneName, type VRMHumanoid } from '@pixiv/three-vrm';

export interface SkinRig {
  /** The bone that carries the position: a humanoid's `hips`, or the topmost mapped node. */
  readonly root: string;
  /** Off, `update` leaves the file's nodes alone, for `'position'` to set them itself. */
  autoUpdate: boolean;
  /** `bone`'s normalized node: where it rests, unturned. Null if the file lacks it. */
  node(bone: string): THREE.Object3D | null;
  /** The file's own node for `bone`. */
  rawNode(bone: string): THREE.Object3D | null;
  /** The bones `bone` hangs from, nearest first; not all of them need be in the file. */
  ancestors(bone: string): string[];
  /** The normalized pose onto the file's nodes. */
  update(): void;
  /** The file's nodes back at rest. */
  resetRawPose(): void;
}

/** A VRM's own humanoid, as three-vrm loaded it. */
export function humanoidRig(humanoid: VRMHumanoid): SkinRig {
  return {
    root: 'hips',
    get autoUpdate() {
      return humanoid.autoUpdateHumanBones;
    },
    set autoUpdate(on: boolean) {
      humanoid.autoUpdateHumanBones = on;
    },
    node: (bone) => humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName),
    rawNode: (bone) => humanoid.getRawBoneNode(bone as VRMHumanBoneName),
    ancestors: (bone) => {
      const out: string[] = [];
      for (let p = VRMHumanBoneParentMap[bone as VRMHumanBoneName]; p; p = VRMHumanBoneParentMap[p]) out.push(p);
      return out;
    },
    update: () => humanoid.update(),
    resetRawPose: () => humanoid.resetRawPose(),
  };
}

type NodeBone = {
  raw: THREE.Object3D;
  normalized: THREE.Object3D;
  ancestors: string[];
  /** At rest: the raw node's parent's world rotation, and its own local transform. */
  parentTurn: THREE.Quaternion;
  parentTurnInverse: THREE.Quaternion;
  turn: THREE.Quaternion;
  position: THREE.Vector3;
};

const _at = new THREE.Vector3();
const _scale = new THREE.Vector3();
const _unused = new THREE.Vector3();

/**
 * three-vrm's normalized humanoid rig, for any nodes, nested as the file nests them: each
 * bone's normalized node sits where its node rests, unturned, under the normalized node
 * of the nearest mapped node above. A .glb's humanoid is one of these too, so the file's
 * nesting need not be VRM's.
 */
export class NodeRig implements SkinRig {
  readonly root: string;
  /** Holds the normalized nodes; goes beside the file's scene, in the same frame. */
  readonly normalizedRoot = new THREE.Object3D();
  autoUpdate = true;
  private readonly bones = new Map<string, NodeBone>();

  /** `nodes` (bone -> the file's node) at rest. */
  constructor(nodes: Record<string, THREE.Object3D>) {
    const byNode = new Map(Object.entries(nodes).map(([bone, node]) => [node, bone]));
    const above = (node: THREE.Object3D) => {
      const out: string[] = [];
      for (let p = node.parent; p; p = p.parent) {
        const bone = byNode.get(p);
        if (bone !== undefined) out.push(bone);
      }
      return out;
    };
    // Parents first, so each normalized node's parent is made before it.
    const order = Object.entries(nodes)
      .map(([bone, node]) => ({ bone, node, ancestors: above(node) }))
      .sort((a, b) => a.ancestors.length - b.ancestors.length);
    const restAt = new Map<string, THREE.Vector3>();
    for (const { bone, node, ancestors } of order) {
      node.updateWorldMatrix(true, false);
      const at = new THREE.Vector3().setFromMatrixPosition(node.matrixWorld);
      restAt.set(bone, at);
      const parentTurn = new THREE.Quaternion();
      node.parent?.matrixWorld.decompose(_unused, parentTurn, _scale);
      const normalized = new THREE.Object3D();
      normalized.name = `Normalized_${node.name}`;
      const parent = ancestors.length ? this.bones.get(ancestors[0])!.normalized : this.normalizedRoot;
      parent.add(normalized);
      normalized.position.copy(at);
      if (ancestors.length) normalized.position.sub(restAt.get(ancestors[0])!);
      this.bones.set(bone, {
        raw: node,
        normalized,
        ancestors,
        parentTurn,
        parentTurnInverse: parentTurn.clone().invert(),
        turn: node.quaternion.clone(),
        position: node.position.clone(),
      });
    }
    this.root = order.find((bone) => !bone.ancestors.length)?.bone ?? '';
  }

  node(bone: string): THREE.Object3D | null {
    return this.bones.get(bone)?.normalized ?? null;
  }

  rawNode(bone: string): THREE.Object3D | null {
    return this.bones.get(bone)?.raw ?? null;
  }

  ancestors(bone: string): string[] {
    return [...(this.bones.get(bone)?.ancestors ?? [])];
  }

  /** As three-vrm's: each node turned in its parent's rest frame as its twin has turned. */
  update(): void {
    if (!this.autoUpdate) return;
    for (const bone of this.bones.values()) {
      bone.raw.quaternion
        .copy(bone.normalized.quaternion)
        .multiply(bone.parentTurn)
        .premultiply(bone.parentTurnInverse)
        .multiply(bone.turn);
    }
    const root = this.bones.get(this.root);
    const parent = root?.raw.parent;
    if (!root || !parent) return;
    root.normalized.getWorldPosition(_at);
    parent.updateWorldMatrix(true, false);
    root.raw.position.copy(parent.worldToLocal(_at));
  }

  resetRawPose(): void {
    for (const bone of this.bones.values()) {
      bone.raw.quaternion.copy(bone.turn);
      bone.raw.position.copy(bone.position);
    }
  }
}
