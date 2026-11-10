/**
 * Limits on what a skin (often a third party's avatar) makes three-vrm do. A skin past one
 * is refused at load, and applySkin runs the scene without it.
 */
import type * as THREE from 'three';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRMSpringBoneLoaderPlugin, type VRM } from '@pixiv/three-vrm';
import type { SpringChainConfig } from './config';

/** Node visits the constraint walk may make per update: an avatar's are a few hundred. */
const MAX_CONSTRAINT_WORK = 1 << 16;
/** That walk rebuilds each node's ancestor list every update, faster than linearly in depth. */
const MAX_CONSTRAINT_DEPTH = 256;
/** Spring joints: three-vrm sorts them at load and steps each up to 4 times a frame. */
const MAX_SPRING_JOINTS = 1024;
/** Per step, over all joints: collider groups looped plus colliders tested. */
const MAX_SPRING_TESTS = 1 << 16;
/** Per step, over all joints: objects below a joint, up to the next, that it re-matrixes. */
const MAX_SPRING_UPDATES = 8192;
/** At load: the joint sort's calls and the ancestors it walks to each dependency. */
const MAX_SPRING_SORT = 1 << 22;
/** Deepest a joint's parent or collider may sit: each walk builds its path by unshift. */
const MAX_SPRING_DEPTH = 256;

function depth(object: THREE.Object3D | null): number {
  let d = 0;
  for (let o = object; o; o = o.parent) d++;
  return d;
}

/** Refuse a VRM whose node constraints would cost more per update than an avatar's. */
export function checkConstraints(vrm: VRM): void {
  let work = 0;
  for (const c of vrm.nodeConstraintManager?.constraints ?? []) {
    for (const node of [...c.dependencies, c.source, c.destination]) {
      const d = depth(node);
      if (d > MAX_CONSTRAINT_DEPTH) throw new Error(`VRM node constraint on a node ${d} deep (at most ${MAX_CONSTRAINT_DEPTH})`);
      work += d;
    }
  }
  if (work > MAX_CONSTRAINT_WORK) {
    throw new Error(`VRM node constraints visit ${work} nodes per update (at most ${MAX_CONSTRAINT_WORK})`);
  }
}

/** Refuse a manifest's spring chains past the joint or joint-collider test budget. */
export function checkSpringChains(chains: readonly SpringChainConfig[]): void {
  let joints = 0;
  let tests = 0;
  for (const c of chains) {
    if (!Array.isArray(c.joints) || !Array.isArray(c.colliders)) throw new Error('a spring chain needs joints and colliders lists');
    const n = Math.max(0, c.joints.length - (c.tip === undefined ? 1 : 0));
    joints += n;
    tests += n * c.colliders.length;
  }
  if (!(joints <= MAX_SPRING_JOINTS && tests <= MAX_SPRING_TESTS)) {
    throw new Error(`a skin's springs make ${joints} joints and ${tests} collider tests per update (at most ${MAX_SPRING_JOINTS} and ${MAX_SPRING_TESTS})`);
  }
}

/** three-vrm's spring import, refused before it imports or sorts anything past the budget. */
export class BoundedSpringBones extends VRMSpringBoneLoaderPlugin {
  override async afterRoot(gltf: GLTF): Promise<void> {
    const nodes = (await gltf.parser.getDependencies('node')) as THREE.Object3D[];
    const over = overBudget(springWork(gltf.parser.json, nodes));
    if (over) throw new Error(`a skin's springs exceed ${over}`);
    return super.afterRoot(gltf);
  }
}

export interface SpringWork { joints: number; tests: number; updates: number; sort: number; depth: number }

/** An object three-vrm's spring import will see: a loaded one, or a collider it attaches. */
interface Placed { parent: Placed | null; children: Placed[]; joints: number }

const list = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);
const field = (v: unknown, key: string): unknown => (typeof v === 'object' && v !== null ? (v as Record<string, unknown>)[key] : undefined);

/**
 * The work three-vrm 3.5.5's spring import and step will do, counted on the loaded nodes:
 * VRM 1.0 makes a joint per spring joint after the first, VRM 0.x one per object under each
 * listed root, including attached colliders and a multi-primitive mesh's child meshes, which
 * the glTF JSON does not show. Stops once past a budget.
 */
export function springWork(json: unknown, nodes: THREE.Object3D[]): SpringWork {
  const placed = new Map<THREE.Object3D, Placed>();
  const place = (node: THREE.Object3D): Placed => {
    let top = node;
    while (top.parent) top = top.parent;
    if (!placed.has(top)) {
      placed.set(top, { parent: null, children: [], joints: 0 });
      for (const stack = [top]; stack.length; ) {
        const object = stack.pop()!;
        const p = placed.get(object)!;
        for (const child of object.children) {
          const c: Placed = { parent: p, children: [], joints: 0 };
          placed.set(child, c);
          p.children.push(c);
          stack.push(child);
        }
      }
    }
    return placed.get(node)!;
  };
  const attach = (node: THREE.Object3D | undefined): Placed | null => {
    if (!node) return null;
    const parent = place(node);
    const collider: Placed = { parent, children: [], joints: 0 };
    parent.children.push(collider);
    return collider;
  };
  // Every reference is looped each step, an empty group's too.
  const groupsOf = (refs: unknown, all: Array<Placed[] | null>) => {
    const groups = list(refs).map((i) => all[i as number]).filter((g): g is Placed[] => !!g);
    return { groups, tests: groups.reduce((n, g) => n + 1 + g.length, 0) };
  };
  const work: SpringWork = { joints: 0, tests: 0, updates: 0, sort: 0, depth: 0 };
  const joints: Array<{ bone: Placed; groups: Placed[][] }> = [];
  const add = (bone: Placed, set: { groups: Placed[][]; tests: number }): boolean => {
    bone.joints++;
    joints.push({ bone, groups: set.groups });
    work.joints++;
    work.tests += set.tests;
    return work.joints > MAX_SPRING_JOINTS || work.tests > MAX_SPRING_TESTS;
  };
  const nodeAt = (index: unknown) => nodes[index as number] as THREE.Object3D | undefined;

  const v1 = field(field(json, 'extensions'), 'VRMC_springBone');
  const v1Colliders = list(field(v1, 'colliders')).map((c) => attach(nodeAt(field(c, 'node'))));
  const v1Groups = list(field(v1, 'colliderGroups')).map((g) =>
    list(field(g, 'colliders')).map((i) => v1Colliders[i as number]).filter((c): c is Placed => !!c),
  );
  for (const spring of list(field(v1, 'springs'))) {
    const set = groupsOf(field(spring, 'colliderGroups'), v1Groups);
    const chain = list(field(spring, 'joints'));
    for (let k = 0; k + 1 < chain.length; k++) {
      const node = nodeAt(field(chain[k], 'node'));
      if (node && add(place(node), set)) return work;
    }
  }
  const v0 = field(field(field(json, 'extensions'), 'VRM'), 'secondaryAnimation');
  const v0Groups = list(field(v0, 'colliderGroups')).map((g) => {
    const node = nodeAt(field(g, 'node'));
    return node ? list(field(g, 'colliders')).map(() => attach(node)!) : null;
  });
  for (const group of list(field(v0, 'boneGroups'))) {
    const set = groupsOf(field(group, 'colliderGroups'), v0Groups);
    for (const root of list(field(group, 'bones'))) {
      const start = nodeAt(root);
      for (const stack = start ? [place(start)] : []; stack.length; ) {
        const object = stack.pop()!;
        if (add(object, set)) return work;
        for (const child of object.children) stack.push(child);
      }
    }
  }

  // Each step re-matrixes, per joint, the objects below its bone up to the next joint's.
  const below = new Map<Placed, number>();
  work.sort = joints.length;
  for (const { bone, groups } of joints) {
    if (!below.has(bone)) {
      let n = 0;
      for (const stack = [...bone.children]; stack.length; ) {
        const object = stack.pop()!;
        if (object.joints) continue;
        n++;
        for (const child of object.children) stack.push(child);
      }
      below.set(bone, n);
    }
    work.updates += below.get(bone)!;
    if (work.updates > MAX_SPRING_UPDATES) return work;
    // The sort walks from the root to each dependency (the bone's parent, each distinct
    // collider) and calls itself for every joint on the way.
    const dependencies = new Set<Placed>(bone.parent ? [bone.parent] : []);
    for (const g of new Set(groups)) for (const c of g) dependencies.add(c);
    for (const dependency of dependencies) {
      let d = 0;
      for (let a: Placed | null = dependency; a; a = a.parent) {
        d++;
        work.sort += 1 + a.joints;
      }
      work.depth = Math.max(work.depth, d);
      if (d > MAX_SPRING_DEPTH || work.sort > MAX_SPRING_SORT) return work;
    }
  }
  return work;
}

function overBudget(w: SpringWork): string | null {
  if (w.joints > MAX_SPRING_JOINTS) return `${MAX_SPRING_JOINTS} joints`;
  if (w.tests > MAX_SPRING_TESTS) return `${MAX_SPRING_TESTS} collider tests a step`;
  if (w.updates > MAX_SPRING_UPDATES) return `${MAX_SPRING_UPDATES} updates a step`;
  if (w.depth > MAX_SPRING_DEPTH) return `a depth of ${MAX_SPRING_DEPTH}`;
  if (w.sort > MAX_SPRING_SORT) return `${MAX_SPRING_SORT} sort steps`;
  return null;
}
