/**
 * A scene's manifest `skins` entry, as `mjswan.SkinConfig` resolves it at build time:
 * every name a node index and every measurement made, so the browser reads no names.
 * Types only, so the manifest and the engine's types bring no skin code with them.
 */

/** One skin (see `mjswan.SkinConfig`). */
export interface SkinConfig {
  /** `'vrm'` names its own bones; `'gltf'`, a `.glb`, has them named by `nodes`. */
  format: 'vrm' | 'gltf';
  /** Bone -> robot body: VRM humanoid bone names for a VRM, `nodes` keys for a `.glb`. */
  bones: Record<string, string>;
  /** `.glb` only: bone -> node index. The topmost of them carries the position. */
  nodes?: Record<string, number>;
  /** The arms and legs `'hybrid'` bends, as `[upper, lower, end]` bones. */
  limbs?: Array<[string, string, string]>;
  /** `.glb` only: `[node, new parent]` moves, in order, each keeping its rest pose. */
  reparent?: Array<[number, number]>;
  /** `.glb` only: metres per unit of the file. Absent: 1. */
  scale?: number;
  /** `.glb` only: the axis the model faces. Absent: `'+z'`. */
  facing?: '+z' | '-z' | '+x' | '-x';
  /** `.glb` only: chains that swing as a VRM's spring bones do. */
  springs?: SpringChainConfig[];
  /** `.glb` only: its PBR materials drawn as MToon. */
  material?: { toon: ToonConfig };
  /** Bone -> `[x, y, z, w]`, in the avatar's VRM-1.0 frame: its turn at the rest pose. */
  rest_offsets?: Record<string, ArrayLike<number>>;
  /** Robot root height -> avatar root height. Absent: the avatar root's height at rest over `root_height`. */
  height_ratio?: number;
  /** The robot root's height above its soles at the rest pose. Absent: at the zero pose. */
  root_height?: number;
  /** Added to the avatar's position, in MuJoCo's world frame (x forward, y left, z up), metres. Absent: zero. */
  offset?: ArrayLike<number>;
  /** Where the root bone sits on its body, in that body's frame (MuJoCo axes), metres. */
  hips_offset?: ArrayLike<number>;
  /** Stop drawing the robot the skin follows, the avatar in its place. Absent: true; false draws both. */
  hide_robot?: boolean;
  /** Scales the toon colours once they are divided by the scene's light. Absent: 1. */
  brightness?: number;
  /** True simulates the spring bones in MuJoCo instead of three-vrm. Absent: false. */
  mujoco_springs?: boolean;
  /**
   * `'rotation'` turns each bone as its body has, only the root placed; `'position'`
   * fixes each to its body, as a robot's own meshes are. Absent: `'hybrid'`, turned,
   * then each limb bent to put its end where `'position'` would.
   */
  retarget?: 'rotation' | 'position' | 'hybrid';
}

/** A chain of nodes swinging as one VRM spring: VRM's settings, lengths in metres. */
export interface SpringChainConfig {
  /** Node indices, each the child of the one before; each swings toward the next. */
  joints: number[];
  stiffness: number;
  drag: number;
  gravity: number;
  gravity_dir: [number, number, number];
  hit_radius: number;
  /** A tail this far past the last node, so it swings too. Absent: the last only ends the chain. */
  tip?: number;
  colliders: SpringColliderConfig[];
}

/** A sphere on a node, or with `tail` a capsule, along the node's axes. */
export interface SpringColliderConfig {
  node: number;
  radius: number;
  offset: [number, number, number];
  tail?: [number, number, number];
}

/** MToon's shading: the shade's share of the lit colour, how hard and where they meet. */
export interface ToonConfig {
  shade: number;
  toony: number;
  shift: number;
}
