/**
 * A skin: a rigged character (a VRM or a .glb) posed from the robot's bodies every frame,
 * outside the scene's model. Its spring bones swing by three-vrm, or in a MuJoCo model of
 * their own (springs.ts).
 *
 * The runtime imports this module only for a scene with a skin, so a scene without one
 * never fetches it, nor three-vrm and the modules that come with it. The file is parsed
 * from bytes the app already holds; nothing is fetched and nothing is evaluated.
 */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { lightBudget } from '../scene/lights';
import { threeToMjcCoordinate } from '../scene/coordinate';
import { TERRAIN_GROUP } from '../xr/grounding';
import { SpringPhysics } from './springs';
import {
  buildRig,
  dropToReach,
  fromMujoco,
  hipsPosition,
  reachLimb,
  retarget,
  zeroPoseHeight,
  zeroPoseRotations,
  type RetargetRig,
} from './retarget';
import { loadSkinModel, type SkinModel } from './skinModel';
import type { SkinConfig } from './config';

export type { SkinConfig } from './config';

/**
 * three-vrm's springs integrate at this fixed step, through poses interpolated between
 * frames: one large step whips a long chain (hair) through the body when the pose jumps,
 * as a spinkick's does at 30 fps.
 */
const SPRING_STEP = 1 / 120;
/** A frame longer than this is not caught up. */
const MAX_FRAME_SECONDS = 1 / 30;
/** A frame longer than this (a hidden or stalled tab) starts the springs over instead. */
const STALL_SECONDS = 0.25;
/** A root jump this large in one frame is a reset, not motion: the springs start over. */
const TELEPORT_DISTANCE = 0.3;
/** Straight down in MuJoCo's frame. */
const DOWN = [0, 0, -1];
/** How many of the robot's own geoms a ray under it may pass before it gives up. */
const MAX_ROBOT_HITS = 8;
/**
 * With `retarget: 'hybrid'`, how far above the lowest hand or foot another still carries
 * the body, as a share of its limb's length: those lower the avatar when they cannot
 * reach, a humanoid's hands held up never do.
 */
const SUPPORT_BAND = 0.25;
const ONE = new THREE.Vector3(1, 1, 1);
const _placed = new THREE.Matrix4();
const _at = new THREE.Vector3();
const _scale = new THREE.Vector3();
/** `mj_ray` writes the geom it hit here. */
const hitGeom = new Int32Array(1);

/** One body's pose at the previous frame, this frame, and the substep between. */
type BodyPose = {
  from: THREE.Quaternion;
  to: THREE.Quaternion;
  now: THREE.Quaternion;
  fromAt: THREE.Vector3;
  toAt: THREE.Vector3;
  nowAt: THREE.Vector3;
};

/** A bone fixed to a body: where it sits in the body's frame, from the calibration pose. */
type Attachment = { name: string; bone: THREE.Object3D; bodyId: number; inBody: THREE.Matrix4 };

/** With `retarget: 'hybrid'`, an arm or leg (normalized bones) and where its end belongs. */
type Limb = {
  upper: THREE.Object3D;
  lower: THREE.Object3D;
  end: THREE.Object3D;
  target: Attachment;
  bend: THREE.Vector3;
  /** Its two bones' lengths, summed. */
  length: number;
  /** This frame's place and world rotation for `end`. */
  at: THREE.Vector3;
  turn: THREE.Quaternion;
};

/** The two colours of an MToon material that its lights multiply. */
type ToonMaterial = THREE.Material & { isMToonMaterial: true; color: THREE.Color; shadeColorFactor: THREE.Color };

/** How low geom `g` reaches in `data` (MuJoCo z): its shape's, a mesh's bounding box's. */
export function geomBottom(mujoco: MainModule, model: MjModel, data: MjData, g: number): number {
  const type = model.geom_type[g];
  const shapes = mujoco.mjtGeom;
  // The world z of each of the geom's axes, and its sizes along them.
  const [zx, zy, zz] = [6, 7, 8].map((k) => data.geom_xmat[g * 9 + k]);
  const [a, b, c] = [0, 1, 2].map((k) => model.geom_size[g * 3 + k]);
  const z = data.geom_xpos[g * 3 + 2];
  if (type === shapes.mjGEOM_SPHERE.value) return z - a;
  if (type === shapes.mjGEOM_CAPSULE.value) return z - Math.abs(zz) * b - a;
  if (type === shapes.mjGEOM_CYLINDER.value) return z - Math.abs(zz) * b - a * Math.sqrt(Math.max(0, 1 - zz * zz));
  if (type === shapes.mjGEOM_ELLIPSOID.value) return z - Math.hypot(zx * a, zy * b, zz * c);
  if (type === shapes.mjGEOM_BOX.value) return z - Math.abs(zx) * a - Math.abs(zy) * b - Math.abs(zz) * c;
  if (type === shapes.mjGEOM_PLANE.value || type === shapes.mjGEOM_HFIELD.value) return Infinity;
  // Centre, then half sizes, in the geom's frame.
  const [cx, cy, cz, hx, hy, hz] = [0, 1, 2, 3, 4, 5].map((k) => model.geom_aabb[g * 6 + k]);
  return z + zx * cx + zy * cy + zz * cz - Math.abs(zx) * hx - Math.abs(zy) * hy - Math.abs(zz) * hz;
}

/** The model's body `name` names: itself, else the one `<entity>/<name>`, as Python's `resolve_body`. */
function findBody(mjModel: MjModel, name: string): number {
  let suffixed = -1;
  let matches = 0;
  for (let b = 0; b < mjModel.nbody; b++) {
    const bodyName = mjModel.body(b).name;
    if (bodyName === name) return b;
    if (bodyName.endsWith(`/${name}`)) {
      suffixed = b;
      matches += 1;
    }
  }
  return matches === 1 ? suffixed : -1;
}

export class Skin {
  /** Added to the scene by the runtime; holds the model at `offset`, facing the robot's way. */
  readonly root = new THREE.Group();
  /** With `hide_robot`, every body of the robot the root bone follows, whose meshes go undrawn. */
  hiddenBodies: number[] = [];
  private rig: RetargetRig | null = null;
  private nodes: THREE.Object3D[] = [];
  private world: THREE.Quaternion[] = [];
  private local: THREE.Quaternion[] = [];
  /** The normalized bone that carries the position, and the body it follows. */
  private rootBone: THREE.Object3D | null = null;
  private rootBody = -1;
  /** The tree the root body is in: the robot, whose own geoms are not its ground. */
  private robotRoot = -1;
  /** The root bone's height above the model's lowest point, at rest. */
  private readonly restHeight: number;
  private heightRatio = 1;
  private mujoco: MainModule | null = null;
  private poses = new Map<number, BodyPose>();
  /** `hips_offset` in three.js axes, in the root body's frame. */
  private readonly hipsOffset = new THREE.Vector3();
  private readonly rootFrom = new THREE.Vector3();
  private readonly rootTo = new THREE.Vector3();
  private readonly rootNow = new THREE.Vector3();
  /** The ground's height (three.js y) under the root: the height ratio scales from it. */
  private groundFrom = 0;
  private groundTo = 0;
  private lastTime: number | null = null;
  /** Every MToon material, with the colours it came with. */
  private readonly toon: Array<{ material: ToonMaterial; lit: THREE.Color; shade: THREE.Color }> = [];
  private springs: SpringPhysics | null = null;
  private springsBuilt = false;
  /** With `retarget: 'position'`, every mapped bone, ancestors first; else empty. */
  private attached: Attachment[] = [];
  /** With `retarget: 'hybrid'`, every limb whose three bones are mapped; else empty. */
  private limbs: Limb[] = [];

  private constructor(
    private readonly model: SkinModel,
    private readonly config: SkinConfig,
  ) {
    this.root.name = 'Skin';
    this.root.add(model.scene);
    const materials = new Set<THREE.Material>();
    model.scene.traverse((object) => {
      // Skinned bounds never follow the pose, and a hit test on one skins every vertex.
      object.frustumCulled = false;
      object.raycast = () => {};
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) materials.add(material);
    });
    for (const material of materials) {
      const toon = material as ToonMaterial;
      if (toon.isMToonMaterial) {
        this.toon.push({ material: toon, lit: toon.color.clone(), shade: toon.shadeColorFactor.clone() });
      }
    }
    this.root.updateMatrixWorld(true);
    const rootBone = model.rig.node(model.rig.root);
    const lowest = this.lowestVertex();
    // A model with no mesh stands on its origin.
    const ground = Number.isFinite(lowest) ? lowest : 0;
    this.restHeight = rootBone ? rootBone.getWorldPosition(new THREE.Vector3()).y - ground : 0;
  }

  static async load(data: ArrayBuffer, config: SkinConfig): Promise<Skin> {
    return new Skin(await loadSkinModel(data, config), config);
  }

  /** Resolve the bone map against a (re)built model. Unresolved bones are left at rest. */
  bind(mujoco: MainModule, mjModel: MjModel): void {
    // First, while the avatar is still at rest: the spring model is built from that pose.
    if (!this.springsBuilt) {
      this.springsBuilt = true;
      this.springs = this.buildSprings(mujoco);
    }
    this.normalizeToon(mjModel);
    this.mujoco = mujoco;

    const skeleton = this.model.rig;
    const mapped: Array<{ name: string; bodyId: number; ancestors: string[] }> = [];
    for (const [bone, body] of Object.entries(this.config.bones)) {
      const node = skeleton.node(bone);
      const bodyId = findBody(mjModel, body);
      if (!node || bodyId < 0) {
        console.warn(`[Skin] ${bone} -> ${body}: ${node ? 'no such body' : 'not in this file'}`);
        continue;
      }
      mapped.push({ name: bone, bodyId, ancestors: skeleton.ancestors(bone) });
    }

    const frame = this.model.frame;
    const zeroPose = zeroPoseRotations(mjModel, mujoco.mjtJoint.mjJNT_FREE.value);
    this.rig = buildRig(mapped, zeroPose, this.config.rest_offsets ?? {}, frame);
    this.nodes = this.rig.bones.map((bone) => skeleton.node(bone.name)!);
    this.world = this.rig.bones.map(() => new THREE.Quaternion());
    this.local = this.rig.bones.map(() => new THREE.Quaternion());
    this.poses = new Map(
      this.rig.bones.map((bone) => [
        bone.bodyId,
        {
          from: new THREE.Quaternion(),
          to: new THREE.Quaternion(),
          now: new THREE.Quaternion(),
          fromAt: new THREE.Vector3(),
          toAt: new THREE.Vector3(),
          nowAt: new THREE.Vector3(),
        },
      ]),
    );

    const root = this.rig.bones.find((bone) => bone.name === skeleton.root);
    this.rootBone = root ? skeleton.node(root.name) : null;
    this.rootBody = root?.bodyId ?? -1;
    this.robotRoot = this.rootBody >= 0 ? mjModel.body_rootid[this.rootBody] : -1;
    this.hiddenBodies = [];
    if (this.config.hide_robot !== false && this.rootBody >= 0) {
      for (let b = 1; b < mjModel.nbody; b++) {
        if (mjModel.body_rootid[b] === this.robotRoot) this.hiddenBodies.push(b);
      }
    }
    const hipsOffset = this.config.hips_offset ?? [0, 0, 0];
    this.hipsOffset.set(hipsOffset[0], hipsOffset[2], -hipsOffset[1]);
    const offsetHeight =
      this.rootBody >= 0 ? this.hipsOffset.clone().applyQuaternion(zeroPose[this.rootBody]).y : 0;
    const rootHeight =
      (this.config.root_height ?? zeroPoseHeight(mjModel, zeroPose, this.rootBody)) + offsetHeight;
    this.heightRatio =
      this.config.height_ratio ?? (this.restHeight > 0 && rootHeight > 0 ? this.restHeight / rootHeight : 1);

    // The container turns the model frame onto the robot's facing; see retarget.ts.
    const offset = this.config.offset ?? [0, 0, 0];
    this.root.position.set(offset[0], offset[2], -offset[1]);
    this.root.quaternion.copy(this.rig.facing).multiply(frame);
    const mode = this.config.retarget ?? 'hybrid';
    // Position drives the raw skeleton; hybrid bends the normalized one before it is copied,
    // and with no limbs to bend is rotation.
    const fixed = mode === 'position' || (mode === 'hybrid' && !!this.config.limbs?.length);
    const attachments = fixed ? this.attach(mujoco, mjModel, mode === 'hybrid') : [];
    this.attached = mode === 'position' ? attachments : [];
    this.limbs = mode === 'hybrid' ? this.limbsOf(attachments) : [];
    // Fixed bones are set on the raw skeleton, which the normalized one would overwrite.
    skeleton.autoUpdate = mode !== 'position';
    this.lastTime = null;
  }

  /** Each limb whose three bones are posed, with where `'position'` puts its end. */
  private limbsOf(attachments: Attachment[]): Limb[] {
    const skeleton = this.model.rig;
    const mapped = new Set(this.rig!.bones.map((bone) => bone.name));
    return (this.config.limbs ?? []).flatMap((names) => {
      const target = attachments.find((attachment) => attachment.name === names[2]);
      if (!target || !names.every((name) => mapped.has(name))) return [];
      const [upper, lower, end] = names.map((name) => skeleton.node(name)!);
      const [a, b, c] = [upper, lower, end].map((node) => node.getWorldPosition(new THREE.Vector3()));
      return [
        {
          upper,
          lower,
          end,
          target,
          bend: new THREE.Vector3(1, 0, 0),
          length: a.distanceTo(b) + b.distanceTo(c),
          at: new THREE.Vector3(),
          turn: new THREE.Quaternion(),
        },
      ];
    });
  }

  /**
   * Each mapped bone's place in its body's frame, raw or normalized, read with the avatar
   * posed by rotation at the model's first keyframe (its zero pose without one), its
   * lowest point moved onto the robot's: a foot lands on the robot's foot even where the
   * keyframe sinks the robot into the floor or holds it off.
   */
  private attach(mujoco: MainModule, mjModel: MjModel, normalized: boolean): Attachment[] {
    const rig = this.rig!;
    const skeleton = this.model.rig;
    const data = new mujoco.MjData(mjModel);
    try {
      if (mjModel.nkey > 0) mujoco.mj_resetDataKeyframe(mjModel, data, 0);
      mujoco.mj_kinematics(mjModel, data);
      const position = (b: number) =>
        new THREE.Vector3(data.xpos[b * 3], data.xpos[b * 3 + 2], -data.xpos[b * 3 + 1]);
      const rotation = (b: number) => fromMujoco(data.xquat, b, new THREE.Quaternion());

      skeleton.autoUpdate = true;
      skeleton.resetRawPose();
      retarget(rig, rotation, this.world, this.local);
      for (let i = 0; i < this.nodes.length; i++) this.nodes[i].quaternion.copy(this.local[i]);
      if (this.rootBone) {
        const root = this.hipsOffset.clone().applyQuaternion(rotation(this.rootBody)).add(position(this.rootBody));
        hipsPosition(rig, root, this.heightRatio, 0, this.rootBone.position);
      }
      skeleton.update();

      // In the robot's frame: the offset is added back as each bone is placed.
      const offset = this.root.position.clone();
      this.root.position.set(0, 0, 0);
      this.root.updateMatrixWorld(true);
      const lift = this.robotBottom(mujoco, mjModel, data) - this.lowestVertex();
      this.root.position.y = Number.isFinite(lift) ? lift : 0;
      this.root.updateMatrixWorld(true);
      const body = new THREE.Matrix4();
      const attached = rig.bones.flatMap((bone) => {
        const node = normalized ? skeleton.node(bone.name) : skeleton.rawNode(bone.name);
        if (!node) return [];
        body.compose(position(bone.bodyId), rotation(bone.bodyId), ONE).invert();
        const inBody = body.clone().multiply(node.matrixWorld);
        return [{ name: bone.name, bone: node, bodyId: bone.bodyId, inBody }];
      });
      this.root.position.copy(offset);
      this.root.updateMatrixWorld(true);
      const depth = (node: THREE.Object3D) => {
        let d = 0;
        for (let p = node.parent; p; p = p.parent) d++;
        return d;
      };
      return attached.sort((a, b) => depth(a.bone) - depth(b.bone));
    } finally {
      data.delete();
    }
  }

  /** The lowest point of the robot's colliding geoms (MuJoCo z); Infinity without one. */
  private robotBottom(mujoco: MainModule, mjModel: MjModel, data: MjData): number {
    let bottom = Infinity;
    for (let g = 0; g < mjModel.ngeom; g++) {
      if (mjModel.body_rootid[mjModel.geom_bodyid[g]] !== this.robotRoot) continue;
      if (!mjModel.geom_contype[g] && !mjModel.geom_conaffinity[g]) continue;
      bottom = Math.min(bottom, geomBottom(mujoco, mjModel, data, g));
    }
    return bottom;
  }

  /** The lowest point of the model's meshes as posed (three.js y); Infinity without one. */
  private lowestVertex(): number {
    let lowest = Infinity;
    const at = new THREE.Vector3();
    this.model.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (!mesh.isMesh) return;
      const count = mesh.geometry.getAttribute('position')?.count ?? 0;
      for (let i = 0; i < count; i++) {
        lowest = Math.min(lowest, mesh.getVertexPosition(i, at).applyMatrix4(mesh.matrixWorld).y);
      }
    });
    return lowest;
  }

  /** MuJoCo's springs, or null to leave them to three-vrm: not asked for, none, or no model. */
  private buildSprings(mujoco: MainModule): SpringPhysics | null {
    const manager = this.model.springBoneManager;
    if (!this.config.mujoco_springs || !manager) return null;
    try {
      return SpringPhysics.build(mujoco, manager, this.model.scene);
    } catch (error) {
      console.warn('[Skin] spring model not built; three-vrm runs the springs:', error);
      return null;
    }
  }

  /**
   * A toon material takes every light it faces in full, with no falloff, so the sum of
   * MuJoCo's lights (2.3 in a default mjlab scene) clips it to white. Dividing its colours
   * by that sum, never brightening, leaves the scene's own materials as they are.
   */
  private normalizeToon(mjModel: MjModel): void {
    const scale = (this.config.brightness ?? 1) / Math.max(lightBudget(mjModel), 1);
    for (const { material, lit, shade } of this.toon) {
      material.color.copy(lit).multiplyScalar(scale);
      material.shadeColorFactor.copy(shade).multiplyScalar(scale);
    }
  }

  /**
   * Pose from the bodies as drawn this frame, then step the springs. `time` in ms. The
   * last substep lands on this frame's pose, so the avatar is never behind the robot.
   * The model and data just drawn find the ground under it; without them the ground is
   * z = 0. They are passed every frame, never kept: the runtime frees a model it rebuilds.
   */
  update(bodies: Record<number, THREE.Object3D>, time: number, mjModel?: MjModel, mjData?: MjData): void {
    const rig = this.rig;
    if (!rig) return;
    const elapsed = this.lastTime === null ? 0 : (time - this.lastTime) / 1000;
    const dt = Math.min(elapsed, MAX_FRAME_SECONDS);
    this.lastTime = time;

    for (const [bodyId, pose] of this.poses) {
      pose.to.copy(bodies[bodyId].quaternion);
      pose.toAt.copy(bodies[bodyId].position);
    }
    if (this.rootBody >= 0) {
      const body = bodies[this.rootBody];
      this.rootTo.copy(this.hipsOffset).applyQuaternion(body.quaternion).add(body.position);
      // Bones fixed to their bodies need no ground.
      if (mjModel && mjData && !this.attached.length) {
        this.groundTo = this.groundUnder(this.rootTo, mjModel, mjData) ?? this.groundTo;
      }
    }
    const restart =
      dt === 0 || elapsed > STALL_SECONDS || this.rootTo.distanceTo(this.rootFrom) > TELEPORT_DISTANCE;
    if (restart) this.settle();

    if (this.springs) {
      // The anchors are interpolated between frames inside the spring model, so the
      // avatar is posed once.
      this.pose(rig, 1);
      this.model.update(dt);
      this.root.updateMatrixWorld(true);
      if (restart) this.springs.reset();
      else this.springs.step(dt);
      this.settle();
      return;
    }

    const springs = this.model.springBoneManager;
    const steps = restart ? 1 : Math.ceil(dt / SPRING_STEP);
    for (let step = 1; step <= steps; step++) {
      this.pose(rig, step / steps);
      if (restart) {
        this.model.rig.update();
        this.root.updateMatrixWorld(true);
        springs?.reset();
      }
      this.model.update(dt / steps);
      springs?.update(dt / steps);
    }
    this.settle();
  }

  /** This frame's pose becomes the one the next frame's substeps start from. */
  private settle(): void {
    for (const pose of this.poses.values()) {
      pose.from.copy(pose.to);
      pose.fromAt.copy(pose.toAt);
    }
    this.rootFrom.copy(this.rootTo);
    this.groundFrom = this.groundTo;
  }

  /**
   * The height (three.js y) of the ground under `point`: the first geom straight below
   * in the terrain group, as mjlab's height scan sees it, that is not the robot's own.
   * Null where there is none.
   */
  private groundUnder(point: THREE.Vector3, model: MjModel, mjData: MjData): number | null {
    const mujoco = this.mujoco!;
    const origin = threeToMjcCoordinate(point).toArray();
    for (let hit = 0; hit < MAX_ROBOT_HITS; hit++) {
      const distance = mujoco.mj_ray(
        model,
        mjData,
        origin,
        DOWN,
        TERRAIN_GROUP as unknown as number[],
        true,
        -1,
        hitGeom,
        null,
      );
      if (distance < 0) return null;
      // MuJoCo's z is three.js's y.
      origin[2] -= distance;
      if (model.body_rootid[model.geom_bodyid[hitGeom[0]]] !== this.robotRoot) return origin[2];
      origin[2] -= 1e-6;
    }
    return null;
  }

  /** Spring joints MuJoCo simulates; zero when three-vrm runs them. */
  get simulatedSpringJoints(): number {
    return this.springs?.jointCount ?? 0;
  }

  /** The bones at `alpha` of the way from the previous frame's pose to this one's. */
  private pose(rig: RetargetRig, alpha: number): void {
    for (const pose of this.poses.values()) {
      pose.now.slerpQuaternions(pose.from, pose.to, alpha);
      pose.nowAt.lerpVectors(pose.fromAt, pose.toAt, alpha);
    }
    if (this.attached.length) {
      this.fix();
      return;
    }
    retarget(rig, (bodyId) => this.poses.get(bodyId)!.now, this.world, this.local);
    for (let i = 0; i < this.nodes.length; i++) this.nodes[i].quaternion.copy(this.local[i]);
    if (this.rootBone) {
      this.rootNow.lerpVectors(this.rootFrom, this.rootTo, alpha);
      const ground = THREE.MathUtils.lerp(this.groundFrom, this.groundTo, alpha);
      hipsPosition(rig, this.rootNow, this.heightRatio, ground, this.rootBone.position);
    }
    if (this.limbs.length) this.reach();
  }

  /**
   * `retarget: 'hybrid'`: each limb, posed by rotation, bent so its hand or foot lands,
   * turned, where its body carries it under `'position'`. Its bones keep their lengths;
   * where the ones it stands on fall short, the whole avatar is lowered until they reach.
   */
  private reach(): void {
    this.root.updateMatrixWorld(true);
    let lowest = Infinity;
    for (const limb of this.limbs) {
      const pose = this.poses.get(limb.target.bodyId)!;
      _placed.compose(_at.copy(pose.nowAt).add(this.root.position), pose.now, ONE).multiply(limb.target.inBody);
      _placed.decompose(limb.at, limb.turn, _scale);
      lowest = Math.min(lowest, limb.at.y);
    }
    // Faded out across the band, so a foot lifting off lets the body back up smoothly.
    let drop = 0;
    for (const limb of this.limbs) {
      const share = 1 - (limb.at.y - lowest) / (SUPPORT_BAND * limb.length);
      if (share > 0) drop = Math.max(drop, share * dropToReach(limb.upper, limb.lower, limb.end, limb.at));
    }
    if (drop > 0 && this.rootBone) {
      this.rootBone.getWorldPosition(_at);
      _at.y -= drop;
      this.rootBone.position.copy(this.rootBone.parent!.worldToLocal(_at));
      this.rootBone.updateMatrixWorld(true);
    }
    for (const limb of this.limbs) reachLimb(limb.upper, limb.lower, limb.end, limb.at, limb.bend, limb.turn);
  }

  /**
   * Every fixed bone where its body carries it, set on the raw bone through its parent,
   * ancestors first. Where the avatar's joints are not the robot's, the mesh across them
   * stretches or folds.
   */
  private fix(): void {
    this.root.updateMatrixWorld(true);
    for (const { bone, bodyId, inBody } of this.attached) {
      const pose = this.poses.get(bodyId)!;
      _placed.compose(_at.copy(pose.nowAt).add(this.root.position), pose.now, ONE).multiply(inBody);
      bone.matrix.copy(bone.parent!.matrixWorld).invert().multiply(_placed);
      bone.matrix.decompose(bone.position, bone.quaternion, bone.scale);
      bone.updateMatrixWorld(true);
    }
  }

  dispose(): void {
    this.springs?.dispose();
    this.springs = null;
    this.root.removeFromParent();
    this.model.dispose();
  }
}
