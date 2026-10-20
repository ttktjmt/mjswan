/**
 * A VRM avatar posed from the robot's bodies every frame: render only, outside the model.
 *
 * three-vrm loads lazily, so a scene without an avatar never fetches it. The file is
 * parsed from bytes the app already holds; nothing is fetched and nothing is evaluated.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { MainModule, MjModel } from 'mujoco';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';
import {
  buildRig,
  hipsPosition,
  modelFrame,
  retarget,
  zeroPoseHeight,
  zeroPoseRotations,
  type RetargetRig,
} from './retarget';

/** A scene's manifest `avatar` entry (see `mjswan.AvatarConfig`). */
export interface AvatarConfig {
  /** VRM humanoid bone -> robot body; `hips` carries the position. */
  bones: Record<string, string>;
  /** VRM bone -> `[x, y, z, w]`, in the avatar's VRM-1.0 frame. */
  rest_offsets?: Record<string, ArrayLike<number>>;
  /** Robot root height -> avatar hips height. Absent: hips height over root height at rest. */
  height_ratio?: number;
  /** Added to the avatar's position, in MuJoCo's world frame (x forward, y left, z up), metres. */
  offset?: ArrayLike<number>;
  /** Stop drawing the robot the avatar follows: with a zero `offset`, the avatar is its skin. */
  hide_robot?: boolean;
}

/**
 * Springs integrate at this fixed step, through poses interpolated between frames: one
 * large step whips a long chain (hair) through the body when the pose jumps, as a
 * spinkick's does at 30 fps.
 */
const SPRING_STEP = 1 / 120;
/** A frame longer than this (a stalled tab) is not caught up. */
const MAX_FRAME_SECONDS = 1 / 30;
/** A root jump this large in one frame is a reset, not motion: the springs start over. */
const TELEPORT_DISTANCE = 0.3;

/** One body's rotation at the previous frame, this frame, and the substep between. */
type BodyPose = { from: THREE.Quaternion; to: THREE.Quaternion; now: THREE.Quaternion };

type VrmModule = typeof import('@pixiv/three-vrm');

/** The model's body `name` names: itself, else the one `<entity>/<name>` (as Python's check). */
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

export class Avatar {
  /** Added to the scene by the runtime; holds the VRM at `offset`, facing the robot's way. */
  readonly root = new THREE.Group();
  /** With `hide_robot`, every body of the robot the hips follow, whose meshes go undrawn. */
  hiddenBodies: number[] = [];
  private rig: RetargetRig | null = null;
  private nodes: THREE.Object3D[] = [];
  private world: THREE.Quaternion[] = [];
  private local: THREE.Quaternion[] = [];
  private hips: THREE.Object3D | null = null;
  private hipsBody = -1;
  private heightRatio = 1;
  private poses = new Map<number, BodyPose>();
  private readonly rootFrom = new THREE.Vector3();
  private readonly rootTo = new THREE.Vector3();
  private readonly rootNow = new THREE.Vector3();
  private lastTime: number | null = null;

  private constructor(
    private readonly module: VrmModule,
    private readonly vrm: VRM,
    private readonly config: AvatarConfig,
  ) {
    this.root.name = 'Avatar';
    this.root.add(vrm.scene);
    vrm.scene.traverse((object) => {
      // Skinned bounds never follow the pose, and a hit test on one skins every vertex.
      object.frustumCulled = false;
      object.raycast = () => {};
      if ((object as THREE.Mesh).isMesh) object.castShadow = true;
    });
  }

  static async load(data: ArrayBuffer, config: AvatarConfig): Promise<Avatar> {
    const module = await import('@pixiv/three-vrm');
    const loader = new GLTFLoader();
    loader.register((parser) => new module.VRMLoaderPlugin(parser));
    const gltf = await loader.parseAsync(data, '');
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) throw new Error('not a VRM file (no VRM or VRMC_vrm extension)');
    module.VRMUtils.removeUnnecessaryVertices(gltf.scene);
    module.VRMUtils.combineSkeletons(gltf.scene);
    return new Avatar(module, vrm, config);
  }

  /** VRM `meta` version: `'0'` or `'1'`. */
  get metaVersion(): string | undefined {
    return this.vrm.meta?.metaVersion;
  }

  /** Resolve the bone map against a (re)built model. Unresolved bones are left at rest. */
  bind(mujoco: MainModule, mjModel: MjModel): void {
    const humanoid = this.vrm.humanoid;
    const parents = this.module.VRMHumanBoneParentMap;
    const mapped: Array<{ name: string; bodyId: number; ancestors: string[] }> = [];
    for (const [bone, body] of Object.entries(this.config.bones)) {
      const node = humanoid.getNormalizedBoneNode(bone as VRMHumanBoneName);
      const bodyId = findBody(mjModel, body);
      if (!node || bodyId < 0) {
        console.warn(`[Avatar] ${bone} -> ${body}: ${node ? 'no such body' : 'not in this VRM'}`);
        continue;
      }
      const ancestors: string[] = [];
      for (let p = parents[bone as VRMHumanBoneName]; p; p = parents[p]) ancestors.push(p);
      mapped.push({ name: bone, bodyId, ancestors });
    }

    const frame = modelFrame(this.metaVersion);
    const zeroPose = zeroPoseRotations(mjModel, mujoco.mjtJoint.mjJNT_FREE.value);
    this.rig = buildRig(mapped, zeroPose, this.config.rest_offsets ?? {}, frame);
    this.nodes = this.rig.bones.map(
      (bone) => humanoid.getNormalizedBoneNode(bone.name as VRMHumanBoneName)!,
    );
    this.world = this.rig.bones.map(() => new THREE.Quaternion());
    this.local = this.rig.bones.map(() => new THREE.Quaternion());
    this.poses = new Map(
      this.rig.bones.map((bone) => [
        bone.bodyId,
        { from: new THREE.Quaternion(), to: new THREE.Quaternion(), now: new THREE.Quaternion() },
      ]),
    );

    const hips = this.rig.bones.find((bone) => bone.name === 'hips');
    this.hips = hips ? humanoid.getNormalizedBoneNode('hips') : null;
    this.hipsBody = hips?.bodyId ?? -1;
    this.hiddenBodies = [];
    if (this.config.hide_robot && this.hipsBody >= 0) {
      const robot = mjModel.body_rootid[this.hipsBody];
      for (let b = 1; b < mjModel.nbody; b++) {
        if (mjModel.body_rootid[b] === robot) this.hiddenBodies.push(b);
      }
    }
    this.heightRatio =
      this.config.height_ratio ?? this.autoHeightRatio(zeroPoseHeight(mjModel, zeroPose, this.hipsBody));

    // The container turns the model frame onto the robot's facing; see retarget.ts.
    const offset = this.config.offset ?? [0, 0, 0];
    this.root.position.set(offset[0], offset[2], -offset[1]);
    this.root.quaternion.copy(this.rig.facing).multiply(frame);
    this.lastTime = null;
  }

  /** The avatar's hips height over the robot root's at its zero pose, standing on z = 0. */
  private autoHeightRatio(rootHeight: number): number {
    const hips = this.vrm.humanoid.normalizedRestPose.hips?.position;
    return hips && rootHeight > 0 ? hips[1] / rootHeight : 1;
  }

  /**
   * Pose from the bodies as drawn this frame, then step the springs. `time` in ms. The
   * last substep is this frame's pose, so the avatar is never behind the robot.
   */
  update(bodies: Record<number, THREE.Object3D>, time: number): void {
    const rig = this.rig;
    if (!rig) return;
    const dt = this.lastTime === null ? 0 : Math.min((time - this.lastTime) / 1000, MAX_FRAME_SECONDS);
    this.lastTime = time;

    for (const [bodyId, pose] of this.poses) pose.to.copy(bodies[bodyId].quaternion);
    if (this.hipsBody >= 0) this.rootTo.copy(bodies[this.hipsBody].position);
    const restart = dt === 0 || this.rootTo.distanceTo(this.rootFrom) > TELEPORT_DISTANCE;
    if (restart) {
      for (const pose of this.poses.values()) pose.from.copy(pose.to);
      this.rootFrom.copy(this.rootTo);
    }

    const steps = restart ? 1 : Math.ceil(dt / SPRING_STEP);
    for (let step = 1; step <= steps; step++) {
      this.pose(rig, step / steps);
      if (restart) {
        this.vrm.humanoid.update();
        this.root.updateMatrixWorld(true);
        this.vrm.springBoneManager?.reset();
      }
      this.vrm.update(dt / steps);
    }

    for (const pose of this.poses.values()) pose.from.copy(pose.to);
    this.rootFrom.copy(this.rootTo);
  }

  /** The normalized bones at `alpha` of the way from the previous frame's pose to this one's. */
  private pose(rig: RetargetRig, alpha: number): void {
    for (const pose of this.poses.values()) pose.now.slerpQuaternions(pose.from, pose.to, alpha);
    retarget(rig, (bodyId) => this.poses.get(bodyId)!.now, this.world, this.local);
    for (let i = 0; i < this.nodes.length; i++) this.nodes[i].quaternion.copy(this.local[i]);
    if (this.hips) {
      this.rootNow.lerpVectors(this.rootFrom, this.rootTo, alpha);
      hipsPosition(rig, this.rootNow, this.heightRatio, this.hips.position);
    }
  }

  dispose(): void {
    this.root.removeFromParent();
    this.module.VRMUtils.deepDispose(this.vrm.scene);
  }
}
