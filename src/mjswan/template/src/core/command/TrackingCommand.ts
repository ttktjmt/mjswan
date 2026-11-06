import * as THREE from 'three';

import { quatApply, quatApplyInv, quatInverse, quatMultiply, yawQuat } from '../observation/math';
import { getPosition, getQuaternion } from '../scene/scene';
import { type NpzEntry, loadNpz } from '../scene/npz';
import { type Bytes, resolveBytes } from '../utils/bytes';
import { OnnxEvent, isOnnxEventConfig } from '../event/OnnxEvent';
import { ArrowBatch, addFrame, viserMaterial, type Rgba } from '../debugViz/primitives';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type TrackingMotionConfig = {
  name: string;
  /** Raw `.npz` bytes (or a lazy loader) supplied by the app. */
  data: Bytes;
  fps: number;
  anchor_body_name: string;
  body_names: string[];
  dataset_joint_names?: string[];
  default?: boolean;
  loop?: boolean;
  clip_format?: 'body_world' | 'qpos';
  time_source?: 'wall' | 'sim';
  /** Arrays of the `.npz`, one row per frame, served as state fields of their own name. */
  frame_fields?: string[];
};

type LoadedTrackingMotion = TrackingMotionConfig & {
  jointPos: Float32Array[];
  jointVel: Float32Array[];
  bodyPosW: Float32Array[];
  bodyQuatW: Float32Array[];
  bodyLinVelW: Float32Array[];
  bodyAngVelW: Float32Array[];
  qposFrames?: Float32Array[];
  frameFields: Map<string, Float32Array[]>;
  frameCount: number;
};

function normalizeQuat(quat: ArrayLike<number>): Float32Array {
  const length = Math.hypot(quat[0] ?? 1, quat[1] ?? 0, quat[2] ?? 0, quat[3] ?? 0) || 1.0;
  return new Float32Array([
    (quat[0] ?? 1) / length,
    (quat[1] ?? 0) / length,
    (quat[2] ?? 0) / length,
    (quat[3] ?? 0) / length,
  ]);
}

function splitFrames(entry: NpzEntry): Float32Array[] {
  const totalFrames = entry.shape[0] ?? 0;
  const width = entry.shape.length <= 1 ? 1 : entry.shape.slice(1).reduce((acc, v) => acc * v, 1);
  const frames: Float32Array[] = [];
  for (let i = 0; i < totalFrames; i++) {
    const out = new Float32Array(width);
    const start = i * width;
    for (let j = 0; j < width; j++) {
      out[j] = entry.data[start + j] ?? 0.0;
    }
    frames.push(out);
  }
  return frames;
}

/** A motion's sample rate above this is refused: no capture runs this fast. */
const MAX_MOTION_FPS = 10_000;
/** mjlab's default `viz.ghost_color`. */
const DEFAULT_GHOST_COLOR: readonly number[] = [0.5, 0.7, 0.5, 0.5];
/** `DebugVisualizer.add_ghost_mesh`'s default `alpha`, which `MotionCommand` leaves as is. */
const GHOST_OPACITY = 0.5;
/** mjlab's `_DESIRED_FRAME_COLORS`, the reference's axes in `"frames"` mode. */
const DESIRED_FRAME_COLORS: readonly Rgba[] = [
  [1.0, 0.5, 0.5, 1],
  [0.5, 1.0, 0.5, 1],
  [0.5, 0.5, 1.0, 1],
];

/** viser's ghost: `ghost_color`'s RGB at `GHOST_OPACITY`; mjlab uses its alpha to pick visual geoms. */
function ghostMaterial(rgba: readonly number[]): THREE.Material {
  return viserMaterial(rgba, GHOST_OPACITY);
}

function hasRenderableMesh(object: THREE.Object3D): boolean {
  let found = false;
  object.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      found = true;
    }
  });
  return found;
}

/**
 * mjlab's `update_relative_body_poses`: place the reference bodies onto the robot by
 * keeping the robot's anchor x/y (but the *reference's* z) and rotating the
 * anchor-relative offsets by the yaw between the two anchors.
 */
export function reanchorBodyPositions(
  bodyPosW: Float32Array,
  anchorPos: ArrayLike<number>,
  anchorQuat: ArrayLike<number>,
  robotAnchorPos: ArrayLike<number>,
  robotAnchorQuat: ArrayLike<number>,
): Float32Array {
  const deltaPos = [robotAnchorPos[0] ?? 0, robotAnchorPos[1] ?? 0, anchorPos[2] ?? 0];
  const deltaOri = yawQuat(quatMultiply(robotAnchorQuat, quatInverse(anchorQuat)));
  const out = new Float32Array(bodyPosW.length);
  for (let i = 0; i + 2 < bodyPosW.length; i += 3) {
    const offset = quatApply(deltaOri, [
      bodyPosW[i] - (anchorPos[0] ?? 0),
      bodyPosW[i + 1] - (anchorPos[1] ?? 0),
      bodyPosW[i + 2] - (anchorPos[2] ?? 0),
    ]);
    for (let j = 0; j < 3; j++) {
      out[i + j] = deltaPos[j] + offset[j];
    }
  }
  return out;
}

export class TrackingCommand implements CommandTerm {
  private readonly context: CommandTermContext;
  private readonly motions: TrackingMotionConfig[];
  private readonly loadedMotions: Map<string, LoadedTrackingMotion>;
  private sampleHz: number;
  private readonly ghostRoot: THREE.Group | null;
  /** mjlab's `viz.mode="frames"`: reference and robot body frames instead of the ghost. */
  private readonly frames: ArrowBatch | null;
  private readonly ghostBodies: Map<number, THREE.Group>;
  /** Model body id by name, filled on demand (see `resolveBodyId`). */
  private readonly bodyIds = new Map<string, number>();
  private readonly ghostData: import('mujoco').MjData | null;
  private refBodyPosW: Float32Array[];
  private refBodyQuatW: Float32Array[];
  private refBodyLinVelW: Float32Array[];
  private refBodyAngVelW: Float32Array[];
  private selectedMotionName: string | null;
  private selectedMotion: LoadedTrackingMotion | null;
  private selectedAnchorBodyIndex: number;
  private selectedRootBodyIndex: number;
  private datasetQposAdr: number[];
  private frameAccumulator: number;
  /** Upstream SuperMimic's command: a reset's first step observes the frame it wrote. */
  private readonly holdResetFrame: boolean;
  /** The pending reset is a clip's wrap. */
  private looped: boolean;
  /** mjlab's `_debug_vis_enabled`. */
  private debugVisOn: boolean;
  private readonly samplingMode: string;
  /** Look-ahead/look-back offsets the `ref_*` window state fields are sampled at. */
  private readonly timeSteps: number[];
  /** Traced reference-state-initialization jitter, or null when the task jitters nothing. */
  private readonly resetJitter: OnnxEvent | null;
  refJointPos: Float32Array[];
  refRootPos: Float32Array[];
  refRootQuat: Float32Array[];
  refIdx: number;
  refLen: number;
  nJoints: number;

  constructor(
    _termName: string,
    config: CommandConfigEntry,
    context: CommandTermContext,
  ) {
    this.context = context;
    this.motions = Array.isArray(config.motions) ? config.motions as TrackingMotionConfig[] : [];
    this.loadedMotions = new Map();
    this.sampleHz = 50.0;
    this.selectedMotionName =
      this.motions.find((motion) => motion.default)?.name ??
      this.motions[0]?.name ??
      null;
    this.selectedMotion = null;
    this.selectedAnchorBodyIndex = 0;
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = [];
    this.frameAccumulator = 0.0;
    this.holdResetFrame = config.hold_reset_frame === true;
    this.looped = false;
    this.debugVisOn = true;
    this.samplingMode = typeof config.sampling_mode === 'string' ? config.sampling_mode : 'start';
    this.timeSteps = Array.isArray(config.time_steps)
      ? (config.time_steps as unknown[]).map((step) => Math.trunc(Number(step) || 0))
      : [0];
    this.resetJitter = this.buildResetJitter(config.reset_graph);
    this.refJointPos = [];
    this.refRootPos = [];
    this.refRootQuat = [];
    this.refIdx = 0;
    this.refLen = 0;
    this.nJoints = this.motions.find((motion) => motion.name === this.selectedMotionName)?.dataset_joint_names?.length ?? 0;

    this.ghostBodies = new Map();
    this.ghostData = context.mjModel ? new context.mujoco.MjData(context.mjModel) : null;
    const ghostColor = Array.isArray(config.ghost_color)
      ? (config.ghost_color as unknown[]).map(Number)
      : DEFAULT_GHOST_COLOR;
    // Absent means on: documents without the field always drew the ghost.
    const draws = config.debug_vis !== false;
    const framesMode = config.viz_mode === 'frames';
    this.ghostRoot = draws && !framesMode ? this.createGhostRoot(ghostColor) : null;
    this.frames = draws && framesMode ? this.createFrames() : null;
    this.refBodyPosW = [];
    this.refBodyQuatW = [];
    this.refBodyLinVelW = [];
    this.refBodyAngVelW = [];
  }

  getCommand(): Float32Array {
    if (!this.selectedMotion || this.refLen === 0) {
      return new Float32Array(this.nJoints * 2);
    }
    if (this.selectedMotion.clip_format === 'qpos') {
      return new Float32Array(0);
    }
    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(this.nJoints);
    const jointVel = this.selectedMotion.jointVel[this.refIdx] ?? new Float32Array(this.nJoints);
    const out = new Float32Array(jointPos.length + jointVel.length);
    out.set(jointPos, 0);
    out.set(jointVel, jointPos.length);
    return out;
  }

  getUiConfig(): CommandUiConfig | null {
    return null;
  }

  async setSelectedMotion(name: string | null): Promise<boolean> {
    if (name === null) {
      this.selectedMotionName = null;
      this.selectedMotion = null;
      this.refJointPos = [];
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      this.refLen = 0;
      this.nJoints = 0;
      this.updateGhostPose();
      return false;
    }

    const config = this.motions.find((motion) => motion.name === name);
    if (!config) {
      return false;
    }

    const loaded = this.loadedMotions.get(name) ?? await this.loadMotion(config);
    this.loadedMotions.set(name, loaded);
    this.selectedMotionName = name;
    this.selectedMotion = loaded;
    this.selectedAnchorBodyIndex = Math.max(
      0,
      loaded.body_names.indexOf(loaded.anchor_body_name),
    );
    this.selectedRootBodyIndex = 0;
    this.datasetQposAdr = this.resolveQposAdr(loaded.dataset_joint_names ?? []);
    this.refLen = loaded.frameCount;
    this.refJointPos = loaded.jointPos;
    this.nJoints = loaded.jointPos[0]?.length ?? 0;
    this.looped = false;
    this.updateReferenceState();
    this.startEpisode();
    return true;
  }

  reset(): void {
    this.updateReferenceState();
    this.startEpisode();
  }

  /**
   * mjlab's resample, which writes the start frame into the sim, then the `dt` 0 compute
   * it runs after every reset, which steps one frame past it. A wrap's reset does not
   * (mjlab resamples a wrap after that step's increment), nor does a command that holds
   * the frame it wrote; a `sim` clock sets the frame itself.
   */
  private startEpisode(): void {
    this.refIdx = this.sampleInitialFrame(this.refLen);
    this.frameAccumulator = 0.0;
    this.applyReferenceStateToSim();
    const steps =
      !this.looped && !this.holdResetFrame && this.selectedMotion?.time_source !== 'sim';
    if (steps && this.refIdx + 1 < this.refLen) this.refIdx += 1;
    this.looped = false;
    this.updateGhostPose();
  }

  update(dt: number, afterReset = false): void {
    // `reset()` already ran the `dt` 0 update mjlab gives a reset env.
    if (afterReset || !this.selectedMotion || this.refLen <= 1) {
      return;
    }
    if (this.selectedMotion.time_source === 'sim') {
      const simTime = this.context.mjData?.time ?? 0;
      this.refIdx = this.sampleHz > 0 ? Math.floor(simTime * this.sampleHz) % this.refLen : 0;
      this.updateGhostPose();
      return;
    }
    const shouldLoop = this.selectedMotion?.loop !== false;
    this.frameAccumulator += dt * this.sampleHz;
    // Arithmetic, not a frame at a time: a step costs the same whatever the rate.
    const steps = Math.floor(this.frameAccumulator);
    let motionLooped = false;
    if (steps > 0) {
      this.frameAccumulator -= steps;
      const next = this.refIdx + steps;
      if (next < this.refLen) {
        this.refIdx = next;
      } else if (!shouldLoop) {
        this.refIdx = this.refLen - 1;
        this.frameAccumulator = 0.0;
      } else {
        this.refIdx = next % this.refLen;
        motionLooped = true;
      }
    }
    if (motionLooped && this.context.requestReset) {
      this.looped = true;
      this.context.requestReset();
    }
    this.updateGhostPose();
  }

  debugVisEnabled(): boolean | null {
    return this.ghostRoot || this.frames ? this.debugVisOn : null;
  }

  setDebugVisEnabled(enabled: boolean): void {
    this.debugVisOn = enabled;
  }

  updateDebugVisuals(shown = true): void {
    const visible = shown && this.debugVisOn && this.isReady();
    if (this.ghostRoot) {
      this.ghostRoot.visible = visible;
    }
    if (this.frames) {
      this.frames.clear();
      if (visible) this.drawFrames(this.frames);
      this.frames.commit();
    }
  }

  /** mjlab's `"frames"` drawing: each tracked body and the anchor, reference and robot. */
  private drawFrames(arrows: ArrowBatch): void {
    const bodyNames = this.getBodyNames();
    const refPos = this.refBodyPosW[this.refIdx];
    const refQuat = this.refBodyQuatW[this.refIdx];
    const robotPos = this.robotBodyField('xpos', 3, bodyNames);
    const robotQuat = this.robotBodyField('xquat', 4, bodyNames);
    for (let i = 0; i < bodyNames.length; i++) {
      if (refPos && refQuat) {
        const quat = normalizeQuat(refQuat.subarray(i * 4, i * 4 + 4));
        addFrame(arrows, refPos.subarray(i * 3, i * 3 + 3), quat, 0.08, DESIRED_FRAME_COLORS);
      }
      if (robotPos && robotQuat) {
        addFrame(arrows, robotPos.subarray(i * 3, i * 3 + 3), robotQuat.subarray(i * 4, i * 4 + 4), 0.12);
      }
    }
    const anchorPos = this.getAnchorPos();
    const anchorQuat = this.getAnchorQuat();
    if (anchorPos && anchorQuat) addFrame(arrows, anchorPos, anchorQuat, 0.1, DESIRED_FRAME_COLORS);
    const anchor = [this.getAnchorBodyName() ?? ''];
    const robotAnchorPos = this.robotBodyField('xpos', 3, anchor);
    const robotAnchorQuat = this.robotBodyField('xquat', 4, anchor);
    if (robotAnchorPos && robotAnchorQuat) addFrame(arrows, robotAnchorPos, robotAnchorQuat, 0.15);
  }

  private createFrames(): ArrowBatch {
    const bodies = Math.max(0, ...this.motions.map((motion) => motion.body_names?.length ?? 0));
    // Three axes for each body's two frames and the anchor's two.
    const arrows = new ArrowBatch(3 * (2 * bodies + 2), 'Tracking Frames');
    (this.context.mujocoRoot ?? this.context.scene).add(arrows.object);
    return arrows;
  }

  dispose(): void {
    if (this.ghostRoot) {
      this.ghostRoot.parent?.remove(this.ghostRoot);
      this.ghostRoot.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          if (Array.isArray(obj.material)) {
            for (const material of obj.material) {
              material.dispose?.();
            }
          } else {
            obj.material?.dispose?.();
          }
        }
      });
    }
    this.frames?.dispose();
    this.ghostData?.delete?.();
  }

  isReady(): boolean {
    return this.selectedMotion !== null && this.refLen > 0;
  }

  getSelectedMotion(): LoadedTrackingMotion | null {
    return this.selectedMotion;
  }

  getSelectedMotionName(): string | null {
    return this.selectedMotionName;
  }

  getAnchorBodyName(): string | null {
    return this.selectedMotion?.anchor_body_name
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.anchor_body_name
      ?? null;
  }

  getBodyNames(): string[] {
    return this.selectedMotion?.body_names
      ?? this.motions.find((motion) => motion.name === this.selectedMotionName)?.body_names
      ?? [];
  }

  getAnchorBodyIndex(): number {
    return this.selectedAnchorBodyIndex;
  }

  getAnchorPos(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  getAnchorQuat(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyQuatW[frameIndex];
    if (!frame) {
      return null;
    }
    const offset = this.selectedAnchorBodyIndex * 4;
    return normalizeQuat(frame.slice(offset, offset + 4));
  }

  /** The anchor body's slice of a per-body world vector field, at the current frame. */
  private anchorVector(frames: Float32Array[]): Float32Array | null {
    const frame = frames[this.refIdx];
    if (!frame) return null;
    const offset = this.selectedAnchorBodyIndex * 3;
    return frame.slice(offset, offset + 3);
  }

  /** {@link anchorVector}, rotated into the anchor's own frame. */
  private anchorFrameVector(frames: Float32Array[]): Float32Array | null {
    const vector = this.anchorVector(frames);
    const anchorQuat = this.getAnchorQuat();
    if (!vector || !anchorQuat) return null;
    return Float32Array.from(quatApplyInv(anchorQuat, vector));
  }

  getBodyPosW(frameIndex = this.refIdx): Float32Array | null {
    const motion = this.selectedMotion;
    if (!motion) {
      return null;
    }
    const frame = this.refBodyPosW[frameIndex];
    return frame ? frame.slice() : null;
  }

  /**
   * mjlab `MotionCommand` state, for traced graphs declaring a `{command: "motion",
   * field}` slot — in mjlab's frame, element order and units (`env_origins` omitted,
   * since the browser runs one env at the origin). An unlisted field returns null and
   * its caller holds the previous value.
   *
   * The `ref_*` fields and `is_ready` are the look-ahead window, which mjlab has no
   * equivalent of: each is the `time_steps` offsets' frames concatenated, and the
   * traced term slices out the ones it wants.
   *
   * Any other field is one of the selected motion's `frame_fields`, at the current frame.
   */
  getStateField(field: string): Float32Array | null {
    switch (field) {
      case 'command':
        return this.getCommand();
      case 'is_ready':
        return new Float32Array([this.isReady() ? 1.0 : 0.0]);
      case 'ref_root_pos_w':
        return this.refWindow(this.refRootPos, 3);
      case 'ref_root_quat_w':
        return this.refWindow(this.refRootQuat, 4, true);
      case 'ref_joint_pos':
        return this.refWindow(this.refJointPos, this.nJoints);
      case 'anchor_pos_w':
        return this.getAnchorPos();
      case 'anchor_quat_w':
        return this.getAnchorQuat();
      case 'anchor_lin_vel_w':
        return this.anchorVector(this.refBodyLinVelW);
      case 'anchor_ang_vel_w':
        return this.anchorVector(this.refBodyAngVelW);
      // Anchor-frame reference features: the world quantities above rotated into the
      // anchor's frame, as mjlab's `quat_apply_inverse(anchor_quat_w, …)` properties do.
      case 'ref_base_height': {
        // `env_origins` is omitted throughout: the browser runs one env at the origin.
        const anchorPos = this.getAnchorPos();
        return anchorPos ? new Float32Array([anchorPos[2]]) : null;
      }
      case 'ref_base_lin_vel_b':
        return this.anchorFrameVector(this.refBodyLinVelW);
      case 'ref_base_ang_vel_b':
        return this.anchorFrameVector(this.refBodyAngVelW);
      case 'ref_gravity_b': {
        const anchorQuat = this.getAnchorQuat();
        return anchorQuat
          ? Float32Array.from(quatApplyInv(anchorQuat, [0, 0, -1]))
          : null;
      }
      case 'joint_pos':
      case 'tracked_joint_pos':
        // The current reference frame alone, where `ref_joint_pos` is the whole
        // look-ahead window.
        return this.refJointPos[this.refIdx]?.slice() ?? null;
      case 'body_pos_w':
        return this.getBodyPosW();
      case 'robot_anchor_pos_w':
        return this.robotBodyField('xpos', 3, [this.getAnchorBodyName() ?? '']);
      case 'robot_anchor_quat_w':
        return this.robotBodyField('xquat', 4, [this.getAnchorBodyName() ?? '']);
      case 'robot_body_pos_w':
        return this.robotBodyField('xpos', 3, this.getBodyNames());
      case 'body_pos_relative_w':
        return this.bodyPosRelativeW();
      default:
        return this.selectedMotion?.frameFields.get(field)?.[this.refIdx]?.slice() ?? null;
    }
  }

  /**
   * One `ref_*` field at every `time_steps` offset, concatenated. Offsets clamp rather
   * than wrap, as in training; not ready gives zeros the `is_ready` gate multiplies away.
   */
  private refWindow(frames: Float32Array[], stride: number, quat = false): Float32Array {
    const out = new Float32Array(this.timeSteps.length * stride);
    if (quat) {
      for (let i = 0; i < this.timeSteps.length; i++) out[i * stride] = 1.0;
    }
    if (!this.isReady()) {
      return out;
    }
    for (let i = 0; i < this.timeSteps.length; i++) {
      const index = Math.min(this.refLen - 1, Math.max(0, this.refIdx + this.timeSteps[i]));
      const frame = frames[index];
      if (!frame) continue;
      const values = quat ? normalizeQuat(frame) : frame;
      for (let j = 0; j < stride && j < values.length; j++) {
        out[i * stride + j] = values[j];
      }
    }
    return out;
  }

  /** `mjData.<source>` rows for the named bodies, flattened — mjlab's `body_link_*_w`. */
  private robotBodyField(
    source: 'xpos' | 'xquat',
    stride: number,
    bodyNames: string[],
  ): Float32Array | null {
    const mjData = this.context.mjData;
    if (!mjData || bodyNames.length === 0) {
      return null;
    }
    const out = new Float32Array(bodyNames.length * stride);
    for (let i = 0; i < bodyNames.length; i++) {
      const bodyId = this.resolveBodyId(bodyNames[i]);
      if (bodyId < 0) {
        return null;
      }
      for (let j = 0; j < stride; j++) {
        out[i * stride + j] = mjData[source][bodyId * stride + j] ?? 0.0;
      }
    }
    return out;
  }

  /** The reference bodies re-anchored onto the robot (`reanchorBodyPositions`). */
  private bodyPosRelativeW(): Float32Array | null {
    const anchorPos = this.getAnchorPos();
    const anchorQuat = this.getAnchorQuat();
    const bodyPos = this.getBodyPosW();
    const anchorBody = [this.getAnchorBodyName() ?? ''];
    const robotAnchorPos = this.robotBodyField('xpos', 3, anchorBody);
    const robotAnchorQuat = this.robotBodyField('xquat', 4, anchorBody);
    if (!anchorPos || !anchorQuat || !bodyPos || !robotAnchorPos || !robotAnchorQuat) {
      return null;
    }
    return reanchorBodyPositions(bodyPos, anchorPos, anchorQuat, robotAnchorPos, robotAnchorQuat);
  }

  /** `findBodyIdByName`, memoized: the slots are read every control step. */
  private resolveBodyId(bodyName: string): number {
    let bodyId = this.bodyIds.get(bodyName);
    if (bodyId === undefined) {
      bodyId = this.findBodyIdByName(bodyName);
      this.bodyIds.set(bodyName, bodyId);
    }
    return bodyId;
  }

  private createGhostRoot(color: readonly number[]): THREE.Group | null {
    const bodies = this.context.bodies ?? null;
    const mjModel = this.context.mjModel;
    if (!bodies || !mjModel) {
      return null;
    }
    const root = new THREE.Group();
    root.name = 'Tracking Ghost';
    root.visible = false;
    for (const [bodyId, body] of Object.entries(bodies)) {
      const numericBodyId = Number(bodyId);
      if (!this.isDynamicBody(numericBodyId)) {
        continue;
      }
      const clone = body.clone(true);
      clone.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.material = Array.isArray(obj.material)
            ? obj.material.map(() => ghostMaterial(color))
            : ghostMaterial(color);
          obj.castShadow = false;
          obj.receiveShadow = false;
          obj.renderOrder = 2;
        }
      });
      if (!hasRenderableMesh(clone)) {
        continue;
      }
      this.ghostBodies.set(numericBodyId, clone);
      root.add(clone);
    }
    (this.context.mujocoRoot ?? this.context.scene).add(root);
    return root;
  }

  private isDynamicBody(bodyId: number): boolean {
    const mjModel = this.context.mjModel;
    if (!mjModel || bodyId <= 0 || bodyId >= mjModel.nbody) {
      return false;
    }

    let current = bodyId;
    while (current > 0) {
      if (mjModel.body_jntnum[current] > 0) {
        return true;
      }
      current = mjModel.body_parentid[current];
    }
    return false;
  }

  private async loadMotion(config: TrackingMotionConfig): Promise<LoadedTrackingMotion> {
    if (!(Number.isFinite(config.fps) && config.fps > 0 && config.fps <= MAX_MOTION_FPS)) {
      throw new Error(`Motion '${config.name}' has fps ${config.fps}; expected a positive rate up to ${MAX_MOTION_FPS}.`);
    }
    this.sampleHz = config.fps;
    const npz = await loadNpz(await resolveBytes(config.data));
    const empty: Float32Array[] = [];
    const frameFields = new Map<string, Float32Array[]>();
    for (const name of config.frame_fields ?? []) {
      const entry = npz[name];
      if (!entry) {
        throw new Error(`Motion asset is missing frame field '${name}'`);
      }
      frameFields.set(name, splitFrames(entry));
    }

    if (config.clip_format === 'qpos') {
      if (!npz['qpos']) {
        throw new Error("Motion asset with clip_format='qpos' is missing 'qpos'");
      }
      const qposFrames = splitFrames(npz['qpos']!);
      return {
        ...config,
        jointPos: empty,
        jointVel: empty,
        bodyPosW: empty,
        bodyQuatW: empty,
        bodyLinVelW: empty,
        bodyAngVelW: empty,
        qposFrames,
        frameFields,
        frameCount: qposFrames.length,
      };
    }

    const required = ['joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'] as const;
    for (const key of required) {
      if (!npz[key]) {
        throw new Error(`Motion asset is missing '${key}'`);
      }
    }
    const jointPos = splitFrames(npz['joint_pos']!);
    const jointVel = splitFrames(npz['joint_vel']!);
    const sourceBodyNames = npz['body_names']?.strings ?? null;
    const bodyPosW = this.selectMotionBodyFrames(splitFrames(npz['body_pos_w']!), config.body_names, 3, sourceBodyNames);
    const bodyQuatW = this.selectMotionBodyFrames(splitFrames(npz['body_quat_w']!), config.body_names, 4, sourceBodyNames);
    const bodyLinVelW = this.selectMotionBodyFrames(splitFrames(npz['body_lin_vel_w']!), config.body_names, 3, sourceBodyNames);
    const bodyAngVelW = this.selectMotionBodyFrames(splitFrames(npz['body_ang_vel_w']!), config.body_names, 3, sourceBodyNames);
    return { ...config, jointPos, jointVel, bodyPosW, bodyQuatW, bodyLinVelW, bodyAngVelW, frameFields, frameCount: jointPos.length };
  }

  private selectMotionBodyFrames(
    frames: Float32Array[],
    bodyNames: string[],
    stride: number,
    sourceBodyNames: string[] | null = null,
  ): Float32Array[] {
    const mjModel = this.context.mjModel;
    const first = frames[0];
    if (!mjModel || !first || bodyNames.length === 0) {
      return frames;
    }

    const sourceBodyCount = Math.floor(first.length / stride);
    if (sourceBodyCount === bodyNames.length) {
      return frames;
    }

    let bodySourceIndices: number[];
    if (sourceBodyNames !== null) {
      // Use the npz's own body-name manifest for unambiguous index lookup.
      bodySourceIndices = bodyNames.map((name) => sourceBodyNames.indexOf(name));
    } else {
      // Fall back to mjModel body-ID order from the first body in body_names.
      const rootBodyId = this.findBodyIdByName(bodyNames[0]);
      const bodyIds = bodyNames.map((name) => this.findBodyIdByName(name));
      bodySourceIndices = bodyIds.map((id) => id - rootBodyId);
    }

    if (bodySourceIndices.some((idx) => idx < 0 || idx >= sourceBodyCount)) {
      console.warn('[TrackingCommand] Could not map all motion body names to source body indices; using raw body frames.');
      return frames;
    }

    return frames.map((frame) => {
      const selected = new Float32Array(bodyNames.length * stride);
      for (let i = 0; i < bodySourceIndices.length; i++) {
        const sourceOffset = bodySourceIndices[i] * stride;
        const targetOffset = i * stride;
        for (let j = 0; j < stride; j++) {
          selected[targetOffset + j] = frame[sourceOffset + j] ?? 0.0;
        }
      }
      return selected;
    });
  }

  private updateReferenceState(): void {
    const motion = this.selectedMotion;
    if (!motion || motion.frameCount === 0 || motion.clip_format === 'qpos') {
      this.refRootPos = [];
      this.refRootQuat = [];
      this.refBodyPosW = [];
      this.refBodyQuatW = [];
      this.refBodyLinVelW = [];
      this.refBodyAngVelW = [];
      return;
    }

    this.refBodyPosW = motion.bodyPosW.map((frame) => frame.slice());
    this.refBodyQuatW = motion.bodyQuatW.map((frame) => frame.slice());
    this.refBodyLinVelW = motion.bodyLinVelW.map((frame) => frame.slice());
    this.refBodyAngVelW = motion.bodyAngVelW.map((frame) => frame.slice());
    this.refRootPos = this.refBodyPosW.map((frame) =>
      frame.slice(this.selectedRootBodyIndex * 3, this.selectedRootBodyIndex * 3 + 3),
    );
    this.refRootQuat = this.refBodyQuatW.map((frame) =>
      normalizeQuat(frame.slice(this.selectedRootBodyIndex * 4, this.selectedRootBodyIndex * 4 + 4)),
    );
  }

  private applyReferenceStateToSim(): void {
    const mjModel = this.context.mjModel;
    const mjData = this.context.mjData;
    const motion = this.selectedMotion;
    if (!mjModel || !mjData || !motion || this.refLen === 0 || motion.clip_format === 'qpos') {
      return;
    }

    const rootPos = this.sampleRootPos(this.refIdx);
    const rootQuat = this.sampleRootQuat(this.refIdx);
    const freeJointIndex = this.findFreeJointIndex();
    if (rootPos && rootQuat && freeJointIndex >= 0) {
      const qposAdr = mjModel.jnt_qposadr[freeJointIndex];
      const qvelAdr = mjModel.jnt_dofadr[freeJointIndex];
      mjData.qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      mjData.qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      mjData.qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      mjData.qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      mjData.qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      mjData.qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      mjData.qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;

      const linVel = this.sampleRootVelocity(this.refIdx, this.refBodyLinVelW);
      const angVel = this.sampleRootAngularVelocity(this.refIdx);
      if (linVel && angVel) {
        // A free joint's qvel holds its angular velocity in the body's frame; the clip's is the world's.
        const angVelB = quatApplyInv(rootQuat, angVel);
        mjData.qvel[qvelAdr + 0] = linVel[0] ?? 0.0;
        mjData.qvel[qvelAdr + 1] = linVel[1] ?? 0.0;
        mjData.qvel[qvelAdr + 2] = linVel[2] ?? 0.0;
        mjData.qvel[qvelAdr + 3] = angVelB[0];
        mjData.qvel[qvelAdr + 4] = angVelB[1];
        mjData.qvel[qvelAdr + 5] = angVelB[2];
      }
    }

    const jointPos = this.sampleJointPos(this.refIdx);
    const jointVel = motion.jointVel[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      mjData.qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }
    for (let i = 0; i < this.datasetQposAdr.length && i < jointVel.length; i++) {
      const dofAdr = this.resolveQvelAdrForQposAdr(this.datasetQposAdr[i]);
      if (dofAdr >= 0) {
        mjData.qvel[dofAdr] = jointVel[i] ?? 0.0;
      }
    }

    this.context.mujoco.mj_forward(mjModel, mjData);
    this.applyResetJitter();
  }

  /**
   * Run the traced reference-state-initialization graph, if the build shipped one.
   *
   * mjlab perturbs the reference frame before writing it; this perturbs it after,
   * reading it back off `asset.data` — same numbers, and the clip stays out of the
   * graph. The `mj_forward` above is what makes that read valid.
   *
   * Fire-and-forget, since `reset()` is sync and ORT is not. A frame of un-jittered
   * reference pose is harmless.
   */
  private applyResetJitter(): void {
    const graph = this.resetJitter;
    if (!graph) return;
    void graph
      .fire({
        mjModel: this.context.mjModel,
        mjData: this.context.mjData,
        terrainData: null,
      })
      .then(() => {
        const { mjModel, mjData } = this.context;
        if (mjModel && mjData) this.context.mujoco.mj_forward(mjModel, mjData);
      });
  }

  /**
   * The RSI graph, run through `OnnxEvent` rather than a second `rand`+`entity_write`
   * evaluator. Skips if it or the PRNG is absent: a less varied start, not a broken one.
   */
  private buildResetJitter(config: unknown): OnnxEvent | null {
    if (!isOnnxEventConfig(config)) return null;
    const session = this.context.onnxSessions?.get(config.onnx);
    const rng = this.context.rng;
    if (!session || !rng) {
      console.warn(
        `[TrackingCommand] reset jitter needs the ONNX session "${config.onnx}" and a ` +
          'seeded rng; starting from the unjittered reference frame.',
      );
      return null;
    }
    return new OnnxEvent(config, { session, rng, readSlot: this.context.readOnnxSlot });
  }

  private sampleRootPos(frameIndex: number): Float32Array | null {
    return this.refRootPos[frameIndex] ?? null;
  }

  private sampleRootQuat(frameIndex: number): Float32Array | null {
    return this.refRootQuat[frameIndex] ?? null;
  }

  private sampleRootVelocity(frameIndex: number, source: Float32Array[]): Float32Array | null {
    return (
      source[frameIndex]?.slice(
        this.selectedRootBodyIndex * 3,
        this.selectedRootBodyIndex * 3 + 3,
      ) ?? null
    );
  }

  private sampleRootAngularVelocity(frameIndex: number): Float32Array | null {
    return this.sampleRootVelocity(frameIndex, this.refBodyAngVelW);
  }

  private sampleInitialFrame(frameCount: number): number {
    if (frameCount <= 1 || this.samplingMode === 'start') {
      return 0;
    }
    if (this.samplingMode === 'uniform') {
      // The seeded PRNG, not `Math.random()`, so a session replays; mjlab uses randint.
      const rng = this.context.rng;
      if (!rng) {
        console.warn('[TrackingCommand] no seeded rng in context; starting at frame 0.');
        return 0;
      }
      return Math.min(frameCount - 1, Math.floor(rng.next() * frameCount));
    }
    return 0;
  }

  private sampleJointPos(frameIndex: number): Float32Array {
    return this.refJointPos[frameIndex] ?? new Float32Array(0);
  }

  private resolveQposAdr(jointNames: string[]): number[] {
    const mjModel = this.context.mjModel;
    if (!mjModel || jointNames.length === 0) {
      return [];
    }
    const resolved: number[] = [];
    for (const jointName of jointNames) {
      let adr = -1;
      for (let j = 0; j < mjModel.njnt; j++) {
        const modelJointName = mjModel.jnt(j).name;
        if (modelJointName === jointName || modelJointName.endsWith(`/${jointName}`)) {
          adr = mjModel.jnt_qposadr[j];
          break;
        }
      }
      if (adr >= 0) {
        resolved.push(adr);
      }
    }
    return resolved;
  }

  private resolveQvelAdrForQposAdr(qposAdr: number): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let j = 0; j < mjModel.njnt; j++) {
      if (mjModel.jnt_qposadr[j] === qposAdr) {
        return mjModel.jnt_dofadr[j];
      }
    }
    return -1;
  }

  private findBodyIdByName(bodyName: string): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let b = 0; b < mjModel.nbody; b++) {
      const name = mjModel.body(b).name;
      if (name === bodyName || name.endsWith(`/${bodyName}`)) {
        return b;
      }
    }
    return -1;
  }

  /** Pose only: `updateDebugVisuals` decides whether the ghost is drawn. */
  private updateGhostPose(): void {
    if (!this.ghostRoot || !this.ghostData || !this.context.mjModel || !this.selectedMotion || !this.refLen) {
      return;
    }

    if (this.selectedMotion.clip_format === 'qpos' && this.selectedMotion.qposFrames) {
      const frame = this.selectedMotion.qposFrames[this.refIdx];
      if (frame) {
        this.ghostData.qpos.set(frame);
      }
      this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);
      for (const [bodyId, body] of this.ghostBodies) {
        getPosition(this.ghostData.xpos, bodyId, body.position);
        getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
      }
      return;
    }

    const qpos = this.ghostData.qpos;
    qpos.set(this.context.mjModel.qpos0);

    const rootPos = this.refRootPos[this.refIdx];
    const rootQuat = this.refRootQuat[this.refIdx];
    const freeJointIndex = this.findFreeJointIndex();
    if (freeJointIndex >= 0 && rootPos && rootQuat) {
      const qposAdr = this.context.mjModel.jnt_qposadr[freeJointIndex];
      qpos[qposAdr + 0] = rootPos[0] ?? 0.0;
      qpos[qposAdr + 1] = rootPos[1] ?? 0.0;
      qpos[qposAdr + 2] = rootPos[2] ?? 0.0;
      qpos[qposAdr + 3] = rootQuat[0] ?? 1.0;
      qpos[qposAdr + 4] = rootQuat[1] ?? 0.0;
      qpos[qposAdr + 5] = rootQuat[2] ?? 0.0;
      qpos[qposAdr + 6] = rootQuat[3] ?? 0.0;
    }

    const jointPos = this.refJointPos[this.refIdx] ?? new Float32Array(0);
    for (let i = 0; i < this.datasetQposAdr.length && i < jointPos.length; i++) {
      qpos[this.datasetQposAdr[i]] = jointPos[i] ?? 0.0;
    }

    this.context.mujoco.mj_forward(this.context.mjModel, this.ghostData);

    for (const [bodyId, body] of this.ghostBodies) {
      getPosition(this.ghostData.xpos, bodyId, body.position);
      getQuaternion(this.ghostData.xquat, bodyId, body.quaternion);
    }
  }

  private findFreeJointIndex(): number {
    const mjModel = this.context.mjModel;
    if (!mjModel) {
      return -1;
    }
    for (let i = 0; i < mjModel.njnt; i++) {
      if (mjModel.jnt_type[i] === 0) {
        return i;
      }
    }
    return -1;
  }
}
