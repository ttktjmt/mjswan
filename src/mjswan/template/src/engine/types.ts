/**
 * Public API surface for the headless engine. One simulation at a time, bytes in directly
 * rather than fetched, and verbs named for their cost: `loadScene` rebuilds, the rest are live.
 */
import type { CameraView, ViewerConfig } from '../core/engine/viewer_config';
import type { TerrainData } from '../core/event/EventBase';
import type { Bytes } from '../core/utils/bytes';
import type { SplatTransform } from '../core/scene/splat';
import type { EnginePlugins } from '../core/plugins';

// Bytes: asset bytes or a lazy loader. EnginePlugins: custom MDP terms, trusted-only.
export type { Bytes, SplatTransform, EnginePlugins };

export interface SplatInput {
  data: Bytes;            // .spz
  collider?: Bytes;       // optional collider mesh
  transform?: SplatTransform;
}

export interface MotionInput {
  name: string;
  data: Bytes;            // .npz (lazy loader preserves today's on-demand load)
  default?: boolean;
}

export interface PolicyInput {
  /**
   * The policy's manifest entry merged with its MDP (observations, actions,
   * terminations, commands and events); opaque to the app, interpreted by the engine.
   */
  config: object;
  onnx: Bytes;
  /**
   * Traced term-body graphs of the whole MDP, events included, keyed by the path the
   * config refers to them by (`"mdp/locomotion/obs/actor.onnx"`). The engine never
   * fetches, so the app delivers the bytes; `policyGraphRefs(config)` enumerates what to
   * load. A missing entry warns and skips that term.
   */
  graphs?: Record<string, Bytes>;
  motions?: MotionInput[];
  /** Policy-scoped custom terms (observations / terminations / commands / events). */
  plugins?: EnginePlugins;
}

export interface SceneInput {
  model: Bytes;           // .mjz (engine unpacks)
  policy?: PolicyInput | null;
  splat?: SplatInput | null;
  viewer?: ViewerConfig;
  /**
   * Spawn positions (flat patches) any policy's event terms may draw from. The events
   * themselves travel with the policy (ADR 0006 §3).
   */
  terrainData?: TerrainData;
  /** mjlab's `timestep * decimation`; the model carries only the physics timestep. */
  controlDt?: number;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
}

/** Camera pose in spherical MuJoCo coordinates (x forward, y left, z up). */
export type { CameraView };

export interface CameraControls {
  /** Overwrite the current pose; body tracking (if any) continues, user drag stays live. */
  set(view: Partial<CameraView>): void;
  get(): CameraView;
  /** Re-fit the camera to the scene bounds. */
  frame(): void;
}

/** A policy command term surfaced for generic UI controls. */
export interface CommandDescriptor {
  id: string;             // "group:name"
  group: string;
  type: 'slider' | 'checkbox' | 'button';
  label: string;
  min?: number;           // slider only
  max?: number;           // slider only
  step?: number;          // slider only
  /** Slider only: name of a sibling checkbox that gates this control. */
  enabledWhen?: string;
  /** Slider only: a presentational companion that rescales this slider's drag range. */
  adjustableRange?: SliderRangeControl;
  /** Button only: tabler icon name the build recorded from the term's own GUI. */
  icon?: string;
}

/** Bounds of an {@link CommandDescriptor.adjustableRange} companion slider. */
export interface SliderRangeControl {
  min: number;
  max: number;
  step: number;
  default: number;
  label?: string;
}

export interface CommandControls {
  set(id: string, value: number): void;
  trigger(id: string): void;
}

/** A command term whose debug drawing (mjlab's `debug_vis`) can be shown or hidden. */
export interface DebugVisDescriptor {
  /** The term's config key, and the id {@link DebugVisControls.set} takes. */
  term: string;
  enabled: boolean;
}

export interface DebugVisControls {
  set(term: string, enabled: boolean): void;
}

/** One event term the operator can drive: a `manual` button or an `interval` schedule. */
export interface EventDescriptor {
  /** The term's config name, and the id {@link EventControls} takes. */
  name: string;
  /** The term's own `label`, or its name when it declared none. */
  label: string;
  kind: 'manual' | 'interval';
  /** `interval`: its schedule is running. `manual`: its button can fire — false while
   * the term's `disabled_when` schedule owns the job. */
  armed: boolean;
}

export interface EventControls {
  /** Fire a `mode="manual"` term now. */
  fire(name: string): void;
  /** Start or stop a `mode="interval"` term's schedule. */
  setArmed(name: string, armed: boolean): void;
}

/** Immutable snapshot pushed to {@link MjswanEngine.subscribe} listeners. */
export interface MjswanEngineState {
  phase: 'running' | 'paused';
  loading: boolean;
  loadingMessage: string | null;
  error: Error | null;
  commands: ReadonlyArray<CommandDescriptor>;
  commandValues: Readonly<Record<string, number>>;
  /** Terms with a debug drawing to toggle; empty when the policy has none. */
  debugVis: ReadonlyArray<DebugVisDescriptor>;
  /** Event terms the operator can drive; empty when the scene has none. */
  events: ReadonlyArray<EventDescriptor>;
  /** Reported so an app recording a session can persist it rather than guess. */
  termSeed: number;
}

export interface CreateEngineOptions {
  /** Load `mujoco/mt` (SharedArrayBuffer; requires COOP/COEP). Default false. */
  multithreaded?: boolean;
  /**
   * Seed for the one PRNG every traced term's `rand` comes from. Pass the value read back
   * from {@link MjswanEngineState.termSeed} to re-run a recorded session.
   */
  termSeed?: number;
  /** Put WebXR-tracked hands in the simulation as mocap-driven fingertips. */
  handTracking?: boolean;
}

/** A headless, instance-scoped simulation engine. Create with {@link createEngine}. */
export interface MjswanEngine {
  // content — verbs match switch cost
  loadScene(input: SceneInput): Promise<void>;          // full model rebuild
  setPolicy(input: PolicyInput | null): Promise<void>;  // live, keeps model
  setSplat(input: SplatInput | null): Promise<void>;    // live
  setMotion(name: string | null): Promise<boolean>;     // live; returns whether accepted
  setReferenceVisible(visible: boolean): void;          // motion ghost toggle
  /** Live-update the current splat's placement (dev calibration; no reload). */
  calibrateSplat(transform: SplatTransform): void;

  // playback
  play(): void;
  pause(): void;
  reset(): void;

  // subsystems
  readonly camera: CameraControls;
  readonly commands: CommandControls;
  readonly debugVis: DebugVisControls;
  readonly events: EventControls;

  // state
  getState(): MjswanEngineState;
  subscribe(listener: (state: MjswanEngineState) => void): () => void;

  // misc
  captureThumbnail(options?: { maxDim?: number; quality?: number }): Promise<Blob>;
  dispose(): void;
}
