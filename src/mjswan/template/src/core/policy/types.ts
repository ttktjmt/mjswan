import type { MainModule, MjData, MjModel } from 'mujoco';
import type { Scene } from 'three';

import type { DelayConfigEntry } from '../action/commandDelay';
import type { OnnxActionGraphConfig, OnnxActionProcessConfig } from '../action/OnnxActionProcess';
import type { CommandsConfig } from '../command';
import type { CommandManager } from '../command/CommandManager';
import type { DebugVisConfig } from '../debugViz/drawings';
import type { EventConfig } from '../event/EventBase';
import type { Bytes } from '../utils/bytes';

export type PolicyRunnerContext = {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene?: Scene | null;
  /** Instance-scoped command manager; command-state input slots read it via the runner. */
  commandManager?: CommandManager;
};

export type PolicyState = {
  jointPos: Float32Array;
  jointVel?: Float32Array;
  rootPos?: Float32Array;
  rootQuat?: Float32Array;
  rootLinVel?: Float32Array;
  rootAngVel?: Float32Array;
  [key: string]: unknown;
};

export type ObservationConfigEntry = {
  name: string;
  [key: string]: unknown;
};

export type ObservationGroupConfig =
  | ObservationConfigEntry[]
  | {
    history_steps?: number;
    interleaved?: boolean;
    components?: ObservationConfigEntry[];
  };

export type ActionConfigEntry = {
  type: string;
  scale?: number | number[] | Record<string, number>;
  offset?: number | Record<string, number>;
  use_default_offset?: boolean;
  stiffness?: number | number[] | Record<string, number>;
  damping?: number | number[] | Record<string, number>;
  /** mjlab's DC motor torque-speed curve on the term's joints (format 10). */
  saturation_effort?: number | number[] | Record<string, number>;
  velocity_limit?: number | number[] | Record<string, number>;
  actuator_names?: string[];
  clip?: Record<string, readonly number[]>;
  /** Ramp each target across the substeps, from the previous processed action. */
  interpolate?: boolean;
  /** Seconds from an episode's start during which every target is the measured position. */
  hold_s?: number;
  /** mjlab's actuator command delays (`ActuatorCfg.delay_*`) on the term's joints. */
  delay?: DelayConfigEntry[];
  /** A traced term's `process_actions`, which replaces the native formula. */
  process?: OnnxActionProcessConfig;
  /** A traced term's `reset`, present when it writes the term's state. */
  reset?: OnnxActionGraphConfig;
  [key: string]: unknown;
};

/** An mjlab actuator's traced torque law (format 11). */
export type ActuatorConfigEntry = {
  /** The model joints the law drives, in its graph's column order. */
  joints: string[];
  /** The law's graph, scene-relative, run by `core/actuator/` rather than ONNX Runtime. */
  onnx: string;
  func?: string;
  doc?: string;
};

export type TerminationConfigEntry = {
  name: string;
  params?: Record<string, unknown>;
  time_out?: boolean;
};

export type PolicyConfig = {
  policy_module?: string;
  policy_joint_names?: string[];
  policy_num_actions?: number;
  default_joint_pos?: number[];
  encoder_bias?: number[];
  action_scale?: number[] | number;
  stiffness?: number[] | number;
  damping?: number[] | number;
  control_type?: string;
  /**
   * Symmetric bound on the raw policy output, mirroring rsl-rl's
   * `RslRlVecEnvWrapper`. It clamps before `env.step`, so the clamped vector is what
   * the action terms and any `last_action` observation see. Not `ActionConfigEntry.clip`,
   * which bounds `raw * scale + offset` per target.
   */
  clip_actions?: number;
  /**
   * The ONNX input slot table: `in_keys[i]` names the tensor that fills the network's
   * i-th input, an observation group or one the runtime synthesizes. Absent for the
   * common single-input policy, whose one input takes the `actor` group.
   */
  in_keys?: string[];
  /** The output slot table, positional like `in_keys`; absent means `['action']`. */
  out_keys?: (string | string[])[];
  commands?: CommandsConfig;
  motions?: Array<{
    name: string;
    /** Injected by the engine from PolicyInput.motions (matched by name). */
    data?: Bytes;
    anchor_body_name: string;
    body_names: string[];
    dataset_joint_names?: string[];
    default?: boolean;
    [key: string]: unknown;
  }>;
  observations?: Record<string, ObservationGroupConfig>;
  actions?: Record<string, ActionConfigEntry>;
  /** The MDP's traced actuator laws, keyed by a name of their class and place. */
  actuators?: Record<string, ActuatorConfigEntry>;
  terminations?: Record<string, TerminationConfigEntry>;
  /** The MDP's event terms, switched with the policy rather than held by the scene. */
  events?: EventConfig[];
  /** The sensors and reward terms Debug Viz draws beside the command terms. */
  debug_vis?: DebugVisConfig;
  [key: string]: unknown;
};
