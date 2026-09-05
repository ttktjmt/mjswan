import type { MainModule, MjData, MjModel } from 'mujoco';
import type { Scene } from 'three';

import type { CommandsConfig } from '../command';
import type { CommandManager } from '../command/CommandManager';
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
  ema_alpha?: number;
  warmup_time_s?: number;
  actuator_names?: string[];
  [key: string]: unknown;
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
   * the action terms and any `last_action` observation see — not `ActionConfigEntry.clip`,
   * which bounds `raw * scale + offset` per target.
   */
  clip_actions?: number;
  /**
   * The ONNX input slot table (ADR 0006 §5): `in_keys[i]` names the tensor that fills the
   * network's i-th input, an observation group or one the runtime synthesizes. Absent
   * for the common single-input policy, whose one input takes the `actor` group.
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
  terminations?: Record<string, TerminationConfigEntry>;
  /** The MDP's event terms, switched with the policy rather than held by the scene (ADR 0006 §3). */
  events?: EventConfig[];
  [key: string]: unknown;
};
