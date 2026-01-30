import type { MjData, MjModel } from '@/mujoco';
import type { Mujoco } from '../../types/mujoco';

export type PolicyRunnerContext = {
  mujoco: Mujoco;
  mjModel: MjModel | null;
  mjData: MjData | null;
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

export type PolicyConfig = {
  policy_module?: string;
  policy_joint_names?: string[];
  default_joint_pos?: number[];
  action_scale?: number[] | number;
  stiffness?: number[] | number;
  damping?: number[] | number;
  control_type?: string;
  onnx?: {
    path: string;
    meta?: {
      in_keys?: string[];
      out_keys?: (string | string[])[];
    };
  };
  obs_config?: Record<string, ObservationGroupConfig>;
  [key: string]: unknown;
};
