import type * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import type { SeededRng } from '../rng';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';

export type CommandType = 'slider' | 'button' | 'checkbox';

/**
 * A companion slider rescaling another's drag range, as mjlab's play GUI pairs each
 * velocity axis with a "Max <label>". Not simulation state: it sends nothing to the engine.
 */
export interface SliderRangeControl {
  min: number;
  max: number;
  step: number;
  default: number;
  /** Label for the companion slider; defaults to `Max <label>`. */
  label?: string;
}

export interface SliderCommandConfig {
  type: 'slider';
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  enabled_when?: string;
  /** Renders a companion range slider and clamps this one to `[-value, value]`. */
  adjustable_range?: SliderRangeControl;
}

export interface ButtonCommandConfig {
  type: 'button';
  name: string;
  label: string;
  /** Tabler icon name, as viser's `Icon` spells it; an unknown one goes undrawn. */
  icon?: string;
}

export interface CheckboxCommandConfig {
  type: 'checkbox';
  name: string;
  label: string;
  default: boolean;
}

export type CommandInputConfig = SliderCommandConfig | ButtonCommandConfig | CheckboxCommandConfig;

export interface CommandUiConfig {
  inputs?: CommandInputConfig[];
}

export interface CommandConfigEntry {
  name: string;
  ui?: CommandUiConfig;
  [key: string]: unknown;
}

export type CommandsConfig = Record<string, CommandConfigEntry>;

export interface CommandDefinition {
  id: string;
  groupName: string;
  config: CommandInputConfig;
}

export function getCommandInputId(groupName: string, inputName: string): string {
  return `${groupName}:${inputName}`;
}

export type CommandEventType =
  | 'change'
  | 'reset'
  | 'button'
  | 'group_registered'
  | 'clear'
  | 'debug_vis';

export interface CommandEvent {
  type: CommandEventType;
  commandId: string;
  groupName?: string;
  value?: number;
}

export type CommandEventListener = (event: CommandEvent) => void;

export interface CommandTermContext {
  mujoco: MainModule;
  mjModel: MjModel | null;
  mjData: MjData | null;
  scene: THREE.Scene;
  bodies?: Record<number, THREE.Group> | null;
  mujocoRoot?: THREE.Group | null;
  requestReset?: () => void;
  /** The seeded PRNG and loaded graphs `OnnxCommand` terms need; absent if none. */
  rng?: SeededRng;
  onnxSessions?: OnnxSessionCache;
  /** Reads a command's declared dynamic runtime input slots. */
  readOnnxSlot?: SlotReader;
  /** The host's `ArchiveLimits.maxInflatedBytes`, which bounds a motion's `.npz` too. */
  maxInflatedBytes?: number;
}

export interface CommandTerm {
  getCommand(): Float32Array;
  getUiConfig?(): CommandUiConfig | null;
  /** One named piece of the term's state; `null` for a field it does not publish. */
  getStateField?(field: string): Float32Array | null;
  /** Episode reset — the resample for a traced term, hence async and awaited. */
  reset?(): void | Promise<void>;
  /** `afterReset`: an auto-reset ran `reset()` earlier in this step. */
  update?(dt: number, afterReset?: boolean): void;
  /** `shown` is Debug Viz "Enabled": false hides the drawing whatever the term's own switch. */
  updateDebugVisuals?(shown?: boolean): void;
  /** mjlab's `_debug_vis_enabled`; `null` when the term draws nothing. */
  debugVisEnabled?(): boolean | null;
  setDebugVisEnabled?(enabled: boolean): void;
  setValue?(inputName: string, value: number): number | void;
  /** Current value of one UI input, for the panel's mirror of it. */
  getUiValue?(inputName: string): number | undefined;
  /** A press. Return false for a name the term does nothing with; nothing means handled. */
  triggerButton?(inputName: string): boolean | void;
  dispose?(): void;
}

export interface TrackingSource {
  isReady(): boolean;
  getAnchorBodyName(): string | null;
  getAnchorPos(frameIndex?: number): Float32Array | null;
  getAnchorQuat(frameIndex?: number): Float32Array | null;
  getBodyNames(): string[];
  getBodyPosW(frameIndex?: number): Float32Array | null;
}

export function isTrackingSource(term: unknown): term is TrackingSource {
  return (
    typeof term === 'object' &&
    term !== null &&
    'isReady' in term &&
    'getAnchorBodyName' in term &&
    'getBodyNames' in term
  );
}

export type CommandTermConstructor = new (
  termName: string,
  config: CommandConfigEntry,
  context: CommandTermContext
) => CommandTerm;
