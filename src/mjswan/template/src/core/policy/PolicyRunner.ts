import { ObservationBase } from '../observation/ObservationBase';
import {
  FusedObservation,
  isFusedObservationConfig,
  type FusedObservationConfig,
} from '../observation/FusedObservation';
import {
  HistoryObservation,
  historyOffsets,
  writeInterleavedFrame,
} from '../observation/HistoryObservation';
import {
  NativeObservation,
  isNativeObservationConfig,
} from '../observation/NativeObservation';
import {
  OnnxObservation,
  isOnnxObservationConfig,
  type OnnxObservationConfig,
} from '../observation/OnnxObservation';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';
import { type Bytes, resolveBytes } from '../utils/bytes';
import { PolicyModule } from './PolicyModule';
import type {
  ObservationConfigEntry,
  PolicyConfig,
  PolicyRunnerContext,
  PolicyState,
} from './types';

/** A frame-major stack re-laid element-major, as `history_interleaved` asks for. */
function interleaveStack(
  buffer: Float32Array,
  width: number,
  steps: number,
): Float32Array {
  const out = new Float32Array(buffer.length);
  for (let i = 0; i < steps; i++) {
    writeInterleavedFrame(
      out,
      buffer.subarray(i * width, (i + 1) * width),
      width,
      i,
      steps,
    );
  }
  return out;
}

export type PolicyModuleConstructor = new (config: PolicyConfig) => PolicyModule;
export type ObservationConstructor = new (
  runner: PolicyRunner,
  config: ObservationConfigEntry
) => ObservationBase;

export type PolicyRunnerOptions = {
  policyModules?: Record<string, PolicyModuleConstructor>;
  observations?: Record<string, ObservationConstructor>;
  /** App-supplied motion clips (name → bytes), exposed to terms via getMotionData. */
  motions?: Array<{ name: string; data: Bytes }>;
  /** The loaded graphs and slot reader traced observations need; absent if none. */
  onnxSessions?: OnnxSessionCache;
  readOnnxSlot?: SlotReader;
};

export class PolicyRunner {
  private config: PolicyConfig;
  private options: PolicyRunnerOptions;
  private policyModule: PolicyModule | null;
  private obsGroups: Record<string, ObservationBase[]>;
  private obsLayouts: Record<string, { name: string; size: number }[]>;
  private obsSizes: Record<string, number>;
  private historyConfig: Record<string, { steps: number; interleaved: boolean }>;
  private historyBuffers: Record<string, Float32Array>;
  /** Groups whose history must be filled with the next frame (set by `reset()`). */
  private historyNeedsPrime: Record<string, boolean> = {};
  private defaultObsKey: string | null;
  private context: PolicyRunnerContext | null;
  private policyJointNames: string[];
  private defaultJointPos: Float32Array;
  private encoderBias: Float32Array;
  private numActions: number;
  private lastActions: Float32Array;
  /**
   * Older entries of mjlab's action window, newest first: `[prev_action,
   * prev_prev_action]`. `lastActions` is `action`, so together they are the three
   * `ActionManager` keeps.
   */
  private olderActions: Float32Array[];
  private motionCache: Map<string, Promise<ArrayBuffer | null>> = new Map();

  constructor(config: PolicyConfig, options: PolicyRunnerOptions = {}) {
    this.config = config;
    this.options = options;
    this.policyModule = null;
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.historyNeedsPrime = {};
    this.defaultObsKey = null;
    this.context = null;

    this.policyJointNames = (config.policy_joint_names ?? []).slice();
    this.numActions = (config.policy_num_actions as number | undefined) ?? this.policyJointNames.length;
    this.lastActions = new Float32Array(this.numActions);
    this.olderActions = [
      new Float32Array(this.numActions),
      new Float32Array(this.numActions),
    ];
    this.defaultJointPos = this.normalizeArray(
      config.default_joint_pos ?? [],
      this.numActions,
      0.0
    );
    this.encoderBias = this.normalizeArray(
      config.encoder_bias ?? [],
      this.numActions,
      0.0
    );
  }

  async init(context: PolicyRunnerContext): Promise<void> {
    this.context = context;
    this.policyModule = await this.buildPolicyModule(context);
    this.buildObservationGroups();
  }

  reset(state?: PolicyState): void {
    this.lastActions.fill(0.0);
    for (const actions of this.olderActions) actions.fill(0.0);
    this.policyModule?.reset();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.reset) {
          obs.reset(state);
        }
      }
    }
    if (state) {
      // Priming computes a frame, which is async for an ONNX term while `reset()` is not.
      // Flag it and prime on the next collect, which uses the frame actually about to run.
      for (const [key, config] of Object.entries(this.historyConfig)) {
        if (config.steps > 1) this.historyNeedsPrime[key] = true;
      }
    }
  }

  update(state: PolicyState): void {
    this.policyModule?.update();
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (obs.update) {
          obs.update(state);
        }
      }
    }
  }

  /** Async because ONNX-backed terms run ORT inference. */
  async collectObservationsByKey(state: PolicyState): Promise<Record<string, Float32Array>> {
    this.update(state);
    const outputs: Record<string, Float32Array> = {};

    for (const [key, obsList] of Object.entries(this.obsGroups)) {
      const history = this.historyConfig[key];
      if (history && history.steps > 1) {
        const frame = await this.buildFrame(obsList, state);
        const buffer = this.historyBuffers[key];
        if (this.historyNeedsPrime[key]) {
          // First frame after a reset: fill every slot, never a history of untrained zeros.
          for (let i = 0; i < history.steps; i++) buffer.set(frame, i * frame.length);
          delete this.historyNeedsPrime[key];
        } else {
          // Shift the oldest frame out the front; the newest lands last, the order
          // `history_length` and `HistoryObservation` use.
          buffer.copyWithin(0, frame.length);
          buffer.set(frame, buffer.length - frame.length);
        }
        // The buffer is frame-major either way; interleaving is an output layout.
        outputs[key] = history.interleaved
          ? interleaveStack(buffer, frame.length, history.steps)
          : new Float32Array(buffer);
      } else {
        outputs[key] = await this.buildFrame(obsList, state);
      }
    }
    return outputs;
  }

  async collectObservations(state: PolicyState): Promise<Float32Array> {
    const outputs = await this.collectObservationsByKey(state);
    if (this.defaultObsKey && outputs[this.defaultObsKey]) {
      return outputs[this.defaultObsKey];
    }
    const first = Object.keys(outputs)[0];
    return first ? outputs[first] : new Float32Array(0);
  }

  /** Await all observation preload() promises before the first inference step. */
  async preloadAll(): Promise<void> {
    const promises: Promise<void>[] = [];
    for (const obsList of Object.values(this.obsGroups)) {
      for (const obs of obsList) {
        if (typeof obs.preload === 'function') {
          promises.push(obs.preload());
        }
      }
    }
    await Promise.all(promises);
  }

  getObservationSize(): number {
    if (this.defaultObsKey && this.obsSizes[this.defaultObsKey] !== undefined) {
      return this.obsSizes[this.defaultObsKey];
    }
    const first = Object.keys(this.obsSizes)[0];
    return first ? this.obsSizes[first] : 0;
  }

  getObservationLayout(): { name: string; size: number }[] {
    if (this.defaultObsKey && this.obsLayouts[this.defaultObsKey]) {
      return this.obsLayouts[this.defaultObsKey].map((entry) => ({ ...entry }));
    }
    const first = Object.keys(this.obsLayouts)[0];
    return first ? this.obsLayouts[first].map((entry) => ({ ...entry })) : [];
  }

  getPolicyModuleContext(): Record<string, unknown> {
    return this.policyModule?.getContext() ?? {};
  }

  getPolicyModule(): PolicyModule | null {
    return this.policyModule;
  }

  getContext(): PolicyRunnerContext | null {
    return this.context;
  }

  getPolicyJointNames(): string[] {
    return this.policyJointNames.slice();
  }

  getNumActions(): number {
    return this.numActions;
  }

  getDefaultJointPos(): Float32Array {
    return new Float32Array(this.defaultJointPos);
  }

  getEncoderBias(): Float32Array {
    return new Float32Array(this.encoderBias);
  }

  getLastActions(): Float32Array {
    return new Float32Array(this.lastActions);
  }

  /**
   * The action `age` control steps back — 0 being the newest, as mjlab's `last_action`
   * reads it. Ages past the window return zeros, which is what mjlab's own buffers
   * hold before that many steps have run.
   */
  getActions(age: number): Float32Array {
    if (age <= 0) return this.getLastActions();
    const older = this.olderActions[age - 1];
    return new Float32Array(older ?? new Float32Array(this.numActions));
  }

  getConfig(): PolicyConfig {
    return this.config;
  }

  /** An app-supplied clip's bytes by name (cached); custom terms read this, never a URL. */
  getMotionData(name: string): Promise<ArrayBuffer | null> {
    const cached = this.motionCache.get(name);
    if (cached) return cached;
    const motion = this.options.motions?.find((m) => m.name === name);
    const promise: Promise<ArrayBuffer | null> = motion
      ? resolveBytes(motion.data)
      : Promise.resolve(null);
    this.motionCache.set(name, promise);
    return promise;
  }

  setLastActions(actions: Float32Array): void {
    // Shift the window before overwriting the newest, as mjlab's `process_action` does.
    if (actions.length !== this.lastActions.length) {
      // A different action width is a different policy; its history starts empty.
      this.lastActions = new Float32Array(actions);
      this.olderActions = [
        new Float32Array(actions.length),
        new Float32Array(actions.length),
      ];
      return;
    }
    this.olderActions[1].set(this.olderActions[0]);
    this.olderActions[0].set(this.lastActions);
    this.lastActions.set(actions);
  }

  private async buildPolicyModule(
    context: PolicyRunnerContext
  ): Promise<PolicyModule | null> {
    const registry = this.options.policyModules ?? {};
    const moduleKey = this.config.policy_module;
    const Module = moduleKey ? registry[moduleKey] : registry.default;

    if (moduleKey && !Module) {
      throw new Error(`Unknown policy module: ${moduleKey}`);
    }

    if (!Module) {
      return null;
    }

    const module = new Module(this.config);
    await module.init(context);
    return module;
  }

  private buildObservationGroups(): void {
    const registry = this.options.observations ?? {};
    const obsConfig = this.config.observations ?? {};
    this.obsGroups = {};
    this.obsLayouts = {};
    this.obsSizes = {};
    this.historyConfig = {};
    this.historyBuffers = {};
    this.historyNeedsPrime = {};
    this.defaultObsKey = null;

    const buildTerm = (entry: ObservationConfigEntry): ObservationBase => {
      // Traced and native terms bypass the registry, so `entry.name` is the term's identity.
      if (isOnnxObservationConfig(entry)) {
        return this.buildOnnxObservation(entry);
      }
      if (isNativeObservationConfig(entry)) {
        return new NativeObservation(this, entry);
      }
      const ObsClass = registry[entry.name];
      if (!ObsClass) {
        throw new Error(`Unknown observation type: ${entry.name}`);
      }
      return new ObsClass(this, entry);
    };

    // mjlab stacks per term. Only ONNX/native entries wrap: registry classes self-stack.
    const buildObservation = (entry: ObservationConfigEntry): ObservationBase => {
      const base = buildTerm(entry);
      const offsets =
        isOnnxObservationConfig(entry) || isNativeObservationConfig(entry)
          ? historyOffsets(entry)
          : null;
      return offsets ? new HistoryObservation(this, entry, base, offsets) : base;
    };

    for (const [key, value] of Object.entries(obsConfig)) {
      // A fused group is one graph for all its terms; below is the unfused path.
      if (isFusedObservationConfig(value)) {
        const fused = this.buildFusedObservation(key, value);
        this.registerGroup(key, [fused], [{ name: key }], undefined, value.layout);
        continue;
      }
      if (Array.isArray(value)) {
        const obsList = value.map(buildObservation);
        this.registerGroup(key, obsList, value);
        continue;
      }
      if (value && typeof value === 'object') {
        const configValue = value as {
          history_steps?: number;
          interleaved?: boolean;
          components?: ObservationConfigEntry[];
        };
        if (Array.isArray(configValue.components)) {
          // Group history owns the stacking, so each component computes one frame.
          const obsList = configValue.components.map((entry) =>
            buildObservation({ ...entry, history_steps: 1 }),
          );
          const steps = Math.max(1, Math.floor(configValue.history_steps ?? 1));
          const interleaved = Boolean(configValue.interleaved);
          this.registerGroup(key, obsList, configValue.components, {
            steps,
            interleaved,
          });
        }
      }
    }

    // The group a single-input policy is fed: the default slot when it exists, else the
    // one group there is (its name is free), else the first declared.
    this.defaultObsKey = this.obsGroups.actor
      ? 'actor'
      : (Object.keys(this.obsGroups)[0] ?? null);
  }

  /**
   * Build a traced-ONNX observation term, or throw: unlike a command or event, dropping one
   * shifts every later term's offset in the policy's input vector.
   */
  private buildOnnxObservation(entry: OnnxObservationConfig): OnnxObservation {
    const session = this.options.onnxSessions?.get(entry.onnx);
    const readSlot = this.options.readOnnxSlot;
    if (!session || !readSlot) {
      throw new Error(
        `Observation "${entry.name}" needs the ONNX session "${entry.onnx}" and a ` +
          'slot reader; pass onnxSessions/readOnnxSlot in PolicyRunnerOptions.'
      );
    }
    return new OnnxObservation(this, entry, { session, readSlot });
  }

  /** Build the single handler for a fused group, or throw, as the per-term case does. */
  private buildFusedObservation(
    key: string,
    config: FusedObservationConfig
  ): FusedObservation {
    const session = this.options.onnxSessions?.get(config.fused);
    const readSlot = this.options.readOnnxSlot;
    if (!session || !readSlot) {
      throw new Error(
        `Observation group "${key}" needs the ONNX session "${config.fused}" and a ` +
          'slot reader; pass onnxSessions/readOnnxSlot in PolicyRunnerOptions.'
      );
    }
    return new FusedObservation(this, { ...config, name: key }, { session, readSlot });
  }

  private registerGroup(
    key: string,
    obsList: ObservationBase[],
    configList: ObservationConfigEntry[],
    history?: { steps: number; interleaved: boolean },
    /** Fused groups only: per-term widths, since one handler covers every term. */
    fusedLayout?: Array<{ name: string; size: number }>
  ): void {
    this.obsGroups[key] = obsList;
    this.obsLayouts[key] = fusedLayout
      ? fusedLayout.map((entry) => ({ ...entry }))
      : obsList.map((obs, index) => ({
          name: configList[index]?.name ?? `obs_${index}`,
          size: obs.size,
        }));
    const baseSize = this.obsLayouts[key].reduce((sum, entry) => sum + entry.size, 0);
    if (history && history.steps > 1) {
      this.historyConfig[key] = history;
      this.historyBuffers[key] = new Float32Array(baseSize * history.steps);
      this.obsSizes[key] = baseSize * history.steps;
    } else {
      this.obsSizes[key] = baseSize;
    }
  }

  private async buildFrame(
    obsList: ObservationBase[],
    state: PolicyState
  ): Promise<Float32Array> {
    // Sized from the actual arrays, since a term's `size` getter may lag its output, and
    // kicked off as a batch so the group's graphs do not serialize.
    const arrays = await Promise.all(
      obsList.map(async (obs) => {
        const value = await obs.compute(state);
        const array = value instanceof Float32Array ? value : Float32Array.from(value);
        if (array.length !== obs.size) {
          throw new Error(
            `Observation size mismatch: expected ${obs.size}, got ${array.length}`
          );
        }
        return array;
      })
    );
    const total = arrays.reduce((sum, array) => sum + array.length, 0);
    const output = new Float32Array(total);
    let offset = 0;
    for (const array of arrays) {
      output.set(array, offset);
      offset += array.length;
    }
    return output;
  }

  private normalizeArray(
    values: number[],
    length: number,
    fallback: number
  ): Float32Array {
    const output = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      output[i] = typeof values[i] === 'number' ? values[i] : fallback;
    }
    return output;
  }
}
