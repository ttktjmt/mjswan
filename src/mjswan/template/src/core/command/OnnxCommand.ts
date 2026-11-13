/**
 * The single generic command handler: every traced command is a data instantiation of
 * this class — a different graph, `state_fields`, `ui` and `write_targets`.
 *
 * The graph owns the math; this owns the native half — the resample timer that sets
 * `resample_mask`, `prev_state` across frames, `rand` drawn from the seeded PRNG, the
 * `entity_write` application, and the `viz` debug drawing.
 *
 * A UI override overwrites the command *after* the autonomous computation, which is
 * never skipped, as mjlab does at play time.
 *
 * **Async boundary.** `update()`/`getCommand()` are sync but ORT-Web is not, so
 * `update()` kicks off inference and `getCommand()` serves the last completed value.
 * A frame arriving mid-flight is skipped, never queued.
 */

import { SeededRng } from '../rng';
import { applyEntityWrites, type WriteTarget, type WriteValues } from '../event/entityWrite';
import { buildFeeds, declaredFeeds, toFloat32 } from '../onnx/session';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { CommandDebugVisuals, type VizPrimitive } from './debugViz';
import type { CommandConfigEntry, CommandTerm, CommandTermContext, CommandUiConfig } from './types';

export type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader };

export interface OnnxStateFieldSpec {
  name: string;
  shape: number[];
  dtype: string;
  /** As the build found it. Absent means zero-fill, wrong for a counter or held value. */
  init?: number[];
}

export interface OnnxCommandConfig extends CommandConfigEntry {
  onnx: string;
  command_field: string;
  rand_dim: number;
  state_fields: OnnxStateFieldSpec[];
  input_slots?: OnnxInputSlot[];
  write_targets?: WriteTarget[];
  resampling_time_range?: [number, number];
  rand_ranges?: Array<[number, number]>;
  debug_vis?: boolean;
  /** What mjlab's `_debug_vis_impl` draws, as data — see `debugViz.ts`. */
  viz?: VizPrimitive[];
}

export interface OnnxCommandDeps {
  session: OnnxSession;
  rng: SeededRng;
  readSlot?: SlotReader;
}

function numel(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** A state field's tensor at its build-time `init`. Shared with traced action terms. */
export function makeStateTensor(spec: OnnxStateFieldSpec): OnnxTensorLike {
  const n = numel(spec.shape);
  const data = spec.dtype === 'bool' ? new Uint8Array(n) : new Float32Array(n);
  const init = spec.init;
  if (init) {
    // Clamped to the declared width: a mismatch means config and graph disagree.
    for (let i = 0; i < Math.min(n, init.length); i++) data[i] = Number(init[i]);
  }
  return { data, dims: [...spec.shape] };
}

export class OnnxCommand implements CommandTerm {
  private readonly cfg: OnnxCommandConfig;
  private readonly deps: OnnxCommandDeps;
  private readonly context: CommandTermContext | null;

  private state = new Map<string, OnnxTensorLike>();
  private command: Float32Array;
  private timeLeft = 0;
  /** The running `step`, if any — `update` skips on it, `reset` waits for it. */
  private inFlight: Promise<void> | null = null;
  /** Set on the first update so the initial frame resamples (reset semantics). */
  private pendingResample = true;
  private uiValues = new Map<string, number>();
  private readonly visuals: CommandDebugVisuals | null;
  /** mjlab's `_debug_vis_enabled` — starts wherever the task's `debug_vis` put it. */
  private debugVisOn: boolean;

  constructor(
    termName: string,
    config: OnnxCommandConfig,
    context: CommandTermContext | null,
    deps: OnnxCommandDeps,
  ) {
    this.cfg = config;
    this.deps = deps;
    this.context = context;
    for (const spec of config.state_fields) this.state.set(spec.name, makeStateTensor(spec));
    const commandSpec = config.state_fields.find(s => s.name === config.command_field);
    this.command = new Float32Array(commandSpec ? numel(commandSpec.shape) : 0);
    this.timeLeft = this.sampleResampleTime();
    this.debugVisOn = config.debug_vis === true;
    for (const input of config.ui?.inputs ?? []) {
      if (input.type === 'slider') this.uiValues.set(input.name, input.default);
      else if (input.type === 'checkbox') this.uiValues.set(input.name, input.default ? 1 : 0);
    }
    this.visuals =
      config.viz?.length && context
        ? // Under the model root, as `TrackingCommand`'s ghost is: a marker in world
          // coordinates has to move with whatever transform the root carries.
          new CommandDebugVisuals(termName, config.viz, context.mujocoRoot ?? context.scene)
        : null;
  }

  getCommand(): Float32Array {
    // UI override: the autonomous value is already computed; overwrite per axis.
    if (this.isUiEnabled()) {
      const sliders = (this.cfg.ui?.inputs ?? []).filter(i => i.type === 'slider');
      const out = Float32Array.from(this.command);
      for (let i = 0; i < sliders.length && i < out.length; i++) {
        out[i] = this.uiValues.get(sliders[i].name) ?? out[i];
      }
      return out;
    }
    return this.command;
  }

  getUiConfig(): CommandUiConfig | null {
    return this.cfg.ui ?? null;
  }

  /**
   * One traced state field, for a `{command, field}` slot on another term's graph. The
   * command, by field `command` (what `get_command()` reads) or its own name, carries the
   * UI override, as mjlab's joystick writes into that state.
   */
  getStateField(field: string): Float32Array | null {
    if (field === 'command' || field === this.cfg.command_field) {
      return Float32Array.from(this.getCommand());
    }
    const tensor = this.state.get(field);
    // Copied: this tensor feeds the graph next frame, so in-place edits would corrupt it.
    return tensor ? Float32Array.from(toFloat32(tensor.data)) : null;
  }

  /** Advance the timer and kick off inference; never blocks (see class docs). */
  update(dt: number, afterReset = false): void {
    // `reset()` already ran the `dt` 0 update mjlab gives a reset env.
    if (afterReset) return;
    this.timeLeft -= dt;
    if (this.timeLeft <= 0) {
      this.pendingResample = true;
      this.timeLeft += this.sampleResampleTime();
      if (this.timeLeft <= 0) this.timeLeft = this.sampleResampleTime();
    }
    if (this.inFlight) return; // skip, never queue
    const resample = this.pendingResample;
    this.pendingResample = false;
    // Caught here and not in `run()`, which `reset()` awaits and whose caller handles it.
    void this.run(resample).catch((error) => {
      console.warn(`[OnnxCommand] "${this.cfg.name ?? this.cfg.command_field}" failed:`, error);
    });
  }

  async settled(): Promise<void> {
    await this.inFlight?.catch(() => {});
  }

  /**
   * Resample **now**, as mjlab's `CommandTerm.reset` does — before the step's single
   * forward, so an `entity_write` it emits is published by that forward rather than
   * leaving the next observation on a stale `xpos`.
   */
  async reset(): Promise<void> {
    this.timeLeft = this.sampleResampleTime();
    this.pendingResample = false;
    // Never interleaved with a step already running: both read and rewrite `state`.
    await this.inFlight?.catch(() => {});
    await this.run(true, true);
  }

  /** One `step`, tracked so `update` can skip it and `reset` can wait for it. */
  private run(resample: boolean, reset = false): Promise<void> {
    const pending = this.step(resample, reset).finally(() => {
      if (this.inFlight === pending) this.inFlight = null;
    });
    this.inFlight = pending;
    return pending;
  }

  /** Whether the drawing is on, or `null` when there is none — no control is offered. */
  debugVisEnabled(): boolean | null {
    if (!this.visuals || !this.cfg.debug_vis) return null;
    return this.debugVisOn;
  }

  setDebugVisEnabled(enabled: boolean): void {
    this.debugVisOn = enabled;
  }

  /** Redraw the `viz` primitives from the current state, as mjlab redraws each frame. */
  updateDebugVisuals(shown = true): void {
    this.visuals?.update(
      shown && this.debugVisEnabled() === true,
      // mjlab's joystick writes into the term's state, so its arrows follow the sliders.
      field =>
        field === this.cfg.command_field ? this.getCommand() : this.getStateField(field),
      this.deps.readSlot,
    );
  }

  dispose(): void {
    this.visuals?.dispose();
  }

  getUiValue(inputName: string): number | undefined {
    return this.uiValues.get(inputName);
  }

  setValue(inputName: string, value: number): number {
    this.uiValues.set(inputName, value);
    return value;
  }

  triggerButton(inputName: string): boolean {
    // mjlab's Zero button.
    if (inputName !== 'zero') return false;
    for (const input of this.cfg.ui?.inputs ?? []) {
      if (input.type === 'slider') this.uiValues.set(input.name, 0);
    }
    return true;
  }

  /**
   * Run one graph evaluation. `reset` marks an episode reset, fed as `reset_mask`, which
   * a graph declares only when its term adds to mjlab's `reset`. Exposed for tests and
   * deterministic stepping.
   */
  async step(resample: boolean, reset = false): Promise<void> {
    const { feeds } = buildFeeds(this.cfg.input_slots, this.deps.readSlot);
    for (const spec of this.cfg.state_fields) {
      feeds[`prev_${spec.name}`] = this.state.get(spec.name)!;
    }
    feeds.resample_mask = { data: new Uint8Array([resample ? 1 : 0]), dims: [1] };
    feeds.reset_mask = { data: new Uint8Array([reset ? 1 : 0]), dims: [1] };
    feeds.rand = {
      data: this.deps.rng.randVector(this.cfg.rand_dim, this.cfg.rand_ranges),
      dims: [this.cfg.rand_dim],
    };

    const outputs = await this.deps.session.run(declaredFeeds(this.deps.session, feeds));

    for (const spec of this.cfg.state_fields) {
      const next = outputs[`next_${spec.name}`];
      if (next) this.state.set(spec.name, next);
    }
    const commandTensor = this.state.get(this.cfg.command_field);
    if (commandTensor) this.command = toFloat32(commandTensor.data);

    // Only on resample, as mjlab writes the entity from `_resample_command` alone. The
    // graph cannot say so — `resample_mask` gates the state fields only — so its write
    // outputs are a fresh draw every frame, which would teleport the object each step.
    if (resample) this.applyWrites(outputs);
  }

  private applyWrites(outputs: Record<string, OnnxTensorLike>): void {
    const targets = this.cfg.write_targets ?? [];
    if (targets.length === 0) return;
    const mjModel = this.context?.mjModel;
    const mjData = this.context?.mjData;
    if (!mjModel || !mjData) return;
    const values: WriteValues = {};
    for (const [key, tensor] of Object.entries(outputs)) {
      if (!key.startsWith('next_')) values[key] = toFloat32(tensor.data);
    }
    applyEntityWrites(mjModel, mjData, targets, values);
  }

  private isUiEnabled(): boolean {
    const hasCheckbox = (this.cfg.ui?.inputs ?? []).some(
      i => i.type === 'checkbox' && i.name === 'enabled',
    );
    return hasCheckbox && (this.uiValues.get('enabled') ?? 0) > 0.5;
  }

  private sampleResampleTime(): number {
    const range = this.cfg.resampling_time_range;
    if (!range) return Number.POSITIVE_INFINITY; // resample only on reset
    return this.deps.rng.uniform(range[0], range[1]);
  }
}
