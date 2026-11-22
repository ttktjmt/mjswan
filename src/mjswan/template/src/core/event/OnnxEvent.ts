/**
 * The generic ONNX-backed event handler, symmetric to `OnnxCommand`: every traced event
 * term is a data instantiation of this class. Unlike a command, an event body is a pure
 * function of its constants, `input_slots` and `rand` — no state across frames.
 *
 * The graph owns the math; this draws `rand` from the seeded PRNG and applies the
 * outputs to `mjData` (`entityWrite.ts`) or `mjModel` (`modelWrite.ts`). *When* it fires
 * is the caller's business (`EventManager`).
 *
 * **Async boundary.** Interval/startup dispatch is fire-and-forget with an in-flight
 * guard, so a slow graph cannot queue a backlog of the same disturbance. Reset firings
 * are awaited, since the reset must be complete before the next frame renders.
 */

import { SeededRng } from '../rng';
import { buildFeeds, declaredFeeds, toFloat32 } from '../onnx/session';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import { applyEntityWrites, type WriteTarget, type WriteValues } from './entityWrite';
import type { EventContext } from './EventBase';
import {
  applyModelWrite,
  isModelWriteTarget,
  setConstKeepingQpos,
  type ModelWriteTarget,
} from './modelWrite';

export type EventMode = 'startup' | 'reset' | 'interval' | 'manual';

export interface OnnxEventConfig {
  name: string;
  mode: EventMode;
  onnx: string;
  rand_dim: number;
  input_slots?: OnnxInputSlot[];
  write_targets?: Array<WriteTarget | ModelWriteTarget>;
  rand_ranges?: Array<[number, number]>;
  /** The model writes leave MuJoCo's derived constants stale, so `mj_setConst` follows. */
  set_const?: boolean;
  /** `mode="interval"` only: `[min, max]` seconds between firings. */
  interval_range_s?: [number, number];
  /** `mode="interval"` only: timer survives episode reset when true. */
  is_global_time?: boolean;
  /** `mode="reset"` only: suppress firing on resets that arrive too soon. */
  min_step_count_between_reset?: number;
  /** Control-panel text: a `manual` term's button, an `interval` term's arm checkbox. */
  label?: string;
  /** `mode="manual"` only: the `mode="interval"` term whose armed schedule greys it out. */
  disabled_when?: string;
}

export function isOnnxEventConfig(config: unknown): config is OnnxEventConfig {
  return (
    typeof config === 'object'
    && config !== null
    && typeof (config as { onnx?: unknown }).onnx === 'string'
    && typeof (config as { mode?: unknown }).mode === 'string'
  );
}

export interface OnnxEventDeps {
  session: OnnxSession;
  rng: SeededRng;
  readSlot?: SlotReader;
}

export class OnnxEvent {
  readonly name: string;
  readonly mode: EventMode;
  private inFlight = false;

  constructor(
    readonly config: OnnxEventConfig,
    private readonly deps: OnnxEventDeps,
  ) {
    this.name = config.name;
    this.mode = config.mode;
  }

  /** True while a `fire()` is in flight (callers use this to skip, not queue). */
  get busy(): boolean {
    return this.inFlight;
  }

  /** Run the graph once and apply its writes; false when a run already in flight drops it. */
  async fire(context: EventContext): Promise<boolean> {
    if (this.inFlight) return false;
    this.inFlight = true;
    try {
      const { feeds } = buildFeeds(this.config.input_slots, this.deps.readSlot);
      feeds.rand = {
        data: this.deps.rng.randVector(this.config.rand_dim, this.config.rand_ranges),
        dims: [this.config.rand_dim],
      };

      const outputs = await this.deps.session.run(declaredFeeds(this.deps.session, feeds));
      this.applyWrites(context, outputs);
      return true;
    } finally {
      this.inFlight = false;
    }
  }

  private applyWrites(context: EventContext, outputs: Record<string, OnnxTensorLike>): void {
    const targets = this.config.write_targets ?? [];
    if (targets.length === 0) return;
    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return;
    const values: WriteValues = {};
    for (const [key, tensor] of Object.entries(outputs)) values[key] = toFloat32(tensor.data);
    const entityTargets = targets.filter((t): t is WriteTarget => !isModelWriteTarget(t));
    applyEntityWrites(mjModel, mjData, entityTargets, values);
    let wroteModel = false;
    for (const target of targets) {
      if (!isModelWriteTarget(target)) continue;
      wroteModel = applyModelWrite(mjModel, target, values, context.modelDefaults) || wroteModel;
    }
    if (!wroteModel || !this.config.set_const) return;
    if (context.mujoco) setConstKeepingQpos(context.mujoco, mjModel, mjData);
    else console.warn(`[OnnxEvent] "${this.name}" owes an mj_setConst but has no MuJoCo module.`);
  }
}
