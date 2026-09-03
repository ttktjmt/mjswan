/**
 * One graph for a whole observation group, replacing its terms with a single
 * `ort.run()`. A per-term graph is often a single node, so the fixed per-call cost is
 * the whole expense, and a shared slot gets marshalled twice.
 *
 * The build folded per-term clip/scale and the concatenation in, so the output *is* the
 * group vector. Native terms (`prev_action`, a generated command) are fed as graph
 * inputs to keep it complete, with a `command` input's name bound at construction —
 * unbound, it would arrive as a zero block inside the policy's input vector.
 *
 * History stays out: a stateless graph cannot hold it, so the build refuses to fuse a
 * group that carries any.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import { assertCommandTermBound, sliceStoredActions } from './NativeObservation';
import { conformToSize } from './pipeline';
import type { OnnxInputSlot, OnnxSession, SlotReader } from '../onnx/session';
import { buildFeeds } from '../onnx/session';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

/** A native term the fused graph takes as an input rather than computing. */
export interface FusedNativeInput {
  name: string;
  native: 'prev_action' | 'command';
  /** Graph input name to feed the value as. */
  input: string;
  size: number;
  /** `command` only: which command term to read. */
  command_name?: string;
  /** `prev_action` only: which action term, when the term names one. */
  action_name?: string;
  /** `prev_action` with an `action_name`: where that term's slice starts. */
  action_offset?: number;
  /** `prev_action` only: control steps back, 0 (the default) being the newest. */
  age?: number;
}

export interface FusedObservationConfig extends ObservationConfig {
  /** Path to the group's graph (the field's presence is what marks a group fused). */
  fused: string;
  size: number;
  input_slots?: OnnxInputSlot[];
  native_inputs?: FusedNativeInput[];
  /** Per-term `{name, size}` in concat order, for the runner's group layout. */
  layout?: Array<{ name: string; size: number }>;
}

export interface FusedObservationDeps {
  session: OnnxSession;
  readSlot: SlotReader;
}

/** Whether a group config is a single fused graph rather than a list of terms. */
export function isFusedObservationConfig(entry: unknown): entry is FusedObservationConfig {
  return (
    typeof entry === 'object' &&
    entry !== null &&
    typeof (entry as { fused?: unknown }).fused === 'string'
  );
}

export class FusedObservation extends ObservationBase<FusedObservationConfig> {
  private readonly deps: FusedObservationDeps;
  /** Last completed vector, served if a later frame's inference cannot run. */
  private last: Float32Array;

  constructor(
    runner: PolicyRunner,
    config: FusedObservationConfig,
    deps: FusedObservationDeps,
  ) {
    super(runner, config);
    for (const native of config.native_inputs ?? []) {
      if (native.native === 'command') {
        assertCommandTermBound(runner, `${config.name}.${native.name}`, native.command_name);
      }
    }
    this.deps = deps;
    this.last = new Float32Array(config.size);
  }

  get size(): number {
    return this.config.size;
  }

  async compute(_state: PolicyState): Promise<Float32Array> {
    const { feeds, missing } = buildFeeds(this.config.input_slots, this.deps.readSlot);
    if (missing) {
      // Serve the last good vector rather than feeding the policy zeros.
      console.warn(
        `[FusedObservation] "${this.config.name}" could not read slot ${missing}; ` +
          'reusing the previous vector.',
      );
      return this.last;
    }
    for (const native of this.config.native_inputs ?? []) {
      const value = this.readNative(native);
      feeds[native.input] = { data: value, dims: [1, value.length] };
    }

    const outputs = await this.deps.session.run(feeds);
    const first = Object.values(outputs)[0];
    if (!first) {
      console.warn(`[FusedObservation] "${this.config.name}" produced no output.`);
      return this.last;
    }
    this.last = conformToSize(Float32Array.from(first.data as Float32Array), this.size);
    return this.last;
  }

  private readNative(native: FusedNativeInput): Float32Array {
    const raw =
      native.native === 'prev_action'
        ? sliceStoredActions(this.runner.getActions(native.age ?? 0), native)
        : this.readCommand(native);
    // `raw` may be the runtime's own buffer, and the graph declared a fixed width.
    return conformToSize(Float32Array.from(raw), native.size);
  }

  /** The constructor asserts the name, so only a manager-less embedding falls back. */
  private readCommand(native: FusedNativeInput): Float32Array {
    const manager = this.runner.getContext()?.commandManager;
    if (!manager || !native.command_name) return new Float32Array(native.size);
    return manager.getCommand(native.command_name);
  }
}
