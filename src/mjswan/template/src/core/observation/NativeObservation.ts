/**
 * Observation terms that are a plain read of state the orchestrator already owns, so
 * tracing them would only wrap an identity graph around a value in hand. The build
 * marks them `native`:
 * - `prev_action` — mjlab's `last_action`, or an older entry of its action window
 *   when `age` is set (mjlab's `prev_action` / `prev_prev_action`).
 * - `command` — mjlab's `generated_commands`, the named term's current value.
 * - `constant` — reads nothing from the env; its value is baked at build time.
 *
 * `size` normally comes from the build, but `prev_action`/`command` can resolve it from
 * the runtime, since a browser-only command has no build-time width.
 */

import { ObservationBase, type ObservationConfig } from './ObservationBase';
import {
  applyObservationPipeline,
  conformToSize,
  type ObservationClip,
  type ObservationScale,
} from './pipeline';
import type { PolicyRunner } from '../policy/PolicyRunner';
import type { PolicyState } from '../policy/types';

export type NativeObservationKind = 'prev_action' | 'command' | 'constant';

export interface NativeObservationConfig extends ObservationConfig {
  native: NativeObservationKind;
  size?: number;
  /** `constant` only: the baked value. */
  value?: number[];
  /** `command` only: which command term to read. */
  command_name?: string;
  /** `prev_action` only: which action term, when the term names one. */
  action_name?: string;
  /** `prev_action` with an `action_name`: where that term's slice starts. */
  action_offset?: number;
  /** `prev_action` only: control steps back, 0 (the default) being the newest. */
  age?: number;
  scale?: ObservationScale;
  clip?: ObservationClip;
}

/** Whether a config entry names a natively-computed observation. */
export function isNativeObservationConfig(
  entry: ObservationConfig,
): entry is NativeObservationConfig {
  const native = (entry as { native?: unknown }).native;
  return native === 'prev_action' || native === 'command' || native === 'constant';
}

/**
 * The stored actions at one term's `action_offset` — mjlab's `get_term(name).raw_action`.
 * An entry with no offset is the bare `last_action`, which is the whole vector.
 */
export function sliceStoredActions(
  actions: Float32Array,
  config: { action_offset?: number; size?: number },
): Float32Array {
  const offset = config.action_offset;
  if (offset === undefined) return actions;
  // A view: both callers copy before the pipeline mutates.
  return actions.subarray(offset, offset + (config.size ?? actions.length - offset));
}

/**
 * Fail at construction if a `command` term names something no command provides, as mjlab
 * asserts the same lookup — otherwise the miss is a silent block of zeros in the policy's
 * input vector, and a slot meant to be empty says so with `native: "constant"`. Shared
 * with `FusedObservation`, and skipped when there is no manager at all.
 */
export function assertCommandTermBound(
  runner: PolicyRunner,
  label: string,
  commandName: string | undefined,
): void {
  if (!commandName) {
    throw new Error(
      `Observation term "${label}" is native:"command" but carries no command_name, ` +
        'so there is nothing to read. The build always emits one (mjlab takes it as a ' +
        'required param), which makes this a malformed config rather than a scene ' +
        'without commands.',
    );
  }
  const manager = runner.getContext()?.commandManager;
  if (!manager) return;
  const available = manager.termNames();
  if (available.includes(commandName)) return;
  throw new Error(
    `Observation term "${label}" reads the command "${commandName}", which this scene ` +
      `does not define. Available: ${available.length ? available.join(', ') : '(none)'}. ` +
      'Unchecked, this feeds the policy a zero block of the declared width — a ' +
      'silently wrong input vector — so the scene fails to load instead. A command ' +
      'slot the scene deliberately does not drive belongs as native:"constant".',
  );
}

export class NativeObservation extends ObservationBase<NativeObservationConfig> {
  private readonly constant: Float32Array | null;
  private cachedSize: number | null;

  constructor(runner: PolicyRunner, config: NativeObservationConfig) {
    super(runner, config);
    if (config.native === 'command') {
      assertCommandTermBound(runner, config.name, config.command_name);
    }
    this.constant =
      config.native === 'constant' ? Float32Array.from(config.value ?? []) : null;
    this.cachedSize = config.size ?? this.constant?.length ?? null;
  }

  get size(): number {
    if (this.cachedSize !== null) return this.cachedSize;
    // A browser-only command: take the width from the live value, once.
    this.cachedSize = this.read().length;
    return this.cachedSize;
  }

  compute(_state: PolicyState): Float32Array {
    const raw = this.read();
    // Copy first: the pipeline mutates, and `read()` may hand back a live buffer.
    const values = conformToSize(Float32Array.from(raw), this.size);
    return applyObservationPipeline(values, this.config);
  }

  private read(): Float32Array {
    switch (this.config.native) {
      case 'constant':
        return this.constant ?? new Float32Array(0);
      case 'prev_action':
        return sliceStoredActions(
          this.runner.getActions(this.config.age ?? 0),
          this.config,
        );
      case 'command': {
        const name = this.config.command_name;
        const manager = this.runner.getContext()?.commandManager;
        // The constructor asserts the name, so only a manager-less embedding lands here.
        if (!name || !manager) return new Float32Array(this.cachedSize ?? 0);
        return manager.getCommand(name);
      }
    }
  }
}
