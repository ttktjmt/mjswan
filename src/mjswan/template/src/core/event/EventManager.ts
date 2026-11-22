import type { SeededRng } from '../rng';
import type { OnnxSessionCache } from '../onnx/session';
import type { SlotReader } from '../onnx/session';
import { EventBase, type EventConfig, type EventContext } from './EventBase';
import type { EventConstructor } from './EventBase';
import { applyModelFieldDr, isModelFieldDrConfig, type ModelFieldDrConfig } from './modelFieldDr';
import { ModelFieldDefaults } from './modelWrite';
import { OnnxEvent, isOnnxEventConfig } from './OnnxEvent';
import { IntervalTrigger, ResetTrigger, StartupTrigger } from './triggers';
import {
  DEFAULT_EVENT,
  mergeEvent,
  type ActivityListener,
  type EventOverride,
  type EventOverridePatch,
} from '../mdp/operator';

/** Deps for constructing ONNX-backed event terms; omit if a scene has none. */
export interface EventManagerDeps {
  sessions: OnnxSessionCache;
  rng: SeededRng;
  readSlot?: SlotReader;
}

/** One control the panel offers for an event term. */
export interface EventControl {
  name: string;
  /** The term's own `label`, or its name when it declared none. */
  label: string;
  /** `manual` renders a button, `interval` an arm checkbox. */
  kind: 'manual' | 'interval';
  /** `interval`: its schedule is running. `manual`: its button can fire — false while
   * the term's `disabled_when` schedule owns the job. */
  armed: boolean;
}

/** The episode clock that times each write, and where activity goes. */
export interface EventManagerHooks {
  /** Episode time, in seconds. */
  now?: () => number;
  onActivity?: ActivityListener;
}

/**
 * One term the operator can switch off or trigger by hand, and when it last wrote. A manual
 * term has no switch, since nothing but a press fires it: its `enabled` stays true.
 */
export interface EventTermReport extends EventOverride {
  name: string;
  label: string;
  /** Startup terms have run before an operator could reach them, so they are not listed. */
  mode: 'reset' | 'interval' | 'manual';
  /** Episode time its write last landed this episode. */
  firedAt: number | null;
}

/** A plugin-supplied event class — reset-only, no traced graph. */
type PluginTerm = { kind: 'plugin'; name: string; term: EventBase; trigger: ResetTrigger };
type OnnxResetTerm = { kind: 'onnx'; name: string; term: OnnxEvent; trigger: ResetTrigger };
type ResetEntry = PluginTerm | OnnxResetTerm;

type TermState = {
  label: string;
  mode: EventTermReport['mode'];
  override: EventOverride;
  firedAt: number | null;
};

/**
 * Mode-aware event dispatch: `mode="reset"` terms fire on episode reset behind a
 * `ResetTrigger`, `mode="interval"` on the frames their `IntervalTrigger` allows,
 * `mode="startup"` once, and `mode="manual"` only when {@link fire} is called.
 *
 * One graph per term, unfused — the reference tasks have at most two reset terms and
 * none of the other modes, and fusing would have to reproduce the config-order write
 * semantics `onReset` gets for free.
 */
export class EventManager {
  private resetTerms: ResetEntry[] = [];
  private intervalTerms: Array<{ term: OnnxEvent; trigger: IntervalTrigger }> = [];
  private manualTerms: OnnxEvent[] = [];
  private startupTerms: Array<{ term: OnnxEvent; trigger: StartupTrigger }> = [];
  /** Legacy model-field descriptors, applied once by `startup()` (see `modelFieldDr`). */
  private modelFieldTerms: Array<{ config: ModelFieldDrConfig; trigger: StartupTrigger }> = [];
  /** The reset, interval and manual terms, in config order. */
  private readonly states = new Map<string, TermState>();

  constructor(
    configs: EventConfig[],
    registry: Record<string, EventConstructor>,
    private readonly deps?: EventManagerDeps,
    private readonly hooks: EventManagerHooks = {},
  ) {
    for (const config of configs) {
      if (isModelFieldDrConfig(config)) {
        // Perturbs `mjModel` rather than running a graph, so it needs no session.
        this.modelFieldTerms.push({ config, trigger: new StartupTrigger() });
        continue;
      }
      if (isOnnxEventConfig(config)) {
        if (!deps) {
          console.warn(
            `[EventManager] ONNX event "${config.name}" needs session/rng deps; none supplied.`,
          );
          continue;
        }
        const session = deps.sessions.get(config.onnx);
        if (!session) {
          console.warn(`[EventManager] No ONNX session loaded for "${config.onnx}".`);
          continue;
        }
        const term = new OnnxEvent(config, { session, rng: deps.rng, readSlot: deps.readSlot });
        if (config.mode !== 'startup') {
          const mode = config.mode === 'manual' || config.mode === 'interval' ? config.mode : 'reset';
          this.track(config.name, config.label, mode);
        }
        switch (config.mode) {
          case 'manual':
            this.manualTerms.push(term);
            break;
          case 'startup':
            this.startupTerms.push({ term, trigger: new StartupTrigger() });
            break;
          case 'interval':
            this.intervalTerms.push({
              term,
              trigger: new IntervalTrigger(
                {
                  intervalRangeS: config.interval_range_s ?? [1.0, 1.0],
                  isGlobalTime: config.is_global_time,
                },
                deps.rng,
              ),
            });
            break;
          case 'reset':
          default:
            this.resetTerms.push({
              kind: 'onnx',
              name: config.name,
              term,
              trigger: new ResetTrigger({
                minStepCountBetweenReset: config.min_step_count_between_reset,
              }),
            });
            break;
        }
        continue;
      }
      if (config.native) {
        // The build declined to trace this term and said why; not a missing plugin.
        continue;
      }
      const EventClass = registry[config.name];
      if (!EventClass) {
        console.warn(`[EventManager] Unknown event type: ${config.name}`);
        continue;
      }
      this.track(config.name, config.label, 'reset');
      this.resetTerms.push({
        kind: 'plugin',
        name: config.name,
        term: new EventClass(config),
        trigger: new ResetTrigger(),
      });
    }
  }

  private track(name: string, label: string | undefined, mode: TermState['mode']): void {
    this.states.set(name, { label: label ?? name, mode, override: { ...DEFAULT_EVENT }, firedAt: null });
  }

  private now(): number {
    return this.hooks.now?.() ?? 0;
  }

  /** Note a write that landed, for the operator. */
  private wrote(name: string, byHand = false): void {
    const state = this.states.get(name);
    if (state) state.firedAt = this.now();
    this.hooks.onActivity?.({ kind: 'event', name, what: 'fired', time: this.now(), ...(byHand ? { byHand } : {}) });
  }

  private isEnabled(name: string): boolean {
    return this.states.get(name)?.override.enabled ?? true;
  }

  /**
   * Fire every `mode="startup"` term once, in config order and after the legacy
   * model-field descriptors.
   */
  async startup(context: EventContext): Promise<void> {
    this.applyModelFieldTerms(context);
    for (const { term, trigger } of this.startupTerms) {
      if (trigger.take()) await term.fire(context);
    }
  }

  /** Once, before the startup terms: `add`/`scale` are relative to the compiled default. */
  private applyModelFieldTerms(context: EventContext): void {
    if (this.modelFieldTerms.length === 0) return;
    const { mujoco, mjModel, mjData } = context;
    if (!mujoco || !mjModel || !mjData) {
      console.warn('[EventManager] model-field randomization needs a live model; skipping.');
      return;
    }
    if (!this.deps?.rng) {
      console.warn('[EventManager] model-field randomization needs the seeded rng; skipping.');
      return;
    }
    // One `defaults` per model, so events on one field share a base across passes.
    const base = context.modelDefaults ?? new ModelFieldDefaults(mjModel);
    for (const { config, trigger } of this.modelFieldTerms) {
      if (!trigger.take()) continue;
      applyModelFieldDr(mujoco, mjModel, mjData, config, this.deps.rng, base);
    }
  }

  /**
   * Advance one control step: fire the `mode="interval"` terms whose timer elapsed **in
   * config order**, and advance every reset-gate counter. Sequential, so overlaps
   * resolve by config order rather than by ORT completion; the counters go first, since
   * a failing graph must not stall them.
   */
  async tick(dt: number, context: EventContext): Promise<void> {
    for (const { trigger } of this.resetTerms) trigger.step();
    for (const { term, trigger } of this.intervalTerms) {
      if (!trigger.tick(dt)) continue;
      if (await term.fire(context)) this.wrote(term.name);
    }
  }

  /** True while a manual term's `disabled_when` interval schedule is armed. */
  private isGated(manual: OnnxEvent): boolean {
    const gate = manual.config.disabled_when;
    if (gate === undefined) return false;
    return this.intervalTerms.find(({ term }) => term.name === gate)?.trigger.isArmed ?? false;
  }

  /**
   * Fire one `mode="manual"` term by name, unless its `disabled_when` schedule is armed.
   * `OnnxEvent.fire` drops a press that lands while its own graph is still running.
   */
  async fire(name: string, context: EventContext): Promise<void> {
    const term = this.manualTerms.find(entry => entry.name === name);
    if (!term) {
      console.warn(`[EventManager] No mode="manual" event named "${name}".`);
      return;
    }
    if (this.isGated(term)) {
      // The panel greys the button out, so only an API caller gets here; say why.
      console.warn(
        `[EventManager] "${name}" is disabled while "${term.config.disabled_when}" is armed.`,
      );
      return;
    }
    // A press the graph is still busy with is dropped, as it always was, and shows as nothing.
    if (await term.fire(context)) this.wrote(name, true);
  }

  /**
   * Run one listed term's write now, whatever its mode and its switch: a reset term
   * re-randomizes mid-episode, an interval term fires off its schedule, a manual term as
   * {@link fire}. Its gate and its schedule are left as they were.
   */
  async trigger(name: string, context: EventContext): Promise<void> {
    const mode = this.states.get(name)?.mode;
    if (mode === 'manual') return this.fire(name, context);
    let wrote = false;
    if (mode === 'interval') {
      const entry = this.intervalTerms.find(({ term }) => term.name === name);
      wrote = (await entry?.term.fire(context)) ?? false;
    } else if (mode === 'reset') {
      const entry = this.resetTerms.find((reset) => reset.name === name);
      if (entry?.kind === 'plugin') {
        entry.term.onReset(context);
        wrote = true;
      } else {
        wrote = (await entry?.term.fire(context)) ?? false;
      }
    } else {
      console.warn(`[EventManager] No reset, interval or manual event named "${name}".`);
    }
    if (wrote) this.wrote(name, true);
  }

  /** Start or stop one `mode="interval"` term's schedule. False if there is no such term. */
  setArmed(name: string, armed: boolean): boolean {
    if (this.states.get(name)?.mode !== 'interval') return false;
    return this.setOverride(name, { enabled: armed });
  }

  /**
   * Switch a reset or interval term on or off; for an interval term it is the schedule's
   * arm switch. False for a manual term, which has none, and for a name not listed.
   */
  setOverride(name: string, patch: EventOverridePatch): boolean {
    const state = this.states.get(name);
    if (!state || state.mode === 'manual') return false;
    state.override = mergeEvent(state.override, patch);
    if (state.mode === 'interval') {
      this.intervalTerms.find(({ term }) => term.name === name)?.trigger.setArmed(state.override.enabled);
    }
    return true;
  }

  /** The reset, interval and manual terms, in config order, as the operator sees them. */
  describe(): EventTermReport[] {
    return [...this.states].map(([name, state]) => ({
      name,
      label: state.label,
      mode: state.mode,
      ...state.override,
      firedAt: state.firedAt,
    }));
  }

  controls(): EventControl[] {
    return [
      ...this.manualTerms.map(term => ({
        name: term.name,
        label: term.config.label ?? term.name,
        kind: 'manual' as const,
        armed: !this.isGated(term),
      })),
      ...this.intervalTerms.map(({ term, trigger }) => ({
        name: term.name,
        label: term.config.label ?? term.name,
        kind: 'interval' as const,
        armed: trigger.isArmed,
      })),
    ];
  }

  /**
   * Fire every `mode="reset"` term whose gate allows it, **in config order**. Sequential
   * rather than `Promise.all`: every write is an assignment over `default_*`, so the
   * order decides which value survives an overlap.
   *
   * A term switched off leaves its gate as it was: it did not fire.
   */
  async onReset(context: EventContext): Promise<void> {
    for (const state of this.states.values()) state.firedAt = null;
    for (const entry of this.resetTerms) {
      if (!this.isEnabled(entry.name)) continue;
      if (!entry.trigger.take()) continue;
      if (entry.kind === 'plugin') entry.term.onReset(context);
      else if (!(await entry.term.fire(context))) continue;
      this.wrote(entry.name);
    }
    for (const { trigger } of this.intervalTerms) trigger.onReset();
  }

  get size(): number {
    return (
      this.resetTerms.length +
      this.intervalTerms.length +
      this.manualTerms.length +
      this.startupTerms.length +
      // Counted: a task whose only events are DR would otherwise report "0 loaded".
      this.modelFieldTerms.length
    );
  }
}
