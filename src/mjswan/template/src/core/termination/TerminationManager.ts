import { TerminationBase, type TerminationConfig } from './TerminationBase';
import type { TerminationConstructor } from './terminations';
import {
  FusedLane,
  FusedTermination,
  isFusedTerminationConfig,
  type FusedTerminationConfig,
} from './FusedTermination';
import { OnnxTermination, type OnnxTerminationConfig } from './OnnxTermination';
import { TimeOutTermination, type TimeOutTerminationConfig } from './TimeOutTermination';
import type { OnnxSessionCache, SlotReader } from '../onnx/session';
import type { PolicyState, TerminationConfigEntry } from '../policy/types';
import type { PolicyRunner } from '../policy/PolicyRunner';
import {
  DEFAULT_TERMINATION,
  delayOf,
  elapsed,
  mergeTermination,
  type ActivityListener,
  type TerminationOverride,
  type TerminationOverridePatch,
} from '../mdp/operator';

export type TerminationResult = {
  done: boolean;
  terminated: boolean;
  truncated: boolean;
  reasons: string[];
  /** Terms set to stop whose condition turned true this step. */
  stopped: string[];
};

/** Absent for a policy whose terminations are all plugin terms. */
export type TerminationManagerDeps = {
  onnxSessions?: OnnxSessionCache;
  readOnnxSlot?: SlotReader;
  /** mjlab's `episode_length_buf`, for a pre-format-4 `time_out`. */
  episodeLength?: () => number;
  /** The control period: a pre-format-4 `time_out`'s step limit, and the operator's delays. */
  stepDt?: number;
  onActivity?: ActivityListener;
};

/** One term as the operator sees it: its override, and what it did this episode. */
export interface TerminationTermReport extends TerminationOverride {
  name: string;
  /** mjlab's `time_out`: a truncation rather than a failure. */
  timeOut: boolean;
  /** Its condition held at the last control step. */
  active: boolean;
  /** Episode time its condition last turned true this episode. */
  firedAt: number | null;
  /** Episode time the reset it holds lands at; null unless it fired with a delay while on. */
  resetAt: number | null;
}

type TermEntry = {
  name: string;
  term: TerminationBase;
  isTimeOut: boolean;
  override: TerminationOverride;
  active: boolean;
  firedAt: number | null;
  /** When the reset this term owes started counting down; null when it owes none. */
  heldSince: number | null;
};

/** Whether an entry names a traced-ONNX termination. */
function isOnnxEntry(entry: TerminationConfigEntry): entry is OnnxTerminationConfig {
  return typeof (entry as { onnx?: unknown }).onnx === 'string';
}

/** A pre-format-4 `time_out` marker, matched on `native` being present: its text is prose. */
function isNativeTimeOutEntry(
  entry: TerminationConfigEntry,
): entry is TimeOutTerminationConfig {
  return typeof (entry as { native?: unknown }).native === 'string';
}

export class TerminationManager {
  private terms: TermEntry[] = [];
  /** Every graph, fused or not, run before any term is read. */
  private graphs: (OnnxTermination | FusedTermination)[] = [];

  constructor(
    config: Record<string, TerminationConfigEntry>,
    registry: Record<string, TerminationConstructor>,
    runner: PolicyRunner,
    private readonly deps: TerminationManagerDeps = {},
  ) {
    for (const [name, entry] of Object.entries(config)) {
      if (isFusedTerminationConfig(entry)) {
        this.addFusedGroup(entry, runner, deps);
        continue;
      }
      if (isOnnxEntry(entry)) {
        const term = this.buildOnnxTermination(name, entry, runner, deps);
        if (term) {
          this.graphs.push(term);
          this.add(name, term, entry.time_out ?? false);
        }
        continue;
      }
      if (isNativeTimeOutEntry(entry)) {
        this.add(
          name,
          new TimeOutTermination(
            runner,
            { ...entry, name },
            deps.episodeLength ?? (() => 0),
            deps.stepDt ?? 0,
          ),
          entry.time_out ?? false,
        );
        continue;
      }
      const TermClass = registry[entry.name];
      if (!TermClass) {
        // Throws like the observation and command registries: continuing would run the
        // episode without a reset condition it is configured to have.
        throw new Error(`Unknown termination type: ${entry.name}`);
      }
      const termConfig: TerminationConfig = {
        name: entry.name,
        params: entry.params,
        time_out: entry.time_out,
      };
      this.add(name, new TermClass(runner, termConfig), entry.time_out ?? false);
    }
  }

  private add(name: string, term: TerminationBase, isTimeOut: boolean): void {
    this.terms.push({
      name,
      term,
      isTimeOut,
      override: { ...DEFAULT_TERMINATION },
      active: false,
      firedAt: null,
      heldSince: null,
    });
  }

  /**
   * Expand a fused graph into one entry per lane, so the logic below need not know about
   * fusion. Skips the group on missing deps: lost reset conditions beat a dead scene.
   */
  private addFusedGroup(
    entry: FusedTerminationConfig,
    runner: PolicyRunner,
    deps: TerminationManagerDeps,
  ): void {
    const session = deps.onnxSessions?.get(entry.fused);
    const readSlot = deps.readOnnxSlot;
    if (!session || !readSlot) {
      console.warn(
        `[TerminationManager] the fused graph "${entry.fused}" needs a session and ` +
          'a slot reader; skipping every term it covers.',
      );
      return;
    }
    const group = new FusedTermination(entry, { session, readSlot });
    this.graphs.push(group);
    entry.lanes.forEach((lane, index) => {
      this.add(lane.name, new FusedLane(runner, { name: lane.name }, group, index), lane.time_out ?? false);
    });
  }

  /**
   * Build a traced-ONNX termination, or warn and skip: unlike an observation, dropping one
   * loses a reset condition rather than reshaping the policy's input vector.
   */
  private buildOnnxTermination(
    name: string,
    entry: OnnxTerminationConfig,
    runner: PolicyRunner,
    deps: TerminationManagerDeps,
  ): OnnxTermination | null {
    const session = deps.onnxSessions?.get(entry.onnx);
    const readSlot = deps.readOnnxSlot;
    if (!session || !readSlot) {
      console.warn(
        `[TerminationManager] "${name}" needs the ONNX session "${entry.onnx}" and a ` +
          'slot reader; skipping.',
      );
      return null;
    }
    return new OnnxTermination(runner, { ...entry, name }, { session, readSlot });
  }

  /** Episode time, as the graphs count it; zero without a clock. */
  private now(): number {
    return (this.deps.episodeLength?.() ?? 0) * (this.deps.stepDt ?? 0);
  }

  /**
   * OR-reduce every term to one verdict, `time_out` split out. Every graph is awaited
   * first, so a verdict is this step's, as in mjlab.
   *
   * A term switched off is still evaluated, so its firing shows, but ends nothing. One with
   * a delay holds the reset from the step it fires, whatever its condition does meanwhile.
   * One set to stop is named in `stopped` on that step, on or off, and owes its reset at
   * once; the caller pauses and lands it on play.
   */
  async evaluate(state: PolicyState): Promise<TerminationResult> {
    await Promise.all(this.graphs.map((graph) => graph.step()));
    const now = this.now();
    // Without a control period a delay cannot be timed, so the reset lands at once.
    const timed = (this.deps.stepDt ?? 0) > 0;
    let terminated = false;
    let truncated = false;
    const reasons: string[] = [];
    const stopped: string[] = [];

    for (const entry of this.terms) {
      const active = entry.term.evaluate(state);
      if (active && !entry.active) {
        entry.firedAt = now;
        this.emit(entry, 'fired', now);
        if (entry.override.onFire === 'stop') stopped.push(entry.name);
      }
      entry.active = active;
      if (active && entry.override.enabled && entry.heldSince === null) entry.heldSince = now;
      if (entry.heldSince === null) continue;
      if (timed && !elapsed(now, entry.heldSince, delayOf(entry.override.onFire))) continue;
      // Cleared once delivered, so a step that follows without a reset judges the term afresh.
      entry.heldSince = null;
      reasons.push(entry.name);
      this.emit(entry, 'reset', now);
      if (entry.isTimeOut) {
        truncated = true;
      } else {
        terminated = true;
      }
    }

    return {
      done: terminated || truncated,
      terminated,
      truncated,
      reasons,
      stopped,
    };
  }

  private emit(entry: TermEntry, what: 'fired' | 'reset', time: number): void {
    this.deps.onActivity?.({
      kind: 'termination',
      name: entry.name,
      what,
      time,
      ...(entry.override.enabled ? {} : { off: true }),
    });
  }

  reset(): void {
    for (const graph of this.graphs) graph.reset();
    for (const entry of this.terms) {
      entry.term.reset?.();
      entry.active = false;
      entry.firedAt = null;
      entry.heldSince = null;
    }
  }

  /** Change one term's override. False when no term has that name. */
  setOverride(name: string, patch: TerminationOverridePatch): boolean {
    const entry = this.terms.find((term) => term.name === name);
    if (!entry) return false;
    entry.override = mergeTermination(entry.override, patch);
    // Off cancels the reset it owed; a new delay re-times it from when it fired.
    if (!entry.override.enabled) entry.heldSince = null;
    return true;
  }

  /** Whether a term is holding a reset back, so a host's countdown needs refreshing. */
  get holding(): boolean {
    return this.terms.some((entry) => entry.heldSince !== null);
  }

  /** Every term, in config order, as the operator sees it. */
  describe(): TerminationTermReport[] {
    return this.terms.map((entry) => ({
      name: entry.name,
      timeOut: entry.isTimeOut,
      ...entry.override,
      active: entry.active,
      firedAt: entry.firedAt,
      resetAt: entry.heldSince === null ? null : entry.heldSince + delayOf(entry.override.onFire),
    }));
  }

  get size(): number {
    return this.terms.length;
  }
}
