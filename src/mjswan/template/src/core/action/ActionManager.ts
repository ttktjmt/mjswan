/**
 * The action terms of one policy, run as mjlab's `ActionManager` runs them: `process`
 * once per control step, `apply` once per substep, `reset` at an episode's start.
 *
 * A native term processes in `processAction`; a traced one in its `OnnxActionProcess`,
 * which this awaits. Either way `applyAction` takes it from there.
 */

import { inHold, processAction, type ProcessContext, type ResolvedActionTerm } from './applyAction';
import type { OnnxActionProcess } from './OnnxActionProcess';
import type { ActuatorSet } from '../actuator/ActuatorSet';

/** One term and, for a traced one, the graph that processes it. */
export interface ManagedActionTerm {
  name: string;
  term: ResolvedActionTerm;
  traced?: OnnxActionProcess | null;
}

export class ActionManager {
  readonly terms: ResolvedActionTerm[];
  private readonly byName: Map<string, ManagedActionTerm>;
  /** Each traced term's last logged failure: one that repeats every step is logged once. */
  private readonly lastError = new Map<string, string>();

  constructor(
    private readonly entries: readonly ManagedActionTerm[],
    /** The policy's traced actuator laws, which `applyAction` runs after the terms. */
    readonly actuators: ActuatorSet | null = null,
  ) {
    this.terms = entries.map((entry) => entry.term);
    this.byName = new Map(entries.map((entry) => [entry.name, entry]));
  }

  /**
   * Once per control step, after the policy: every term's `processed`, in config order.
   *
   * Without a `fresh` action (the policy produced none this step), a traced term keeps its
   * targets rather than integrate the held action again; a native term recomputes the same
   * targets from it, and a command term goes on following its command. A traced term that
   * fails keeps its targets too.
   */
  async process(
    actions: Float32Array,
    context: ProcessContext,
    { fresh = true }: { fresh?: boolean } = {},
  ): Promise<void> {
    for (const { name, term, traced } of this.entries) {
      if (!traced) {
        processAction(term, actions, context);
        continue;
      }
      term.holding = inHold(term.holdS, context);
      // Kept targets hold flat rather than replay the last step's ramp; a run that
      // succeeds sets the ramp's start itself.
      term.rampFrom?.set(term.processed);
      if (!fresh) continue;
      const raw = new Float32Array(term.actionIndices.length);
      for (let i = 0; i < raw.length; i++) raw[i] = actions[term.actionIndices[i]] ?? 0;
      try {
        if (await traced.process(raw, term.processed, term.rampFrom)) {
          term.active = true;
          this.lastError.delete(name);
        }
      } catch (error) {
        const message = String(error);
        if (this.lastError.get(name) !== message) {
          this.lastError.set(name, message);
          console.error(`[ActionManager] Action term "${name}" failed; its targets hold:`, error);
        }
      }
    }
  }

  /** mjlab's `action_manager.reset`: each traced term's own `reset`, in config order. */
  async reset(): Promise<void> {
    for (const { traced } of this.entries) await traced?.reset();
  }

  /**
   * mjlab's actuator reset, which comes with the sim state's: the command delays start
   * over, and every actuator target is zero again.
   */
  resetActuators(): void {
    for (const term of this.terms) for (const delay of term.delays) delay.reset();
    this.actuators?.reset();
  }

  /**
   * What an observation reads off a term (`get_term(name).<field>`): a traced term's
   * state field, else any term's processed action.
   */
  getStateField(name: string, field: string): Float32Array | null {
    const entry = this.byName.get(name);
    if (!entry) return null;
    const state = entry.traced?.getStateField(field);
    if (state) return state;
    return field === '_processed_actions' ? Float32Array.from(entry.term.processed) : null;
  }
}
