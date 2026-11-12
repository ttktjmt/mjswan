/**
 * mjlab's viewer budget (`viewer/base.py`), spent a physics step at a time rather than a
 * control step at a time, so the renderer gets the poses between two policy steps too.
 */

export interface ControlStepHooks {
  decimation(): number;
  /** Observation, inference and the action's targets, before the first physics step. */
  begin(): Promise<void>;
  /** Physics steps `from` up to `to`; true when `mj_step` reset the data, ending the step. */
  physics(from: number, to: number): boolean;
  /** Terminations, the one forward, commands and events, after the last physics step. */
  end(diverged: boolean): Promise<void>;
  /** The poses the renderer draws, taken in the middle of a control step. */
  snapshot(): void;
}

export class ControlStep {
  /** Physics steps of the current control step already taken. */
  private substep = 0;

  constructor(private readonly hooks: ControlStepHooks) {}

  /** Abandon the control step under way: the next starts with `begin`. */
  restart(): void {
    this.substep = 0;
  }

  /**
   * Spend `budget` ms of real time on physics steps of `stepMs` each and return what is left.
   * An unfinished control step resumes on the next call, after a pause too. Past `deadline`
   * it stops and keeps one step of budget, as mjlab's viewer drops a backlog it cannot run
   * within a frame.
   */
  async spend(
    budget: number,
    stepMs: number,
    deadline: number,
    now: () => number = () => performance.now(),
  ): Promise<number> {
    let midStep = false;
    while (budget >= stepMs) {
      if (this.substep === 0) await this.hooks.begin();
      const from = this.substep;
      const decimation = this.hooks.decimation();
      const to = Math.min(decimation, from + Math.floor(budget / stepMs));
      const diverged = this.hooks.physics(from, to);
      budget -= (to - from) * stepMs;
      midStep = !diverged && to < decimation;
      this.substep = midStep ? to : 0;
      if (!midStep) await this.hooks.end(diverged);
      if (now() > deadline) {
        budget = Math.min(budget, stepMs);
        break;
      }
    }
    if (midStep) this.hooks.snapshot();
    return budget;
  }
}
