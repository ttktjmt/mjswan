/**
 * The operator's overrides of termination terms: a term turned off still shows its firing
 * but ends nothing; what follows a firing is one of the reset at once, the reset a delay
 * later (held from the step the term fires, whatever its condition does meanwhile), or a
 * stop, which names the term on that step so the runtime pauses there.
 */
import { describe, expect, it } from 'vitest';

import { TerminationBase } from '../TerminationBase';
import { TerminationManager } from '../TerminationManager';
import type { TermActivity } from '../../mdp/operator';
import type { PolicyRunner } from '../../policy/PolicyRunner';
import type { PolicyState } from '../../policy/types';

const runner = {} as unknown as PolicyRunner;
const state = {} as PolicyState;
const STEP_DT = 0.02;

/** Two plugin terms whose verdicts the test sets, on a control-step clock it advances. */
function rig() {
  const verdicts: Record<string, boolean> = { fell: false, time_out: false };
  class Scripted extends TerminationBase {
    evaluate(): boolean {
      return verdicts[this.config.name];
    }
  }
  let steps = 0;
  const activity: TermActivity[] = [];
  const manager = new TerminationManager(
    { fell: { name: 'fell' }, time_out: { name: 'time_out', time_out: true } },
    { fell: Scripted, time_out: Scripted },
    runner,
    { episodeLength: () => steps, stepDt: STEP_DT, onActivity: (a) => activity.push(a) },
  );
  const step = () => {
    steps += 1;
    return manager.evaluate(state);
  };
  const reset = () => {
    steps = 0;
    manager.reset();
  };
  return { verdicts, manager, step, reset, activity };
}

const termNamed = (manager: TerminationManager, name: string) =>
  manager.describe().find((term) => term.name === name)!;

describe('TerminationManager overrides', () => {
  it('by default ends the episode on the step a term fires, as mjlab does', async () => {
    const { verdicts, manager, step } = rig();
    expect((await step()).done).toBe(false);
    verdicts.fell = true;
    const result = await step();
    expect(result).toMatchObject({ done: true, terminated: true, truncated: false, reasons: ['fell'] });
    expect(termNamed(manager, 'fell')).toMatchObject({ enabled: true, onFire: 'reset', active: true });
  });

  it('at no delay, judges a step without a reset after it on that step alone', async () => {
    const { verdicts, step } = rig();
    verdicts.fell = true;
    expect((await step()).done).toBe(true);
    verdicts.fell = false;
    expect((await step()).done).toBe(false);
  });

  it('evaluates a term that is off and reports it, but never ends the episode', async () => {
    const { verdicts, manager, step, activity } = rig();
    expect(manager.setOverride('fell', { enabled: false })).toBe(true);
    verdicts.fell = true;
    for (let i = 0; i < 10; i++) expect((await step()).done).toBe(false);
    expect(termNamed(manager, 'fell')).toMatchObject({ active: true, firedAt: STEP_DT, resetAt: null });
    expect(activity).toEqual([{ kind: 'termination', name: 'fell', what: 'fired', time: STEP_DT, off: true }]);
  });

  it('holds the reset for its delay from the firing, even once the condition clears', async () => {
    const { verdicts, manager, step, activity } = rig();
    manager.setOverride('fell', { onFire: { delay: 0.1 } });
    await step();
    verdicts.fell = true;
    expect((await step()).done).toBe(false);
    expect(termNamed(manager, 'fell').resetAt).toBeCloseTo(0.04 + 0.1);
    verdicts.fell = false;
    expect(manager.holding).toBe(true);
    for (let i = 0; i < 4; i++) expect((await step()).done).toBe(false);
    const result = await step();
    expect(result).toMatchObject({ done: true, terminated: true, reasons: ['fell'] });
    expect(activity.map(({ what, time }) => [what, +time.toFixed(2)])).toEqual([
      ['fired', 0.04],
      ['reset', 0.14],
    ]);
  });

  it('keeps a truncation a truncation when it is delayed', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('time_out', { onFire: { delay: 0.04 } });
    verdicts.time_out = true;
    expect((await step()).done).toBe(false);
    await step();
    expect(await step()).toMatchObject({ done: true, terminated: false, truncated: true });
  });

  it('turning a held term off cancels its reset; back on while it holds starts a new wait', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('fell', { onFire: { delay: 0.04 } });
    verdicts.fell = true;
    await step();
    manager.setOverride('fell', { enabled: false });
    expect(manager.holding).toBe(false);
    for (let i = 0; i < 5; i++) expect((await step()).done).toBe(false);
    manager.setOverride('fell', { enabled: true });
    expect((await step()).done).toBe(false);
    expect((await step()).done).toBe(false);
    expect((await step()).done).toBe(true);
  });

  it('re-times a held reset from the firing when the delay changes', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('fell', { onFire: { delay: 1 } });
    verdicts.fell = true;
    await step();
    await step();
    await step();
    manager.setOverride('fell', { onFire: { delay: 0.02 } });
    expect((await step()).done).toBe(true);
  });

  it('forgets what a term did when the episode resets, and keeps its override', async () => {
    const { verdicts, manager, step, reset } = rig();
    manager.setOverride('fell', { onFire: { delay: 0.5 } });
    verdicts.fell = true;
    await step();
    reset();
    expect(termNamed(manager, 'fell')).toMatchObject({
      enabled: true,
      onFire: { delay: 0.5 },
      active: false,
      firedAt: null,
      resetAt: null,
    });
    expect(manager.holding).toBe(false);
  });

  it('refuses a name it does not have, keeps a delay that is not a number, and clamps the rest', async () => {
    const { manager } = rig();
    expect(manager.setOverride('nope', { enabled: false })).toBe(false);
    manager.setOverride('fell', { onFire: { delay: 0.3 } });
    manager.setOverride('fell', { onFire: { delay: Number.NaN } });
    expect(termNamed(manager, 'fell').onFire).toEqual({ delay: 0.3 });
    manager.setOverride('fell', { onFire: { delay: -1 } });
    expect(termNamed(manager, 'fell').onFire).toBe('reset');
    manager.setOverride('fell', { onFire: { delay: 1e6 } });
    expect(termNamed(manager, 'fell').onFire).toEqual({ delay: 60 });
  });

  it('names a term set to freeze on the step it fires, on or off, timed or not, and only on that step', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('fell', { onFire: 'freeze' });
    verdicts.fell = true;
    expect(await step()).toMatchObject({ done: true, frozen: ['fell'] });
    expect((await step()).frozen).toEqual([]);
    manager.setOverride('time_out', { onFire: { freeze: 3 }, enabled: false });
    verdicts.time_out = true;
    expect(await step()).toMatchObject({ frozen: ['time_out'], truncated: false });
    expect(termNamed(manager, 'time_out')).toMatchObject({ enabled: false, onFire: { freeze: 3 }, active: true });
  });

  it('holds a delay or a freeze, never both: setting one replaces the other', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('fell', { onFire: { delay: 1 } });
    manager.setOverride('fell', { onFire: { freeze: 2 } });
    expect(termNamed(manager, 'fell').onFire).toEqual({ freeze: 2 });
    verdicts.fell = true;
    // Frozen, it owes its reset at once: the runtime lands it when the freeze ends.
    expect(await step()).toMatchObject({ done: true, frozen: ['fell'] });
    manager.reset();
    verdicts.fell = false;
    manager.setOverride('fell', { onFire: { delay: 0.04 } });
    expect(termNamed(manager, 'fell').onFire).toEqual({ delay: 0.04 });
    verdicts.fell = true;
    expect(await step()).toMatchObject({ done: false, frozen: [] });
  });

  it('keeps a freeze timed in seconds, clamped as a delay is, and a zero one as a reset', () => {
    const { manager } = rig();
    manager.setOverride('fell', { onFire: { freeze: 1e6 } });
    expect(termNamed(manager, 'fell').onFire).toEqual({ freeze: 60 });
    manager.setOverride('fell', { onFire: { freeze: 0 } });
    expect(termNamed(manager, 'fell').onFire).toBe('reset');
    manager.setOverride('fell', { onFire: { delay: 1, freeze: 1 } as never });
    expect(termNamed(manager, 'fell').onFire).toBe('reset');
  });

  it('lands a held reset on the next step when switched to freeze or reset: a freeze pauses only on a firing', async () => {
    const { verdicts, manager, step } = rig();
    manager.setOverride('fell', { onFire: { delay: 1 } });
    verdicts.fell = true;
    await step();
    manager.setOverride('fell', { onFire: 'freeze' });
    expect(await step()).toMatchObject({ done: true, reasons: ['fell'], frozen: [] });
  });

  it('cannot time a delay without a control period, so it does not hold one', async () => {
    let fell = false;
    class Scripted extends TerminationBase {
      evaluate(): boolean {
        return fell;
      }
    }
    const manager = new TerminationManager({ fell: { name: 'fell' } }, { fell: Scripted }, runner);
    manager.setOverride('fell', { onFire: { delay: 5 } });
    fell = true;
    expect((await manager.evaluate(state)).done).toBe(true);
  });
});
