import { describe, expect, it } from 'vitest';

import { ControlStep } from '../controlStep';

/**
 * A control step of `decimation` physics steps, logging every hook into `calls`; the physics
 * reports a divergence on any slice that reaches `divergeAt`.
 */
function recording(decimation: number, divergeAt = Infinity): { step: ControlStep; calls: string[] } {
  const calls: string[] = [];
  const step = new ControlStep({
    decimation: () => decimation,
    begin: async () => {
      calls.push('begin');
    },
    physics: (from, to) => {
      calls.push(`physics ${from}-${to}`);
      return to > divergeAt;
    },
    end: async (diverged) => {
      calls.push(diverged ? 'end diverged' : 'end');
    },
    snapshot: () => {
      calls.push('snapshot');
    },
  });
  return { step, calls };
}

const NO_DEADLINE = Infinity;

describe('ControlStep', () => {
  it('runs a control step as begin, its physics steps, then end', async () => {
    const { step, calls } = recording(4);
    expect(await step.spend(20, 5, NO_DEADLINE)).toBe(0);
    expect(calls).toEqual(['begin', 'physics 0-4', 'end']);
  });

  it('spends what has accrued and resumes the control step on the next call', async () => {
    const { step, calls } = recording(20);
    expect(await step.spend(17, 5, NO_DEADLINE)).toBe(2);
    expect(await step.spend(7, 5, NO_DEADLINE)).toBe(2);
    expect(calls).toEqual(['begin', 'physics 0-3', 'snapshot', 'physics 3-4', 'snapshot']);
  });

  it('runs into the next control step when the budget reaches it', async () => {
    const { step, calls } = recording(4);
    expect(await step.spend(30, 5, NO_DEADLINE)).toBe(0);
    expect(calls).toEqual(['begin', 'physics 0-4', 'end', 'begin', 'physics 0-2', 'snapshot']);
  });

  it('leaves less than a step of budget for later', async () => {
    const { step, calls } = recording(4);
    expect(await step.spend(4.5, 5, NO_DEADLINE)).toBe(4.5);
    expect(calls).toEqual([]);
  });

  it('starts a fresh control step after a restart', async () => {
    const { step, calls } = recording(20);
    await step.spend(10, 5, NO_DEADLINE);
    step.restart();
    await step.spend(5, 5, NO_DEADLINE);
    expect(calls).toEqual(['begin', 'physics 0-2', 'snapshot', 'begin', 'physics 0-1', 'snapshot']);
  });

  it('ends a control step early when the physics resets itself', async () => {
    const { step, calls } = recording(20, 2);
    expect(await step.spend(25, 5, NO_DEADLINE)).toBe(0);
    expect(calls).toEqual(['begin', 'physics 0-5', 'end diverged']);
  });

  it('keeps one step of budget once past the deadline', async () => {
    const { step, calls } = recording(4);
    expect(await step.spend(1000, 5, 0, () => 1)).toBe(5);
    expect(calls).toEqual(['begin', 'physics 0-4', 'end']);
  });
});
