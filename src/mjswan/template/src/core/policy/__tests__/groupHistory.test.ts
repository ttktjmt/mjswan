/**
 * `PolicyRunner`'s group-level frame stack: the `{components, history_steps,
 * interleaved}` config shape a hand-authored `policy.json` can declare.
 *
 * Both layouts are the same length, so getting `interleaved` — or the direction the
 * stack grows in — wrong reaches the policy as reordered numbers rather than as a size
 * error, hence a test per layout and one pinning the size.
 */
import { describe, expect, it } from 'vitest';

import { ObservationBase } from '../../observation/ObservationBase';
import { PolicyRunner } from '../PolicyRunner';
import type { PolicyRunnerContext, PolicyState } from '../types';

const STATE = {} as PolicyState;
const CONTEXT = {} as PolicyRunnerContext;

/** A term whose value is `[n, n * 10]` on its n-th frame, so frames are identifiable. */
class Counter extends ObservationBase {
  private frames = 0;

  get size(): number {
    return 2;
  }

  compute(): Float32Array {
    this.frames += 1;
    return new Float32Array([this.frames, this.frames * 10]);
  }
}

async function runner(interleaved: boolean): Promise<PolicyRunner> {
  const instance = new PolicyRunner(
    {
      observations: {
        policy: {
          history_steps: 2,
          interleaved,
          components: [{ name: 'counter' }],
        },
      },
    },
    { observations: { counter: Counter } },
  );
  await instance.init(CONTEXT);
  instance.reset(STATE);
  return instance;
}

async function collect(instance: PolicyRunner): Promise<number[]> {
  const outputs = await instance.collectObservationsByKey(STATE);
  return Array.from(outputs.policy);
}

describe('PolicyRunner group history', () => {
  it('stacks frame-major, oldest first, as a per-term stack does', async () => {
    const instance = await runner(false);
    // Primed: every slot holds frame 1.
    expect(await collect(instance)).toEqual([1, 10, 1, 10]);
    expect(await collect(instance)).toEqual([1, 10, 2, 20]);
  });

  it('lays the stack out element-major when interleaved', async () => {
    const instance = await runner(true);
    expect(await collect(instance)).toEqual([1, 1, 10, 10]);
    // [e0_t-1, e0_t, e1_t-1, e1_t] rather than [e0_t-1, e1_t-1, e0_t, e1_t].
    expect(await collect(instance)).toEqual([1, 2, 10, 20]);
  });

  it('keeps the group size the same either way', async () => {
    for (const interleaved of [false, true]) {
      const instance = await runner(interleaved);
      expect((await collect(instance)).length).toBe(4);
    }
  });
});
