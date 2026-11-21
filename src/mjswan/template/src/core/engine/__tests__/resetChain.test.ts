/**
 * The reset chain's ordering (mjlab's `_reset_idx`).
 *
 * `runtime.ts` needs the MuJoCo WASM module to instantiate, so the order it used to
 * express inline — `void eventManager.onReset(…)`, then `commandManager.resetTerms()`,
 * with `policyRunner.reset` trailing both — had no test and no way to get one. These
 * pin the extracted sequence: the ordering cases fail the moment the await is dropped
 * or the policy reset drifts back past the commands.
 */
import { describe, expect, it } from 'vitest';

import { applyResetTerms, type ResetChain } from '../resetChain';
import type { EventContext } from '../../event/EventBase';
import type { PolicyState } from '../../policy/types';

/** The chain only forwards this, so an empty scene context is enough. */
const CONTEXT = {
  mujoco: null,
  mjModel: null,
  mjData: null,
  terrainData: null,
} as unknown as EventContext;

/** Yield past the microtask queue, the way an awaited ORT run does. */
function settle(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/** A chain with every sink recording into `order`, overridable per test. */
function chainRecording(order: string[], overrides: Partial<ResetChain> = {}): ResetChain {
  return {
    events: {
      onReset: async () => {
        await settle();
        order.push('event');
      },
    },
    policy: { reset: () => order.push('policy') },
    commands: { resetTerms: () => { order.push('command'); } },
    context: CONTEXT,
    ...overrides,
  };
}

describe('applyResetTerms', () => {
  it('runs event -> policy -> command, mjlab\'s _reset_idx order', async () => {
    const order: string[] = [];
    await applyResetTerms(chainRecording(order));
    // Not ['command', 'event', 'policy'], which un-awaited events would give.
    expect(order).toEqual(['event', 'policy', 'command']);
  });

  it('resets the action terms between the policy and the commands, and waits for them', async () => {
    const order: string[] = [];
    await applyResetTerms(
      chainRecording(order, {
        actions: {
          reset: async () => {
            await settle();
            order.push('action');
          },
        },
      }),
    );
    expect(order).toEqual(['event', 'policy', 'action', 'command']);
  });

  it('resets the command terms only after the reset events have landed', async () => {
    const order: string[] = [];
    await applyResetTerms(chainRecording(order, { policy: null }));
    expect(order).toEqual(['event', 'command']);
  });

  it('lets the command term win where both write the same element', async () => {
    // A reset event writes `qpos` async while `TrackingCommand.reset` writes it sync;
    // mjlab assigns in `_reset_idx` order, so the command's value survives.
    const qpos = [0];
    await applyResetTerms({
      events: {
        onReset: async () => {
          await settle();
          qpos[0] = 1;
        },
      },
      commands: { resetTerms: () => { qpos[0] = 2; } },
      context: CONTEXT,
    });
    // Settle after the call too, or an un-awaited write lands here and the test passes.
    await settle();

    // Un-awaited leaves 1: the event overwrites the command, mjlab's order backwards.
    expect(qpos[0]).toBe(2);
  });

  it('builds the policy state after the events, not before', async () => {
    // Why `buildState` is a thunk: mjlab primes the observation history from the state
    // the reset events wrote, not the one before them.
    const qpos = [0];
    let seen: number | null = null;
    await applyResetTerms({
      events: {
        onReset: async () => {
          await settle();
          qpos[0] = 7;
        },
      },
      policy: { reset: state => { seen = (state?.jointPos?.[0] as number) ?? null; } },
      commands: { resetTerms: () => {} },
      context: CONTEXT,
      buildState: () => ({ jointPos: Float32Array.from(qpos) }) as unknown as PolicyState,
    });
    expect(seen).toBe(7);
  });

  it('waits for an async command reset before returning to the caller', async () => {
    // A traced term's reset *is* its resample — an `ort.run()` that may write to the
    // sim — and the caller's `mj_forward` has to publish those writes, not precede them.
    const order: string[] = [];
    await applyResetTerms({
      events: null,
      policy: null,
      commands: {
        resetTerms: async () => {
          await settle();
          order.push('resample');
        },
      },
      context: CONTEXT,
    });
    order.push('forward');
    expect(order).toEqual(['resample', 'forward']);
  });

  it('skips absent managers rather than failing', async () => {
    // `loadPolicyConfig` runs this before a PolicyRunner exists; a scene may have no events.
    const order: string[] = [];
    await applyResetTerms({
      events: null,
      policy: null,
      commands: { resetTerms: () => { order.push('command'); } },
      context: CONTEXT,
    });
    expect(order).toEqual(['command']);
  });

  it('propagates a failing reset event instead of resetting on top of it', async () => {
    // The caller decides what a failed reset means; swallowing it here would reset the
    // policy and commands against a half-applied scene and report nothing.
    const order: string[] = [];
    await expect(
      applyResetTerms(
        chainRecording(order, {
          events: { onReset: () => Promise.reject(new Error('graph missing')) },
        }),
      ),
    ).rejects.toThrow('graph missing');
    expect(order).toEqual([]);
  });
});
