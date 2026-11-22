/**
 * The operator's overrides of event terms: a reset or interval term switched off makes no
 * write at its trigger, and any listed term can be triggered by hand, whatever its mode or
 * switch.
 */
import { describe, expect, it, vi } from 'vitest';

import { SeededRng } from '../../rng';
import { OnnxSessionCache, type OnnxTensorLike } from '../../onnx/session';
import { EventManager } from '../EventManager';
import type { EventConfig, EventContext } from '../EventBase';
import type { TermActivity } from '../../mdp/operator';

const NO_MODEL: EventContext = { mjModel: null, mjData: null };
const DT = 0.25;

const CONFIGS: EventConfig[] = [
  { name: 'reset_base', mode: 'reset', onnx: 'event/reset_base.onnx', rand_dim: 0 },
  {
    name: 'push_robot',
    mode: 'interval',
    onnx: 'event/push.onnx',
    rand_dim: 0,
    interval_range_s: [1.0, 1.0],
    label: 'Push',
  },
  { name: 'throw', mode: 'manual', onnx: 'event/throw.onnx', rand_dim: 0 },
  { name: 'foot_friction', mode: 'startup', onnx: 'event/friction.onnx', rand_dim: 0 },
];

/** Every graph counts its runs by path; the clock is the episode's, which the test drives. */
async function rig() {
  const runs: string[] = [];
  const paths = ['reset_base', 'push', 'throw', 'friction'].map((name) => `event/${name}.onnx`);
  // The cache hands its factory bytes alone, so each graph is told apart by its buffer.
  const buffers = new Map(paths.map((path) => [new ArrayBuffer(0), path]));
  const sessions = new OnnxSessionCache((bytes) =>
    Promise.resolve({
      run: (_feeds: Record<string, OnnxTensorLike>) => {
        runs.push(buffers.get(bytes)!);
        return Promise.resolve({});
      },
    }),
  );
  await sessions.load([...buffers].map(([data, name]) => ({ name, data })));
  let time = 0;
  const activity: TermActivity[] = [];
  const manager = new EventManager(
    CONFIGS,
    {},
    { sessions, rng: new SeededRng(1) },
    { now: () => time, onActivity: (a) => activity.push(a) },
  );
  const tick = async () => {
    time += DT;
    await manager.tick(DT, NO_MODEL);
  };
  const reset = async () => {
    time = 0;
    await manager.onReset(NO_MODEL);
  };
  const count = (path: string) => runs.filter((run) => run === `event/${path}.onnx`).length;
  return { manager, tick, reset, count, activity };
}

const termNamed = (manager: EventManager, name: string) =>
  manager.describe().find((term) => term.name === name)!;

describe('EventManager overrides', () => {
  it('lists the reset, interval and manual terms in config order, startup left out', async () => {
    const { manager } = await rig();
    expect(manager.describe().map(({ name, mode, label }) => [name, mode, label])).toEqual([
      ['reset_base', 'reset', 'reset_base'],
      ['push_robot', 'interval', 'Push'],
      ['throw', 'manual', 'throw'],
    ]);
    expect(manager.setOverride('foot_friction', { enabled: false })).toBe(false);
  });

  it('records when each write landed this episode, and forgets it at the next reset', async () => {
    const { manager, tick, reset, activity } = await rig();
    await reset();
    for (let i = 0; i < 4; i++) await tick();
    expect(termNamed(manager, 'reset_base').firedAt).toBe(0);
    expect(termNamed(manager, 'push_robot').firedAt).toBe(1);
    expect(activity.map(({ name, what, time }) => [name, what, time])).toEqual([
      ['reset_base', 'fired', 0],
      ['push_robot', 'fired', 1],
    ]);
    await reset();
    expect(termNamed(manager, 'push_robot').firedAt).toBeNull();
  });

  it('skips a reset term that is off, and leaves its gate as if it had not fired', async () => {
    const { manager, reset, count } = await rig();
    manager.setOverride('reset_base', { enabled: false });
    await reset();
    expect(count('reset_base')).toBe(0);
    manager.setOverride('reset_base', { enabled: true });
    await reset();
    expect(count('reset_base')).toBe(1);
  });

  it('makes off and the interval schedule one switch', async () => {
    const { manager, tick, reset, count } = await rig();
    await reset();
    manager.setArmed('push_robot', false);
    expect(termNamed(manager, 'push_robot').enabled).toBe(false);
    manager.setOverride('push_robot', { enabled: true });
    expect(manager.controls().find(({ name }) => name === 'push_robot')?.armed).toBe(true);
    for (let i = 0; i < 4; i++) await tick();
    expect(count('push')).toBe(1);
  });

  it('has no switch for a manual term, which only a press fires', async () => {
    const { manager } = await rig();
    expect(manager.setOverride('throw', { enabled: false })).toBe(false);
    expect(termNamed(manager, 'throw').enabled).toBe(true);
  });

  it('triggers a reset term mid-episode, even switched off, leaving its gate alone', async () => {
    const { manager, tick, reset, count, activity } = await rig();
    manager.setOverride('reset_base', { enabled: false });
    await reset();
    await tick();
    await manager.trigger('reset_base', NO_MODEL);
    expect(count('reset_base')).toBe(1);
    expect(termNamed(manager, 'reset_base').firedAt).toBe(0.25);
    expect(activity[activity.length - 1]).toEqual({ kind: 'event', name: 'reset_base', what: 'fired', time: 0.25, byHand: true });
  });

  it('triggers an interval term off its schedule, with the schedule off too, and keeps its timer', async () => {
    const { manager, tick, reset, count } = await rig();
    await reset();
    manager.setArmed('push_robot', false);
    await manager.trigger('push_robot', NO_MODEL);
    expect(count('push')).toBe(1);
    manager.setArmed('push_robot', true);
    for (let i = 0; i < 3; i++) await tick();
    expect(count('push')).toBe(1);
    await tick();
    expect(count('push')).toBe(2);
  });

  it('triggers a manual term as its button does, and warns for a name it does not list', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { manager, count } = await rig();
    await manager.trigger('throw', NO_MODEL);
    expect(count('throw')).toBe(1);
    await manager.trigger('foot_friction', NO_MODEL);
    expect(count('friction')).toBe(0);
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
