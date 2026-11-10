/** One budget per policy load bounds every buffer a document sizes. */
import { describe, expect, it } from 'vitest';
import { substepOptions } from '../../action/applyAction';
import { ContactSensorSet } from '../../onnx/contact';
import { BufferBudget, MAX_BUFFERED, MAX_FRAMES, reserveTermFeeds } from '../bufferBudget';
import { PolicyRunner } from '../PolicyRunner';
import type { PolicyConfig, PolicyRunnerContext } from '../types';

const rng = { uniform: () => 0, randint: () => 0 } as never;

describe('BufferBudget', () => {
  it('admits what a policy declares', () => {
    const budget = new BufferBudget();
    expect(() => budget.reserve(10, 48, 'history')).not.toThrow();
    expect(() => budget.reserve(MAX_FRAMES, 1, 'ring')).not.toThrow();
  });

  it('refuses one ring past the frame limit', () => {
    expect(() => new BufferBudget().reserve(MAX_FRAMES + 1, 1, 'ring')).toThrow(/buffer limit/);
  });

  it('refuses rings that add up past the total, empty frames included', () => {
    const budget = new BufferBudget();
    budget.reserve(MAX_FRAMES, MAX_BUFFERED / MAX_FRAMES - 64, 'one');
    expect(() => budget.reserve(1, 0, 'two')).toThrow(/buffer limit/);
  });
});

describe('a declared count past the limit', () => {
  it('stops PolicyRunner before it allocates a term history', async () => {
    const config = {
      observations: {
        actor: [{ name: 'c', native: 'constant', size: 1, value: [0], history_offsets: [100_000_000] }],
      },
    } as unknown as PolicyConfig;
    await expect(new PolicyRunner(config, {}).init({} as PolicyRunnerContext)).rejects.toThrow(/buffer limit/);
  });

  it('stops a command delay', () => {
    const entry = { delay: [{ joints: ['knee'], max_lag: 100_000_000 }] };
    expect(() => substepOptions(entry, ['knee'], 'joint_pos', rng)).toThrow(/buffer limit/);
  });

  it('stops a contact history', () => {
    const descriptor = {
      kind: 'contact' as const,
      num_slots: 1,
      history_length: 100_000_000,
      history_fields: ['force'],
      fields: { force: { sensors: ['foot_force'], dim: 3 } },
    };
    expect(() => new ContactSensorSet({ feet: descriptor })).toThrow(/buffer limit/);
  });

  // With no history, every read still allocates num_slots frames per sensor.
  it('stops a contact reading', () => {
    const descriptor = {
      kind: 'contact' as const,
      num_slots: 100_000_000,
      history_length: 0,
      history_fields: [],
      fields: { force: { sensors: ['foot_force'], dim: 3 } },
    };
    expect(() => new ContactSensorSet({ feet: descriptor })).toThrow(/buffer limit/);
  });
});

describe('a count that would refill the budget', () => {
  it.each([
    [-1e15, 0],
    [1, -1e12],
    [1.5, 1],
    [1, Number.NaN],
  ])('refuses %s frames of width %s', (frames, width) => {
    expect(() => new BufferBudget().reserve(frames, width, 'refill')).toThrow(/buffer limit/);
  });

  it('stops a contact refill before a large sibling', () => {
    const big = {
      kind: 'contact' as const,
      num_slots: 1,
      history_length: 9999,
      history_fields: ['force'],
      fields: { force: { sensors: ['s0'], dim: 5000 } },
    };
    const refill = { ...big, num_slots: -1e15, history_length: 0, history_fields: [] };
    expect(() => new ContactSensorSet({ refill, big })).toThrow(/buffer limit/);
  });
});

describe('reserveTermFeeds', () => {
  const scan = (frames: number, offsets: number) => ({
    kind: 'raycast',
    frames: Array.from({ length: frames }, () => ({ type: 'body', name: 'base' })),
    local_offsets: Array.from({ length: offsets }, () => [0, 0, 0]),
  });
  const reads = (n: number) =>
    Array.from({ length: n }, (_, i) => ({ sensor: 'scan', field: 'distances', input: `x${i}` }));
  const terminations = (sensor: object, slots: object[]) => ({ t: { name: 't', onnx: 't.onnx', sensors: { scan: sensor }, input_slots: slots } });
  const reserve = (config: object) => () => reserveTermFeeds(config, new BufferBudget());

  it('admits a shipped-size policy', () => {
    const config = {
      observations: { actor: [{ name: 'h', size: 187, input_slots: [{ sim: 'qpos', rows: [0, 1], shape: [1, 2, 7], input: 'q' }] }] },
      commands: [{ name: 'c', rand_dim: 3 }],
      terminations: terminations(scan(1, 187), reads(1)),
      debug_vis: { sensors: { scan: { ...scan(1, 187), debug_vis: true } }, rewards: { up: { terrain_sensors: ['scan'] } } },
    };
    expect(reserve(config)).not.toThrow();
  });

  it.each([
    ['a rand_dim past the total', { commands: [{ rand_dim: 16_000_001 }] }],
    ['a negative rand_dim', { commands: [{ rand_dim: -1 }] }],
    ['a fractional rand_dim', { events: [{ rand_dim: 1.5 }] }],
    ['a sim slot past the total', { terminations: { t: { input_slots: [{ sim: 'qpos', rows: [0], shape: [1, 1, 16_000_001] }] } } }],
    ['an observation size past the total', { observations: { actor: [{ name: 'c', native: 'constant', size: 16_000_001 }] } }],
    ['a scan read past the ray limit', { terminations: terminations(scan(300, 300), reads(1)) }],
    ['one scan read by many slots', { terminations: terminations(scan(10, 1000), reads(100)) }],
  ])('refuses %s', (_, config) => {
    expect(reserve(config)).toThrow(/limit/);
  });

  it('counts a debug drawing per reward that lists the sensor', () => {
    const sensors = { scan: { ...scan(10, 1000), debug_vis: false } };
    const rewards = Object.fromEntries(Array.from({ length: 100 }, (_, i) => [`r${i}`, { terrain_sensors: ['scan'] }]));
    expect(reserve({ debug_vis: { sensors, rewards } })).toThrow(/ray limit/);
    expect(reserve({ terminations: terminations(scan(10, 1000), reads(1)) })).not.toThrow();
  });
});
