/** One budget per policy load bounds every buffer a document sizes. */
import { describe, expect, it } from 'vitest';
import { substepOptions } from '../../action/applyAction';
import { ContactSensorSet } from '../../onnx/contact';
import { BufferBudget, MAX_BUFFERED, MAX_FRAMES } from '../bufferBudget';
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
