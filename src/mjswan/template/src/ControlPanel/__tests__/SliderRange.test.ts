/**
 * The adjustable-range companion slider.
 *
 * mjlab's play GUI pairs each velocity axis with a "Max <label>" slider that only
 * rescales how far the value slider drags. The property worth pinning is that it
 * is *presentational*: it must never reach the engine. A version that called `set`
 * for it would silently feed the policy a bogus command.
 */
import { describe, expect, it } from 'vitest';

import type { SliderCommandConfig } from '../../core/command/types';
import type { CommandDefinition } from '../../core/command';
import type { CommandDescriptor } from '../../engine/types';

/**
 * The engine's descriptor mapping, duplicated rather than imported: importing
 * `createEngine` pulls in the MuJoCo WASM loader. The field names are what is pinned.
 */
function toDescriptor(def: CommandDefinition): CommandDescriptor {
  const config = def.config;
  const base = { id: def.id, group: def.groupName, type: config.type, label: config.label };
  return config.type === 'slider'
    ? {
        ...base,
        min: config.min,
        max: config.max,
        step: config.step,
        enabledWhen: config.enabled_when,
        adjustableRange: config.adjustable_range,
      }
    : base;
}

const SLIDER: SliderCommandConfig = {
  type: 'slider',
  name: 'lin_vel_x',
  label: 'Forward Velocity',
  min: -1.5,
  max: 1.5,
  step: 0.05,
  default: 0.5,
  enabled_when: 'enabled',
  adjustable_range: { min: 0, max: 1.5, step: 0.05, default: 1.5 },
};

describe('adjustable_range in the command descriptor', () => {
  it('reaches the app through the descriptor', () => {
    const descriptor = toDescriptor({
      id: 'velocity:lin_vel_x',
      groupName: 'velocity',
      config: SLIDER,
    });
    expect(descriptor.adjustableRange).toEqual({
      min: 0,
      max: 1.5,
      step: 0.05,
      default: 1.5,
    });
  });

  it('is absent when the build did not ask for one', () => {
    const plain = { ...SLIDER };
    delete plain.adjustable_range;
    const descriptor = toDescriptor({
      id: 'velocity:lin_vel_x',
      groupName: 'velocity',
      config: plain,
    });
    expect(descriptor.adjustableRange).toBeUndefined();
  });

  it('carries no id of its own — it is not a settable command', () => {
    // `set` is keyed by command id, and the range control has none — it stays app-local.
    const descriptor = toDescriptor({
      id: 'velocity:lin_vel_x',
      groupName: 'velocity',
      config: SLIDER,
    });
    expect(Object.keys(descriptor.adjustableRange!).sort()).toEqual([
      'default',
      'max',
      'min',
      'step',
    ]);
  });
});
