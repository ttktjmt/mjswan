/** The build's defaults are where a term starts and where `clear` returns; the operator's changes sit on top. */
import { describe, expect, it } from 'vitest';

import { TermOverrides, freezeOf, isFreeze, normalOnFire } from '../operator';

describe('onFire', () => {
  it('keeps a reset, a delay and a freeze, timed or not', () => {
    expect(normalOnFire('reset')).toBe('reset');
    expect(normalOnFire('freeze')).toBe('freeze');
    expect(normalOnFire({ delay: 0.2 })).toEqual({ delay: 0.2 });
    expect(normalOnFire({ freeze: 3 })).toEqual({ freeze: 3 });
  });

  it('refuses what it cannot keep', () => {
    expect(normalOnFire('stop')).toBeNull();
    expect(normalOnFire({ delay: 'soon' })).toBeNull();
    expect(normalOnFire({ wait: 1 })).toBeNull();
    expect(normalOnFire(null)).toBeNull();
  });

  it('tells a freeze and its seconds apart', () => {
    expect(isFreeze('freeze')).toBe(true);
    expect(isFreeze({ freeze: 1 })).toBe(true);
    expect(isFreeze({ delay: 1 })).toBe(false);
    expect(freezeOf({ freeze: 1.5 })).toBe(1.5);
    expect(freezeOf('freeze')).toBeNull();
  });
});

describe('TermOverrides with build defaults', () => {
  const build = () => {
    const overrides = new TermOverrides();
    overrides.setDefaults({
      terminations: { bad_orientation: { onFire: { delay: 0.2 } }, time_out: { enabled: false } },
      events: { push_robot: { enabled: false } },
    });
    return overrides;
  };

  it('starts a term at its default, and mjlab’s for one the build left alone', () => {
    const overrides = build();
    expect(overrides.termination('bad_orientation')).toEqual({ enabled: true, onFire: { delay: 0.2 } });
    expect(overrides.termination('illegal_contact')).toEqual({ enabled: true, onFire: 'reset' });
    expect(overrides.event('push_robot')).toEqual({ enabled: false });
    expect(overrides.terminationEntries()).toEqual([
      ['bad_orientation', { enabled: true, onFire: { delay: 0.2 } }],
      ['time_out', { enabled: false, onFire: 'reset' }],
    ]);
  });

  it('counts a term changed only while it differs from its default', () => {
    const overrides = build();
    expect(overrides.changed('termination', 'bad_orientation')).toBe(false);
    overrides.setTermination('bad_orientation', { onFire: { freeze: 3 } });
    expect(overrides.changed('termination', 'bad_orientation')).toBe(true);
    overrides.setTermination('bad_orientation', { onFire: { delay: 0.2 } });
    expect(overrides.changed('termination', 'bad_orientation')).toBe(false);
    overrides.setEvent('push_robot', { enabled: true });
    expect(overrides.changed('event', 'push_robot')).toBe(true);
  });

  it('returns to the defaults on clear, not to mjlab’s', () => {
    const overrides = build();
    overrides.setTermination('bad_orientation', { onFire: 'freeze' });
    overrides.setEvent('push_robot', { enabled: true });
    overrides.clear();
    expect(overrides.termination('bad_orientation').onFire).toEqual({ delay: 0.2 });
    expect(overrides.event('push_robot').enabled).toBe(false);
  });

  it('keeps the operator’s changes across a new set of defaults, dropping one the defaults now match', () => {
    const overrides = build();
    overrides.setTermination('bad_orientation', { onFire: 'freeze' });
    overrides.setTermination('illegal_contact', { enabled: false });
    overrides.setDefaults({ terminations: { illegal_contact: { enabled: false } } });
    expect(overrides.termination('bad_orientation').onFire).toBe('freeze');
    expect(overrides.changed('termination', 'illegal_contact')).toBe(false);
    overrides.setDefaults(null);
    expect(overrides.termination('time_out')).toEqual({ enabled: true, onFire: 'reset' });
  });
});
