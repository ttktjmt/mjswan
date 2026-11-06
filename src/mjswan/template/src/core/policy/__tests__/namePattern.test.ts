import { describe, expect, it } from 'vitest';
import { compileNamePattern } from '../namePattern';

describe('compileNamePattern', () => {
  it.each([
    ['.*', 'left_hip_joint'],
    ['.*_(hip|knee)_.*', 'left_hip_joint'],
    ['.*_(hip|knee)_.*', 'left_ankle_joint'],
    ['(?:FL|FR|RL|RR)_.*_joint', 'RR_calf_joint'],
    ['[A-Z]{2}_thigh_joint', 'FL_thigh_joint'],
    ['joint\\d{1,2}', 'joint12'],
    ['joint\\d{1,2}', 'joint123'],
    ['left_.*|right_.*', 'right_wrist'],
    ['waist_(yaw|roll|pitch)_joint', 'waist_pitch_joint'],
    ['a{2,}b?', 'aaab'],
    ['robot/.*', 'robot/torso'],
  ])('matches %s on %s as ^(?:p)$ does', (pattern, name) => {
    expect(compileNamePattern(pattern)(name)).toBe(new RegExp(`^(?:${pattern})$`).test(name));
  });

  // V8's RegExp takes seconds on these by a few dozen characters.
  it('stays linear on backtracking payloads', () => {
    const start = performance.now();
    expect(compileNamePattern('(a+)+$')('a'.repeat(4000) + '!')).toBe(false);
    expect(compileNamePattern('(a|a)'.repeat(51))('a'.repeat(51) + '!')).toBe(false);
    expect(performance.now() - start).toBeLessThan(100);
  });

  it('refuses lookaround and backreferences', () => {
    expect(() => compileNamePattern('(?=a)')).toThrow();
    expect(() => compileNamePattern('(a)\\1')).toThrow();
  });
});
