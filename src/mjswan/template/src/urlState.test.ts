import { describe, it, expect } from 'vitest';
import { applyUrlState, type UrlState } from './urlState';

// The URL carries ids, not display names (ADR 0006 §4).
const SELECTED: UrlState = {
  project: 'demo_two',
  scene: 'humanoid',
  policy: 'walk',
  panel: true,
  ref: true,
};

describe('applyUrlState', () => {
  it('writes the whole selection', () => {
    expect(applyUrlState('', SELECTED)).toBe('project=demo_two&scene=humanoid&policy=walk');
  });

  it('replaces a stale selection wholesale', () => {
    // The bug: a project switch left the previous project pinned.
    const search = '?project=demo_one&scene=cartpole&policy=balance';
    expect(applyUrlState(search, SELECTED)).toBe('project=demo_two&scene=humanoid&policy=walk');
  });

  it('drops a parameter whose selection is null', () => {
    // Single-project build, and a scene run with no policy.
    const state = { ...SELECTED, project: null, policy: null };
    expect(applyUrlState('?project=demo_two&policy=walk', state)).toBe('scene=humanoid');
  });

  it('leaves parameters it does not own alone', () => {
    const search = '?manifest=/other/manifest.json&utm_source=paper';
    expect(applyUrlState(search, SELECTED)).toBe(
      'manifest=%2Fother%2Fmanifest.json&utm_source=paper&project=demo_two&scene=humanoid&policy=walk',
    );
  });

  it('pins the chrome flags only when they are off', () => {
    expect(applyUrlState('', { ...SELECTED, panel: false, ref: false })).toBe(
      'project=demo_two&scene=humanoid&policy=walk&panel=0&ref=0',
    );
    expect(applyUrlState('?panel=0&ref=0', SELECTED)).toBe(
      'project=demo_two&scene=humanoid&policy=walk',
    );
  });
});
