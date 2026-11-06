import { describe, it, expect } from 'vitest';
import { applyUrlState, pluginModuleUrl, type UrlState } from './urlState';

// The URL carries ids, not display names (ADR 0006 §4).
const SELECTED: UrlState = {
  project: 'demo_two',
  scene: 'humanoid',
  policy: 'walk',
  skin: null,
  panel: true,
  debug: true,
  xray: false,
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

  it("writes the skin's id, and an empty value for None", () => {
    expect(applyUrlState('', { ...SELECTED, skin: 'fox' })).toBe('project=demo_two&scene=humanoid&policy=walk&skin=fox');
    expect(applyUrlState('?skin=fox', { ...SELECTED, skin: '' })).toBe(
      'skin=&project=demo_two&scene=humanoid&policy=walk',
    );
    // A scene with no skins.
    expect(applyUrlState('?skin=fox', SELECTED)).toBe('project=demo_two&scene=humanoid&policy=walk');
  });

  it('pins xray only when it is on', () => {
    expect(applyUrlState('', { ...SELECTED, xray: true })).toBe('project=demo_two&scene=humanoid&policy=walk&xray=1');
    expect(applyUrlState('?xray=1', SELECTED)).toBe('project=demo_two&scene=humanoid&policy=walk');
  });

  it('pins the chrome flags only when they are off', () => {
    expect(applyUrlState('', { ...SELECTED, panel: false, debug: false })).toBe(
      'project=demo_two&scene=humanoid&policy=walk&panel=0&debug=0',
    );
    expect(applyUrlState('?panel=0&debug=0', SELECTED)).toBe(
      'project=demo_two&scene=humanoid&policy=walk',
    );
  });
});

describe('pluginModuleUrl', () => {
  const page = 'https://example.org/app/?scene=a';

  it("loads the build's own module", () => {
    expect(pluginModuleUrl('/app/', 'assets/plugins.js', '?scene=a', page)).toBe(
      'https://example.org/app/assets/plugins.js',
    );
  });

  it('loads nothing a ?manifest= link names', () => {
    const search = '?manifest=https%3A%2F%2Fother.test%2Fmanifest.json';
    expect(pluginModuleUrl('/app/', 'assets/plugins.js', search, page)).toBeNull();
  });

  // Joined as strings, base '/' and a leading slash would form a scheme-relative URL.
  it('keeps a leading slash on the page origin', () => {
    expect(pluginModuleUrl('/', '/other.test/p.js', '', page)).toBe('https://example.org/other.test/p.js');
  });

  it.each(['//other.test/p.js', '/\\other.test/p.js', 'https://other.test/p.js'])(
    'refuses a module on another origin (%s)',
    (path) => {
      expect(pluginModuleUrl('/', path, '', page)).toBeNull();
    },
  );
});
