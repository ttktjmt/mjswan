import { describe, expect, it } from 'vitest';
import { documentPath, documentUrl } from './path';

describe('documentUrl', () => {
  it.each([
    ['/', '/demo/s/scene.mjz'],
    ['/app/', '/app/demo/s/scene.mjz'],
    ['/app', '/app/demo/s/scene.mjz'],
    ['https://h.test/app', 'https://h.test/app/demo/s/scene.mjz'],
    ['https://h.test/app/', 'https://h.test/app/demo/s/scene.mjz'],
    ['https://h.test/app/?v=1#top', 'https://h.test/app/demo/s/scene.mjz'],
    [new URL('https://h.test/app/'), 'https://h.test/app/demo/s/scene.mjz'],
  ])('joins a file onto %s as a directory', (base, url) => {
    expect(documentUrl(base, 'demo/s/scene.mjz')).toBe(url);
  });

  // What the URL parser would take somewhere else: another origin, out of the base path.
  it.each(['\\evil.test/x', '/x', '//evil.test/x', '%2e%2e/x', '.\t./x', '../x', 'a//b', 'x?y', 'https://evil.test/x', ''])(
    'refuses %j, which is no file of the document',
    (path) => {
      expect(() => documentUrl('/', path)).toThrow(/is not a file of the document/);
    },
  );

  it('stays under its base whatever the path, once it is accepted', () => {
    for (const base of ['https://h.test/', 'https://h.test/app/']) {
      const url = new URL(documentUrl(base, 'a/b-c.d_e'));
      expect(url.origin).toBe('https://h.test');
      expect(url.pathname.startsWith(new URL(base).pathname)).toBe(true);
    }
  });
});

describe('documentPath', () => {
  it("resolves a scene's path against its directory, `..` included", () => {
    expect(documentPath('demo/s/', 'assets/a.npz', 'motion')).toBe('demo/s/assets/a.npz');
    expect(documentPath('demo/s/', '../licenses/LICENSE.x', 'license')).toBe('demo/licenses/LICENSE.x');
    expect(documentPath('', 'assets/plugins.js', 'plugins')).toBe('assets/plugins.js');
  });

  it('refuses a path climbing above the root', () => {
    expect(() => documentPath('demo/', '../../x', 'license')).toThrow(/climbs out of the document/);
    expect(() => documentPath('', '../x', 'plugins')).toThrow(/climbs out of the document/);
  });

  it.each([undefined, null, 3, ['a']])('refuses a value that is not a path (%j)', (value) => {
    expect(() => documentPath('demo/s/', value, 'scene')).toThrow(/is not a path/);
  });
});
