import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { compareVersions, engineUrl, isReleaseVersion } from './version';

const RELEASES = ['0.13.1', '0.14.0', '0.14.0-beta.1', '1.2.3-rc.1.2', '10.20.30'];
const NOT_RELEASES = [
  'v1.2.3',
  '1.2',
  '1.2.3-',
  '1.2.3+build',
  'latest',
  '1.2.3/../../x',
  '../../evil/x',
  '1.2.3\n',
  ' 1.2.3',
  '１.2.3',
  '',
];

describe('isReleaseVersion', () => {
  it.each(RELEASES)('takes %j', (version) => {
    expect(isReleaseVersion(version)).toBe(true);
  });

  it.each([...NOT_RELEASES, 1.2, null, undefined, ['1.2.3']])('refuses %j', (version) => {
    expect(isReleaseVersion(version)).toBe(false);
  });

  // The one form, read off the workflow that tags a release.
  it("matches release.yml's own check", () => {
    const workflow = readFileSync(new URL('../../../../../.github/workflows/release.yml', import.meta.url), 'utf8');
    const checks = [...workflow.matchAll(/=~ (\^\S+\$) \]\]/g)].map((m) => new RegExp(m[1]));
    expect(checks).toHaveLength(2);
    for (const version of [...RELEASES, ...NOT_RELEASES]) {
      expect(isReleaseVersion(version)).toBe(checks.some((check) => check.test(version)));
    }
  });
});

describe('compareVersions', () => {
  it('orders releases as semver does', () => {
    const ordered = ['0.9.9', '0.13.1', '0.14.0-alpha', '0.14.0-beta.2', '0.14.0-beta.10', '0.14.0-rc.1', '0.14.0', '1.0.0'];
    const shuffled = [...ordered].reverse();
    expect(shuffled.sort(compareVersions)).toEqual(ordered);
    expect(compareVersions('0.14.0', '0.14.0')).toBe(0);
    expect(compareVersions('0.14.0-beta', '0.14.0-beta.1')).toBeLessThan(0);
    expect(compareVersions('0.14.0-1', '0.14.0-beta')).toBeLessThan(0);
  });

  it('refuses a version no release has', () => {
    expect(() => compareVersions('0.14.0', 'latest')).toThrow(/not a release version/);
  });
});

describe('engineUrl', () => {
  it("names the release's file on the CDN", () => {
    expect(engineUrl('0.14.0').href).toBe('https://cdn.jsdelivr.net/npm/mjswan@0.14.0/dist/mjswan.js');
    expect(engineUrl('0.14.0-beta.1', 'manifest.js', 'https://cdn.test/npm').href).toBe(
      'https://cdn.test/npm/mjswan@0.14.0-beta.1/dist/manifest.js',
    );
  });

  it.each(NOT_RELEASES)('refuses %j rather than name another path', (version) => {
    expect(() => engineUrl(version)).toThrow(/not a release version/);
  });
});
