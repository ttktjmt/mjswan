/**
 * mjswan release versions. A document's `version` names the release that wrote it, and a
 * host picks an engine by it, so it must be one a release could have, never a path or a
 * URL: an engine URL built from it stays at that release.
 *
 * See context/document.md.
 */

/** `X.Y.Z`, or `X.Y.Z-<prerelease>`: the forms `release.yml` publishes. */
const RELEASE = /^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.]+))?$/;

/** The CDN a host loads a released engine from. */
export const ENGINE_CDN = 'https://cdn.jsdelivr.net/npm/';

/** Whether `version` is one a release could have: `0.14.0`, `0.14.0-beta.1`. */
export function isReleaseVersion(version: unknown): version is string {
  return typeof version === 'string' && RELEASE.test(version);
}

function parts(version: string): [number[], string[] | null] {
  const match = RELEASE.exec(version);
  if (!match) throw new Error(`mjswan: ${JSON.stringify(version)} is not a release version.`);
  return [[Number(match[1]), Number(match[2]), Number(match[3])], match[4]?.split('.') ?? null];
}

/** One prerelease identifier against another: numbers numerically and before words. */
function compareIdentifiers(a: string, b: string): number {
  const numeric = /^\d+$/;
  if (numeric.test(a) && numeric.test(b)) return Math.sign(Number(a) - Number(b));
  if (numeric.test(a) !== numeric.test(b)) return numeric.test(a) ? -1 : 1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Negative when release `a` comes before `b`, positive after, zero when they are one:
 * semver's precedence, so `0.14.0-beta.2` < `0.14.0-beta.10` < `0.14.0`. Throws for a
 * version no release has.
 */
export function compareVersions(a: string, b: string): number {
  const [coreA, preA] = parts(a);
  const [coreB, preB] = parts(b);
  for (let i = 0; i < 3; i++) {
    if (coreA[i] !== coreB[i]) return Math.sign(coreA[i] - coreB[i]);
  }
  if (!preA || !preB) return preA ? -1 : preB ? 1 : 0;
  for (let i = 0; i < Math.min(preA.length, preB.length); i++) {
    const order = compareIdentifiers(preA[i], preB[i]);
    if (order) return order;
  }
  return Math.sign(preA.length - preB.length);
}

/**
 * Where release `version`'s engine (`mjswan.js`) or catalog parser (`manifest.js`) is
 * served from `cdn`. Throws for a version no release has, so a document cannot choose
 * what the URL names beyond the release.
 */
export function engineUrl(
  version: string,
  file: 'mjswan.js' | 'manifest.js' = 'mjswan.js',
  cdn: string = ENGINE_CDN,
): URL {
  if (!isReleaseVersion(version)) {
    throw new Error(`mjswan: ${JSON.stringify(version)} is not a release version.`);
  }
  return new URL(`mjswan@${version}/dist/${file}`, cdn.endsWith('/') ? cdn : `${cdn}/`);
}
