/**
 * Paths inside a document, the URL a host fetches one from, and the URLs a document may
 * name outside itself.
 *
 * A path is checked as written, never normalized: each segment is `..` or a name a build
 * writes, and `..` never climbs above the root. What a URL parser reads otherwise (a
 * backslash, `%2e%2e`, a tab in `..`, a scheme, an empty segment) is no such name, so a
 * checked path stays under the directory URL it is joined to.
 */

/** A name a build writes: an id, a term name, `scene.mjz`, `LICENSE.<component>`. */
const NAME = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*$/;

function refuse(what: string, value: unknown, why: string): Error {
  return new Error(`mjswan/manifest: ${what} ${JSON.stringify(value)} ${why}.`);
}

/** `value` if it is one name a build writes (an id: a directory or a file), else an error. */
export function documentName(value: unknown, what: string): string {
  if (typeof value !== 'string' || !NAME.test(value)) {
    throw refuse(what, value, 'is not a name a build writes');
  }
  return value;
}

/**
 * `rel` resolved against `dir` (`''` for the root, else ending in `/`); throws for a path
 * a build would not write or one climbing above the root.
 */
export function documentPath(dir: string, rel: unknown, what: string): string {
  if (typeof rel !== 'string') throw refuse(what, rel, 'is not a path');
  const parts = dir ? dir.slice(0, -1).split('/') : [];
  for (const segment of rel.split('/')) {
    if (segment === '..') {
      if (!parts.length) throw refuse(what, rel, 'climbs out of the document');
      parts.pop();
    } else if (NAME.test(segment)) {
      parts.push(segment);
    } else {
      throw refuse(what, rel, 'is not a path inside the document');
    }
  }
  return parts.join('/');
}

/**
 * Where a host fetches the document file at `path` (as a {@link ByteSource} gets it): under
 * `base`, read as a directory. Throws unless every segment is a name a build writes.
 */
export function documentUrl(base: string | URL, path: string): string {
  for (const segment of path.split('/')) {
    if (!NAME.test(segment)) throw refuse('path', path, 'is not a file of the document');
  }
  const root = String(base).replace(/[?#][^]*$/, '');
  return `${root.endsWith('/') ? root : `${root}/`}${path}`;
}

/** Whether a document's reference names a URL (it has a scheme) rather than a path in it. */
export function isUrl(value: string): boolean {
  return /^[A-Za-z][A-Za-z0-9+.-]*:/.test(value);
}

/** A URL a document names outside itself, refused unless it is `http(s)`. */
export function remoteUrl(value: unknown, what: string): URL {
  let url: URL | undefined;
  try {
    url = typeof value === 'string' ? new URL(value) : undefined;
  } catch {
    url = undefined;
  }
  if (url?.protocol !== 'https:' && url?.protocol !== 'http:') {
    throw refuse(what, value, 'is not an http(s) URL');
  }
  return url;
}
