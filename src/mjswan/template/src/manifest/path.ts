/**
 * Paths inside a document, the URL a host fetches one from, and the URLs a document may
 * name outside itself.
 *
 * A document names its files by path: a scene's against `<project-id>/<scene-id>/`, a
 * project's work against `<project-id>/`, `plugins` against the root. A path is checked
 * as written, never normalized first: each segment is `..` or a name a build writes
 * (letters, digits, `_` and `-`, then `.` too), and `..` never climbs above the document's
 * root. What a URL parser would read as something else (a backslash, `%2e%2e`, a tab
 * inside `..`, a scheme, an empty segment) is no such name, so a checked path joined onto
 * a directory's URL stays under it.
 *
 * See context/document.md.
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
 * `rel` against `dir`, a document-relative directory (`''` for the root, else ending in
 * `/`), as the document-relative path a {@link ByteSource} is asked for. Refuses a `rel`
 * that is not a path a build writes, or that would climb above the document's root.
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
 * The URL a host fetches a document's file from: `path`, as a {@link ByteSource} is given
 * it, under `base`, the document's root read as a directory (`https://host/app` is
 * `https://host/app/`). Throws unless every segment of `path` is a name a build writes, so
 * the URL stays under `base` whatever a document says.
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
