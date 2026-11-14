// TODO: Remove this module after mujoco wasm mj_loadXML() supports mjz format loading.

import JSZip from 'jszip';
import type { MainModule } from 'mujoco';

/**
 * What a `.mjz` may hold, checked as it unpacks. A host that renders archives it did not
 * build passes its own upload rules, and they hold whatever the archive's directory claims:
 * a file's size is counted as it inflates, and its name is the one it is written under.
 */
export interface ArchiveLimits {
  /** Most files, directories not counted. */
  maxEntries: number;
  /** Most bytes the files inflate to, all together. */
  maxInflatedBytes: number;
  /** Extensions a file's name may end in, lowercase with the dot (`'.xml'`). */
  extensions: readonly string[];
}

/** Where a model's files are written for MuJoCo to open. */
export const WORKING = '/working';

/**
 * Give the next model an empty /working, and return the time to date its files with: past
 * `after`, the last model's, by a whole second. MuJoCo resolves a model's includes and
 * assets against everything in /working, and caches a mesh by its path and its modification
 * time in whole seconds, so a file an earlier model left, or one written at the same path
 * in the same second, would stand in for the next model's own.
 */
export function resetWorking(mujoco: MainModule, after: number): number {
  mujoco.FS.unmount(WORKING);
  mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, WORKING);
  return Math.max(Date.now(), after + 1000);
}

/** Inside the archive and nowhere else: no root, no `..`, nothing a reader takes differently. */
const isSafePath = (name: string) =>
  !name.startsWith('/') &&
  !name.includes('\\') &&
  !/\p{Cc}/u.test(name) &&
  name.split('/').every((seg) => seg !== '' && seg !== '.' && seg !== '..');

const extensionOf = (name: string) => {
  const dot = name.lastIndexOf('.');
  return dot === -1 ? '' : name.slice(dot).toLowerCase();
};

/** JSZip's `ZipObject#internalStream`, documented by JSZip and missing from its typings. */
interface StreamHelper {
  on(event: 'data', listener: (chunk: Uint8Array) => void): StreamHelper;
  on(event: 'error', listener: (error: Error) => void): StreamHelper;
  on(event: 'end', listener: () => void): StreamHelper;
  pause(): StreamHelper;
  resume(): StreamHelper;
}

/** A file's bytes, inflated in JSZip's chunks so the count can stop it partway. */
function inflate(file: JSZip.JSZipObject, room: number): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const chunks: Uint8Array[] = [];
    let size = 0;
    const stream = (file as unknown as { internalStream(type: 'uint8array'): StreamHelper }).internalStream('uint8array');
    stream
      .on('data', (chunk: Uint8Array) => {
        size += chunk.length;
        if (size > room) {
          stream.pause();
          reject(new RangeError(`inflates past ${room} bytes`));
          return;
        }
        chunks.push(chunk);
      })
      .on('error', reject)
      .on('end', () => {
        const out = new Uint8Array(size);
        let at = 0;
        for (const c of chunks) {
          out.set(c, at);
          at += c.length;
        }
        resolve(out);
      })
      .resume();
  });
}

/**
 * Unzip an .mjz archive into the Emscripten VFS under /working and return the
 * root XML path **relative to /working** (e.g. `scene.xml`) — the form
 * loadSceneFromURL expects, which prefixes /working itself.
 *
 * With `limits`, the archive is refused, before MuJoCo compiles any of it, for a file
 * past them or at a name they do not take. JSZip names a file from its local header, or
 * the Unicode path in its directory record, so the name checked is the one written.
 * `mtime` dates every file written (`resetWorking`).
 */
export async function loadMjzFile(
    mujoco: MainModule,
    mjzBytes: ArrayBuffer,
    limits?: ArchiveLimits,
    mtime?: number
): Promise<string> {
    const zip = await JSZip.loadAsync(mjzBytes);
    const files = Object.entries(zip.files).filter(([, file]) => !file.dir);

    if (limits) {
        if (files.length > limits.maxEntries) {
            throw new Error(`The .mjz holds ${files.length} files; this host opens at most ${limits.maxEntries}.`);
        }
        for (const [name] of files) {
            if (!isSafePath(name) || !limits.extensions.includes(extensionOf(name))) {
                throw new Error(`The .mjz holds "${name}", which this host does not open.`);
            }
        }
    }

    // Root XML path, relative to /working.
    let xmlPath: string | null = null;
    let room = limits?.maxInflatedBytes ?? Infinity;

    // Extract all files to Emscripten FS
    for (const [relativePath, file] of files) {
        // Sanitize ZIP entry path to prevent path traversal
        const sanitized = relativePath
            .replace(/\\/g, '/')
            .split('/')
            .filter((seg) => seg && seg !== '..' && seg !== '.')
            .join('/');
        if (!sanitized) continue;

        let content: Uint8Array;
        try {
            content = await inflate(file, room);
        } catch (error) {
            if (error instanceof RangeError && limits) {
                throw new Error(`The .mjz inflates to more than ${limits.maxInflatedBytes / 1048576} MB, which this host does not open.`);
            }
            throw error;
        }
        room -= content.length;
        const fsPath = `${WORKING}/${sanitized}`;

        // Create directories
        const dirPath = fsPath.substring(0, fsPath.lastIndexOf('/'));
        try {
            mujoco.FS.mkdirTree(dirPath, 0o777);
        } catch {
            // Directory exists
        }

        // Write file
        mujoco.FS.writeFile(fsPath, content);
        if (mtime !== undefined) mujoco.FS.utime(fsPath, mtime, mtime);

        // Track root XML
        if (sanitized.endsWith('.xml') && !sanitized.includes('/')) {
            xmlPath = sanitized;
        }
    }

    if (!xmlPath) {
        throw new Error('No root XML found in .mjz archive');
    }

    return xmlPath;
}
