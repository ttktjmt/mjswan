// TODO: Remove this module after mujoco wasm mj_loadXML() supports mjz format loading.

import { Inflate } from 'fflate';
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

/**
 * One file or directory, read as JSZip (which this module used before) read it: listed,
 * sized and typed by the central directory, named by its local header.
 */
interface ZipEntry {
  name: string;
  dir: boolean;
  method: number;
  /** As stored, so compressed when `method` is DEFLATED. */
  data: Uint8Array;
  /** What the central directory says the file inflates to. */
  size: number;
}

const LOCAL_HEADER = 0x04034b50;
const CENTRAL_HEADER = 0x02014b50;
const END = 0x06054b50;
const ZIP64_END = 0x06064b50;
const ZIP64_END_LOCATOR = 0x07064b50;
const ZIP64_FIELD = 0x0001;
const UNICODE_PATH_FIELD = 0x7075;
const MAX16 = 0xffff;
const MAX32 = 0xffffffff;
const STORED = 0;
const DEFLATED = 8;
const MADE_BY_UNIX = 3;

// JSZip kept a leading BOM in a name, which TextDecoder strips by default.
const utf8 = new TextDecoder('utf-8', { ignoreBOM: true });

let crcTable: Uint32Array | undefined;
function crc32(bytes: Uint8Array): number {
  crcTable ??= Uint32Array.from({ length: 256 }, (_, n) => {
    for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
    return n;
  });
  let crc = MAX32;
  for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ MAX32) >>> 0;
}

/**
 * The archive's entries in central directory order, with JSZip's reading of each: the end
 * record found from the back, zip64 records, data prepended to the archive (a self-extractor)
 * skipped, a name decoded as UTF-8 unless an up-to-date Unicode path field gives it.
 */
function readEntries(bytes: Uint8Array): ZipEntry[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const check = (at: number, length: number): number => {
    if (at < 0 || at + length > bytes.length) {
      throw new Error(`End of data reached (data length = ${bytes.length}, asked index = ${at + length}). Corrupted zip ?`);
    }
    return at;
  };
  const u16 = (at: number) => view.getUint16(check(at, 2), true);
  const u32 = (at: number) => view.getUint32(check(at, 4), true);
  const u64 = (at: number) => u32(at) + u32(at + 4) * 2 ** 32;
  const slice = (at: number, length: number) => bytes.subarray(check(at, length), at + length);
  const isSignature = (at: number, signature: number) =>
    at >= 0 && at + 4 <= bytes.length && view.getUint32(at, true) === signature;
  const lastSignature = (signature: number) => {
    for (let at = bytes.length - 4; at >= 0; at--) if (view.getUint32(at, true) === signature) return at;
    return -1;
  };

  const end = lastSignature(END);
  if (end < 0) {
    throw new Error(
      isSignature(0, LOCAL_HEADER)
        ? "Corrupted zip: can't find end of central directory"
        : "Can't find end of central directory : is this a zip file ?",
    );
  }
  let records = u16(end + 10);
  let directorySize = u32(end + 12);
  let directoryOffset = u32(end + 16);
  let expectedEnd = directoryOffset + directorySize;
  if (
    [u16(end + 4), u16(end + 6), u16(end + 8), records].includes(MAX16) ||
    directorySize === MAX32 ||
    directoryOffset === MAX32
  ) {
    const locator = lastSignature(ZIP64_END_LOCATOR);
    if (locator < 0) throw new Error("Corrupted zip: can't find the ZIP64 end of central directory locator");
    if (u32(locator + 16) > 1) throw new Error('Multi-volumes zip are not supported');
    let end64 = u64(locator + 8);
    if (!isSignature(end64, ZIP64_END)) {
      end64 = lastSignature(ZIP64_END);
      if (end64 < 0) throw new Error("Corrupted zip: can't find the ZIP64 end of central directory");
    }
    records = u64(end64 + 32);
    directorySize = u64(end64 + 40);
    directoryOffset = u64(end64 + 48);
    expectedEnd = directoryOffset + directorySize + 20 + 12 + u64(end64 + 4);
  }
  const prepended = end - expectedEnd;
  if (prepended < 0) throw new Error(`Corrupted zip: missing ${-prepended} bytes.`);

  const entries: ZipEntry[] = [];
  for (let at = prepended + directoryOffset; isSignature(at, CENTRAL_HEADER); ) {
    const madeBy = u16(at + 4) >> 8;
    const flags = u16(at + 8);
    const method = u16(at + 10);
    let compressedSize = u32(at + 20);
    let size = u32(at + 24);
    const nameLength = u16(at + 28);
    const extraLength = u16(at + 30);
    const external = u32(at + 38);
    let localOffset = u32(at + 42);
    if (flags & 1) throw new Error('Encrypted zip are not supported');

    const fields = new Map<number, Uint8Array>();
    const extraEnd = at + 46 + nameLength + extraLength;
    for (let field = at + 46 + nameLength; field + 4 < extraEnd; field += 4 + u16(field + 2)) {
      fields.set(u16(field), slice(field + 4, u16(field + 2)));
    }
    const zip64 = fields.get(ZIP64_FIELD);
    if (zip64) {
      let read = 0;
      const next = () => {
        if (read + 8 > zip64.length) throw new Error('End of data reached in a ZIP64 extra field. Corrupted zip ?');
        const z = new DataView(zip64.buffer, zip64.byteOffset + read, 8);
        read += 8;
        return z.getUint32(0, true) + z.getUint32(4, true) * 2 ** 32;
      };
      if (size === MAX32) size = next();
      if (compressedSize === MAX32) compressedSize = next();
      if (localOffset === MAX32) localOffset = next();
    }
    if (compressedSize === MAX32 || size === MAX32) {
      throw new Error("Bug or corrupted zip : didn't get enough information from the central directory");
    }

    const local = prepended + localOffset;
    if (!isSignature(local, LOCAL_HEADER)) throw new Error('Corrupted zip or bug: unexpected signature at a local file header');
    const localNameLength = u16(local + 26);
    const nameBytes = slice(local + 30, localNameLength);
    const unicodePath = fields.get(UNICODE_PATH_FIELD);
    const name =
      !(flags & 0x0800) &&
      unicodePath &&
      unicodePath.length >= 5 &&
      unicodePath[0] === 1 &&
      new DataView(unicodePath.buffer, unicodePath.byteOffset + 1, 4).getUint32(0, true) === crc32(nameBytes)
        ? utf8.decode(unicodePath.subarray(5))
        : utf8.decode(nameBytes);
    if (method !== STORED && method !== DEFLATED) {
      throw new Error(`Corrupted zip : compression ${method} unknown (inner file : ${name})`);
    }
    entries.push({
      name,
      dir: (external & 0x10) !== 0 || name.endsWith('/') || (madeBy === MADE_BY_UNIX && ((external >>> 16) & 0x4000) !== 0),
      method,
      data: slice(local + 30 + localNameLength + u16(local + 28), compressedSize),
      size,
    });
    at = extraEnd + u16(at + 32);
  }
  if (records !== 0 && entries.length === 0) {
    throw new Error(`Corrupted zip or bug: expected ${records} records in central dir, got 0`);
  }
  return entries;
}

/** JSZip's folding of a loaded name: `.` and empty segments dropped, `..` taking one back. */
function resolve(name: string): string {
  const parts = name.split('/');
  const kept: string[] = [];
  parts.forEach((part, index) => {
    if (part === '.' || (part === '' && index !== 0 && index !== parts.length - 1)) return;
    if (part === '..') kept.pop();
    else kept.push(part);
  });
  return kept.join('/');
}

/** Thrown from inside the inflate, so it stops there; never reaches the caller. */
class PastRoom extends Error {}

/** JSZip's read size: a bomb is stopped within one chunk's worth of output. */
const CHUNK = 16 * 1024;

/** A file's bytes, counted as they inflate so the count can stop it partway. */
function inflate(entry: ZipEntry, room: number): Uint8Array {
  // JSZip took a file its directory sizes at 0 for empty, whatever the archive holds.
  if (entry.size === 0) return new Uint8Array(0);
  if (entry.method === STORED) {
    if (entry.data.length > room) throw new PastRoom();
    return entry.data;
  }
  const chunks: Uint8Array[] = [];
  let size = 0;
  const inflater = new Inflate((chunk) => {
    size += chunk.length;
    if (size > room) throw new PastRoom();
    chunks.push(chunk);
  });
  let at = 0;
  do {
    inflater.push(entry.data.subarray(at, at + CHUNK), at + CHUNK >= entry.data.length);
    at += CHUNK;
  } while (at < entry.data.length);
  const out = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * Unzip an .mjz archive into the Emscripten VFS under /working and return the
 * root XML path **relative to /working** (e.g. `scene.xml`) — the form
 * loadSceneFromURL expects, which prefixes /working itself.
 *
 * With `limits`, the archive is refused, before MuJoCo compiles any of it, for a file
 * past them or at a name they do not take. A file is named from its local header, or the
 * Unicode path in its directory record, so the name checked is the one written. Every
 * name is checked before any file inflates. `mtime` dates every file written
 * (`resetWorking`).
 */
export async function loadMjzFile(
    mujoco: MainModule,
    mjzBytes: ArrayBuffer,
    limits?: ArchiveLimits,
    mtime?: number
): Promise<string> {
    // Keyed as JSZip keyed them: a later file under a name replaces the earlier one in place.
    const byName: Record<string, ZipEntry> = Object.create(null);
    for (const entry of readEntries(new Uint8Array(mjzBytes))) {
        const name = resolve(entry.name);
        byName[entry.dir && !name.endsWith('/') ? `${name}/` : name] = entry;
    }
    const files = Object.entries(byName).filter(([, file]) => !file.dir);

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
            content = inflate(file, room);
        } catch (error) {
            if (error instanceof PastRoom && limits) {
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
