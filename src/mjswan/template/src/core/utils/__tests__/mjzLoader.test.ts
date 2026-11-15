/**
 * The archive a host did not build. Its directory is the archive's own word for what it
 * holds, and a host's upload check can read only that, so these forge the directory and
 * hold the loader to what the archive really unpacks to.
 */
import JSZip from 'jszip';
import type { MainModule } from 'mujoco';
import { beforeAll, describe, expect, it } from 'vitest';

import { loadMjzFile, resetWorking, WORKING, type ArchiveLimits } from '../mjzLoader';

const LIMITS: ArchiveLimits = { maxEntries: 3, maxInflatedBytes: 1024 * 1024, extensions: ['.xml', '.stl'] };
const SCENE = '<mujoco><compiler meshdir="."/></mujoco>';

/** A stand-in for MuJoCo's VFS that notes what was written where. */
function vfs() {
  const written = new Map<string, number>();
  const mujoco = {
    FS: { mkdirTree: () => {}, writeFile: (path: string, data: Uint8Array) => written.set(path, data.length) },
  } as unknown as MainModule;
  return { mujoco, written };
}

async function zip(files: Record<string, string | Uint8Array>): Promise<Uint8Array> {
  const archive = new JSZip();
  for (const [name, data] of Object.entries(files)) archive.file(name, data);
  return archive.generateAsync({ type: 'uint8array', compression: 'DEFLATE' });
}

const ascii = (s: string) => new TextEncoder().encode(s);

/** Where `name` starts in the bytes, the local header's copy first and the directory's after. */
function at(bytes: Uint8Array, name: string, nth: number): number {
  const needle = ascii(name);
  let seen = 0;
  for (let i = 0; i <= bytes.length - needle.length; i++) {
    if (needle.every((b, k) => bytes[i + k] === b) && seen++ === nth) return i;
  }
  throw new Error(`${name} is not in the archive`);
}

/** The archive with what `name` inflates to written as `size`, in its local header and its directory. */
function claiming(bytes: Uint8Array, name: string, size: number): Uint8Array {
  const out = bytes.slice();
  const view = new DataView(out.buffer);
  view.setUint32(at(out, name, 0) - 30 + 22, size, true);
  view.setUint32(at(out, name, 1) - 46 + 24, size, true);
  return out;
}

describe('loadMjzFile', () => {
  it('unpacks every file and opens the last .xml at the top', async () => {
    const { mujoco, written } = vfs();
    const root = await loadMjzFile(mujoco, (await zip({ 'meshes/base.stl': 'solid', 'scene.xml': SCENE })).buffer as ArrayBuffer);
    expect(root).toBe('scene.xml');
    expect([...written.keys()]).toEqual(['/working/meshes/base.stl', '/working/scene.xml']);
  });

  it('passes an archive within the limits a host gives', async () => {
    const { mujoco } = vfs();
    const bytes = await zip({ 'base.stl': new Uint8Array(4096), 'scene.xml': SCENE });
    await expect(loadMjzFile(mujoco, bytes.buffer as ArrayBuffer, LIMITS)).resolves.toBe('scene.xml');
  });

  // A bomb: 8 MB of zeros deflate to a few KB, and the directory says 100 bytes.
  it('stops a file that inflates past the limit, whatever the directory says it holds', async () => {
    const { mujoco, written } = vfs();
    const bomb = claiming(await zip({ 'scene.xml': SCENE, 'base.stl': new Uint8Array(8 * 1024 * 1024) }), 'base.stl', 100);
    expect(bomb.length).toBeLessThan(64 * 1024);
    await expect(loadMjzFile(mujoco, bomb.buffer as ArrayBuffer, LIMITS)).rejects.toThrow(
      'The .mjz inflates to more than 1 MB, which this host does not open.',
    );
    expect(written.has('/working/base.stl')).toBe(false);
  });

  // The directory names `model.stl`, which a host's check passes; JSZip writes the name
  // in the local header.
  it('holds the name a file is written under to the limits, not the one the directory gives', async () => {
    const bytes = await zip({ 'scene.xml': SCENE, 'model.stl': 'solid' });
    bytes.set(ascii('evil.html'), at(bytes, 'model.stl', 0));

    const open = vfs();
    await loadMjzFile(open.mujoco, bytes.buffer as ArrayBuffer);
    expect(open.written.has('/working/evil.html')).toBe(true);

    const held = vfs();
    await expect(loadMjzFile(held.mujoco, bytes.buffer as ArrayBuffer, LIMITS)).rejects.toThrow(
      'The .mjz holds "evil.html", which this host does not open.',
    );
    expect(held.written.size).toBe(0);
  });

  // JSZip itself resolves `..` inside the archive as it loads; a backslash it keeps.
  it('refuses more files than the limit, and a path readers take differently', async () => {
    const { mujoco } = vfs();
    const many = await zip({ 'a.stl': 's', 'b.stl': 's', 'c.stl': 's', 'scene.xml': SCENE });
    await expect(loadMjzFile(mujoco, many.buffer as ArrayBuffer, LIMITS)).rejects.toThrow('holds 4 files; this host opens at most 3');
    const slanted = await zip({ 'meshes\\base.stl': 's', 'scene.xml': SCENE });
    await expect(loadMjzFile(mujoco, slanted.buffer as ArrayBuffer, LIMITS)).rejects.toThrow('holds "meshes\\base.stl"');
  });
});

describe('resetWorking', () => {
  let mujoco: MainModule;
  let at = 0;
  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
    mujoco.FS.mkdir(WORKING);
    mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, WORKING);
  });

  /** One model loaded as the runtime loads it: a fresh /working, then its files dated. */
  async function load(files: Record<string, string>): Promise<string> {
    at = resetWorking(mujoco, at);
    return loadMjzFile(mujoco, (await zip(files)).buffer as ArrayBuffer, undefined, at);
  }

  function compiled<T>(root: string, read: (model: import('mujoco').MjModel) => T): T {
    const model = mujoco.MjModel.mj_loadXML(`${WORKING}/${root}`);
    try {
      return read(model);
    } finally {
      model.delete();
    }
  }

  it("leaves the next model none of the last one's files", async () => {
    await load({ 'p.xml': '<mujoco/>', 'lib/limb.xml': '<mujoco><worldbody><body name="planted"/></worldbody></mujoco>' });
    const root = await load({ 'q.xml': '<mujoco><include file="lib/limb.xml"/></mujoco>' });
    expect(() => compiled(root, () => null)).toThrow();
  });

  it('opens a root the last model held as a directory', async () => {
    await load({ 'p.xml': '<mujoco/>', 'scene.xml/pad.xml': '<mujoco/>' });
    const root = await load({ 'scene.xml': '<mujoco><worldbody><body name="own"/></worldbody></mujoco>' });
    expect(compiled(root, (m) => m.nbody)).toBe(2);
  });

  it('compiles its own mesh at a path the last model wrote in the same second', async () => {
    // MuJoCo caches a mesh by path and modification time in whole seconds.
    const cube = (s: number) =>
      [[0, 0, 0], [s, 0, 0], [0, s, 0], [0, 0, s]].map((v) => `v ${v.join(' ')}`).join('\n') + '\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n';
    const scene = '<mujoco><asset><mesh name="m" file="m.obj"/></asset><worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>';
    const extent = (m: import('mujoco').MjModel) => Math.max(...m.mesh_vert) - Math.min(...m.mesh_vert);
    const small = compiled(await load({ 's.xml': scene, 'm.obj': cube(1) }), extent);
    const large = compiled(await load({ 's.xml': scene, 'm.obj': cube(3) }), extent);
    expect(large / small).toBeCloseTo(3);
  });
});
