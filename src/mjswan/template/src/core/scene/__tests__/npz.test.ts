import JSZip from 'jszip';
import { describe, expect, it, vi } from 'vitest';
import { loadNpz } from '../npz';

function npy(shape: number[], values: Float32Array): Uint8Array {
  const header = new TextEncoder().encode(
    `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')},), }\n`,
  );
  const out = new Uint8Array(10 + header.length + values.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(header, 10);
  out.set(new Uint8Array(values.buffer), 10 + header.length);
  return out;
}

function raw(header: string, data: Uint8Array, major = 1): Uint8Array {
  const text = new TextEncoder().encode(header);
  const start = major === 1 ? 10 : 12;
  const out = new Uint8Array(start + text.length + data.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, major, 0]);
  const view = new DataView(out.buffer);
  if (major === 1) view.setUint16(8, text.length, true);
  else view.setUint32(8, text.length, true);
  out.set(text, start);
  out.set(data, start + text.length);
  return out;
}

async function stored(entries: Record<string, Uint8Array>): Promise<ArrayBuffer> {
  const zip = new JSZip();
  for (const [name, bytes] of Object.entries(entries)) zip.file(name, bytes);
  return zip.generateAsync({ type: 'arraybuffer', compression: 'STORE' });
}

describe('loadNpz', () => {
  it.each(['<U0', 'S0'])('drops a %s array whose shape overflows to Infinity', async (descr) => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const huge = `1${'0'.repeat(400)}`;
    const bytes = await stored({
      'junk.npy': raw(`{'descr': '${descr}', 'fortran_order': False, 'shape': (${huge},), }\n`, new Uint8Array([0x41])),
      'fps.npy': npy([1], new Float32Array([30])),
    });

    const start = performance.now();
    const npz = await loadNpz(bytes);

    expect(npz.junk).toBeUndefined();
    expect(Array.from(npz.fps.data)).toEqual([30]);
    expect(performance.now() - start).toBeLessThan(100);
    warn.mockRestore();
  });

  it('drops a header past numpy size limit before matching it', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const header = `{'descr': '<f4', ${"'shape':(".repeat(29_000)}`;
    const bytes = await stored({ 'x.npy': raw(header, new Uint8Array(0), 2) });

    const start = performance.now();
    const npz = await loadNpz(bytes);

    expect(npz.x).toBeUndefined();
    expect(performance.now() - start).toBeLessThan(100);
    warn.mockRestore();
  });


  it('drops an array whose header claims more values than it holds', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const zip = new JSZip();
    zip.file('qpos.npy', npy([400_000, 8], new Float32Array(0)));
    zip.file('fps.npy', npy([1], new Float32Array([30])));

    const npz = await loadNpz(await zip.generateAsync({ type: 'arraybuffer' }));

    expect(npz.qpos).toBeUndefined();
    expect(Array.from(npz.fps.data)).toEqual([30]);
    warn.mockRestore();
  });

  it('stops inflating at the host limit', async () => {
    const zip = new JSZip();
    zip.file('big.npy', npy([262_144], new Float32Array(262_144)));
    const bytes = await zip.generateAsync({ type: 'arraybuffer', compression: 'DEFLATE' });

    await expect(loadNpz(bytes, 64 * 1024)).rejects.toThrow(/inflates past/);
    expect((await loadNpz(bytes)).big.data.length).toBe(262_144);
  });
});
