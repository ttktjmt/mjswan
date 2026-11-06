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

describe('loadNpz', () => {
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
