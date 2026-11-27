/** A glTF naming a buffer or image elsewhere is refused, and nothing it names is fetched. */
import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseSealedGltf } from '../gltf';

/** One triangle's positions, as a buffer holds them. */
const TRIANGLE = new Uint8Array(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]).buffer);

/** A binary glTF of one triangle whose buffer is `uri` (the GLB's own BIN chunk when absent). */
function triangle(uri?: string, extra: object = {}): ArrayBuffer {
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, ...('materials' in extra ? { material: 0 } : {}) }] }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3', min: [0, 0, 0], max: [1, 1, 0] }],
    bufferViews: [{ buffer: 0, byteLength: TRIANGLE.length }],
    buffers: [{ byteLength: TRIANGLE.length, ...(uri === undefined ? {} : { uri }) }],
    ...extra,
  };
  let text = JSON.stringify(json);
  while (text.length % 4) text += ' ';
  const chunk = new TextEncoder().encode(text);
  const bin = uri === undefined ? TRIANGLE : undefined;
  const out = new DataView(new ArrayBuffer(20 + chunk.length + (bin ? 8 + bin.length : 0)));
  out.setUint32(0, 0x46546c67, true);
  out.setUint32(4, 2, true);
  out.setUint32(8, out.byteLength, true);
  out.setUint32(12, chunk.length, true);
  out.setUint32(16, 0x4e4f534a, true);
  new Uint8Array(out.buffer, 20).set(chunk);
  if (bin) {
    out.setUint32(20 + chunk.length, bin.length, true);
    out.setUint32(24 + chunk.length, 0x004e4942, true);
    new Uint8Array(out.buffer, 28 + chunk.length).set(bin);
  }
  return out.buffer;
}

const DATA_URI = `data:application/octet-stream;base64,${Buffer.from(TRIANGLE).toString('base64')}`;

describe('parseSealedGltf', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it.each([
    ['embedded in the GLB', undefined],
    ['a data: URI', DATA_URI],
  ])('reads a buffer %s', async (_, uri) => {
    // Node has no ProgressEvent, which three's FileLoader uses.
    vi.stubGlobal(
      'ProgressEvent',
      class {
        constructor(
          readonly type: string,
          init: object,
        ) {
          Object.assign(this, init);
        }
      },
    );
    const gltf = await parseSealedGltf(triangle(uri));
    let vertices = 0;
    gltf.scene.traverse((object) => {
      const mesh = object as { geometry?: { attributes: { position: { count: number } } } };
      vertices += mesh.geometry?.attributes.position.count ?? 0;
    });
    expect(vertices).toBe(3);
  });

  it.each(['https://evil.test/b.bin', '//evil.test/b.bin', 'b.bin', '../b.bin'])(
    'refuses a buffer at %s and fetches nothing',
    async (uri) => {
      const fetching = vi.spyOn(globalThis, 'fetch');
      await expect(parseSealedGltf(triangle(uri))).rejects.toThrow(/names files outside itself/);
      expect(fetching).not.toHaveBeenCalled();
    },
  );

  it.each(['t.png', 'https://evil.test/t.png'])('refuses an image at %s and fetches nothing', async (uri) => {
    // Node has no `self`, which three reads for an image.
    vi.stubGlobal('self', globalThis);
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const fetching = vi.spyOn(globalThis, 'fetch');
    const textured = triangle(undefined, {
      materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
      textures: [{ source: 0 }],
      images: [{ uri }],
    });
    await expect(parseSealedGltf(textured)).rejects.toThrow(/names files outside itself/);
    expect(fetching).not.toHaveBeenCalled();
  });
});
