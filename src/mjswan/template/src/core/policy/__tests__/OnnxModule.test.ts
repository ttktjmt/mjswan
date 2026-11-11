/**
 * The provider choice in `OnnxModule.init()`: wasm, the only backend the bundled ORT
 * build carries. `../../onnx/__tests__/ortRuntimeFiles.test.ts` pins that build.
 */
import * as ort from 'onnxruntime-web/wasm';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { OnnxModule } from '../OnnxModule';

const fakeSession = {
  inputNames: ['obs'],
  outputNames: ['action'],
  release: async () => {},
} as unknown as ort.InferenceSession;

const providersOf = (call: unknown[]): unknown =>
  (call[1] as ort.InferenceSession.SessionOptions).executionProviders;

describe('OnnxModule.init', () => {
  afterEach(() => vi.restoreAllMocks());

  it('asks for wasm, the one backend the shipped build registers', async () => {
    const create = vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(fakeSession);
    await new OnnxModule(new ArrayBuffer(8)).init();
    expect(create).toHaveBeenCalledTimes(1);
    expect(providersOf(create.mock.calls[0])).toEqual(['wasm']);
  });

  it('surfaces a creation failure instead of retrying, since there is nothing to fall back to', async () => {
    vi.spyOn(ort.InferenceSession, 'create').mockRejectedValue(new Error('bad model'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(new OnnxModule(new ArrayBuffer(8)).init()).rejects.toThrow('bad model');
    expect(warn).not.toHaveBeenCalled();
  });
});

describe('OnnxModule.dispose', () => {
  afterEach(() => vi.restoreAllMocks());

  /** Yields mid-run, as ORT's wasm run does. */
  function heldSession() {
    let running = false;
    let onStart!: () => void;
    const started = new Promise<void>((resolve) => {
      onStart = resolve;
    });
    const session = {
      inputNames: ['obs'],
      outputNames: ['action'],
      released: 0,
      releasedDuringRun: false,
      async run() {
        running = true;
        onStart();
        await new Promise((resolve) => setTimeout(resolve, 0));
        running = false;
        return { action: new ort.Tensor('float32', new Float32Array([1]), [1, 1]) };
      },
      release: async () => {
        session.released += 1;
        session.releasedDuringRun ||= running;
      },
    };
    return { session, started };
  }

  async function moduleOn(session: object): Promise<OnnxModule> {
    vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(session as ort.InferenceSession);
    const module = new OnnxModule(new ArrayBuffer(8));
    await module.init();
    return module;
  }

  it('releases its session once', async () => {
    const { session } = heldSession();
    const module = await moduleOn(session);
    await module.dispose();
    await module.dispose();
    expect(session.released).toBe(1);
  });

  it('waits for an in-flight run rather than releasing under it', async () => {
    const { session, started } = heldSession();
    const module = await moduleOn(session);
    const actor = new ort.Tensor('float32', new Float32Array([0]), [1, 1]);
    const run = module.runInference({ actor });
    await started;
    await module.dispose();
    const [result] = await run;
    expect(session.releasedDuringRun).toBe(false);
    expect(session.released).toBe(1);
    expect(result.action).toBeDefined();
  });

  it('warns rather than throws when the release fails', async () => {
    const { session } = heldSession();
    session.release = () => Promise.reject(new Error('boom'));
    const module = await moduleOn(session);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(module.dispose()).resolves.toBeUndefined();
    expect(warn).toHaveBeenCalled();
  });
});

describe('OnnxModule.initInput', () => {
  afterEach(() => vi.restoreAllMocks());

  /** A recurrent session whose `adapt_hx` input declares `shape`. */
  async function recurrent(shape: Array<number | string>): Promise<OnnxModule> {
    const session = {
      inputNames: ['obs', 'hx'],
      outputNames: ['action', 'next_hx'],
      inputMetadata: [
        { name: 'obs', isTensor: true, type: 'float32', shape: [1, 4] },
        { name: 'hx', isTensor: true, type: 'float32', shape },
      ],
      release: async () => {},
    };
    vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(session as never);
    const module = new OnnxModule(new ArrayBuffer(8), { in_keys: ['actor', 'adapt_hx'] });
    await module.init();
    return module;
  }

  it('starts the carry at the width the network declares', async () => {
    // A carry seeded at 128 would fail every run of a 730-wide network.
    const module = await recurrent([1, 730]);
    expect(module.initInput().adapt_hx.dims).toEqual([1, 730]);
  });

  it('falls back to 128 where the width is dynamic', async () => {
    const module = await recurrent(['batch', 'hidden']);
    expect(module.initInput().adapt_hx.dims).toEqual([1, 128]);
  });
});

describe('OnnxModule.inputDims', () => {
  afterEach(() => vi.restoreAllMocks());

  /** rsl-rl's CNN export: the flat group, then a 3 x 32 x 32 image. */
  async function cnn(): Promise<OnnxModule> {
    const session = {
      inputNames: ['obs', 'camera'],
      outputNames: ['actions'],
      inputMetadata: [
        { name: 'obs', isTensor: true, type: 'float32', shape: [1, 26] },
        { name: 'camera', isTensor: true, type: 'float32', shape: [1, 3, 32, 32] },
      ],
      release: async () => {},
    };
    vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(session as never);
    const module = new OnnxModule(new ArrayBuffer(8), { in_keys: ['actor', 'camera'] });
    await module.init();
    return module;
  }

  it('feeds an image group in the shape the network declares', async () => {
    expect((await cnn()).inputDims('camera', 3 * 32 * 32)).toEqual([1, 3, 32, 32]);
  });

  it('feeds a flat group, or one whose size disagrees, as a row', async () => {
    const module = await cnn();
    expect(module.inputDims('actor', 26)).toEqual([1, 26]);
    expect(module.inputDims('camera', 10)).toEqual([1, 10]);
  });
});
