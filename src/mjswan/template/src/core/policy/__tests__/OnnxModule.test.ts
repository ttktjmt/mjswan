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
