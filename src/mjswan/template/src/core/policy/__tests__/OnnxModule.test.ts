/**
 * The provider choice in `OnnxModule.init()`: WebGPU first, and the one fallback ORT does
 * not perform itself. ORT drops a provider whose *init* fails; an adapter that exists but
 * fails at session creation rejects the whole `create`, and that case retries on wasm.
 */
import * as ort from 'onnxruntime-web';
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

  it('asks for WebGPU first and lets ORT settle on wasm where there is none', async () => {
    const create = vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(fakeSession);
    await new OnnxModule(new ArrayBuffer(8)).init();
    expect(create).toHaveBeenCalledTimes(1);
    expect(providersOf(create.mock.calls[0])).toEqual(['webgpu', 'wasm']);
  });

  it('retries the session on wasm when WebGPU is present but fails at creation', async () => {
    const create = vi
      .spyOn(ort.InferenceSession, 'create')
      .mockRejectedValueOnce(new Error('WebGPU device lost'))
      .mockResolvedValueOnce(fakeSession);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await new OnnxModule(new ArrayBuffer(8)).init();
    expect(create).toHaveBeenCalledTimes(2);
    expect(providersOf(create.mock.calls[1])).toEqual(['wasm']);
    expect(warn).toHaveBeenCalledOnce();
  });

  it('surfaces the second error, not a WebGPU warning, when wasm fails as well', async () => {
    // Both attempts failing means the model is at fault; the warning would mislead.
    vi.spyOn(ort.InferenceSession, 'create').mockRejectedValue(new Error('bad model'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(new OnnxModule(new ArrayBuffer(8)).init()).rejects.toThrow('bad model');
    expect(warn).not.toHaveBeenCalled();
  });
});
