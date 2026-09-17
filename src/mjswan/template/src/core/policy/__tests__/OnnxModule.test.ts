/**
 * The provider choice in `OnnxModule.init()`: the engine carries ORT's CPU-only build, so
 * there is one provider and no fallback to arrange.
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

  // Asking for a provider this bundle does not carry would cost a failed session and a
  // retry on every policy load, so the ask has to match what shipped (see vite.wasm.ts).
  it('asks only for the provider the shipped bundle has', async () => {
    const create = vi.spyOn(ort.InferenceSession, 'create').mockResolvedValue(fakeSession);
    await new OnnxModule(new ArrayBuffer(8)).init();
    expect(create).toHaveBeenCalledTimes(1);
    expect(providersOf(create.mock.calls[0])).toEqual(['wasm']);
  });

  it('surfaces a session failure rather than swallowing it', async () => {
    vi.spyOn(ort.InferenceSession, 'create').mockRejectedValue(new Error('bad model'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await expect(new OnnxModule(new ArrayBuffer(8)).init()).rejects.toThrow('bad model');
    expect(warn).not.toHaveBeenCalled();
  });
});
