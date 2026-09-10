import * as ort from 'onnxruntime-web';

// Configures ort.env before any session is created; see ortEnv.ts.
import '../onnx/ortEnv';
import { queueOrtRun } from '../onnx/runQueue';
import DEFAULT_SLOTS from './default_slots.json';

/**
 * The policy's slot tables (ADR 0006 §5). `in_keys[i]` names the tensor that fills the
 * session's i-th input, an observation group or one the runtime synthesizes (`is_init`,
 * `adapt_hx`, `time_step`); `out_keys[i]` names its i-th output. The mapping is
 * positional, so the model's own tensor names appear in no config.
 */
export type OnnxSlotTables = {
  in_keys?: string[];
  out_keys?: (string | string[])[];
};

/**
 * What a policy that declares no tables gets: one input fed by the `actor` group. Python
 * holds the same default and a mismatch leaves every such policy inert with no error, so
 * both read `default_slots.json` (the Python side via `tests/test_mjlab_adapter.py`).
 */
const DEFAULT_IN_KEYS: readonly string[] = DEFAULT_SLOTS.in_keys;
const DEFAULT_OUT_KEYS: readonly string[] = DEFAULT_SLOTS.out_keys;

export class OnnxModule {
  private bytes: ArrayBuffer;
  private session: ort.InferenceSession | null;
  private configuredInKeys: string[];
  inKeys: string[];
  outKeys: string[];
  isRecurrent: boolean;

  constructor(bytes: ArrayBuffer, slots?: OnnxSlotTables) {
    this.bytes = bytes;
    this.session = null;
    const inKeys = slots?.in_keys ?? [...DEFAULT_IN_KEYS];
    const outKeys = slots?.out_keys ?? [...DEFAULT_OUT_KEYS];
    this.configuredInKeys = inKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.inKeys = [...this.configuredInKeys];
    this.outKeys = outKeys.map((key) => (Array.isArray(key) ? key.join(',') : key));
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  async init(): Promise<void> {
    const create = (executionProviders: string[]) =>
      ort.InferenceSession.create(this.bytes, { executionProviders, graphOptimizationLevel: 'all' });
    try {
      // WebGPU where the browser has it, wasm everywhere else. ORT drops a provider whose
      // init fails on its own; what it does not survive is an adapter that exists but fails
      // at session creation, hence the retry. See docs/docs/api/engine.md, "Where inference runs".
      this.session = await create(['webgpu', 'wasm']);
    } catch (error) {
      this.session = await create(['wasm']);
      console.warn('[OnnxModule] WebGPU session failed, running on wasm:', error);
    }
    this.inferInputKeys();
    this.isRecurrent = this.inKeys.includes('adapt_hx');
  }

  initInput(): Record<string, ort.Tensor> {
    const inputs: Record<string, ort.Tensor> = {};
    if (this.isRecurrent) {
      inputs.is_init = new ort.Tensor('bool', [true], [1]);
      inputs.adapt_hx = new ort.Tensor('float32', new Float32Array(128), [1, 128]);
    }
    if (this.inKeys.includes('time_step')) {
      inputs.time_step = new ort.Tensor('float32', new Float32Array([0.0]), [1, 1]);
    }
    return inputs;
  }

  async runInference(
    input: Record<string, ort.Tensor>
  ): Promise<[Record<string, ort.Tensor>, Record<string, ort.Tensor>]> {
    if (!this.session) {
      throw new Error('OnnxModule not initialized.');
    }

    const onnxInput: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.inKeys.length; i++) {
      const key = this.inKeys[i];
      const name = this.session.inputNames[i];
      if (!name || !input[key]) {
        throw new Error(`Missing ONNX input for key: ${key}`);
      }
      onnxInput[name] = input[key];
    }

    const session = this.session;
    const onnxOutput = await queueOrtRun(() => session.run(onnxInput));
    const result: Record<string, ort.Tensor> = {};
    for (let i = 0; i < this.outKeys.length; i++) {
      const key = this.outKeys[i];
      const name = this.session.outputNames[i];
      if (name && onnxOutput[name]) {
        result[key] = onnxOutput[name];
      }
    }

    const carry: Record<string, ort.Tensor> = {};
    if (this.isRecurrent && result['next,adapt_hx']) {
      carry.is_init = new ort.Tensor('bool', [false], [1]);
      carry.adapt_hx = result['next,adapt_hx'];
    }

    return [result, carry];
  }

  /** Release the ONNX Runtime session, freeing its WASM memory. */
  dispose(): void {
    void this.session?.release?.();
    this.session = null;
  }

  /**
   * Settle the input table against the session. A single-input model takes the configured
   * key as is, and so does a table exactly as long as the model's input list, which is
   * what the build guarantees. Anything else is a hand-written config disagreeing with
   * its network, so the model's own input names are used instead.
   */
  private inferInputKeys(): void {
    if (!this.session) {
      return;
    }
    const modelInputs = this.session.inputNames;
    this.inKeys =
      modelInputs.length <= 1 || this.configuredInKeys.length === modelInputs.length
        ? [...this.configuredInKeys]
        : [...modelInputs];
  }
}
