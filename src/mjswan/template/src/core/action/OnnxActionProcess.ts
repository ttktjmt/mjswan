/**
 * A traced action term's `process`: its own `process_actions` and `reset`, exported as
 * graphs with the term's state promoted to explicit I/O, as a traced command's is:
 *
 *     process: (raw, slots..., prev_<f>..., [rand]) -> (next_<f>..., processed, [processed_from])
 *     reset:   (slots..., prev_<f>..., [rand])      -> (next_<f>...)
 *
 * This owns the state across control steps; `applyAction` stays native and synchronous.
 * Unlike a command's, the run is awaited: the substeps need this step's targets.
 */

import { makeStateTensor, type OnnxStateFieldSpec } from '../command/OnnxCommand';
import { buildFeeds, declaredFeeds, toFloat32 } from '../onnx/session';
import type { OnnxInputSlot, OnnxSession, OnnxTensorLike, SlotReader } from '../onnx/session';
import type { SeededRng } from '../rng';

/** One traced graph of an action term, as the manifest describes it. */
export interface OnnxActionGraphConfig {
  onnx: string;
  input_slots?: OnnxInputSlot[];
  rand_dim?: number;
  rand_ranges?: Array<[number, number]>;
}

export interface OnnxActionProcessConfig extends OnnxActionGraphConfig {
  state_fields: OnnxStateFieldSpec[];
}

export interface OnnxActionProcessDeps {
  process: OnnxSession;
  /** Absent when the term's `reset` writes none of its state. */
  reset?: OnnxSession | null;
  rng: SeededRng;
  readSlot?: SlotReader;
}

export class OnnxActionProcess {
  private state = new Map<string, OnnxTensorLike>();
  /** The last run queued, which the next waits for (see `run`). */
  private queue: Promise<unknown> = Promise.resolve();

  constructor(
    private readonly name: string,
    private readonly config: OnnxActionProcessConfig,
    private readonly resetConfig: OnnxActionGraphConfig | null,
    private readonly deps: OnnxActionProcessDeps,
  ) {
    for (const spec of config.state_fields) this.state.set(spec.name, makeStateTensor(spec));
  }

  /**
   * One control step: `raw`, the term's slice of the policy's action, into `processed`
   * (and the ramp's start into `rampFrom`, when there is one). False when a slot could
   * not be read or the graph gave no `processed`: the previous step's targets stay.
   */
  async process(
    raw: Float32Array,
    processed: Float32Array,
    rampFrom: Float32Array | null,
  ): Promise<boolean> {
    const outputs = await this.run(this.deps.process, this.config, {
      raw: { data: raw, dims: [1, raw.length] },
    });
    if (!outputs) return false;
    const next = outputs.processed;
    if (!next) {
      console.warn(`[ActionManager] Action term "${this.name}": the graph returned no \`processed\`.`);
      return false;
    }
    copyInto(processed, toFloat32(next.data));
    if (rampFrom && outputs.processed_from) copyInto(rampFrom, toFloat32(outputs.processed_from.data));
    return true;
  }

  /** mjlab's `ActionTerm.reset` for this term: its reset graph, if the reset writes state. */
  async reset(): Promise<void> {
    if (!this.deps.reset || !this.resetConfig) return;
    await this.run(this.deps.reset, this.resetConfig, {});
  }

  /** One state field, for an observation that reads the term's internals. */
  getStateField(field: string): Float32Array | null {
    const tensor = this.state.get(field);
    // Copied: the tensor feeds the graph next step, so an in-place edit would corrupt it.
    return tensor ? Float32Array.from(toFloat32(tensor.data)) : null;
  }

  /**
   * In call order, each run after the last: a run reads the state the one before it
   * writes, and a UI reset starts without waiting for the step's process.
   */
  private run(
    session: OnnxSession,
    graph: OnnxActionGraphConfig,
    extra: Record<string, OnnxTensorLike>,
  ): Promise<Record<string, OnnxTensorLike> | null> {
    const pending = this.queue.then(() => this.runNow(session, graph, extra));
    this.queue = pending.catch(() => {});
    return pending;
  }

  /** Feed slots, state and `rand`; adopt the next state. Null when a slot is missing. */
  private async runNow(
    session: OnnxSession,
    graph: OnnxActionGraphConfig,
    extra: Record<string, OnnxTensorLike>,
  ): Promise<Record<string, OnnxTensorLike> | null> {
    const { feeds, missing } = buildFeeds(graph.input_slots, this.deps.readSlot);
    if (missing) {
      console.warn(`[ActionManager] Action term "${this.name}": cannot read input "${missing}".`);
      return null;
    }
    Object.assign(feeds, extra);
    for (const spec of this.config.state_fields) {
      feeds[`prev_${spec.name}`] = this.state.get(spec.name)!;
    }
    const randDim = graph.rand_dim ?? 0;
    if (randDim > 0) {
      feeds.rand = { data: this.deps.rng.randVector(randDim, graph.rand_ranges), dims: [randDim] };
    }
    const outputs = await session.run(declaredFeeds(session, feeds));
    for (const spec of this.config.state_fields) {
      const next = outputs[`next_${spec.name}`];
      if (next) this.state.set(spec.name, next);
    }
    return outputs;
  }
}

/** `source` into `target`, clamped to the shorter: a mismatch means config and graph disagree. */
function copyInto(target: Float32Array, source: Float32Array): void {
  if (source.length !== target.length) {
    console.warn(`[ActionManager] expected ${target.length} values, the graph gave ${source.length}.`);
  }
  target.set(source.subarray(0, Math.min(source.length, target.length)));
}
