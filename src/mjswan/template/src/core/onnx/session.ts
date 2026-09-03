/**
 * The one place that talks to `onnxruntime-web`, kept minimal so a term handler stays
 * testable with a fake — no ORT, browser or WASM.
 */
import * as ort from 'onnxruntime-web';

import { queueOrtRun } from './runQueue';

/** Minimal ORT-Web surface a command/event handler needs. */
export interface OnnxSession {
  run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>>;
  /** The graph's input names, when the session knows them — see {@link declaredFeeds}. */
  readonly inputNames?: readonly string[];
  /** Free the backing WASM memory. */
  release?(): Promise<void>;
}

export interface OnnxTensorLike {
  data: Float32Array | BigInt64Array | Uint8Array;
  dims: readonly number[];
}

/**
 * A dynamic runtime read a term's graph declares as an input, mirroring
 * `mjswan.compile.tracer.slot_to_json`. Distinguished by which field is set:
 * `entity`+`field`, `sensor`, `sensor`+`field`, or `command`+`field`.
 *
 * `input` is the graph input name, build-supplied because sensor and command names
 * carry dots the build folds to identifiers — not reproducible here. `shape` is the
 * traced shape, batch axis included, since a slot reader hands back a flat array.
 */
export interface OnnxInputSlot {
  entity?: string | null;
  field?: string;
  sensor?: string;
  command?: string;
  /** A raw `mjData` field the term read off mjlab's `SimData` (`act`, `time`, ...). */
  sim?: string;
  input?: string;
  shape?: number[];
}

/** The graph input name for a slot: the build-supplied one, else the legacy scheme. */
export function slotInputName(slot: OnnxInputSlot): string {
  if (slot.input) return slot.input;
  return `${slot.entity ?? 'entity'}__${slot.field ?? ''}`;
}

/**
 * The dims to feed a slot's flat value as. Not every field is rank 2 (`site_pos_w` is
 * `(batch, sites, 3)`) and ORT rejects a rank mismatch, so the traced shape wins, batch
 * pinned to 1 — except where it disagrees with the actual element count.
 */
export function slotDims(slot: OnnxInputSlot, length: number): number[] {
  const shape = slot.shape;
  if (!shape || shape.length === 0) return [1, length];
  const declared = shape.slice(1).reduce((a, b) => a * b, 1);
  if (declared !== length) return [1, length];
  return [1, ...shape.slice(1)];
}

/** Reads the dynamic runtime state an `OnnxInputSlot` declares, or null if absent. */
export type SlotReader = (slot: OnnxInputSlot) => Float32Array | null;

/** An output tensor's data as float32; ORT hands back bool as `Uint8Array`, int64 as bigint. */
export function toFloat32(data: OnnxTensorLike['data']): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = Number(data[i]);
  return out;
}

/**
 * Feed the slots a graph declares, stopping at the first the reader cannot serve and
 * naming it in `missing`. Callers that must not run on partial state bail on it; the
 * rest ignore it and let ORT reject the incomplete feed, which it does either way.
 *
 * Stopping rather than reading on: a slot read is not always cheap (a raycast sensor
 * casts its rays), and the feed is already unusable.
 */
export function buildFeeds(
  slots: readonly OnnxInputSlot[] | undefined,
  readSlot: SlotReader | undefined,
): { feeds: Record<string, OnnxTensorLike>; missing: string | null } {
  const feeds: Record<string, OnnxTensorLike> = {};
  for (const slot of slots ?? []) {
    const value = readSlot?.(slot) ?? null;
    if (!value) return { feeds, missing: slotInputName(slot) };
    feeds[slotInputName(slot)] = { data: value, dims: slotDims(slot, value.length) };
  }
  return { feeds, missing: null };
}

/**
 * `feeds` less anything the graph does not declare.
 *
 * The export prunes an input the body never reads — a term that draws nothing has no
 * `rand` — and ORT rejects a feed it cannot place. Callers assemble everything they
 * *might* owe; this drops the rest. A session not reporting its inputs is fed unchanged.
 */
export function declaredFeeds(
  session: OnnxSession,
  feeds: Record<string, OnnxTensorLike>,
): Record<string, OnnxTensorLike> {
  const names = session.inputNames;
  if (!names) return feeds;
  const declared = new Set(names);
  const filtered: Record<string, OnnxTensorLike> = {};
  for (const [name, tensor] of Object.entries(feeds)) {
    if (declared.has(name)) filtered[name] = tensor;
  }
  return filtered;
}

function toOrtTensor(t: OnnxTensorLike): ort.Tensor {
  if (t.data instanceof Uint8Array) {
    // ORT's 'bool' dtype takes a Uint8Array of 0/1 — a direct pass-through.
    return new ort.Tensor('bool', t.data, t.dims);
  }
  if (t.data instanceof BigInt64Array) {
    return new ort.Tensor('int64', t.data, t.dims);
  }
  return new ort.Tensor('float32', t.data, t.dims);
}

function fromOrtTensor(t: ort.Tensor): OnnxTensorLike {
  if (t.type === 'bool') return { data: t.data as Uint8Array, dims: t.dims };
  if (t.type === 'int64') return { data: t.data as BigInt64Array, dims: t.dims };
  return { data: t.data as Float32Array, dims: t.dims };
}

/** Wraps a real ORT-Web `InferenceSession` behind the minimal `OnnxSession` shape. */
class OrtSession implements OnnxSession {
  constructor(private readonly session: ort.InferenceSession) {}

  get inputNames(): readonly string[] {
    return this.session.inputNames;
  }

  async run(feeds: Record<string, OnnxTensorLike>): Promise<Record<string, OnnxTensorLike>> {
    const ortFeeds: Record<string, ort.Tensor> = {};
    for (const [name, tensor] of Object.entries(feeds)) ortFeeds[name] = toOrtTensor(tensor);
    const outputs = await queueOrtRun(() => this.session.run(ortFeeds));
    const result: Record<string, OnnxTensorLike> = {};
    for (const [name, tensor] of Object.entries(outputs)) result[name] = fromOrtTensor(tensor);
    return result;
  }

  async release(): Promise<void> {
    await this.session.release();
  }
}

/** Create a real ORT-Web-backed session from graph bytes; never fetches. */
export async function createOnnxSession(bytes: ArrayBuffer): Promise<OnnxSession> {
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  return new OrtSession(session);
}

/**
 * Term-body graphs built once from resolved bytes, keyed by config-relative path.
 * `sessionFactory` is injectable for tests.
 */
export class OnnxSessionCache {
  private sessions = new Map<string, OnnxSession>();

  constructor(
    private readonly sessionFactory: (bytes: ArrayBuffer) => Promise<OnnxSession> = createOnnxSession,
  ) {}

  /** Build (or replace) sessions for the given `{name, data}` entries. */
  async load(entries: ReadonlyArray<{ name: string; data: ArrayBuffer }>): Promise<void> {
    await Promise.all(
      entries.map(async (entry) => {
        this.sessions.set(entry.name, await this.sessionFactory(entry.data));
      }),
    );
  }

  get(path: string): OnnxSession | undefined {
    return this.sessions.get(path);
  }

  get size(): number {
    return this.sessions.size;
  }

  /**
   * Released, not just dropped: ORT-Web holds WASM heap that JS GC cannot reach.
   * Through `queueOrtRun` so a swap mid-inference lands between runs, not during one.
   */
  async clear(): Promise<void> {
    const stale = [...this.sessions.values()];
    this.sessions.clear();
    if (stale.length === 0) return;
    await queueOrtRun(async () => {
      for (const session of stale) {
        // Warned, not thrown: `dispose()` awaits this and must still finish.
        try {
          await session.release?.();
        } catch (error) {
          console.warn('[OnnxSessionCache] session release failed:', error);
        }
      }
    });
  }
}
