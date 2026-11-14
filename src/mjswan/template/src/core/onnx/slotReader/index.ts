/**
 * Fills a traced graph's `input_slots` from `mjModel`/`mjData`. A field read wrong here
 * makes the graph compute the right function of the wrong numbers, silently.
 *
 * Every entity slot must be the **whole** field, flattened (the graph carries its own
 * baked-in indexing), in mjlab's element order (MJCF spec order, i.e. ascending model id
 * within an entity, free joint excluded), as float32.
 *
 * A `sim` slot is a raw `mjData` field — whole, or narrowed to the `rows` the term
 * indexes, so the graph need not gather them. An `EntityData` property with no reader in
 * `fields/` is traced through to the `sim` fields it reads, with mjlab's math in the
 * graph; a reader replaces that math with one input. The build emits a value slot only
 * for the fields `READER_FIELDS` (compile/slot.py) lists, kept in step by hand.
 *
 * An `env` slot is the runtime's step counter, fed as float32 like every slot; the graph
 * casts it back to mjlab's int64, exactly below 2^24 steps.
 *
 * A `model` slot is the rows of an `mjModel` field for the elements it names (`model.ts`).
 *
 * Entities resolve as `indexing.ts` describes; an unknown field returns null and the
 * caller holds its previous value.
 */

import type { CameraSensorSet } from '../camera';
import type { ContactSensorSet } from '../contact';
import { RaycastSensor, isRaycastField, type RaycastSensorDescriptor } from '../raycast';
import type { OnnxInputSlot, SlotReader } from '../session';
import { FIELD_READERS } from './fields';
import { buildEntityIndex, decodeNames, unprefixed, type EntityIndex } from './indexing';
import { readModelSlot, type ModelDefaultsSource } from './model';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

export type { EntityIndex } from './indexing';

/** A command term that can hand back one of its traced state fields by name. */
export interface CommandStateSource {
  getStateField(field: string): Float32Array | null;
}

/** Shaped to match `PolicyRunnerContext`, so the runtime can pass one straight through. */
export type SlotReaderContext = {
  mjModel: MjModel | null;
  mjData: MjData | null;
  /** Needed to cast a `RayCastSensor`'s rays (`mj_ray`); absent before load. */
  mujoco?: MainModule | null;
  commandManager?: { getTerm(name: string): unknown } | null;
  /** Control steps since the last reset: mjlab's `episode_length_buf`. */
  episodeLength?: number;
  /** The policy's last action, zeroed on reset: mjlab's `action_manager.action`. */
  lastActions?: () => Float32Array | null;
  /** An action term's state field or processed action, by term and field name. */
  actionTermField?: (term: string, field: string) => Float32Array | null;
  /** The model's compiled field values, for a `model` slot that reads the default. */
  modelDefaults?: ModelDefaultsSource | null;
};

export type SlotReaderOptions = {
  /** By *unprefixed* joint name: `policy.json` stores it in action, not entity, order. */
  jointBias?: (jointName: string) => number;
  /** A function because descriptors arrive with the policy, the reader with the runtime. */
  raycastSensors?: () => Record<string, RaycastSensorDescriptor>;
  /** The engine owns these, since it advances their history per substep. */
  contactSensors?: () => ContactSensorSet | null;
  /** The engine owns these too: rendering one needs its scene. */
  cameraSensors?: () => CameraSensorSet | null;
};

/** Whether an `Entity.data` field can be served — useful for a build-time check. */
export function isReadableEntityField(field: string): boolean {
  return field in FIELD_READERS;
}

/**
 * A raw `mjData` field as float32 — whole, or the `rows` of its element axis the graph
 * takes, each row `shape[2:]` wide (`cvel` is 6 per body) — or null when `mjData` has no
 * such array.
 */
export function readSimField(
  mjData: MjData,
  field: string,
  rows?: readonly number[],
  shape?: readonly number[],
): Float32Array | null {
  const value = (mjData as unknown as Record<string, unknown>)[field];
  if (typeof value === 'number') return new Float32Array([value]);
  if (!value || typeof (value as ArrayLike<number>).length !== 'number') return null;
  const array = value as ArrayLike<number>;
  // Only what reserveTermFeeds charged: the whole field at numel(shape[1:]), or rows.
  if (!rows) {
    const charged = (Array.isArray(shape) ? shape : []).slice(1).reduce((a, b) => a * b, 1);
    return Array.isArray(shape) && array.length === charged ? Float32Array.from(array) : null;
  }
  if (!Array.isArray(rows)) return null;
  const width = (shape ?? []).slice(2).reduce((a, b) => a * b, 1);
  const out = new Float32Array(rows.length * width);
  for (let i = 0; i < rows.length; i++) {
    const base = rows[i] * width;
    for (let k = 0; k < width; k++) out[i * width + k] = array[base + k] ?? 0;
  }
  return out;
}

/**
 * A named MuJoCo sensor's `sensordata` window. The build records mjlab's prefixed name
 * (`robot/imu_lin_vel`), while a plain-MJCF model has the bare one, so try both.
 */
export function sensorWindow(
  mjModel: MjModel,
  sensor: string,
): { adr: number; dim: number } | null {
  const names = decodeNames(mjModel, mjModel.nsensor, mjModel.name_sensoradr);
  let idx = names.indexOf(sensor);
  if (idx < 0) {
    const bare = unprefixed(sensor);
    idx = names.findIndex(name => name === bare || unprefixed(name) === bare);
  }
  if (idx < 0) return null;
  return { adr: mjModel.sensor_adr[idx], dim: mjModel.sensor_dim[idx] };
}

function isCommandStateSource(value: unknown): value is CommandStateSource {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as CommandStateSource).getStateField === 'function'
  );
}

/** `values` zero-padded or cut to the traced width: a UI command is sized by its own inputs. */
function toTracedWidth(values: Float32Array, slot: OnnxInputSlot): Float32Array {
  const shape = slot.shape;
  if (!shape || shape.length < 2) return values;
  const width = shape.slice(1).reduce((a, b) => a * b, 1);
  if (values.length === width) return values;
  const out = new Float32Array(width);
  out.set(values.subarray(0, Math.min(values.length, width)));
  return out;
}

function hasGetCommand(value: unknown): value is { getCommand(): ArrayLike<number> } {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as { getCommand?: unknown }).getCommand === 'function'
  );
}

/**
 * Build the `SlotReader` every ONNX-backed term reads its graph inputs through.
 *
 * `getContext` is called per read so a scene reload is picked up without rebuilding
 * the reader; the per-entity index is cached against the model it came from.
 */
export function createSlotReader(
  getContext: () => SlotReaderContext | null,
  options: SlotReaderOptions = {},
): SlotReader {
  let cachedModel: MjModel | null = null;
  const indices = new Map<string, EntityIndex>();
  // One per sensor, held for its frame resolution and ray buffers: ~200 rays a step.
  const casters = new Map<string, RaycastSensor | null>();

  /** Drop what the previous scene resolved: both maps hold its model indices. */
  const forModel = (mjModel: MjModel): void => {
    if (mjModel === cachedModel) return;
    cachedModel = mjModel;
    indices.clear();
    casters.clear();
  };

  const readRaycast = (
    sensor: string,
    field: string,
    context: SlotReaderContext,
  ): Float32Array | null => {
    const { mjModel, mjData, mujoco } = context;
    if (!mjModel || !mjData || !mujoco) return null;
    forModel(mjModel);
    if (!casters.has(sensor)) {
      const descriptor = options.raycastSensors?.()[sensor];
      if (!descriptor) {
        console.warn(
          `[slotReader] no raycast descriptor for sensor "${sensor}"; the build ` +
            'did not emit one.',
        );
      }
      casters.set(sensor, descriptor ? new RaycastSensor(mujoco, descriptor) : null);
    }
    const caster = casters.get(sensor);
    if (!caster) return null;
    if (!isRaycastField(field)) {
      console.warn(`[slotReader] raycast sensor "${sensor}" cannot serve "${field}".`);
      return null;
    }
    return caster.read(field, mjModel, mjData);
  };

  const indexFor = (mjModel: MjModel, entity: string | null | undefined): EntityIndex => {
    forModel(mjModel);
    const key = entity ?? '';
    let index = indices.get(key);
    if (!index) {
      index = buildEntityIndex(mjModel, entity, options.jointBias);
      indices.set(key, index);
    }
    return index;
  };

  return (slot: OnnxInputSlot): Float32Array | null => {
    const context = getContext();
    if (!context) return null;

    if (slot.command) {
      const term = context.commandManager?.getTerm(slot.command);
      const state = isCommandStateSource(term) ? term.getStateField(slot.field ?? '') : null;
      if (state) return state;
      // mjlab's `get_command()`, which every term answers, traced state or not.
      if (slot.field === 'command' && hasGetCommand(term)) {
        return toTracedWidth(Float32Array.from(term.getCommand()), slot);
      }
      return null;
    }

    if (slot.env) {
      if (slot.env !== 'episode_length_buf' || context.episodeLength === undefined) return null;
      return new Float32Array([context.episodeLength]);
    }

    if (slot.action) {
      if (slot.action !== 'action') return null;
      return context.lastActions?.() ?? null;
    }

    if (slot.action_term) {
      return context.actionTermField?.(slot.action_term, slot.field ?? '') ?? null;
    }

    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return null;

    if (slot.sim) return readSimField(mjData, slot.sim, slot.rows, slot.shape);

    if (slot.model) return readModelSlot(mjModel, slot, context.modelDefaults);

    if (slot.sensor) {
      if (slot.field) {
        // A structured sensor: no single `sensordata` window to fall through to, so
        // each kind is served by the module that knows its layout.
        const contacts = options.contactSensors?.();
        const contact = contacts?.read(slot.sensor, slot.field, mjModel, mjData, sensor =>
          sensorWindow(mjModel, sensor),
        );
        if (contact) return contact;
        const cameras = options.cameraSensors?.();
        if (cameras?.has(slot.sensor)) return cameras.read(slot.sensor, slot.field, mjModel);
        return readRaycast(slot.sensor, slot.field, context);
      }
      const window = sensorWindow(mjModel, slot.sensor);
      if (!window) return null;
      const out = new Float32Array(window.dim);
      for (let i = 0; i < window.dim; i++) out[i] = mjData.sensordata[window.adr + i] ?? 0;
      return out;
    }

    const read = slot.field ? FIELD_READERS[slot.field] : undefined;
    if (!read) return null;
    return read(indexFor(mjModel, slot.entity), mjData);
  };
}
