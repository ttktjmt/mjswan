/**
 * Serves a `ContactSensor`'s fields to the slot reader (mjlab's
 * `_extract_sensor_data` / `_update_history`).
 *
 * No contact physics here: the values are MuJoCo's own `sensordata` and the build ships
 * the layout. Only the rolling history is ours, advancing per *physics substep* as
 * mjlab's `scene.update(dt=physics_dt)` does — hence `advance()` from the step loop.
 */

import { BufferBudget } from '../policy/bufferBudget';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

/** One field's `sensordata` windows, in mjlab's primary-major order. */
export interface ContactFieldDescriptor {
  /** MuJoCo sensor names, one per primary; each window packs `num_slots * dim`. */
  sensors: string[];
  dim: number;
}

/** One `ContactSensor`'s layout, from the build. */
export interface ContactSensorDescriptor {
  kind: 'contact';
  num_slots: number;
  history_length: number;
  /** Fields with a history buffer — a subset of `fields`. */
  history_fields: string[];
  fields: Record<string, ContactFieldDescriptor>;
}

type Window = { adr: number; dim: number };
type WindowLookup = (sensor: string) => Window | null;

/** One sensor: its `sensordata` windows and its history buffers. */
class ContactSensorReader {
  private readonly descriptor: ContactSensorDescriptor;
  private readonly windows = new Map<string, Window[] | null>();
  /** Per history field, `[N][H][dim]` flattened; index `h = 0` is the newest. */
  private readonly history = new Map<string, Float32Array>();
  private model: MjModel | null = null;

  constructor(descriptor: ContactSensorDescriptor) {
    this.descriptor = descriptor;
  }

  /** Current value of `field`, flattened `[N, dim]`, or null if the model lacks it. */
  current(field: string, mjModel: MjModel, mjData: MjData, lookup: WindowLookup): Float32Array | null {
    const spec = this.descriptor.fields[field];
    if (!spec) return null;
    const windows = this.windowsFor(field, mjModel, lookup);
    if (!windows) return null;
    const out = new Float32Array(this.contactCount(field) * spec.dim);
    let at = 0;
    for (const window of windows) {
      for (let i = 0; i < window.dim; i++) out[at++] = mjData.sensordata[window.adr + i] ?? 0;
    }
    return out;
  }

  /** Buffered value of `field`, flattened `[N, H, dim]`, or null if it has no buffer. */
  historyOf(field: string): Float32Array | null {
    return this.history.get(field) ?? null;
  }

  /** Roll every buffer by one and write the current reading at index 0. */
  advance(mjModel: MjModel, mjData: MjData, lookup: WindowLookup): void {
    const h = this.descriptor.history_length;
    if (h <= 0) return;
    for (const field of this.descriptor.history_fields) {
      const spec = this.descriptor.fields[field];
      if (!spec) continue;
      const current = this.current(field, mjModel, mjData, lookup);
      if (!current) continue;
      const n = this.contactCount(field);
      let buffer = this.history.get(field);
      if (!buffer || buffer.length !== n * h * spec.dim) {
        buffer = new Float32Array(n * h * spec.dim);
        this.history.set(field, buffer);
      }
      for (let contact = 0; contact < n; contact++) {
        const row = contact * h * spec.dim;
        // Backwards, so a step reads the slot it has not overwritten yet.
        for (let step = h - 1; step > 0; step--) {
          buffer.copyWithin(row + step * spec.dim, row + (step - 1) * spec.dim, row + step * spec.dim);
        }
        for (let d = 0; d < spec.dim; d++) buffer[row + d] = current[contact * spec.dim + d] ?? 0;
      }
    }
  }

  /** Zero the buffers, as mjlab's `reset` does per env. */
  reset(): void {
    for (const buffer of this.history.values()) buffer.fill(0);
  }

  /** `N` = primaries × slots. */
  private contactCount(field: string): number {
    const spec = this.descriptor.fields[field];
    return (spec?.sensors.length ?? 0) * this.descriptor.num_slots;
  }

  private windowsFor(field: string, mjModel: MjModel, lookup: WindowLookup): Window[] | null {
    if (mjModel !== this.model) {
      this.model = mjModel;
      this.windows.clear();
    }
    if (!this.windows.has(field)) {
      const resolved: Window[] = [];
      for (const name of this.descriptor.fields[field]?.sensors ?? []) {
        const window = lookup(name);
        if (!window) {
          console.warn(
            `[contactSensor] the scene has no MuJoCo sensor "${name}"; the build and ` +
              'the model disagree.',
          );
          this.windows.set(field, null);
          return null;
        }
        resolved.push(window);
      }
      this.windows.set(field, resolved);
    }
    return this.windows.get(field) ?? null;
  }
}

/** The scene's contact sensors, so the engine can advance them all once per substep. */
export class ContactSensorSet {
  private readonly readers = new Map<string, ContactSensorReader>();

  constructor(descriptors: Record<string, ContactSensorDescriptor> = {}, budget = new BufferBudget()) {
    for (const [name, descriptor] of Object.entries(descriptors)) {
      // advance() keeps n * h frames of dim per history field and rolls them every substep.
      if (descriptor.history_length > 0) {
        for (const field of descriptor.history_fields ?? []) {
          const spec = descriptor.fields[field];
          if (!spec) continue;
          const n = (spec.sensors?.length ?? 0) * descriptor.num_slots;
          budget.reserve(n * descriptor.history_length, spec.dim, `contact sensor "${name}" ${field} history`);
        }
      }
      this.readers.set(name, new ContactSensorReader(descriptor));
    }
  }

  get size(): number {
    return this.readers.size;
  }

  /**
   * One field: `force` reads `sensordata`, `force_history` the buffer. Null when this
   * set does not own the sensor or cannot serve the field — the caller's cue to warn.
   */
  read(
    sensor: string,
    field: string,
    mjModel: MjModel,
    mjData: MjData,
    lookup: WindowLookup,
  ): Float32Array | null {
    const reader = this.readers.get(sensor);
    if (!reader) return null;
    const base = field.endsWith('_history') ? field.slice(0, -'_history'.length) : null;
    return base === null
      ? reader.current(field, mjModel, mjData, lookup)
      : reader.historyOf(base);
  }

  advance(mjModel: MjModel, mjData: MjData, lookup: WindowLookup): void {
    for (const reader of this.readers.values()) reader.advance(mjModel, mjData, lookup);
  }

  reset(): void {
    for (const reader of this.readers.values()) reader.reset();
  }
}
