/**
 * Shared pieces of the field readers: a row of an `mjData` array, rows over the
 * entity's elements, and mjlab's `compute_velocity_from_cvel` (`entity/data.py`).
 */
import type { EntityIndex } from '../indexing';

type MjData = import('mujoco').MjData;

/**
 * One `EntityData` field off `mjData`, flattened in mjlab's element order; null when the
 * entity cannot report it (no root body, say) rather than a plausible-looking zero.
 */
export type FieldReader = (index: EntityIndex, mjData: MjData) => Float32Array | null;

/** Row `i` of an array packed `width` per element. */
export function rowAt(source: ArrayLike<number>, i: number, width: number): Float32Array {
  const out = new Float32Array(width);
  const base = i * width;
  for (let k = 0; k < width; k++) out[k] = source[base + k] ?? 0;
  return out;
}

export function vec3At(source: ArrayLike<number>, i: number): Float32Array {
  return rowAt(source, i, 3);
}

/** Identity where the array has no such row, so a bad index rotates nothing. */
export function quatAt(source: ArrayLike<number>, i: number): Float32Array {
  const base = i * 4;
  return new Float32Array([
    source[base] ?? 1,
    source[base + 1] ?? 0,
    source[base + 2] ?? 0,
    source[base + 3] ?? 0,
  ]);
}

export function concat(...parts: ArrayLike<number>[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

export function gather(source: ArrayLike<number>, addresses: readonly number[]): Float32Array {
  const out = new Float32Array(addresses.length);
  for (let i = 0; i < addresses.length; i++) out[i] = source[addresses[i]] ?? 0;
  return out;
}

/**
 * mjlab's `compute_velocity_from_cvel`: MuJoCo's `cvel` packs angular then linear and
 * is about the subtree COM, so the linear part is moved to `pos`. Returns `lin ++ ang`.
 */
export function velocityFromCvel(
  pos: ArrayLike<number>,
  subtreeCom: ArrayLike<number>,
  cvel: ArrayLike<number>,
): Float32Array {
  const ax = cvel[0] ?? 0;
  const ay = cvel[1] ?? 0;
  const az = cvel[2] ?? 0;
  const ox = (subtreeCom[0] ?? 0) - (pos[0] ?? 0);
  const oy = (subtreeCom[1] ?? 0) - (pos[1] ?? 0);
  const oz = (subtreeCom[2] ?? 0) - (pos[2] ?? 0);
  return new Float32Array([
    (cvel[3] ?? 0) - (ay * oz - az * oy),
    (cvel[4] ?? 0) - (az * ox - ax * oz),
    (cvel[5] ?? 0) - (ax * oy - ay * ox),
    ax,
    ay,
    az,
  ]);
}

/** Wrap a reader needing a root body. Null, since indexing at -1 gives plausible zeros. */
export function rootField(
  read: (root: number, mjData: MjData, index: EntityIndex) => Float32Array,
): FieldReader {
  return (index, mjData) =>
    index.rootBodyId < 0 ? null : read(index.rootBodyId, mjData, index);
}

/**
 * One row per element, concatenated: mjlab's `(num_envs, n, k)` flattened. `read` also
 * gets the element's position in the entity's list, which is how the per-element
 * constants `EntityIndex` carries (`bodyIquat`, `siteBodyIds`) are aligned.
 */
export function perElement(
  ids: (index: EntityIndex) => readonly number[],
  width: number,
  read: (id: number, at: number, mjData: MjData, index: EntityIndex) => ArrayLike<number>,
): FieldReader {
  return (index, mjData) => {
    const list = ids(index);
    const out = new Float32Array(list.length * width);
    for (let i = 0; i < list.length; i++) out.set(read(list[i], i, mjData, index), i * width);
    return out;
  };
}

/** Columns `[from, to)` of every `width`-wide row: mjlab's `[..., from:to]` accessors. */
export function columns(read: FieldReader, width: number, from: number, to: number): FieldReader {
  const span = to - from;
  return (index, mjData) => {
    const whole = read(index, mjData);
    if (!whole) return null;
    const rows = whole.length / width;
    const out = new Float32Array(rows * span);
    for (let r = 0; r < rows; r++) {
      out.set(whole.subarray(r * width + from, r * width + to), r * span);
    }
    return out;
  };
}
