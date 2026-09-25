/**
 * Startup domain randomization that perturbs `mjModel` rather than `mjData`: no graph
 * needed, just draw, combine with the base, and write back once from the seeded PRNG.
 *
 * As mjlab's `_randomize_model_field` does, `add`/`scale` combine against the *compiled
 * default* so events on one axis do not accumulate, only targeted axes are written so
 * events on different axes compose, and a field the build flags with `set_const` needs
 * an `mj_setConst` afterwards.
 */

import type { SeededRng } from '../rng';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

export type DrOperation = 'abs' | 'add' | 'scale';
export type DrDistribution = 'uniform' | 'log_uniform' | 'gaussian';

export interface ModelFieldDrConfig {
  name: string;
  kind: 'model_field';
  /** `mjModel` field to perturb, e.g. `geom_friction`. */
  field: string;
  entity_type: 'geom' | 'body' | 'site';
  /** Model names of the elements to perturb — ids differ between build and browser. */
  entity_names: string[];
  /** Axis index → `[lo, hi]`. Only these axes are written. */
  axis_ranges: Record<string, [number, number]>;
  operation: DrOperation;
  distribution: DrDistribution;
  /** One draw shared by every element, rather than one each. */
  shared_random: boolean;
  /** Combine against the compiled default rather than the live value. */
  uses_defaults: boolean;
  set_const: boolean;
  /** `geom_size` only: redo `geom_rbound`/`geom_aabb`, as mjlab does in the same call. */
  recompute_bounds?: boolean;
}

/** `mjtGeom` values whose bounds follow from `geom_size` — mjlab's supported set. */
const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;
const GEOM_ELLIPSOID = 4;
const GEOM_CYLINDER = 5;
const GEOM_BOX = 6;

/**
 * The compiled field values, snapshotted on first touch, so a second `add`/`scale` event
 * on one axis offsets the compiled value rather than the first event's output.
 *
 * One per model, not per pass (ADR 0006 §9): an MDP switch re-runs `mode="startup"`
 * randomization, which must start from the compiled values rather than from what the
 * previous MDP left behind, so `restore()` puts every touched field back first.
 */
export class ModelFieldDefaults {
  private readonly snapshots = new Map<string, Float64Array>();

  constructor(private readonly mjModel: MjModel) {}

  /** The field as compiled. Snapshots it if this is the first read. */
  base(field: string): ArrayLike<number> | undefined {
    const cached = this.snapshots.get(field);
    if (cached) return cached;
    const live = (this.mjModel as unknown as Record<string, ArrayLike<number> | undefined>)[
      field
    ];
    if (!live) return undefined;
    const copy = Float64Array.from(live as ArrayLike<number>);
    this.snapshots.set(field, copy);
    return copy;
  }

  /**
   * Write every snapshotted field back to its compiled value; false when there was
   * nothing to write. A true return leaves the caller owing an `mj_setConst`.
   */
  restore(): boolean {
    if (this.snapshots.size === 0) return false;
    const model = this.mjModel as unknown as Record<string, { [index: number]: number } | undefined>;
    for (const [field, compiled] of this.snapshots) {
      const live = model[field];
      if (!live) continue;
      for (let i = 0; i < compiled.length; i++) live[i] = compiled[i];
    }
    return true;
  }
}

/** Whether an event config is a model-field randomization rather than a graph. */
export function isModelFieldDrConfig(config: unknown): config is ModelFieldDrConfig {
  return (
    typeof config === 'object' &&
    config !== null &&
    (config as { kind?: unknown }).kind === 'model_field' &&
    typeof (config as { field?: unknown }).field === 'string'
  );
}

function decodeNames(mjModel: MjModel, count: number, adr: ArrayLike<number>): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    const start = adr[i];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

function nameTable(mjModel: MjModel, entityType: ModelFieldDrConfig['entity_type']): string[] {
  if (entityType === 'geom') return decodeNames(mjModel, mjModel.ngeom, mjModel.name_geomadr);
  if (entityType === 'body') return decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr);
  return decodeNames(mjModel, mjModel.nsite, mjModel.name_siteadr);
}

/** One draw, by mjlab's distribution semantics. */
function draw(rng: SeededRng, distribution: DrDistribution, lo: number, hi: number): number {
  if (distribution === 'log_uniform') {
    // mjlab's `sample_log_uniform`: equal probability per decade, not per unit.
    return Math.exp(rng.uniform(Math.log(lo), Math.log(hi)));
  }
  if (distribution === 'gaussian') {
    // mjlab's `sample_gaussian(lo, hi)` reads the pair as (mean, std).
    return lo + hi * gaussianUnit(rng);
  }
  return rng.uniform(lo, hi);
}

/** Standard normal via Box–Muller, from the seeded stream. */
function gaussianUnit(rng: SeededRng): number {
  // `next()` is [0, 1); log(0) would be -Infinity, so nudge off zero.
  const u = 1 - rng.next();
  const v = rng.next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function combine(operation: DrOperation, base: number, random: number): number {
  if (operation === 'add') return base + random;
  if (operation === 'scale') return base * random;
  return random;
}

/** Apply one randomization, returning whether the caller now owes `mj_setConst`. */
export function applyModelFieldDr(
  mujoco: MainModule,
  mjModel: MjModel,
  mjData: MjData,
  config: ModelFieldDrConfig,
  rng: SeededRng,
  defaults: ModelFieldDefaults,
): boolean {
  const field = (mjModel as unknown as Record<string, ArrayLike<number> | undefined>)[
    config.field
  ];
  if (!field) {
    console.warn(`[modelFieldDr] "${config.name}": no mjModel.${config.field} to write.`);
    return false;
  }
  // `abs` overwrites, so its base is irrelevant; `add`/`scale` need the compiled value.
  const base = config.uses_defaults ? (defaults.base(config.field) ?? field) : field;
  const names = nameTable(mjModel, config.entity_type);
  const indices = config.entity_names.map(name => {
    const exact = names.indexOf(name);
    if (exact >= 0) return exact;
    const bare = name.slice(name.lastIndexOf('/') + 1);
    return names.findIndex(n => n === bare || n.endsWith(`/${bare}`));
  });
  if (indices.some(i => i < 0)) {
    console.warn(
      `[modelFieldDr] "${config.name}": some ${config.entity_type}s are not in this ` +
        'model; skipping.',
    );
    return false;
  }
  // Derived from the field's own length, so an unseen field still works.
  const count = { geom: mjModel.ngeom, body: mjModel.nbody, site: mjModel.nsite }[
    config.entity_type
  ];
  const stride = Math.max(1, Math.floor(field.length / count));
  const writable = field as unknown as { [index: number]: number };

  const axes = Object.keys(config.axis_ranges).map(Number).sort((a, b) => a - b);
  for (const axis of axes) {
    if (axis >= stride) {
      console.warn(
        `[modelFieldDr] "${config.name}": axis ${axis} is past ${config.field}'s ` +
          `width of ${stride}; skipping that axis.`,
      );
      continue;
    }
    const [lo, hi] = config.axis_ranges[String(axis)];
    // `shared_random`: one draw for the set, as mjlab's foot friction does.
    const shared = config.shared_random ? draw(rng, config.distribution, lo, hi) : 0;
    for (const index of indices) {
      const at = index * stride + axis;
      const random = config.shared_random ? shared : draw(rng, config.distribution, lo, hi);
      writable[at] = combine(config.operation, base[at], random);
    }
  }

  if (config.recompute_bounds) recomputeGeomBounds(mjModel, config.name, indices, defaults);

  if (config.set_const) {
    // Inertial fields feed precomputed constants, so without this it is half-applied.
    mujoco.mj_setConst(mjModel, mjData);
  }
  return true;
}

/**
 * `geom_rbound` and `geom_aabb` from the sizes just written, as
 * `dr.geom_size._recompute_geom_bounds` computes them — without it a grown geom keeps its
 * compiled bound and stops colliding at its own surface. `geom_aabb` is `(ngeom, 2, 3)`,
 * centre then half-size, and a primitive's centre stays at its origin.
 */
function recomputeGeomBounds(
  mjModel: MjModel,
  name: string,
  indices: number[],
  defaults: ModelFieldDefaults,
): void {
  const size = mjModel.geom_size as ArrayLike<number> | undefined;
  const types = mjModel.geom_type as ArrayLike<number> | undefined;
  // Snapshotted before they are written, so `restore()` covers the bounds as well as the
  // sizes they follow from; otherwise a size would be restored against a stale bound.
  defaults.base('geom_rbound');
  defaults.base('geom_aabb');
  const rbound = mjModel.geom_rbound as unknown as { [index: number]: number } | undefined;
  const aabb = mjModel.geom_aabb as unknown as { [index: number]: number } | undefined;
  if (!size || !types || !rbound || !aabb) {
    console.warn(`[modelFieldDr] "${name}": no geom bounds to recompute in this model.`);
    return;
  }
  for (const index of indices) {
    const s0 = size[index * 3];
    const s1 = size[index * 3 + 1];
    const s2 = size[index * 3 + 2];
    let bound: number;
    let half: [number, number, number];
    switch (types[index]) {
      case GEOM_SPHERE:
        bound = s0;
        half = [s0, s0, s0];
        break;
      case GEOM_CAPSULE:
        bound = s0 + s1;
        half = [s0, s0, s0 + s1];
        break;
      case GEOM_ELLIPSOID:
        bound = Math.max(s0, s1, s2);
        half = [s0, s1, s2];
        break;
      case GEOM_CYLINDER:
        bound = Math.sqrt(s0 * s0 + s1 * s1);
        half = [s0, s0, s1];
        break;
      case GEOM_BOX:
        bound = Math.sqrt(s0 * s0 + s1 * s1 + s2 * s2);
        half = [s0, s1, s2];
        break;
      default:
        // The build refuses these; this is the backstop.
        console.warn(
          `[modelFieldDr] "${name}": geom ${index} is type ${types[index]}, whose ` +
            'bounds do not follow from its size; leaving them as compiled.',
        );
        continue;
    }
    rbound[index] = bound;
    aabb[index * 6 + 3] = half[0];
    aabb[index * 6 + 4] = half[1];
    aabb[index * 6 + 5] = half[2];
  }
}
