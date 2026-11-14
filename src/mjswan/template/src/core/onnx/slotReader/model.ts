/**
 * `mjModel` fields as a traced event reads and writes them: by element *name*, since the
 * build's trace env numbers elements apart from the browser's model, and a field's
 * row width from its own length, as MuJoCo lays each one out `(count, width)`.
 */

import { decodeNames, unprefixed } from './indexing';

type MjModel = import('mujoco').MjModel;

/** The elements of a model field a slot or write target names, as the build emits them. */
export interface ModelElements {
  /** `body`, `geom`, `site`, `joint`, `dof`, `qpos`, `actuator`, `tendon`, `camera`,
   * `light`, `material`, `texture`, `pair` or `equality`. */
  element?: string;
  /** A `dof` or `qpos` entry names its joint; `offsets` give the address within it. */
  names?: string[];
  offsets?: number[];
}

/** The compiled field values, snapshotted on first read: mjlab's `get_default_field`. */
export interface ModelDefaultsSource {
  base(field: string): ArrayLike<number> | undefined;
}

type CountKey = 'nbody' | 'ngeom' | 'nsite' | 'njnt' | 'nv' | 'nq' | 'nu' | 'ntendon' | 'ncam'
  | 'nlight' | 'nmat' | 'ntex' | 'npair' | 'neq';

/** Per element kind: its `mjModel` count and name addresses; a `dof` or `qpos` is unnamed. */
const ELEMENTS: Record<string, { count: CountKey; nameAdr?: keyof MjModel }> = {
  body: { count: 'nbody', nameAdr: 'name_bodyadr' },
  geom: { count: 'ngeom', nameAdr: 'name_geomadr' },
  site: { count: 'nsite', nameAdr: 'name_siteadr' },
  joint: { count: 'njnt', nameAdr: 'name_jntadr' },
  dof: { count: 'nv' },
  qpos: { count: 'nq' },
  actuator: { count: 'nu', nameAdr: 'name_actuatoradr' },
  tendon: { count: 'ntendon', nameAdr: 'name_tendonadr' },
  camera: { count: 'ncam', nameAdr: 'name_camadr' },
  light: { count: 'nlight', nameAdr: 'name_lightadr' },
  material: { count: 'nmat', nameAdr: 'name_matadr' },
  texture: { count: 'ntex', nameAdr: 'name_texadr' },
  pair: { count: 'npair', nameAdr: 'name_pairadr' },
  equality: { count: 'neq', nameAdr: 'name_eqadr' },
};

/** `mjModel.<field>`, or undefined when this build of MuJoCo has no such array. */
export function modelField(mjModel: MjModel, field: string): ArrayLike<number> | undefined {
  const value = (mjModel as unknown as Record<string, unknown>)[field];
  if (!value || typeof (value as ArrayLike<number>).length !== 'number') return undefined;
  return value as ArrayLike<number>;
}

/** `mjModel.<field>` if it is a float parameter: integer arrays are ids MuJoCo indexes memory with. */
export function writableModelField(mjModel: MjModel, field: string): Float64Array | Float32Array | undefined {
  const value = (mjModel as unknown as Record<string, unknown>)[field];
  return value instanceof Float64Array || value instanceof Float32Array ? value : undefined;
}

/**
 * Where `name` is in `names`: exactly, else by its unprefixed name, since a plain scene's
 * trace env namespaces its elements `robot/…` while the browser's model does not.
 */
function findName(names: readonly string[], name: string): number {
  const exact = names.indexOf(name);
  if (exact >= 0) return exact;
  const bare = unprefixed(name);
  return names.findIndex(candidate => unprefixed(candidate) === bare);
}

/** The field rows `elements` name, or null when one is not in this model. */
export function modelRows(mjModel: MjModel, elements: ModelElements): number[] | null {
  const kind = ELEMENTS[elements.element ?? ''];
  const names = elements.names ?? [];
  if (!kind) return null;
  let rows: number[];
  if (elements.element === 'dof' || elements.element === 'qpos') {
    const joints = decodeNames(mjModel, mjModel.njnt, mjModel.name_jntadr);
    const starts = elements.element === 'dof' ? mjModel.jnt_dofadr : mjModel.jnt_qposadr;
    rows = names.map((name, i) => {
      const joint = findName(joints, name);
      return joint < 0 ? -1 : starts[joint] + (elements.offsets?.[i] ?? 0);
    });
  } else {
    const adr = mjModel[kind.nameAdr as keyof MjModel] as ArrayLike<number>;
    const table = decodeNames(mjModel, mjModel[kind.count] as number, adr);
    rows = names.map(name => findName(table, name));
  }
  return rows.some(row => row < 0) ? null : rows;
}

/** How many numbers one element of `values` holds, or null when they do not divide. */
export function modelRowWidth(
  mjModel: MjModel,
  values: ArrayLike<number>,
  element: string | undefined,
): number | null {
  const kind = ELEMENTS[element ?? ''];
  const count = kind ? (mjModel[kind.count] as number) : 0;
  if (!count || values.length % count !== 0) return null;
  return values.length / count;
}

/**
 * A `model` slot: the rows of the elements it names, live or as compiled, each as wide
 * as the traced `shape` says. Null for anything this model cannot serve that way.
 */
export function readModelSlot(
  mjModel: MjModel,
  slot: ModelElements & { model?: string; default?: boolean; shape?: number[] },
  defaults?: ModelDefaultsSource | null,
): Float32Array | null {
  const field = slot.model ?? '';
  const values = slot.default ? defaults?.base(field) : modelField(mjModel, field);
  if (!values) return null;
  const rows = modelRows(mjModel, slot);
  const width = modelRowWidth(mjModel, values, slot.element);
  if (!rows || width === null) return null;
  const traced = (slot.shape ?? []).slice(2).reduce((a, b) => a * b, 1);
  // What reserveTermFeeds charged: shape[1] rows of shape[2:].
  if (traced !== width || slot.shape?.[1] !== rows.length) {
    console.warn(
      `[slotReader] mjModel.${field} is ${rows.length} x ${width} for these ${slot.element}s, ` +
        `but the graph was traced with ${JSON.stringify(slot.shape)}.`,
    );
    return null;
  }
  const out = new Float32Array(rows.length * width);
  for (let i = 0; i < rows.length; i++) {
    for (let k = 0; k < width; k++) out[i * width + k] = values[rows[i] * width + k];
  }
  return out;
}
