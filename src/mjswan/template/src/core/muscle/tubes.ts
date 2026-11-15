// Ported from MyoSuite (myosuite/viz/muscle_tubes.py, blender_render.py), Copyright (c)
// MyoSuite Authors, Apache-2.0; modified for mjswan. See NOTICE.
import type { MainModule, MjModel } from 'mujoco';

/** MyoSuite's muscle colours, linear RGB. */
export const MUSCLE_RELAXED = [0.5, 0.16, 0.15] as const;
export const MUSCLE_ACTIVE = [0.62, 0.012, 0.02] as const;
export const MUSCLE_UNIFORM = [0.42, 0.11, 0.1] as const;
export const PATH_RELAXED = [0.05, 0.05, 0.05] as const;
export const PATH_ACTIVE = [0.95, 0.3, 0.3] as const;

const CROSS_SECTION_STRESS = 1.6e6; // N/m^2
const BELLY_FRACTION = [0.4, 0.85] as const;
const SLENDERNESS = 0.16;
const TENDON_RADIUS_RATIO = 0.2;
export const PATH_POINTS = 32;
// Adult whole-muscle volumes (m^3), shared among a muscle's parts by force.
const REFERENCE_VOLUMES: Record<string, number> = {
  glmax: 849e-6, glmed: 323e-6, addmag: 559e-6, vaslat: 514e-6, vasmed: 424e-6, vasint: 375e-6,
  recfem: 237e-6, semimem: 245e-6, semiten: 199e-6, bflh: 192e-6, tfl: 69e-6, soleus: 476e-6,
  gasmed: 258e-6, gaslat: 141e-6, DELT: 275e-6, PECM: 210e-6, LAT: 190e-6,
};
// MuJoCo's `wrap_obj` for the (0, 0, 0) point that starts each branch after a pulley.
const PULLEY_MARKER = -2;

/**
 * The muscles MyoSuite draws: muscle actuators on a spatial tendon of two or more
 * elements that starts at a site, in model order.
 */
export function muscleActuators(mujoco: MainModule, model: MjModel): number[] {
  const muscle = mujoco.mjtGain.mjGAIN_MUSCLE.value;
  const tendon = mujoco.mjtTrn.mjTRN_TENDON.value;
  const site = mujoco.mjtWrap.mjWRAP_SITE.value;
  // Each embind getter allocates a view, so read every array once.
  const gaintype = model.actuator_gaintype;
  const trntype = model.actuator_trntype;
  const trnid = model.actuator_trnid;
  const tendonNum = model.tendon_num;
  const tendonAdr = model.tendon_adr;
  const wrapType = model.wrap_type;
  const actuators: number[] = [];
  for (let a = 0; a < model.nu; a++) {
    if (gaintype[a] !== muscle || trntype[a] !== tendon) continue;
    const t = trnid[2 * a];
    if (tendonNum[t] > 1 && wrapType[tendonAdr[t]] === site) actuators.push(a);
  }
  return actuators;
}

/** Branches of a tendon path: one more than its pulleys. */
export function branchCount(mujoco: MainModule, model: MjModel, tendon: number): number {
  const pulley = mujoco.mjtWrap.mjWRAP_PULLEY.value;
  const adr = model.tendon_adr[tendon];
  const wrapType = model.wrap_type;
  let branches = 1;
  for (let w = adr; w < adr + model.tendon_num[tendon]; w++) if (wrapType[w] === pulley) branches++;
  return branches;
}

/** MyoSuite's `_muscle_colour` into `out` at `offset`: a path brightens by the activation's fourth root. */
export function muscleColour(activation: number, paths: boolean, out: Float32Array, offset: number): void {
  const clamped = Math.min(Math.max(activation, 0), 1);
  const a = paths ? clamped ** 0.25 : clamped;
  const [relaxed, active] = paths ? [PATH_RELAXED, PATH_ACTIVE] : [MUSCLE_RELAXED, MUSCLE_ACTIVE];
  for (let c = 0; c < 3; c++) out[offset + c] = relaxed[c] + (active[c] - relaxed[c]) * a;
}

/** A muscle's belly as fixed at its reference pose: the volume then holds as the path changes. */
export interface MuscleShape {
  /** Belly radius over its peak at each of the `PATH_POINTS` along the first branch. */
  profile: Float64Array;
  /** Mean square of `profile`, by MyoSuite's trapezoid. */
  meanSquare: number;
  volume: number;
}

function referenceGroup(name: string): [string, string] {
  const tail = name.slice(-2);
  const side = tail === '_r' || tail === '_l' ? tail : '';
  const stem = name.slice(0, name.length - side.length);
  for (const key of Object.keys(REFERENCE_VOLUMES)) {
    if (new RegExp(`^${key}(\\d*|Dist|Isch|Mid|Prox)$`).test(stem)) return [key, side];
  }
  return [stem, side];
}

function radiusProfile(fraction: number): Float64Array {
  const profile = new Float64Array(PATH_POINTS);
  for (let i = 0; i < PATH_POINTS; i++) {
    const s = i === PATH_POINTS - 1 ? 1 : i * (1 / (PATH_POINTS - 1));
    const u = Math.min(Math.max((s - (1 - fraction) / 4) / fraction, 0), 1);
    profile[i] = TENDON_RADIUS_RATIO + (1 - TENDON_RADIUS_RATIO) * Math.sin(Math.PI * u);
  }
  return profile;
}

function meanSquare(profile: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < profile.length - 1; i++) sum += (profile[i] ** 2 + profile[i + 1] ** 2) / 2;
  return sum / (profile.length - 1);
}

const bellyRadius = (volume: number, length: number, ms: number) =>
  Math.sqrt(volume / (Math.PI * Math.max(length, 1e-6) * ms));

/**
 * MyoSuite's `muscle_shape`, and the volume `_write_muscle_tubes` keeps, from each muscle's
 * first-branch length at the reference pose.
 */
export function muscleShapes(
  model: MjModel,
  actuators: readonly number[],
  lengths: ArrayLike<number>,
): MuscleShape[] {
  const n = actuators.length;
  // Names first: each accessor allocates on the WASM heap, which can detach the views below.
  const groups = actuators.map((a) => referenceGroup(model.actuator(a).name));
  const gainprm = model.actuator_gainprm;
  const stride = gainprm.length / model.nu;
  const acc0 = model.actuator_acc0;
  const lengthrange = model.actuator_lengthrange;
  const force = new Float64Array(n);
  const l0 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const g = actuators[i] * stride;
    // A negative force means MuJoCo derives F_max as scale / acc0.
    force[i] = gainprm[g + 2] > 0 ? gainprm[g + 2] : gainprm[g + 3] / Math.max(acc0[actuators[i]], 1e-10);
    const a = 2 * actuators[i];
    l0[i] = (lengthrange[a + 1] - lengthrange[a]) / Math.max(gainprm[g + 1] - gainprm[g], 1e-6);
  }
  return actuators.map((_, i) => {
    const [key, side] = groups[i];
    let share = 0;
    for (let j = 0; j < n; j++) if (groups[j][0] === key && groups[j][1] === side) share += force[j];
    const target = ((REFERENCE_VOLUMES[key] ?? 0) * force[i]) / Math.max(share, 1e-9);
    const radius =
      target > 0
        ? Math.sqrt(target / (Math.PI * Math.max(l0[i], 1e-3)))
        : Math.sqrt(force[i] / (Math.PI * CROSS_SECTION_STRESS));
    const belly = Math.max(l0[i], radius / SLENDERNESS);
    const length = lengths[i];
    const fraction = Math.min(Math.max(belly / Math.max(length, 1e-6), BELLY_FRACTION[0]), BELLY_FRACTION[1]);
    const profile = radiusProfile(fraction);
    const ms = meanSquare(profile);
    const cap = SLENDERNESS * fraction * length;
    // Referenced muscles keep their volume whatever belly length they end up with.
    const peak = target > 0 ? Math.min(bellyRadius(target, length, ms), cap) : Math.min(radius, cap);
    return { profile, meanSquare: ms, volume: Math.PI * (peak * peak) * length * ms };
  });
}

/** Length of each muscle's first branch, MyoSuite's belly path, from the forward that left these arrays. */
export function bellyLengths(
  wrapXpos: Float64Array,
  wrapObj: Int32Array,
  wrapadr: Int32Array,
  wrapnum: Int32Array,
  tendons: Int32Array,
): Float64Array {
  return Float64Array.from(tendons, (t) => {
    let end = wrapadr[t];
    while (end < wrapadr[t] + wrapnum[t] && wrapObj[end] !== PULLEY_MARKER) end++;
    let length = 0;
    for (let p = wrapadr[t] + 1; p < end; p++) length += distance(wrapXpos, p - 1, p);
    return length;
  });
}

function distance(points: Float64Array, a: number, b: number): number {
  const dx = points[3 * b] - points[3 * a];
  const dy = points[3 * b + 1] - points[3 * a + 1];
  const dz = points[3 * b + 2] - points[3 * a + 2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Scratch for one branch at a time: tubes are written one after another.
const centres = new Float64Array(3 * PATH_POINTS);
const radii = new Float64Array(PATH_POINTS);
const tangent = new Float64Array(3 * PATH_POINTS);
const normal = new Float64Array(3 * PATH_POINTS);
let arc = new Float64Array(64);

/** MyoSuite's `resample_path` of points `start` to `start + n`: `np.interp` over arc length. */
function resamplePath(points: Float64Array, start: number, n: number): number {
  if (n > arc.length) arc = new Float64Array(n);
  arc[0] = 0;
  for (let i = 1; i < n; i++) arc[i] = arc[i - 1] + distance(points, start + i - 1, start + i);
  const total = arc[n - 1];
  const step = total / (PATH_POINTS - 1);
  let j = 0;
  for (let k = 0; k < PATH_POINTS; k++) {
    const x = k === PATH_POINTS - 1 ? total : k * step;
    while (j + 1 < n && arc[j + 1] <= x) j++;
    for (let c = 0; c < 3; c++) {
      const p = points[3 * (start + j) + c];
      centres[3 * k + c] =
        j >= n - 1 || arc[j] === x
          ? p
          : ((points[3 * (start + j + 1) + c] - p) / (arc[j + 1] - arc[j])) * (x - arc[j]) + p;
    }
  }
  return total;
}

/**
 * MyoSuite's `tube_mesh` around `centres` with `radii`: rings of `k` points on a
 * parallel-transported frame, then the two end centres. Writes positions, and the ring's
 * outward direction as each normal, from `vertex` on; returns the next vertex.
 */
function tubeMesh(k: number, positions: Float32Array | Float64Array, normals: Float32Array | Float64Array | null, vertex: number): number {
  const n = PATH_POINTS;
  for (let i = 0; i < n; i++) {
    const a = i === 0 ? 0 : i - 1;
    const b = i === n - 1 ? n - 1 : i + 1;
    const div = b - a === 2 ? 2 : 1;
    const x = (centres[3 * b] - centres[3 * a]) / div;
    const y = (centres[3 * b + 1] - centres[3 * a + 1]) / div;
    const z = (centres[3 * b + 2] - centres[3 * a + 2]) / div;
    const len = Math.max(Math.sqrt(x * x + y * y + z * z), 1e-12);
    tangent[3 * i] = x / len;
    tangent[3 * i + 1] = y / len;
    tangent[3 * i + 2] = z / len;
  }
  const [rx, ry, rz] = Math.abs(tangent[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
  const cx = tangent[1] * rz - tangent[2] * ry;
  const cy = tangent[2] * rx - tangent[0] * rz;
  const cz = tangent[0] * ry - tangent[1] * rx;
  normal[0] = cy * tangent[2] - cz * tangent[1];
  normal[1] = cz * tangent[0] - cx * tangent[2];
  normal[2] = cx * tangent[1] - cy * tangent[0];
  for (let i = 1; i < n; i++) {
    const dot =
      normal[3 * i - 3] * tangent[3 * i] + normal[3 * i - 2] * tangent[3 * i + 1] + normal[3 * i - 1] * tangent[3 * i + 2];
    for (let c = 0; c < 3; c++) normal[3 * i + c] = normal[3 * i - 3 + c] - dot * tangent[3 * i + c];
  }
  for (let i = 0; i < n; i++) {
    const nx = normal[3 * i];
    const ny = normal[3 * i + 1];
    const nz = normal[3 * i + 2];
    const len = Math.max(Math.sqrt(nx * nx + ny * ny + nz * nz), 1e-12);
    for (let c = 0; c < 3; c++) normal[3 * i + c] /= len;
  }
  const step = (2 * Math.PI) / k;
  let o = 3 * vertex;
  for (let i = 0; i < n; i++) {
    const tx = tangent[3 * i], ty = tangent[3 * i + 1], tz = tangent[3 * i + 2];
    const nx = normal[3 * i], ny = normal[3 * i + 1], nz = normal[3 * i + 2];
    const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
    for (let j = 0; j < k; j++) {
      const cos = Math.cos(j * step);
      const sin = Math.sin(j * step);
      const dx = cos * nx + sin * bx, dy = cos * ny + sin * by, dz = cos * nz + sin * bz;
      positions[o] = centres[3 * i] + radii[i] * dx;
      positions[o + 1] = centres[3 * i + 1] + radii[i] * dy;
      positions[o + 2] = centres[3 * i + 2] + radii[i] * dz;
      if (normals) {
        normals[o] = dx;
        normals[o + 1] = dy;
        normals[o + 2] = dz;
      }
      o += 3;
    }
  }
  for (const [i, sign] of [[0, -1], [n - 1, 1]]) {
    for (let c = 0; c < 3; c++) {
      positions[o + c] = centres[3 * i + c];
      if (normals) normals[o + c] = sign * tangent[3 * i + c];
    }
    o += 3;
  }
  return o / 3;
}

/** Vertices of one tube with rings of `k` points. */
export const tubeVertices = (k: number) => PATH_POINTS * k + 2;

/**
 * One muscle's tubes, as MyoSuite's `_write_muscle_tubes` draws them: each pulley branch
 * resampled to `PATH_POINTS`, the first carrying the belly at the volume `shape` keeps and
 * the rest as thick as the belly's tendon ends, every radius times `scale`. Writes at most
 * `branches` tubes from `vertex` on; returns the next vertex.
 */
export function muscleTube(
  wrapXpos: Float64Array,
  wrapObj: Int32Array,
  adr: number,
  num: number,
  branches: number,
  shape: MuscleShape,
  scale: number,
  k: number,
  positions: Float32Array | Float64Array,
  normals: Float32Array | Float64Array | null,
  vertex: number,
): number {
  let peak = 0;
  let branch = 0;
  let start = adr;
  for (let p = adr; p <= adr + num && branch < branches; p++) {
    if (p < adr + num && wrapObj[p] !== PULLEY_MARKER) continue;
    const length = resamplePath(wrapXpos, start, p - start);
    if (branch === 0) {
      peak = bellyRadius(shape.volume, length, shape.meanSquare) * scale;
      for (let i = 0; i < PATH_POINTS; i++) radii[i] = peak * shape.profile[i];
    } else {
      radii.fill(peak * TENDON_RADIUS_RATIO);
    }
    vertex = tubeMesh(k, positions, normals, vertex);
    branch++;
    start = p + 1;
  }
  return vertex;
}

/** Triangles of `tubes` tubes with rings of `k` points: MyoSuite's faces, each quad split. */
export function tubeIndices(tubes: number, k: number): Uint32Array {
  const n = PATH_POINTS;
  const perTube = 6 * (n - 1) * k + 6 * k;
  const indices = new Uint32Array(tubes * perTube);
  let o = 0;
  for (let tube = 0; tube < tubes; tube++) {
    const base = tube * tubeVertices(k);
    const first = base + n * k;
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < k; j++) {
        const a = base + i * k + j, b = base + i * k + ((j + 1) % k);
        const c = base + (i + 1) * k + ((j + 1) % k), d = base + (i + 1) * k + j;
        indices.set([a, b, c, a, c, d], o);
        o += 6;
      }
    }
    for (let j = 0; j < k; j++) {
      indices.set([first, base + ((j + 1) % k), base + j], o);
      indices.set([first + 1, base + (n - 1) * k + j, base + (n - 1) * k + ((j + 1) % k)], o + 3);
      o += 6;
    }
  }
  return indices;
}
