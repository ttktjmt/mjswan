/**
 * Reproduces mjlab's `RayCastSensor._compute_data` with `mj_ray`. A structured sensor
 * has no `sensordata` window to read, so the rays are cast here; the term's own
 * arithmetic stays in the traced graph.
 *
 * The build ships the generated ray offsets, so every mjlab pattern works here.
 *
 * `normals_w` is not produced — the emscripten binding does not marshal `mj_ray`'s
 * output pointers back, and no traced term reads normals.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

export type RayAlignment = 'base' | 'yaw' | 'world';

export interface RaycastFrame {
  type: 'body' | 'site' | 'geom';
  /** Model name, since ids differ between the build model and this one. */
  name: string;
}

/** Everything needed to reproduce one sensor's readings, from the build. */
export interface RaycastSensorDescriptor {
  kind: 'raycast';
  /** `[N][3]` ray origin offsets in frame-local coordinates. */
  local_offsets: number[][];
  /** `[N][3]` ray directions in frame-local coordinates. */
  local_directions: number[][];
  frames: RaycastFrame[];
  ray_alignment: RayAlignment;
  max_distance: number;
  exclude_parent_body: boolean;
  /** Geom groups the rays may hit; null/absent means all. A terrain scan is `[0]`. */
  include_geom_groups?: number[] | null;
}

/** Fields of `RayCastData` this module can serve, by the tracer's slot names. */
export type RaycastField =
  | 'distances'
  | 'hit_pos_w'
  | 'frame_pos_w'
  | 'frame_quat_w'
  | 'pos_w'
  | 'quat_w';

const RAYCAST_FIELDS: ReadonlySet<string> = new Set<RaycastField>([
  'distances',
  'hit_pos_w',
  'frame_pos_w',
  'frame_quat_w',
  'pos_w',
  'quat_w',
]);

export function isRaycastField(field: string): boolean {
  return RAYCAST_FIELDS.has(field);
}

/** One frame: its model index, the body to exclude, and its live world pose. */
type FramePose = {
  /** Index into the body/site/geom tables, resolved once per model. */
  index: number;
  /** Parent body, excluded from the cast so a ray cannot hit its own frame. */
  bodyId: number;
  pos: Float64Array;
  /** Row-major 3x3 world rotation. */
  mat: Float64Array;
};

/**
 * mjlab's `_extract_yaw_rotation`: project the x-axis onto the world xy-plane, falling
 * back to the y-axis below mjlab's 0.1 threshold where that is degenerate.
 */
function yawRotation(mat: Float64Array): Float64Array {
  // Column-major access: mat is row-major 3x3, so column j is mat[j], mat[3+j], mat[6+j].
  const xAxis = [mat[0], mat[3], mat[6]];
  let px = xAxis[0];
  let py = xAxis[1];
  if (Math.hypot(px, py) < 0.1) {
    const yAxis = [mat[1], mat[4], mat[7]];
    // The y-axis leads the fallback frame, so its projection becomes the new x.
    px = yAxis[1];
    py = -yAxis[0];
  }
  const norm = Math.hypot(px, py) || 1;
  const c = px / norm;
  const s = py / norm;
  // Rotation about +z by atan2(s, c), row-major.
  return Float64Array.from([c, -s, 0, s, c, 0, 0, 0, 1]);
}

function alignmentRotation(mat: Float64Array, alignment: RayAlignment): Float64Array {
  if (alignment === 'world') return Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  if (alignment === 'yaw') return yawRotation(mat);
  return mat;
}

/** `R · v` for a row-major 3x3. */
function apply(mat: Float64Array, v: readonly number[]): [number, number, number] {
  return [
    mat[0] * v[0] + mat[1] * v[1] + mat[2] * v[2],
    mat[3] * v[0] + mat[4] * v[1] + mat[5] * v[2],
    mat[6] * v[0] + mat[7] * v[1] + mat[8] * v[2],
  ];
}

/** Quaternion (w, x, y, z) from a row-major 3x3, matching mjlab's `quat_from_matrix`. */
function quatFromMatrix(m: Float64Array): [number, number, number, number] {
  const trace = m[0] + m[4] + m[8];
  if (trace > 0) {
    const s = Math.sqrt(trace + 1.0) * 2;
    return [0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s];
  }
  if (m[0] > m[4] && m[0] > m[8]) {
    const s = Math.sqrt(1.0 + m[0] - m[4] - m[8]) * 2;
    return [(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s];
  }
  if (m[4] > m[8]) {
    const s = Math.sqrt(1.0 + m[4] - m[0] - m[8]) * 2;
    return [(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s];
  }
  const s = Math.sqrt(1.0 + m[8] - m[0] - m[4]) * 2;
  return [(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s];
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

/** mjlab's `include_geom_groups` as the mask `mj_ray` takes, or null for every group. */
export function geomGroupMask(groups: number[] | null | undefined): Uint8Array | null {
  if (!groups) return null;
  const mask = new Uint8Array(6);
  for (const group of groups) {
    if (group >= 0 && group < mask.length) mask[group] = 1;
  }
  return mask;
}

/** Resolve a frame's model index, tolerating a missing `entity/` prefix. */
function findByName(names: string[], wanted: string): number {
  const exact = names.indexOf(wanted);
  if (exact >= 0) return exact;
  const bare = wanted.slice(wanted.lastIndexOf('/') + 1);
  return names.findIndex(name => name === bare || name.endsWith(`/${bare}`));
}

/** Cast one sensor's rays once, serving every field that shares them. */
export class RaycastSensor {
  private readonly frames: FramePose[] = [];
  private resolvedFor: MjModel | null = null;
  /** `mj_ray` writes a geom id here; unused, but the binding wants the slot. */
  private readonly geomId = new Int32Array(1);
  /** Reused across frames — a height scan is ~200 rays every control step. */
  private readonly distances: Float32Array;
  private readonly hitPos: Float32Array;

  /** `mj_ray`'s per-group mask (nonzero = castable), or null for every group. */
  private readonly geomGroup: Uint8Array | null;

  constructor(
    private readonly mujoco: MainModule,
    private readonly descriptor: RaycastSensorDescriptor,
  ) {
    const total = descriptor.frames.length * descriptor.local_offsets.length;
    this.distances = new Float32Array(total);
    this.hitPos = new Float32Array(total * 3);
    this.geomGroup = geomGroupMask(descriptor.include_geom_groups);
  }

  /** One of the sensor's fields, or null if this model has no such frame. */
  read(field: string, mjModel: MjModel, mjData: MjData): Float32Array | null {
    if (!this.resolve(mjModel)) return null;
    // The frame poses are needed for every field; the rays only for two.
    const poses = this.framePoses(mjData);
    switch (field) {
      case 'frame_pos_w':
        return flatten(poses.map(p => [p.pos[0], p.pos[1], p.pos[2]]));
      case 'pos_w':
        return Float32Array.from(poses[0].pos);
      case 'frame_quat_w':
        return flatten(poses.map(p => quatFromMatrix(p.mat)));
      case 'quat_w':
        return Float32Array.from(quatFromMatrix(poses[0].mat));
      case 'distances':
        this.cast(poses, mjModel, mjData);
        return this.distances;
      case 'hit_pos_w':
        this.cast(poses, mjModel, mjData);
        return this.hitPos;
      default:
        return null;
    }
  }

  /** Look up each frame's model index once per model. */
  private resolve(mjModel: MjModel): boolean {
    if (this.resolvedFor === mjModel) return this.frames.length > 0;
    this.resolvedFor = mjModel;
    this.frames.length = 0;
    const tables: Record<string, () => string[]> = {
      body: () => decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr),
      site: () => decodeNames(mjModel, mjModel.nsite, mjModel.name_siteadr),
      geom: () => decodeNames(mjModel, mjModel.ngeom, mjModel.name_geomadr),
    };
    for (const frame of this.descriptor.frames) {
      const index = findByName(tables[frame.type](), frame.name);
      if (index < 0) {
        console.warn(
          `[raycast] ${frame.type} "${frame.name}" is not in this model; the sensor ` +
            'cannot be read.',
        );
        this.frames.length = 0;
        return false;
      }
      const bodyId =
        frame.type === 'body'
          ? index
          : frame.type === 'site'
            ? mjModel.site_bodyid[index]
            : mjModel.geom_bodyid[index];
      this.frames.push({
        index,
        bodyId,
        pos: new Float64Array(3),
        mat: new Float64Array(9),
      });
    }
    return true;
  }

  private framePoses(mjData: MjData): FramePose[] {
    this.descriptor.frames.forEach((frame, i) => {
      const pose = this.frames[i];
      const { index } = pose;
      const [xpos, xmat] =
        frame.type === 'body'
          ? [mjData.xpos, mjData.xmat]
          : frame.type === 'site'
            ? [mjData.site_xpos, mjData.site_xmat]
            : [mjData.geom_xpos, mjData.geom_xmat];
      for (let k = 0; k < 3; k++) pose.pos[k] = xpos[index * 3 + k];
      for (let k = 0; k < 9; k++) pose.mat[k] = xmat[index * 9 + k];
    });
    return this.frames;
  }

  private cast(poses: FramePose[], mjModel: MjModel, mjData: MjData): void {
    const { local_offsets, local_directions, ray_alignment, max_distance } =
      this.descriptor;
    const perFrame = local_offsets.length;
    let ray = 0;
    for (let f = 0; f < poses.length; f++) {
      const pose = poses[f];
      const rot = alignmentRotation(pose.mat, ray_alignment);
      const exclude = this.descriptor.exclude_parent_body ? pose.bodyId : -1;
      for (let n = 0; n < perFrame; n++, ray++) {
        const offset = apply(rot, local_offsets[n]);
        const direction = apply(rot, local_directions[n]);
        const origin: [number, number, number] = [
          pose.pos[0] + offset[0],
          pose.pos[1] + offset[1],
          pose.pos[2] + offset[2],
        ];
        let distance = this.mujoco.mj_ray(
          mjModel,
          mjData,
          origin as unknown as number[],
          direction as unknown as number[],
          this.geomGroup as unknown as number[],
          true,
          exclude,
          this.geomId,
          null,
        );
        // mjlab treats an over-range hit as a miss too, not just "no geom".
        if (distance > max_distance) distance = -1;
        this.distances[ray] = distance;
        const travelled = Math.max(distance, 0);
        for (let k = 0; k < 3; k++) {
          this.hitPos[ray * 3 + k] = origin[k] + direction[k] * travelled;
        }
      }
    }
  }
}

function flatten(rows: ReadonlyArray<ArrayLike<number>>): Float32Array {
  const width = rows[0]?.length ?? 0;
  const out = new Float32Array(rows.length * width);
  rows.forEach((row, i) => out.set(Float32Array.from(row as number[]), i * width));
  return out;
}
