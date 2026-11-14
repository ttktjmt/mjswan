/** Most frames one buffer may declare. */
export const MAX_FRAMES = 10_000;
/** Floats one policy load may buffer in all (about 64 MB). */
export const MAX_BUFFERED = 16_000_000;
/** Rays one policy may cast a control step, every cast site together (a height scan is ~200). */
export const MAX_RAYS = 65_536;
/** Camera sensors one policy may declare: each keeps a feed image, a render target and a frustum. */
export const MAX_CAMERA_SENSORS = 16;
/** Camera images one policy may render a control step, every slot together: each is a whole-scene draw. */
export const MAX_CAMERA_READS = 64;
/** Floats a camera read allocates per pixel: the readback, the rgb image, the feed's copy, a canvas's ImageData. */
const READ_FLOATS_PER_PIXEL = 6;
/** A frame object's heap beyond its data, in floats, so empty frames still count. */
const FRAME_OVERHEAD = 64;

const size = (n: unknown): n is number => Number.isSafeInteger(n) && (n as number) >= 0;

/**
 * One policy load's allowance, drawn down for each buffer the document sizes.
 * A per-buffer ceiling alone would not hold: counts under it still multiply and add up.
 */
export class BufferBudget {
  private remaining = MAX_BUFFERED;
  private rays = MAX_RAYS;
  private renders = MAX_CAMERA_READS;

  reserve(frames: number, width: number, label: string): void {
    const cost = frames * (width + FRAME_OVERHEAD);
    // A negative count would refill the allowance.
    if (!size(frames) || !size(width) || frames > MAX_FRAMES || !(cost <= this.remaining)) {
      throw new Error(`mjswan: ${label} declares ${frames} frames of width ${width}, past this policy's buffer limit.`);
    }
    this.remaining -= cost;
  }

  reserveRays(count: number, label: string): void {
    if (!size(count) || count > this.rays) {
      throw new Error(`mjswan: ${label} casts ${count} rays a step, past this policy's ray limit.`);
    }
    this.rays -= count;
  }

  reserveRender(label: string): void {
    if (this.renders < 1) {
      throw new Error(`mjswan: ${label} renders past this policy's limit of ${MAX_CAMERA_READS} camera images a step.`);
    }
    this.renders -= 1;
  }
}

type Raycast = { frames: unknown[]; local_offsets: unknown[] };
type Entry = Record<string, unknown>;
const RAY_FIELDS = new Set(['distances', 'hit_pos_w', 'normals_w']);
const SECTIONS = ['observations', 'actions', 'terminations', 'commands', 'events'];
const isEntry = (v: unknown): v is Entry => typeof v === 'object' && v !== null;
const isRaycast = (d: unknown): d is Raycast =>
  isEntry(d) && d.kind !== 'contact' && Array.isArray(d.frames) && Array.isArray(d.local_offsets);
const raysOf = (d: Raycast) => d.frames.length * d.local_offsets.length;
const numel = (dims: unknown) => (Array.isArray(dims) ? dims : []).reduce((a: number, b) => a * b, 1);

function walk(value: unknown, visit: (entry: Entry) => void): void {
  if (Array.isArray(value)) value.forEach((v) => walk(v, visit));
  else if (isEntry(value)) {
    visit(value);
    Object.values(value).forEach((v) => walk(v, visit));
  }
}

/**
 * Reserve what a policy's term graphs are fed each step, before any of them runs: each
 * rand vector and input slot, and each raycast sensor's buffers (13 floats a ray). Rays count
 * per cast site, since a sensor recasts on every read of a ray field and every drawing.
 * `cameras` are the camera sensors the runtime keeps, each slot reading one rendering its
 * image at the descriptor's size, whatever the slot's shape says.
 */
export function reserveTermFeeds(
  config: object,
  budget: BufferBudget,
  cameras: Record<string, { width: unknown; height: unknown }> = {},
): void {
  const sections = SECTIONS.map((s) => (config as Entry)[s]);
  const named = Object.entries(cameras);
  if (named.length > MAX_CAMERA_SENSORS) {
    throw new Error(`mjswan: the policy declares ${named.length} camera sensors, past its limit of ${MAX_CAMERA_SENSORS}.`);
  }
  const pixels = new Map<string, number>();
  for (const [name, { width, height }] of named) {
    if (!size(width) || !size(height) || width < 1 || height < 1) {
      throw new Error(`mjswan: camera sensor ${JSON.stringify(name)} declares a ${width}x${height} image.`);
    }
    budget.reserve(1, 2 * width * height, `camera sensor ${JSON.stringify(name)}`);
    pixels.set(name, width * height);
  }
  const raycasts = new Map<string, Raycast>();
  walk(sections, (entry) => {
    for (const [name, d] of Object.entries(isEntry(entry.sensors) ? entry.sensors : {})) {
      if (isRaycast(d)) raycasts.set(name, d);
    }
  });
  for (const [name, d] of raycasts) budget.reserve(1, 13 * raysOf(d), `raycast sensor "${name}"`);
  // An observation term's output is allocated at its declared size every step.
  walk((config as Entry).observations, (entry) => {
    if (typeof entry.size === 'number') budget.reserve(1, entry.size, `observation ${JSON.stringify(entry.name)}`);
  });
  walk(sections, (entry) => {
    if (entry.rand_dim !== undefined) budget.reserve(1, entry.rand_dim as number, 'a term\'s rand_dim');
    for (const slot of Array.isArray(entry.input_slots) ? entry.input_slots : []) {
      const s = isEntry(slot) ? slot : {};
      const shape = Array.isArray(s.shape) ? s.shape : [];
      const width = Array.isArray(s.rows) ? s.rows.length * numel(shape.slice(2)) : numel(shape.slice(1));
      budget.reserve(1, width, `input slot ${JSON.stringify(s.input)}`);
      const ray = typeof s.sensor === 'string' ? raycasts.get(s.sensor) : undefined;
      if (ray && RAY_FIELDS.has(s.field as string)) budget.reserveRays(raysOf(ray), `sensor "${s.sensor}"`);
      const image = s.field === 'rgb' && typeof s.sensor === 'string' ? pixels.get(s.sensor) : undefined;
      if (image !== undefined) {
        budget.reserveRender(`sensor ${JSON.stringify(s.sensor)}`);
        budget.reserve(1, READ_FLOATS_PER_PIXEL * image, `sensor ${JSON.stringify(s.sensor)} image`);
      }
    }
  });
  // DebugViz holds its own casters, and each drawing traces on every update.
  const vis = isEntry((config as Entry).debug_vis) ? ((config as Entry).debug_vis as Entry) : {};
  const sensors = isEntry(vis.sensors) ? vis.sensors : {};
  for (const [name, d] of Object.entries(sensors)) {
    if (!isRaycast(d)) continue;
    budget.reserve(1, 13 * raysOf(d), `debug_vis sensor "${name}"`);
    if ((d as Entry).debug_vis) budget.reserveRays(raysOf(d), `debug_vis sensor "${name}"`);
  }
  for (const [name, reward] of Object.entries(isEntry(vis.rewards) ? vis.rewards : {})) {
    const listed = isEntry(reward) && Array.isArray(reward.terrain_sensors) ? reward.terrain_sensors : [];
    for (const s of listed) {
      const d = sensors[s as string];
      if (isRaycast(d)) budget.reserveRays(raysOf(d), `debug_vis reward "${name}"`);
    }
  }
}
