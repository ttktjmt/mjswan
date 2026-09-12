/**
 * What an entity's elements are in the compiled model: mjlab's `EntityIndexing`
 * (`entity.py: _compute_indexing`), resolved once per `(mjModel, entity)`.
 *
 * mjlab indexes by the entity's own `MjSpec` — `spec.bodies[1:]`, `spec.geoms`, … — in
 * spec order, which is ascending compiled id. The browser has only the compiled model,
 * so membership is recovered from what `attach` left behind: every named element carries
 * the `name/` prefix, and an unnamed geom or site (common for visual meshes) belongs to
 * whichever entity owns its body. One an entity declared on its own worldbody (a floor
 * plane in a robot's MJCF) lands on body 0, where only the prefix says whose it is — so
 * an unnamed one there is attributed to the entity only when it is the sole prefixed
 * entity in the model, as it is for every `add_scene` trace env. A plain-MJCF model has
 * no prefix and is single-entity, so the whole model is the entity.
 */

type MjModel = import('mujoco').MjModel;

/** Everything about one entity that resolving its fields needs, computed once. */
export type EntityIndex = {
  /** qpos addresses of the entity's non-free joints, in spec order. */
  qposAdr: number[];
  /** qvel (dof) addresses of the same joints, same order. */
  qvelAdr: number[];
  /** Encoder bias per joint, aligned to `qposAdr`. */
  jointBias: Float32Array;
  /** mjlab's `root_body_id`: the entity's first non-world body; -1 when it has none. */
  rootBodyId: number;
  /** `body_ids`: the entity's bodies, world excluded, ascending. */
  bodyIds: number[];
  /** `mjModel.body_iquat` row per `bodyIds` entry (4 each): the COM frame in its body. */
  bodyIquat: Float32Array;
  /** `geom_ids`, and the body each geom sits on (its `cvel` is the geom's). */
  geomIds: number[];
  geomBodyIds: number[];
  /** `site_ids`, and the body each site sits on. */
  siteIds: number[];
  siteBodyIds: number[];
  /** `tendon_ids`. */
  tendonIds: number[];
  /** `ctrl_ids`: the entity's actuators, in `mjData.ctrl` order. */
  ctrlIds: number[];
};

/** Widths of a joint's qpos/qvel block by `mjtJoint` (free, ball, slide, hinge). */
const QPOS_WIDTH = [7, 4, 1, 1];
const DOF_WIDTH = [6, 3, 1, 1];
const MJ_JNT_FREE = 0;

export function decodeNames(mjModel: MjModel, count: number, adr: ArrayLike<number>): string[] {
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

/**
 * Indices of the elements belonging to `entity`, in ascending model id.
 *
 * The whole-model fallback is gated on `prefixed`, not on an empty match: an entity
 * with no sites is common, and answering that with every other entity's is worse.
 */
function scopedIndices(
  names: string[],
  entity: string | null | undefined,
  prefixed: boolean,
): number[] {
  const all = names.map((_, i) => i);
  // No asset named in the term's params, or a plain-MJCF model: the whole model.
  if (!entity || !prefixed) return all;
  const prefix = `${entity}/`;
  return all.filter(i => names[i].startsWith(prefix));
}

/**
 * Like {@link scopedIndices}, for elements that sit on a body: an unnamed one is the
 * entity's when its body is, a named one when it carries the prefix. `ownsWorld` adds
 * the unnamed ones on body 0, for the sole prefixed entity (see the module note).
 */
function scopedByBody(
  names: string[],
  bodyOf: ArrayLike<number> | undefined,
  bodies: ReadonlySet<number>,
  entity: string | null | undefined,
  prefixed: boolean,
  ownsWorld: boolean,
): number[] {
  const all = names.map((_, i) => i);
  if (!entity || !prefixed) return all;
  const prefix = `${entity}/`;
  return all.filter(i => {
    if (names[i].startsWith(prefix)) return true;
    const body = bodyOf?.[i] ?? -1;
    return bodies.has(body) || (ownsWorld && body === 0);
  });
}

/** The element's name with its `entity/` prefix removed, as mjlab reports it. */
export function unprefixed(name: string): string {
  const slash = name.lastIndexOf('/');
  return slash < 0 ? name : name.slice(slash + 1);
}

/** A model array the fixture may not carry, read as empty rather than as a crash. */
function names(mjModel: MjModel, count: number | undefined, adr: ArrayLike<number> | undefined): string[] {
  if (!count || !adr) return [];
  return decodeNames(mjModel, count, adr);
}

export function buildEntityIndex(
  mjModel: MjModel,
  entity: string | null | undefined,
  jointBias?: (jointName: string) => number,
): EntityIndex {
  const jointNames = decodeNames(mjModel, mjModel.njnt, mjModel.name_jntadr);
  const bodyNames = decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr);
  const siteNames = names(mjModel, mjModel.nsite, mjModel.name_siteadr);
  const geomNames = names(mjModel, mjModel.ngeom, mjModel.name_geomadr);
  const tendonNames = names(mjModel, mjModel.ntendon, mjModel.name_tendonadr);
  const actuatorNames = names(mjModel, mjModel.nu, mjModel.name_actuatoradr);
  // One verdict for the whole model, not per element kind.
  const prefixes = new Set(
    [...jointNames, ...bodyNames, ...siteNames].filter(n => n.includes('/')).map(n => n.slice(0, n.indexOf('/'))),
  );
  const prefixed = prefixes.size > 0;
  const ownsWorld = prefixes.size === 1 && prefixes.has(entity ?? '');

  const qposAdr: number[] = [];
  const qvelAdr: number[] = [];
  const bias: number[] = [];
  for (const j of scopedIndices(jointNames, entity, prefixed)) {
    const type = mjModel.jnt_type[j];
    if (type === MJ_JNT_FREE) continue; // mjlab keeps the free joint separate
    const qWidth = QPOS_WIDTH[type] ?? 1;
    const vWidth = DOF_WIDTH[type] ?? 1;
    const perJoint = jointBias?.(unprefixed(jointNames[j])) ?? 0;
    for (let k = 0; k < qWidth; k++) {
      qposAdr.push(mjModel.jnt_qposadr[j] + k);
      bias.push(perJoint);
    }
    for (let k = 0; k < vWidth; k++) qvelAdr.push(mjModel.jnt_dofadr[j] + k);
  }

  // Skip the worldbody (id 0): mjlab's `bodies` tuple is `spec.bodies[1:]`.
  const bodyIds = scopedIndices(bodyNames, entity, prefixed).filter(i => i !== 0);
  const bodySet = new Set(bodyIds);
  const bodyIquat = new Float32Array(bodyIds.length * 4);
  const iquat = mjModel.body_iquat as ArrayLike<number> | undefined;
  for (let i = 0; i < bodyIds.length; i++) {
    const base = bodyIds[i] * 4;
    bodyIquat[i * 4] = iquat?.[base] ?? 1;
    bodyIquat[i * 4 + 1] = iquat?.[base + 1] ?? 0;
    bodyIquat[i * 4 + 2] = iquat?.[base + 2] ?? 0;
    bodyIquat[i * 4 + 3] = iquat?.[base + 3] ?? 0;
  }

  const geomIds = scopedByBody(geomNames, mjModel.geom_bodyid, bodySet, entity, prefixed, ownsWorld);
  const siteIds = scopedByBody(siteNames, mjModel.site_bodyid, bodySet, entity, prefixed, ownsWorld);

  return {
    qposAdr,
    qvelAdr,
    jointBias: Float32Array.from(bias),
    rootBodyId: bodyIds.length > 0 ? bodyIds[0] : -1,
    bodyIds,
    bodyIquat,
    geomIds,
    geomBodyIds: geomIds.map(g => mjModel.geom_bodyid?.[g] ?? 0),
    siteIds,
    siteBodyIds: siteIds.map(s => mjModel.site_bodyid?.[s] ?? 0),
    tendonIds: scopedIndices(tendonNames, entity, prefixed),
    ctrlIds: scopedIndices(actuatorNames, entity, prefixed),
  };
}
