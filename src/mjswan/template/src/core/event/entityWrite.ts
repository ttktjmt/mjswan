/**
 * Writes an already-computed `entity_write` value to mjData. Apply-only: the term body
 * sampled and computed inside its graph, so nothing here draws a random number.
 *
 * The kinds mirror `mjswan.compile.tracer._WRITE_FIELDS`.
 */

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

export type WriteKind = 'joint_state' | 'root_pose' | 'root_velocity';

export interface WriteTarget {
  kind: WriteKind;
  /** Entity the write applies to. A scene with two of them (a robot and a thrown ball)
   * has a free joint each, and the first one is not always the right one. */
  entity?: string | null;
  fields: string[];
  /** Resolved joint indices for `joint_state`; `"all"` means every joint. */
  joint_ids?: number[] | 'all';
}

/** Graph outputs keyed by the tracer's `"<kind>__<field>"` naming. */
export type WriteValues = Record<string, Float32Array | Float64Array | number[]>;

function decodeNames(mjModel: MjModel, addresses: Int32Array, count: number): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let i = 0; i < count; i++) {
    const start = addresses[i];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

export function decodeJointNames(mjModel: MjModel): string[] {
  return decodeNames(mjModel, mjModel.name_jntadr, mjModel.njnt);
}

export function findFreeJoint(mjModel: MjModel): number {
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] === 0) return j; // mjJNT_FREE
  }
  return -1;
}

/**
 * The free joint carrying `entity`'s root, or the model's first one.
 *
 * mjlab prefixes an entity's elements with its name, so the owner of a free joint is the
 * prefix on its body. Falling back to the first free joint covers the single-entity
 * scene, whose names carry no prefix at all — the same rule `entityJointIds` follows —
 * and a model that carries no body names to read.
 */
export function findEntityFreeJoint(mjModel: MjModel, entity?: string | null): number {
  if (!entity || !mjModel.name_bodyadr || !mjModel.jnt_bodyid) return findFreeJoint(mjModel);
  const bodies = decodeNames(mjModel, mjModel.name_bodyadr, mjModel.nbody);
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] !== 0) continue; // mjJNT_FREE
    const body = bodies[mjModel.jnt_bodyid[j]] ?? '';
    if (body === entity || body.startsWith(`${entity}/`)) return j;
  }
  return findFreeJoint(mjModel);
}

/**
 * Apply one write target to `mjData`, returning whether anything was written. False for a
 * model with no such target, so a scene mismatch degrades rather than throwing.
 */
export function applyEntityWrite(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  values: WriteValues,
): boolean {
  switch (target.kind) {
    case 'joint_state':
      return writeJointState(mjModel, mjData, target, values);
    case 'root_pose':
      return writeRootPose(mjModel, mjData, target, values['root_pose__pose']);
    case 'root_velocity':
      return writeRootVelocity(mjModel, mjData, target, values['root_velocity__velocity']);
    default:
      return false;
  }
}

/** Apply every write target a term emitted, in order. */
export function applyEntityWrites(
  mjModel: MjModel,
  mjData: MjData,
  targets: readonly WriteTarget[],
  values: WriteValues,
): number {
  let applied = 0;
  for (const target of targets) {
    if (applyEntityWrite(mjModel, mjData, target, values)) applied++;
  }
  return applied;
}

/**
 * What `joint_ids` indexes: the entity's 1-DoF joints in model order, as mjlab's
 * `Entity.joint_names` lists them. The model's own joint list is not the same — it
 * includes the free joint, whose `qpos[0]` is the root's x, not an angle.
 */
function entityJointIds(mjModel: MjModel, entity: string | null | undefined): number[] {
  const ids: number[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] >= 2) ids.push(j); // not mjJNT_FREE / mjJNT_BALL
  }
  if (!entity) return ids;
  const names = decodeJointNames(mjModel);
  const owned = ids.filter(j => names[j].startsWith(`${entity}/`));
  // Unprefixed names mean a single-entity scene, where every joint is the entity's.
  return owned.length > 0 ? owned : ids;
}

function resolveJointIds(mjModel: MjModel, target: WriteTarget): number[] {
  const all = entityJointIds(mjModel, target.entity);
  const ids = target.joint_ids;
  if (ids === undefined || ids === 'all') return all;
  return ids.map(i => all[i] ?? -1);
}

function writeJointState(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  values: WriteValues,
): boolean {
  const position = values['joint_state__position'];
  const velocity = values['joint_state__velocity'];
  if (!position && !velocity) return false;
  const jointIds = resolveJointIds(mjModel, target);
  for (let i = 0; i < jointIds.length; i++) {
    const j = jointIds[i];
    if (j < 0 || j >= mjModel.njnt) continue;
    if (position && i < position.length) {
      mjData.qpos[mjModel.jnt_qposadr[j]] = position[i];
    }
    if (velocity && i < velocity.length) {
      mjData.qvel[mjModel.jnt_dofadr[j]] = velocity[i];
    }
  }
  return true;
}

function writeRootPose(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  pose: WriteValues[string] | undefined,
): boolean {
  if (!pose || pose.length < 7) return false;
  const j = findEntityFreeJoint(mjModel, target.entity);
  if (j < 0) return false;
  const adr = mjModel.jnt_qposadr[j];
  for (let i = 0; i < 7; i++) mjData.qpos[adr + i] = pose[i];
  return true;
}

function writeRootVelocity(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  velocity: WriteValues[string] | undefined,
): boolean {
  if (!velocity || velocity.length < 6) return false;
  const j = findEntityFreeJoint(mjModel, target.entity);
  if (j < 0) return false;
  const adr = mjModel.jnt_dofadr[j];
  for (let i = 0; i < 6; i++) mjData.qvel[adr + i] = velocity[i];
  return true;
}
