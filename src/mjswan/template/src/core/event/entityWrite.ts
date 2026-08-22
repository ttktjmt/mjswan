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
  /** Entity the write applies to. One term may write several — mjlab resolves each
   * entity's own addresses — and the model's first free joint is not always the one. */
  entity?: string | null;
  fields: string[];
  /** Graph outputs holding `fields`, in the same order. Absent in bundles built before
   * a term could write two entities, whose outputs are named `"<kind>__<field>"`. */
  outputs?: string[];
  /** Resolved joint indices for `joint_state`; `"all"` (or absent) means every joint. */
  joint_ids?: number[] | 'all' | null;
}

/** Graph outputs, keyed as `WriteTarget.outputs` names them. */
export type WriteValues = Record<string, Float32Array | Float64Array | number[]>;

export function decodeJointNames(mjModel: MjModel): string[] {
  const bytes = new Uint8Array(mjModel.names);
  const decoder = new TextDecoder();
  const names: string[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    const start = mjModel.name_jntadr[j];
    let end = start;
    while (end < bytes.length && bytes[end] !== 0) end++;
    names.push(decoder.decode(bytes.subarray(start, end)));
  }
  return names;
}

export function findFreeJoint(mjModel: MjModel): number {
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] === 0) return j; // mjJNT_FREE
  }
  return -1;
}

/**
 * Which of `entity`'s joints a write reaches, by mjlab's rule: it resolves an entity's
 * addresses from that entity's own spec joints, whose names it namespaces `entity/…`.
 *
 * An unprefixed model is a single-entity scene, where every joint is the entity's — the
 * hand-built scenes mjswan also serves. A *prefixed* model with no such entity owns
 * nothing the write belongs to, so it gets nothing: mjlab would raise on the scene
 * lookup, and landing on another entity is how a thrown ball launches the robot.
 */
function entityJoints(
  mjModel: MjModel,
  entity: string | null | undefined,
  candidates: number[],
): number[] {
  if (!entity) return candidates;
  const names = decodeJointNames(mjModel);
  const owned = candidates.filter(j => names[j].startsWith(`${entity}/`));
  if (owned.length > 0) return owned;
  return names.some(name => name.includes('/')) ? [] : candidates;
}

/** The free joint carrying `entity`'s root, or -1 when the model holds none of its own. */
export function findEntityFreeJoint(mjModel: MjModel, entity?: string | null): number {
  const free: number[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] === 0) free.push(j); // mjJNT_FREE
  }
  return entityJoints(mjModel, entity, free)[0] ?? -1;
}

/**
 * What `joint_ids` indexes: the entity's 1-DoF joints in model order, as mjlab's
 * `Entity.joint_names` lists them. The model's own joint list is not the same — it
 * includes the free joint, whose `qpos[0]` is the root's x, not an angle.
 */
function entityJointIds(mjModel: MjModel, entity: string | null | undefined): number[] {
  const hinges: number[] = [];
  for (let j = 0; j < mjModel.njnt; j++) {
    if (mjModel.jnt_type[j] >= 2) hinges.push(j); // not mjJNT_FREE / mjJNT_BALL
  }
  return entityJoints(mjModel, entity, hinges);
}

/** The value a target's `field` was written with, or undefined. */
function writtenValue(
  target: WriteTarget,
  values: WriteValues,
  field: string,
): WriteValues[string] | undefined {
  const index = target.fields?.indexOf(field) ?? -1;
  const key = (index >= 0 ? target.outputs?.[index] : undefined) ?? `${target.kind}__${field}`;
  return values[key];
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
      return writeRootPose(mjModel, mjData, target, writtenValue(target, values, 'pose'));
    case 'root_velocity':
      return writeRootVelocity(
        mjModel,
        mjData,
        target,
        writtenValue(target, values, 'velocity'),
      );
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

function resolveJointIds(mjModel: MjModel, target: WriteTarget): number[] {
  const all = entityJointIds(mjModel, target.entity);
  const ids = target.joint_ids;
  if (ids === undefined || ids === null || ids === 'all') return all;
  return ids.map(i => all[i] ?? -1);
}

function writeJointState(
  mjModel: MjModel,
  mjData: MjData,
  target: WriteTarget,
  values: WriteValues,
): boolean {
  const position = writtenValue(target, values, 'position');
  const velocity = writtenValue(target, values, 'velocity');
  if (!position && !velocity) return false;
  let wrote = 0;
  const jointIds = resolveJointIds(mjModel, target);
  for (let i = 0; i < jointIds.length; i++) {
    const j = jointIds[i];
    if (j < 0 || j >= mjModel.njnt) continue;
    if (position && i < position.length) {
      mjData.qpos[mjModel.jnt_qposadr[j]] = position[i];
      wrote++;
    }
    if (velocity && i < velocity.length) {
      mjData.qvel[mjModel.jnt_dofadr[j]] = velocity[i];
      wrote++;
    }
  }
  return wrote > 0;
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

/** Rotate `v` by the conjugate of the (w, x, y, z) quaternion `q` — mjlab's `quat_apply_inverse`. */
function rotateByQuatInverse(
  q: ArrayLike<number>,
  qAdr: number,
  v: ArrayLike<number>,
  vAdr: number,
): [number, number, number] {
  // Conjugate, so the rotation runs world -> body.
  const w = q[qAdr];
  const x = -q[qAdr + 1];
  const y = -q[qAdr + 2];
  const z = -q[qAdr + 3];
  const vx = v[vAdr];
  const vy = v[vAdr + 1];
  const vz = v[vAdr + 2];
  // v' = v + w * t + u x t, with t = 2 * (u x v) and u = (x, y, z).
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
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
  // A free joint's qvel holds world-frame linear and *body*-frame angular velocity, while
  // the term wrote both in the world frame, as mjlab's `write_root_velocity` takes them.
  // The quaternion comes from qpos, so a pose written first is the one used — mjlab's
  // order too, and what `write_root_state_to_sim` splits into.
  const angB = rotateByQuatInverse(mjData.qpos, mjModel.jnt_qposadr[j] + 3, velocity, 3);
  const adr = mjModel.jnt_dofadr[j];
  for (let i = 0; i < 3; i++) mjData.qvel[adr + i] = velocity[i];
  for (let i = 0; i < 3; i++) mjData.qvel[adr + 3 + i] = angB[i];
  return true;
}
