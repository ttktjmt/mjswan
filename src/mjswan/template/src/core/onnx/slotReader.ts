/**
 * Fills a traced graph's `input_slots` from `mjModel`/`mjData`, reproducing mjlab's
 * `EntityData` semantics natively. A field read wrong here makes the graph compute the
 * right function of the wrong numbers, silently.
 *
 * Every slot must be the **whole** field, flattened (the graph carries its own baked-in
 * indexing), in mjlab's element order (MJCF spec order, i.e. ascending model id within
 * an entity, free joint excluded), as float32.
 *
 * Entities resolve by the `name/` prefix mjlab's `attach` adds; an unprefixed model
 * falls back to the whole model, which is correct because such a scene is
 * single-entity. An unknown field returns null and the caller holds its previous value.
 */

import { quatApply, quatApplyInv } from '../observation/math';
import type { ContactSensorSet } from './contact';
import { RaycastSensor, isRaycastField, type RaycastSensorDescriptor } from './raycast';
import type { OnnxInputSlot, SlotReader } from './session';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

/** A command term that can hand back one of its traced state fields by name. */
export interface CommandStateSource {
  getStateField(field: string): Float32Array | null;
}

/** Shaped to match `PolicyRunnerContext`, so the runtime can pass one straight through. */
export type SlotReaderContext = {
  mjModel: MjModel | null;
  mjData: MjData | null;
  /** Needed to cast a `RayCastSensor`'s rays (`mj_ray`); absent before load. */
  mujoco?: MainModule | null;
  commandManager?: { getTerm(name: string): unknown } | null;
};

export type SlotReaderOptions = {
  /** By *unprefixed* joint name: `policy.json` stores it in action, not entity, order. */
  jointBias?: (jointName: string) => number;
  /** A function because descriptors arrive with the policy, the reader with the runtime. */
  raycastSensors?: () => Record<string, RaycastSensorDescriptor>;
  /** The engine owns these, since it advances their history per substep. */
  contactSensors?: () => ContactSensorSet | null;
};

/** Everything about one entity that resolving its fields needs, computed once. */
type EntityIndex = {
  /** qpos addresses of the entity's non-free joints, in spec order. */
  qposAdr: number[];
  /** qvel (dof) addresses of the same joints, same order. */
  qvelAdr: number[];
  /** Encoder bias per joint, aligned to `qposAdr`. */
  jointBias: Float32Array;
  /**
   * `mjData.ctrl` address of each joint's position actuator, aligned to `qposAdr`;
   * `-1` where the joint has none, or has one that commands force rather than a
   * target. `null` if that is true of any of them, which is what makes
   * `joint_pos_target` unreadable for the entity as a whole.
   */
  posCtrlAdr: number[] | null;
  /** mjlab's `root_body_id`: the entity's first non-world body. */
  rootBodyId: number;
  /** Model site ids belonging to the entity, in spec order. */
  siteIds: number[];
};

/** Widths of a joint's qpos/qvel block by `mjtJoint` (free, ball, slide, hinge). */
const QPOS_WIDTH = [7, 4, 1, 1];
const DOF_WIDTH = [6, 3, 1, 1];
const MJ_JNT_FREE = 0;
const MJ_TRN_JOINT = 0;
const MJ_BIAS_AFFINE = 1;

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

/** The element's name with its `entity/` prefix removed, as mjlab reports it. */
function unprefixed(name: string): string {
  const slash = name.lastIndexOf('/');
  return slash < 0 ? name : name.slice(slash + 1);
}

function buildEntityIndex(
  mjModel: MjModel,
  entity: string | null | undefined,
  options: SlotReaderOptions,
): EntityIndex {
  const jointNames = decodeNames(mjModel, mjModel.njnt, mjModel.name_jntadr);
  const bodyNames = decodeNames(mjModel, mjModel.nbody, mjModel.name_bodyadr);
  const siteNames = decodeNames(mjModel, mjModel.nsite, mjModel.name_siteadr);
  // One verdict for the whole model, not per element kind.
  const prefixed = [...jointNames, ...bodyNames, ...siteNames].some(n => n.includes('/'));

  const positionCtrl = positionActuatorByJoint(mjModel);
  const qposAdr: number[] = [];
  const qvelAdr: number[] = [];
  const bias: number[] = [];
  const posCtrl: number[] = [];
  for (const j of scopedIndices(jointNames, entity, prefixed)) {
    const type = mjModel.jnt_type[j];
    if (type === MJ_JNT_FREE) continue; // mjlab keeps the free joint separate
    const qWidth = QPOS_WIDTH[type] ?? 1;
    const vWidth = DOF_WIDTH[type] ?? 1;
    const jointBias = options.jointBias?.(unprefixed(jointNames[j])) ?? 0;
    for (let k = 0; k < qWidth; k++) {
      qposAdr.push(mjModel.jnt_qposadr[j] + k);
      bias.push(jointBias);
      // Only a 1-qpos joint maps to one `ctrl`; a ball's actuator would be replicated.
      posCtrl.push(qWidth === 1 ? (positionCtrl.get(j) ?? -1) : -1);
    }
    for (let k = 0; k < vWidth; k++) qvelAdr.push(mjModel.jnt_dofadr[j] + k);
  }

  // Skip the worldbody (id 0): mjlab's `bodies` tuple is `spec.bodies[1:]`.
  const bodyIds = scopedIndices(bodyNames, entity, prefixed).filter(i => i !== 0);

  return {
    qposAdr,
    qvelAdr,
    jointBias: Float32Array.from(bias),
    posCtrlAdr: posCtrl.every(adr => adr >= 0) ? posCtrl : null,
    rootBodyId: bodyIds.length > 0 ? bodyIds[0] : -1,
    siteIds: scopedIndices(siteNames, entity, prefixed),
  };
}

/**
 * Joint id -> `ctrl` address, for position actuators only (`biastype=affine`).
 *
 * A motor actuator's `ctrl` is a force, so it says nothing about where the joint was
 * told to go; only the affine ones carry mjlab's `joint_pos_target` in `ctrl`.
 */
function positionActuatorByJoint(mjModel: MjModel): Map<number, number> {
  const byJoint = new Map<number, number>();
  for (let a = 0; a < mjModel.nu; a++) {
    if (mjModel.actuator_trntype[a] !== MJ_TRN_JOINT) continue;
    if (mjModel.actuator_biastype[a] !== MJ_BIAS_AFFINE) continue;
    byJoint.set(mjModel.actuator_trnid[a * 2], a);
  }
  return byJoint;
}

function gather(source: ArrayLike<number>, addresses: readonly number[]): Float32Array {
  const out = new Float32Array(addresses.length);
  for (let i = 0; i < addresses.length; i++) out[i] = source[addresses[i]] ?? 0;
  return out;
}

function vec3At(source: ArrayLike<number>, index: number): Float32Array {
  const base = index * 3;
  return new Float32Array([source[base] ?? 0, source[base + 1] ?? 0, source[base + 2] ?? 0]);
}

function quatAt(source: ArrayLike<number>, index: number): Float32Array {
  const base = index * 4;
  return new Float32Array([
    source[base] ?? 1,
    source[base + 1] ?? 0,
    source[base + 2] ?? 0,
    source[base + 3] ?? 0,
  ]);
}

function concat(...parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/**
 * mjlab's `root_link_vel_w` (lin ++ ang, 6): `cvel` is about the subtree COM, so
 * `compute_velocity_from_cvel` removes the rotational offset to the body origin.
 */
function rootLinkVelW(rootBodyId: number, mjData: MjData): Float32Array {
  const pos = vec3At(mjData.xpos, rootBodyId);
  const com = vec3At(mjData.subtree_com, rootBodyId);
  const base = rootBodyId * 6;
  // MuJoCo's cvel packs angular first, then linear.
  const ang = angVelW(mjData, rootBodyId);
  const linC = new Float32Array([
    mjData.cvel[base + 3] ?? 0,
    mjData.cvel[base + 4] ?? 0,
    mjData.cvel[base + 5] ?? 0,
  ]);
  const ox = com[0] - pos[0];
  const oy = com[1] - pos[1];
  const oz = com[2] - pos[2];
  const lin = new Float32Array([
    linC[0] - (ang[1] * oz - ang[2] * oy),
    linC[1] - (ang[2] * ox - ang[0] * oz),
    linC[2] - (ang[0] * oy - ang[1] * ox),
  ]);
  return concat(lin, ang);
}

type FieldReader = (index: EntityIndex, mjData: MjData) => Float32Array | null;

/** Wrap a reader needing a root body. Null, since indexing at -1 gives plausible zeros. */
function rootField(read: (root: number, mjData: MjData) => Float32Array): FieldReader {
  return (index, mjData) => (index.rootBodyId < 0 ? null : read(index.rootBodyId, mjData));
}

/** `cvel`'s angular part for a body — packed before the linear part in MuJoCo. */
function angVelW(mjData: MjData, root: number): Float32Array {
  const base = root * 6;
  return new Float32Array([
    mjData.cvel[base] ?? 0,
    mjData.cvel[base + 1] ?? 0,
    mjData.cvel[base + 2] ?? 0,
  ]);
}

const FIELD_READERS: Record<string, FieldReader> = {
  joint_pos: (index, mjData) => gather(mjData.qpos, index.qposAdr),
  joint_pos_biased: (index, mjData) => {
    const out = gather(mjData.qpos, index.qposAdr);
    for (let i = 0; i < out.length; i++) out[i] += index.jointBias[i] ?? 0;
    return out;
  },
  joint_vel: (index, mjData) => gather(mjData.qvel, index.qvelAdr),
  // The last target the action layer wrote. mjlab stores `processed - encoder_bias`
  // here and writes the same value to `ctrl`, so for a position actuator `ctrl` is it.
  joint_pos_target: (index, mjData) =>
    index.posCtrlAdr && gather(mjData.ctrl, index.posCtrlAdr),

  root_link_pos_w: rootField((root, mjData) => vec3At(mjData.xpos, root)),
  root_link_quat_w: rootField((root, mjData) => quatAt(mjData.xquat, root)),
  root_link_pose_w: rootField((root, mjData) =>
    concat(vec3At(mjData.xpos, root), quatAt(mjData.xquat, root)),
  ),

  root_link_vel_w: rootField(rootLinkVelW),
  root_link_lin_vel_w: rootField((root, mjData) => rootLinkVelW(root, mjData).slice(0, 3)),
  // Straight off cvel, as mjlab does: the COM offset only shifts the linear part.
  root_link_ang_vel_w: rootField((root, mjData) => angVelW(mjData, root)),
  root_link_lin_vel_b: rootField((root, mjData) =>
    Float32Array.from(
      quatApplyInv(quatAt(mjData.xquat, root), rootLinkVelW(root, mjData).subarray(0, 3)),
    ),
  ),
  root_link_ang_vel_b: rootField((root, mjData) =>
    Float32Array.from(quatApplyInv(quatAt(mjData.xquat, root), angVelW(mjData, root))),
  ),

  // A constant, not `mjModel.opt.gravity`: mjlab fills it with the world's down (0,0,-1).
  gravity_vec_w: () => new Float32Array([0, 0, -1]),
  projected_gravity_b: rootField((root, mjData) =>
    Float32Array.from(quatApplyInv(quatAt(mjData.xquat, root), [0, 0, -1])),
  ),
  heading_w: rootField((root, mjData) => {
    const forward = quatApply(quatAt(mjData.xquat, root), [1, 0, 0]);
    return new Float32Array([Math.atan2(forward[1], forward[0])]);
  }),

  site_pos_w: (index, mjData) => {
    const out = new Float32Array(index.siteIds.length * 3);
    for (let i = 0; i < index.siteIds.length; i++) {
      out.set(vec3At(mjData.site_xpos, index.siteIds[i]), i * 3);
    }
    return out;
  },
};

/** Whether an `Entity.data` field can be served — useful for a build-time check. */
export function isReadableEntityField(field: string): boolean {
  return field in FIELD_READERS;
}

/**
 * A named MuJoCo sensor's `sensordata` window. The build records mjlab's prefixed name
 * (`robot/imu_lin_vel`), while a plain-MJCF model has the bare one, so try both.
 */
export function sensorWindow(
  mjModel: MjModel,
  sensor: string,
): { adr: number; dim: number } | null {
  const names = decodeNames(mjModel, mjModel.nsensor, mjModel.name_sensoradr);
  let idx = names.indexOf(sensor);
  if (idx < 0) {
    const bare = unprefixed(sensor);
    idx = names.findIndex(name => name === bare || unprefixed(name) === bare);
  }
  if (idx < 0) return null;
  return { adr: mjModel.sensor_adr[idx], dim: mjModel.sensor_dim[idx] };
}

function isCommandStateSource(value: unknown): value is CommandStateSource {
  return (
    typeof value === 'object' &&
    value !== null &&
    typeof (value as CommandStateSource).getStateField === 'function'
  );
}

/**
 * Build the `SlotReader` every ONNX-backed term reads its graph inputs through.
 *
 * `getContext` is called per read so a scene reload is picked up without rebuilding
 * the reader; the per-entity index is cached against the model it came from.
 */
export function createSlotReader(
  getContext: () => SlotReaderContext | null,
  options: SlotReaderOptions = {},
): SlotReader {
  let cachedModel: MjModel | null = null;
  const indices = new Map<string, EntityIndex>();
  // One per sensor, held for its frame resolution and ray buffers: ~200 rays a step.
  const casters = new Map<string, RaycastSensor | null>();

  /** Drop what the previous scene resolved: both maps hold its model indices. */
  const forModel = (mjModel: MjModel): void => {
    if (mjModel === cachedModel) return;
    cachedModel = mjModel;
    indices.clear();
    casters.clear();
  };

  const readRaycast = (
    sensor: string,
    field: string,
    context: SlotReaderContext,
  ): Float32Array | null => {
    const { mjModel, mjData, mujoco } = context;
    if (!mjModel || !mjData || !mujoco) return null;
    forModel(mjModel);
    if (!casters.has(sensor)) {
      const descriptor = options.raycastSensors?.()[sensor];
      if (!descriptor) {
        console.warn(
          `[slotReader] no raycast descriptor for sensor "${sensor}"; the build ` +
            'did not emit one.',
        );
      }
      casters.set(sensor, descriptor ? new RaycastSensor(mujoco, descriptor) : null);
    }
    const caster = casters.get(sensor);
    if (!caster) return null;
    if (!isRaycastField(field)) {
      console.warn(`[slotReader] raycast sensor "${sensor}" cannot serve "${field}".`);
      return null;
    }
    return caster.read(field, mjModel, mjData);
  };

  const indexFor = (mjModel: MjModel, entity: string | null | undefined): EntityIndex => {
    forModel(mjModel);
    const key = entity ?? '';
    let index = indices.get(key);
    if (!index) {
      index = buildEntityIndex(mjModel, entity, options);
      indices.set(key, index);
    }
    return index;
  };

  return (slot: OnnxInputSlot): Float32Array | null => {
    const context = getContext();
    if (!context) return null;

    if (slot.command) {
      const term = context.commandManager?.getTerm(slot.command);
      if (!isCommandStateSource(term)) return null;
      return term.getStateField(slot.field ?? '');
    }

    const { mjModel, mjData } = context;
    if (!mjModel || !mjData) return null;

    if (slot.sensor) {
      if (slot.field) {
        // A structured sensor: no single `sensordata` window to fall through to, so
        // each kind is served by the module that knows its layout.
        const contacts = options.contactSensors?.();
        const contact = contacts?.read(slot.sensor, slot.field, mjModel, mjData, sensor =>
          sensorWindow(mjModel, sensor),
        );
        if (contact) return contact;
        return readRaycast(slot.sensor, slot.field, context);
      }
      const window = sensorWindow(mjModel, slot.sensor);
      if (!window) return null;
      const out = new Float32Array(window.dim);
      for (let i = 0; i < window.dim; i++) out[i] = mjData.sensordata[window.adr + i] ?? 0;
      return out;
    }

    const read = slot.field ? FIELD_READERS[slot.field] : undefined;
    if (!read) return null;
    return read(indexFor(mjModel, slot.entity), mjData);
  };
}
