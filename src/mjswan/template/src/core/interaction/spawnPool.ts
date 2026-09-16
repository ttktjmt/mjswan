/**
 * Boxes to throw into a running scene.
 *
 * MuJoCo compiles a model once and has no way to add a body to a live one — the spec API
 * can build a *new* `mjModel`, but that means a new `mjData` and a rebuild of everything
 * holding a reference to either, which is the one thing throwing something into a running
 * sim must not do. So the boxes are declared with the model and parked out of sight, the
 * same trick the tracked hands use for their bones, and a throw is a handful of writes.
 *
 * Three things about that are not obvious:
 *
 * - **A parked box still falls.** It has a free joint, so an idle slot is written back to
 *   its parking spot every control step rather than left alone.
 * - **Mass only takes effect through `mj_setConst`.** Writing `body_mass` alone changes
 *   nothing the dynamics read. And `mj_setConst` puts `qpos` back to `qpos0` — it would
 *   teleport the whole scene — so the state is saved across it.
 * - **`geom_size` moves the collider, not the drawing.** Meshes are built once at load, so
 *   a resized box needs its geometry rebuilt; {@link SpawnPool.syncMeshes} does that, and
 *   hides the slots that are parked.
 */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';

import { appendToMjcf } from '../scene/mjcfInject';

/** Flat, like every injected name: a `/` would change how the whole model is indexed. */
const SPAWN_BODY = (index: number): string => `mjswan_spawn${index}`;

export const DEFAULT_SPAWN_POOL = 8;
export const MAX_SPAWN_POOL = 32;
/** Above any scene, and clear of a floor plane, which is solid all the way down. */
const PARK_Z = 100;
/** Wider than two boxes at the largest size, so parked slots never touch each other. */
const PARK_SPACING = 2;

/** What a thrown box is made of. */
export interface SpawnSpec {
  /** Half-extent, metres. */
  size: number;
  /** kg/m^3. Mass and inertia follow from this and the size. */
  density: number;
}

export interface SpawnPose {
  /** World position, MuJoCo axes. */
  position: readonly [number, number, number];
  /** World linear velocity, MuJoCo axes. */
  velocity: readonly [number, number, number];
}

/**
 * The pool's MJCF. Contact parameters are spelled out rather than inherited: a host model
 * ships its own top-level `<default>`, and a box that quietly picked up a robot's contype
 * would fall through the floor.
 */
export function injectSpawnPoolXml(xml: string, count: number): string {
  const slots = Math.max(0, Math.min(MAX_SPAWN_POOL, Math.floor(count)));
  if (slots === 0) return xml;
  const bodies: string[] = [];
  for (let i = 0; i < slots; i++) {
    bodies.push(
      `    <body name="${SPAWN_BODY(i)}" pos="${i * PARK_SPACING} 0 ${PARK_Z}">\n` +
        `      <freejoint/>\n` +
        `      <geom name="${SPAWN_BODY(i)}_g" type="box" size="0.06 0.06 0.06" mass="0.7"` +
        ` group="0" contype="1" conaffinity="1" condim="3" friction="0.9 0.005 0.0001"` +
        ` rgba="0.85 0.45 0.2 1"/>\n` +
        `    </body>`,
    );
  }
  return appendToMjcf(xml, `  <worldbody>\n${bodies.join('\n')}\n  </worldbody>\n`);
}

interface Slot {
  bodyId: number;
  geomId: number;
  qposAdr: number;
  dofAdr: number;
  parked: [number, number, number];
  live: boolean;
  /** What the model currently carries for this slot, so a throw only rebuilds on change. */
  size: number;
  /** Half-extent the drawn mesh was last built for. */
  drawnSize: number;
}

export class SpawnPool {
  private slots: Slot[] = [];
  /** Next slot to use; wraps, so throw N+1 recycles the oldest box. */
  private next = 0;

  get size(): number {
    return this.slots.length;
  }

  /** Resolve the pool against a freshly compiled model. A scene without one binds empty. */
  bind(mujoco: MainModule, mjModel: MjModel): void {
    this.slots = [];
    this.next = 0;
    const body = mujoco.mjtObj.mjOBJ_BODY.value;
    for (let i = 0; i < MAX_SPAWN_POOL; i++) {
      const bodyId = mujoco.mj_name2id(mjModel, body, SPAWN_BODY(i));
      if (bodyId < 0) break;
      const jointAdr = mjModel.body_jntadr[bodyId];
      this.slots.push({
        bodyId,
        geomId: mjModel.body_geomadr[bodyId],
        qposAdr: mjModel.jnt_qposadr[jointAdr],
        dofAdr: mjModel.jnt_dofadr[jointAdr],
        parked: [i * PARK_SPACING, 0, PARK_Z],
        live: false,
        size: mjModel.geom_size[mjModel.body_geomadr[bodyId] * 3],
        drawnSize: mjModel.geom_size[mjModel.body_geomadr[bodyId] * 3],
      });
    }
  }

  /** Every body the pool occupies, so the viewer can keep parked boxes out of its bounds. */
  bodyIds(): number[] {
    return this.slots.map((slot) => slot.bodyId);
  }

  /**
   * Send every box back to the parking row. Called after a reset: a keyframe written for
   * the model before injection is zero-padded for the appended free joints, so a reset
   * would otherwise spawn the whole pool at the world origin, inside the scene.
   */
  park(mjData: MjData): void {
    for (const slot of this.slots) {
      slot.live = false;
      this.writeSlot(mjData, slot, slot.parked, [0, 0, 0]);
    }
    this.next = 0;
  }

  /** Hold the parked ones still. They have free joints, so "parked" is not a resting state. */
  holdIdle(mjData: MjData): void {
    for (const slot of this.slots) {
      if (!slot.live) this.writeSlot(mjData, slot, slot.parked, [0, 0, 0]);
    }
  }

  /**
   * Put a box into the world. Returns its body id, or 0 when the scene carries no pool.
   *
   * Recycles the oldest slot once every box is out, rather than refusing: the alternative
   * is a throw that silently does nothing.
   */
  materialize(
    mujoco: MainModule,
    mjModel: MjModel,
    mjData: MjData,
    spec: SpawnSpec,
    pose: SpawnPose,
  ): number {
    if (this.slots.length === 0) return 0;
    const slot = this.slots[this.next % this.slots.length];
    this.next = (this.next + 1) % this.slots.length;

    if (slot.size !== spec.size || massOf(mjModel, slot) !== massFor(spec)) {
      this.reshape(mujoco, mjModel, mjData, slot, spec);
    }
    slot.live = true;
    this.writeSlot(mjData, slot, pose.position, pose.velocity);
    return slot.bodyId;
  }

  /**
   * Bring the drawing in line with the model: rebuild a box whose collider was resized,
   * and hide the slots that are parked rather than leaving them visible 100 m up.
   */
  syncMeshes(mjModel: MjModel, bodies: Record<number, THREE.Group> | null): void {
    if (!bodies) return;
    for (const slot of this.slots) {
      const group = bodies[slot.bodyId];
      if (!group) continue;
      group.visible = slot.live;
      if (!slot.live || slot.drawnSize === slot.size) continue;
      const mesh = group.children[0] as THREE.Mesh | undefined;
      if (!mesh) continue;
      slot.drawnSize = slot.size;
      mesh.geometry.dispose();
      // The scene builder's own box: full extents, with MuJoCo's y/z swizzle.
      const full = slot.size * 2;
      mesh.geometry = new THREE.BoxGeometry(full, full, full);
      void mjModel;
    }
  }

  /**
   * Resize and re-mass one slot.
   *
   * `body_mass` is inert until `mj_setConst` recomputes what the dynamics actually read,
   * and `mj_setConst` leaves `mjData` in the `qpos0` configuration — so the running state
   * is carried across it by hand. Everything else `mj_setConst` recomputes is derived from
   * the model, which is exactly what needs updating.
   */
  private reshape(
    mujoco: MainModule,
    mjModel: MjModel,
    mjData: MjData,
    slot: Slot,
    spec: SpawnSpec,
  ): void {
    for (let i = 0; i < 3; i++) mjModel.geom_size[slot.geomId * 3 + i] = spec.size;
    const mass = massFor(spec);
    mjModel.body_mass[slot.bodyId] = mass;
    // A cube's principal moments, all equal: m/12 * ((2s)^2 + (2s)^2).
    const inertia = (2 / 3) * mass * spec.size * spec.size;
    for (let i = 0; i < 3; i++) mjModel.body_inertia[slot.bodyId * 3 + i] = inertia;
    slot.size = spec.size;

    const qpos = Array.from(mjData.qpos.slice(0, mjModel.nq) as ArrayLike<number>);
    const qvel = Array.from(mjData.qvel.slice(0, mjModel.nv) as ArrayLike<number>);
    const act = Array.from(mjData.act.slice(0, mjModel.na) as ArrayLike<number>);
    const time = mjData.time;
    mujoco.mj_setConst(mjModel, mjData);
    for (let i = 0; i < mjModel.nq; i++) mjData.qpos[i] = qpos[i];
    for (let i = 0; i < mjModel.nv; i++) mjData.qvel[i] = qvel[i];
    for (let i = 0; i < mjModel.na; i++) mjData.act[i] = act[i];
    mjData.time = time;
  }

  private writeSlot(
    mjData: MjData,
    slot: Slot,
    position: readonly [number, number, number],
    velocity: readonly [number, number, number],
  ): void {
    for (let i = 0; i < 3; i++) mjData.qpos[slot.qposAdr + i] = position[i];
    mjData.qpos[slot.qposAdr + 3] = 1;
    for (let i = 1; i < 4; i++) mjData.qpos[slot.qposAdr + 3 + i] = 0;
    for (let i = 0; i < 3; i++) mjData.qvel[slot.dofAdr + i] = velocity[i];
    for (let i = 3; i < 6; i++) mjData.qvel[slot.dofAdr + i] = 0;
  }
}

function massFor(spec: SpawnSpec): number {
  return spec.density * 8 * spec.size * spec.size * spec.size;
}

function massOf(mjModel: MjModel, slot: Slot): number {
  return mjModel.body_mass[slot.bodyId];
}
