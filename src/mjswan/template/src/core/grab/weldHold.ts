/**
 * Holding a body by retargeting a weld declared with the model.
 *
 * A compiled model cannot gain a constraint, so each weld is injected inactive before the
 * scene loads and aimed at a pair of bodies when something grabs. Shared by the XR hand
 * and the pointer; what triggers a grab stays with each caller.
 */
import type { MainModule, MjData, MjModel } from 'mujoco';

import { quatApplyInv, quatInverse, quatMultiply } from '../observation/math';

export interface HoldOptions {
  /** `eq_data[10]`: 1 pins the orientation too, 0 lets the body swing from the grab. */
  torqueScale?: number;
  /** `eq_solref[0]`, seconds. Bigger is softer. Left alone when absent. */
  solrefTime?: number;
}

/** One slot's compiled values, for `restore`. */
interface SlotDefaults {
  obj1: number;
  obj2: number;
  data: number[];
  solref: number[];
}

/**
 * The weld slots a scene carries, and which body each holds. A body is held by at most one
 * slot, since two welds on it would fight in the solver.
 */
export class WeldHold {
  private neqData = 11;
  private nRef = 2;
  private readonly slots = new Map<string, number>();
  private readonly defaults = new Map<string, SlotDefaults>();
  private readonly holding = new Map<string, number>();

  /** Resolve the named slots on a freshly compiled model; names it lacks are skipped. */
  bind(mujoco: MainModule, mjModel: MjModel, names: readonly string[]): void {
    this.neqData = mujoco.mjNEQDATA;
    this.nRef = mujoco.mjNREF;
    this.slots.clear();
    this.defaults.clear();
    this.holding.clear();
    const equality = mujoco.mjtObj.mjOBJ_EQUALITY.value;
    for (const name of names) {
      const id = mujoco.mj_name2id(mjModel, equality, name);
      if (id < 0) continue;
      this.slots.set(name, id);
      this.defaults.set(name, {
        obj1: mjModel.eq_obj1id[id],
        obj2: mjModel.eq_obj2id[id],
        data: readRow(mjModel.eq_data, id, this.neqData),
        solref: readRow(mjModel.eq_solref, id, this.nRef),
      });
    }
  }

  has(name: string): boolean {
    return this.slots.has(name);
  }

  /** The slot holding `bodyId`, or null. */
  holderOf(bodyId: number): string | null {
    for (const [name, held] of this.holding) {
      if (held === bodyId) return name;
    }
    return null;
  }

  heldBy(name: string): number | null {
    return this.holding.get(name) ?? null;
  }

  /**
   * Weld `target` to `anchor` in its current pose, so the constraint holds it rather than
   * snapping it. Refuses a body another slot already holds.
   *
   * A weld's `eq_data` is `[anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]`,
   * and its relpose is body2 in body1's frame, the opposite of the obvious reading.
   */
  hold(
    mjModel: MjModel,
    mjData: MjData,
    name: string,
    anchor: number,
    target: number,
    options: HoldOptions = {},
  ): boolean {
    const id = this.slots.get(name);
    if (id === undefined || target <= 0 || anchor < 0) return false;
    const holder = this.holderOf(target);
    if (holder !== null && holder !== name) return false;

    const anchorQuat = [0, 1, 2, 3].map((i) => mjData.xquat[anchor * 4 + i]);
    const relPos = quatApplyInv(
      anchorQuat,
      [0, 1, 2].map((i) => mjData.xpos[target * 3 + i] - mjData.xpos[anchor * 3 + i]),
    );
    const relQuat = quatMultiply(
      quatInverse(anchorQuat),
      [0, 1, 2, 3].map((i) => mjData.xquat[target * 4 + i]),
    );
    const at = id * this.neqData;
    for (let i = 0; i < 3; i++) mjModel.eq_data[at + i] = 0;
    for (let i = 0; i < 3; i++) mjModel.eq_data[at + 3 + i] = relPos[i];
    for (let i = 0; i < 4; i++) mjModel.eq_data[at + 6 + i] = relQuat[i];
    mjModel.eq_data[at + 10] = options.torqueScale ?? 1;
    if (options.solrefTime !== undefined) {
      mjModel.eq_solref[id * this.nRef] = options.solrefTime;
    }
    mjModel.eq_obj1id[id] = anchor;
    mjModel.eq_obj2id[id] = target;
    mjData.eq_active[id] = 1;
    this.holding.set(name, target);
    return true;
  }

  /** Let go; the body keeps the momentum the hold gave it. */
  release(mjData: MjData, name: string): void {
    const id = this.slots.get(name);
    if (id === undefined) return;
    mjData.eq_active[id] = 0;
    this.holding.delete(name);
  }

  /**
   * Put every slot back as it compiled. `mj_resetData` restores `eq_active` but not
   * `eq_obj*id` / `eq_data` / `eq_solref`, which live on the model, where startup
   * randomization or a `modelFieldDefaults` snapshot would find a retargeted weld.
   */
  restore(mjModel: MjModel, mjData: MjData | null): void {
    for (const [name, id] of this.slots) {
      const compiled = this.defaults.get(name);
      if (!compiled) continue;
      mjModel.eq_obj1id[id] = compiled.obj1;
      mjModel.eq_obj2id[id] = compiled.obj2;
      writeRow(mjModel.eq_data, id, this.neqData, compiled.data);
      writeRow(mjModel.eq_solref, id, this.nRef, compiled.solref);
      if (mjData) mjData.eq_active[id] = 0;
    }
    this.holding.clear();
  }
}

function readRow(array: ArrayLike<number>, index: number, stride: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < stride; i++) out.push(array[index * stride + i]);
  return out;
}

function writeRow(
  array: { [index: number]: number },
  index: number,
  stride: number,
  values: readonly number[],
): void {
  for (let i = 0; i < stride; i++) array[index * stride + i] = values[i];
}
