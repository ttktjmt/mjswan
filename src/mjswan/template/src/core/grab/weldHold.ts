/**
 * Holding a body where you put it, by retargeting a weld that was declared with the model.
 *
 * MuJoCo compiles a model once and offers no way to add a constraint to a live one, so the
 * weld has to exist before the scene loads (declared `active="false"`, pointing at
 * nothing in particular) and is aimed at a pair of bodies when something actually grabs.
 * The XR hand has worked this way since it learned to carry a load rather than shove one;
 * this is that machinery with the hand taken out of it, so the pointer can use it too.
 *
 * What is **not** shared is the trigger. A hand grabs from contact and a pinch gesture; a
 * pointer grabs from a raycast. Those are different questions about the world and they
 * stay in their own callers. Only the hold is here.
 */
import type { MainModule, MjData, MjModel } from 'mujoco';

import { quatApplyInv, quatInverse, quatMultiply } from '../observation/math';

export interface HoldOptions {
  /** `eq_data[10]`: 1 pins the orientation too, 0 lets the body swing from the grab. */
  torqueScale?: number;
  /** `eq_solref[0]`, seconds. Bigger is softer. Left alone when absent. */
  solrefTime?: number;
}

/** One slot's compiled values, so a release can put the model back as it was. */
interface SlotDefaults {
  obj1: number;
  obj2: number;
  data: number[];
  solref: number[];
}

/**
 * The weld slots a scene carries, and which body each is holding.
 *
 * Bodies are held by at most one slot: a hand and a pointer pulling the same crate through
 * two welds is a fight the solver has no reason to win.
 */
export class WeldHold {
  private neqData = 11;
  private nRef = 2;
  private readonly slots = new Map<string, number>();
  private readonly defaults = new Map<string, SlotDefaults>();
  private readonly holding = new Map<string, number>();

  /**
   * Resolve every named slot against a freshly compiled model and remember what it
   * compiled to. Slots the model does not carry are dropped, so a scene built without an
   * injection simply has none.
   */
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
   * Weld `target` to `anchor` in the pose it is already in, so activating the constraint
   * holds it rather than snapping it. Refuses a body another slot already has.
   *
   * `eq_data` for a weld is `[anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]`,
   * and its relpose is body2 expressed in body1's frame, the opposite of the obvious
   * reading.
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

  /** Let go. The body keeps whatever momentum the hold gave it, which is how a throw works. */
  release(mjData: MjData, name: string): void {
    const id = this.slots.get(name);
    if (id === undefined) return;
    mjData.eq_active[id] = 0;
    this.holding.delete(name);
  }

  /**
   * Put every slot back exactly as it compiled.
   *
   * `mj_resetData` restores `eq_active` on its own, but `eq_obj*id` / `eq_data` /
   * `eq_solref` live on the **model**, so a reset leaves a retargeted weld aimed at
   * whatever it last grabbed, where the next policy's startup randomization, or a
   * `modelFieldDefaults` snapshot, would find it.
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
