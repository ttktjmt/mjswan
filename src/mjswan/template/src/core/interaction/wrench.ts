/**
 * The one writer of `mjData.xfrc_applied`.
 *
 * Nothing else in the engine writes that field (MDP event terms reach `qpos` / `qvel`
 * and stop there), but plenty *reads* it: `body_external_force` / `_torque` / `_wrench`
 * are real slot readers, so whatever a mode puts here is visible to a policy one step
 * later. That makes "exactly one thing writes it" worth stating rather than inheriting.
 *
 * The drag this replaces kept that property by zeroing the whole array every control
 * step. This zeroes only the rows it wrote last step, which is the same guarantee for the
 * modes and leaves any future writer alone.
 */
import type { MainModule, MjData, MjModel, MjvPerturb } from 'mujoco';

import { applyDragPull, applyPointForce, clearWrench, wrenchForce, type DragPull } from './perturbForce';

export class InteractionWrench {
  private written = new Set<number>();

  /** Start a control step: undo last step's rows. Call before any mode writes. */
  begin(mjData: MjData): void {
    for (const bodyId of this.written) clearWrench(mjData, bodyId);
    this.written.clear();
  }

  /** Pull a body toward a point. Returns the force it now carries, newtons. */
  pull(mujoco: MainModule, mjModel: MjModel, mjData: MjData, perturb: MjvPerturb, pull: DragPull): number {
    applyDragPull(mujoco, mjModel, mjData, perturb, pull);
    this.written.add(pull.bodyId);
    return wrenchForce(mjData, pull.bodyId);
  }

  /** Shove a body at a point along a direction. */
  push(
    mujoco: MainModule,
    mjData: MjData,
    bodyId: number,
    worldPoint: readonly [number, number, number],
    force: readonly [number, number, number],
  ): void {
    applyPointForce(mujoco, mjData, bodyId, worldPoint, force);
    this.written.add(bodyId);
  }

  /** Drop everything, for a cancelled gesture or a paused viewer. */
  clear(mjData: MjData | null): void {
    if (mjData) for (const bodyId of this.written) clearWrench(mjData, bodyId);
    this.written.clear();
  }

  /** A new model: the old ids mean nothing, and its rows are already zero. */
  forget(): void {
    this.written.clear();
  }
}
