/**
 * The engine's one writer of `mjData.xfrc_applied`, which a policy reads through the
 * `body_external_*` slots. Each step zeroes only the rows it wrote the step before, so
 * rows another writer set are left alone.
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

  /** Apply a world-frame force at a point. */
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

  /** Zero every row it wrote. */
  clear(mjData: MjData | null): void {
    if (mjData) for (const bodyId of this.written) clearWrench(mjData, bodyId);
    this.written.clear();
  }

  /** A new model: the old ids mean nothing, and its rows are already zero. */
  forget(): void {
    this.written.clear();
  }
}
