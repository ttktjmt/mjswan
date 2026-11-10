/**
 * Applies a traced event's `kind: "model"` write targets to `mjModel`. The graph did the
 * sampling and the math, so this only puts each value in its element's cell.
 */

import {
  modelField,
  modelRowWidth,
  modelRows,
  writableModelField,
  type ModelElements,
} from '../onnx/slotReader/model';
import type { WriteTarget, WriteValues } from './entityWrite';

type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MainModule = import('mujoco').MainModule;

export interface ModelWriteTarget extends ModelElements {
  kind: 'model';
  /** The `mjModel` field written, e.g. `geom_friction`. */
  field: string;
  /** Per output value, in order: its element (an index into `names`) and its offset
   * within that element's row. */
  cells: Array<[number, number]>;
  /** The one graph output holding the values. */
  outputs: string[];
}

export function isModelWriteTarget(t: WriteTarget | ModelWriteTarget): t is ModelWriteTarget {
  return t.kind === 'model';
}

/**
 * The compiled field values, snapshotted on first touch, so a second `add`/`scale` event
 * on one axis offsets the compiled value rather than the first event's output.
 *
 * One per model, not per pass (ADR 0006 §9): an MDP switch re-runs `mode="startup"`
 * randomization, which must start from the compiled values rather than from what the
 * previous MDP left behind, so `restore()` puts every touched field back first.
 */
export class ModelFieldDefaults {
  private readonly snapshots = new Map<string, Float64Array>();

  constructor(private readonly mjModel: MjModel) {}

  /** The field as compiled. Snapshots it if this is the first read. */
  base(field: string): ArrayLike<number> | undefined {
    const cached = this.snapshots.get(field);
    if (cached) return cached;
    const live = modelField(this.mjModel, field);
    if (!live) return undefined;
    const copy = Float64Array.from(live);
    this.snapshots.set(field, copy);
    return copy;
  }

  /**
   * Write every snapshotted field back to its compiled value; false when there was
   * nothing to write. A true return leaves the caller owing an `mj_setConst`.
   */
  restore(): boolean {
    if (this.snapshots.size === 0) return false;
    for (const [field, compiled] of this.snapshots) {
      const live = modelField(this.mjModel, field) as { [index: number]: number } | undefined;
      if (!live) continue;
      for (let i = 0; i < compiled.length; i++) live[i] = compiled[i];
    }
    return true;
  }
}

/**
 * Write one target into `mjModel`, returning whether it was written. A target naming an
 * element this model lacks, or a cell past its row, writes nothing rather than part.
 */
export function applyModelWrite(
  mjModel: MjModel,
  target: ModelWriteTarget,
  values: WriteValues,
  defaults?: ModelFieldDefaults | null,
): boolean {
  const live = writableModelField(mjModel, target.field);
  const written = values[target.outputs[0]];
  const rows = live ? modelRows(mjModel, target) : null;
  const width = live ? modelRowWidth(mjModel, live, target.element) : null;
  if (
    !rows ||
    width === null ||
    !written ||
    written.length < target.cells.length ||
    target.cells.some(([element, offset]) => element >= rows.length || offset >= width)
  ) {
    console.warn(
      `[modelWrite] cannot write mjModel.${target.field} for ${target.element} ` +
        `${(target.names ?? []).join(', ')} in this model; skipping.`,
    );
    return false;
  }
  // Before the first write, so `restore()` puts back the compiled value.
  defaults?.base(target.field);
  const field = live as unknown as { [index: number]: number };
  target.cells.forEach(([element, offset], i) => {
    field[rows[element] * width + offset] = written[i];
  });
  return true;
}

/**
 * `mj_setConst` after a write that leaves MuJoCo's derived constants stale, keeping
 * `qpos`: MuJoCo's C call leaves it at `qpos0`, while mjlab's (mujoco_warp's) restores it.
 */
export function setConstKeepingQpos(mujoco: MainModule, mjModel: MjModel, mjData: MjData): void {
  const qpos = Float64Array.from(mjData.qpos);
  mujoco.mj_setConst(mjModel, mjData);
  const live = mjData.qpos;
  for (let i = 0; i < qpos.length; i++) live[i] = qpos[i];
}
