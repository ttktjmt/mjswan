import type { MainModule, MjData, MjModel } from 'mujoco';
import { TENDON_RGB, type TendonPaint } from '../scene/tendons';
import { MUSCLE_UNIFORM, muscleActuators, muscleColour } from './tubes';

/** How muscles are drawn, in the terms of MyoSuite's `RenderConfig` (ADR 0012). */
export interface MuscleView {
  /** MyoSuite's `muscles`: `paths` colours each muscle's tendon path. */
  style: 'paths';
  /** MyoSuite's `muscle_color`: by activation, or one anatomical red. */
  color: 'activation' | 'uniform';
}

export const DEFAULT_MUSCLE_VIEW: MuscleView = { style: 'paths', color: 'activation' };

const STYLES: readonly unknown[] = ['paths'];
const COLORS: readonly unknown[] = ['activation', 'uniform'];

/** `view` over `base`; a value `view` gets wrong keeps `base`'s, with a warning. */
export function mergeMuscleView(base: MuscleView, view: Partial<MuscleView>): MuscleView {
  const pick = <K extends keyof MuscleView>(key: K, valid: (value: unknown) => boolean): MuscleView[K] => {
    const value = view[key];
    if (value === undefined) return base[key];
    if (valid(value)) return value as MuscleView[K];
    console.warn(`[muscles] ignored ${key}: ${String(value)}`);
    return base[key];
  };
  return {
    style: pick('style', (v) => STYLES.includes(v)),
    color: pick('color', (v) => COLORS.includes(v)),
  };
}

/** A model's muscles, drawn as the view says; built with the model and dropped with it. */
export class Muscles {
  /** Read by the tendon drawing: each muscle's tendon in the view's colour. */
  readonly paint: TendonPaint;
  private readonly actuators: Int32Array;
  private readonly actadr: Int32Array;
  private readonly tendons: Int32Array;
  private view: MuscleView | null = null;

  /** Null for a model without a muscle MyoSuite would draw. */
  static create(mujoco: MainModule, model: MjModel): Muscles | null {
    const actuators = muscleActuators(mujoco, model);
    return actuators.length ? new Muscles(model, actuators) : null;
  }

  private constructor(model: MjModel, actuators: number[]) {
    const trnid = model.actuator_trnid;
    const actadr = model.actuator_actadr;
    this.actuators = Int32Array.from(actuators);
    this.tendons = Int32Array.from(actuators, (a) => trnid[2 * a]);
    this.actadr = Int32Array.from(actuators, (a) => actadr[a]);
    const rgb = new Float32Array(3 * model.ntendon);
    for (let t = 0; t < model.ntendon; t++) rgb.set(TENDON_RGB, 3 * t);
    this.paint = { rgb, hidden: new Uint8Array(model.ntendon) };
  }

  get count(): number {
    return this.actuators.length;
  }

  setView(view: MuscleView | null): void {
    this.view = view;
    const rgb = view?.color === 'uniform' ? MUSCLE_UNIFORM : TENDON_RGB;
    for (const t of this.tendons) this.paint.rgb.set(rgb, 3 * t);
  }

  /** After each forward: colours from the activations it left. */
  update(data: MjData): void {
    if (this.view?.color !== 'activation') return;
    const act = data.act;
    const ctrl = data.ctrl;
    for (let i = 0; i < this.actuators.length; i++) {
      // Muscle gain without muscle dynamics keeps no activation: the control stands in.
      const adr = this.actadr[i];
      muscleColour(adr >= 0 ? act[adr] : ctrl[this.actuators[i]], true, this.paint.rgb, 3 * this.tendons[i]);
    }
  }
}
