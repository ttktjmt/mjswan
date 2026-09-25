/**
 * What each pointer mode is, and every number it exposes.
 *
 * One table, because three places have to agree on it: the manager clamps against it, the
 * engine publishes it as descriptors, and the panel draws a control per entry. A mode with
 * no entry here has no UI and no API surface, which is the point: modes are a closed set
 * (ADR-less by design: there is no plugin path in, and adding one is a code change here).
 */

export type InteractionModeId = 'pull' | 'push' | 'weld';

export const INTERACTION_MODE_IDS: readonly InteractionModeId[] = ['pull', 'push', 'weld'];

export interface InteractionParamSpec {
  name: string;
  label: string;
  /** Printed beside the label; '' for a dimensionless one. */
  unit: string;
  /** A `checkbox` holds 0 or 1. */
  type: 'slider' | 'checkbox';
  /** What `setParam` clamps to, and so what the number box accepts. */
  min: number;
  max: number;
  /**
   * The span the slider drags over, inside `min`..`max`; absent means the same. A value
   * typed past it pins the thumb at the end, as Blender's soft limits do.
   */
  softMin?: number;
  softMax?: number;
  step: number;
  default: number;
}

export interface InteractionModeSpec {
  id: InteractionModeId;
  label: string;
  params: readonly InteractionParamSpec[];
}

// The hard limits are there to keep a typo out of the solver, not to judge what is too
// much: finite and non-negative, at about ten times what the slider reaches.
export const INTERACTION_MODES: readonly InteractionModeSpec[] = [
  {
    id: 'pull',
    label: 'Pull',
    params: [
      // The pre-mode viewer pulled with a hard-coded 100 N/m, so this default is the
      // drag people already know.
      {
        name: 'gain', label: 'Force', unit: 'N/m', type: 'slider',
        min: 0, max: 2000, softMin: 0, softMax: 200, step: 1, default: 100,
      },
      // There was no clamp before, and a camera that follows a body can jump the pointer
      // offset far enough in one frame to launch whatever is held.
      {
        name: 'maxForce', label: 'Max force', unit: 'N', type: 'slider',
        min: 1, max: 20000, softMin: 1, softMax: 500, step: 1, default: 500,
      },
    ],
  },
  {
    id: 'push',
    label: 'Push',
    params: [
      // An impulse, not a force: divided by the control step on the way in, so the shove
      // feels the same whatever a scene's decimation is.
      {
        name: 'impulse', label: 'Impulse', unit: 'N·s', type: 'slider',
        min: 0.1, max: 2000, softMin: 0.1, softMax: 200, step: 0.1, default: 10,
      },
    ],
  },
  {
    id: 'weld',
    label: 'Grab',
    params: [
      // `eq_solref[0]`, the weld's time constant: how long the body takes to catch up with
      // the pointer. Below ~4 ms the constraint fights the solver.
      {
        name: 'softness', label: 'Softness', unit: 's', type: 'slider',
        min: 0.004, max: 0.2, step: 0.002, default: 0.02,
      },
      // `eq_data[10]`, continuous to MuJoCo, but between 0 (hang from the grab point) and 1
      // (keep the pose) nothing reads as a different grip.
      {
        name: 'torqueScale', label: 'Hold rotation', unit: '', type: 'checkbox',
        min: 0, max: 1, step: 1, default: 1,
      },
    ],
  },
];

export const DEFAULT_INTERACTION_MODE: InteractionModeId = 'pull';

export function modeSpec(id: InteractionModeId): InteractionModeSpec {
  const spec = INTERACTION_MODES.find((m) => m.id === id);
  if (!spec) throw new Error(`Unknown interaction mode: ${id}`);
  return spec;
}

/** Every mode's parameters at their defaults. */
export function defaultParams(): Record<InteractionModeId, Record<string, number>> {
  const out = {} as Record<InteractionModeId, Record<string, number>>;
  for (const mode of INTERACTION_MODES) {
    const values: Record<string, number> = {};
    for (const param of mode.params) values[param.name] = param.default;
    out[mode.id] = values;
  }
  return out;
}

/** The value a parameter will actually take: clamped to its hard range, never NaN. */
export function clampParam(spec: InteractionParamSpec, value: number): number {
  if (!Number.isFinite(value)) return spec.default;
  if (spec.type === 'checkbox') return value >= 0.5 ? 1 : 0;
  return Math.min(spec.max, Math.max(spec.min, value));
}
