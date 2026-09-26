/**
 * Every pointer mode and the parameters it exposes: the one table the manager clamps
 * against, the engine publishes as descriptors and the panel draws. The set is closed, so
 * a new mode is a code change here.
 */

export type InteractionModeId = 'view' | 'pull' | 'push' | 'weld';

export const INTERACTION_MODE_IDS: readonly InteractionModeId[] = ['view', 'pull', 'push', 'weld'];

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
  /** The slider's span inside `min`..`max`, like Blender's soft limits; absent means the same. */
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

// Hard limits only keep a typo out of the solver, so where a slider has a soft range they
// sit about ten times past it.
export const INTERACTION_MODES: readonly InteractionModeSpec[] = [
  // Every press is the camera's, for a scene whose bodies leave no empty sky to orbit from.
  { id: 'view', label: 'View', params: [] },
  {
    id: 'pull',
    label: 'Pull',
    params: [
      // Between the grab point and the pointer.
      {
        name: 'spring', label: 'Spring', unit: 'N/m', type: 'slider',
        min: 0, max: 2000, softMin: 0, softMax: 200, step: 1, default: 100,
      },
    ],
  },
  {
    id: 'push',
    label: 'Push',
    params: [
      {
        name: 'impulse', label: 'Impulse', unit: 'N·s', type: 'slider',
        min: 0.1, max: 500, softMin: 0.1, softMax: 50, step: 0.1, default: 10,
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
