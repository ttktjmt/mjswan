/**
 * What each pointer mode is, and every number it exposes.
 *
 * One table, because three places have to agree on it: the manager clamps against it, the
 * engine publishes it as descriptors, and the panel draws a slider per entry. A mode with
 * no entry here has no UI and no API surface, which is the point — modes are a closed set
 * (ADR-less by design: there is no plugin path in, and adding one is a code change here).
 */

export type InteractionModeId = 'pull' | 'push' | 'weld';

export const INTERACTION_MODE_IDS: readonly InteractionModeId[] = ['pull', 'push', 'weld'];

export interface InteractionParamSpec {
  name: string;
  label: string;
  /** Printed beside the value; '' for a dimensionless one. */
  unit: string;
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface InteractionModeSpec {
  id: InteractionModeId;
  label: string;
  /** One line, shown under the mode switch. */
  hint: string;
  params: readonly InteractionParamSpec[];
}

export const INTERACTION_MODES: readonly InteractionModeSpec[] = [
  {
    id: 'pull',
    label: 'Pull',
    hint: 'Drag a body toward the pointer.',
    params: [
      // The pre-mode viewer pulled with a hard-coded 100 N/m, so this default is the
      // drag people already know.
      { name: 'gain', label: 'Force', unit: 'N/m', min: 0, max: 2000, step: 10, default: 100 },
      // There was no clamp before, and a camera that follows a body can jump the pointer
      // offset far enough in one frame to launch whatever is held.
      { name: 'maxForce', label: 'Max force', unit: 'N', min: 1, max: 20000, step: 10, default: 500 },
    ],
  },
  {
    id: 'push',
    label: 'Push',
    hint: 'Tap a body to shove it along the surface you hit.',
    params: [
      // An impulse, not a force: divided by the control step on the way in, so the shove
      // feels the same whatever a scene's decimation is.
      { name: 'impulse', label: 'Impulse', unit: 'N·s', min: 0.1, max: 500, step: 0.5, default: 10 },
    ],
  },
  {
    id: 'weld',
    label: 'Grab',
    hint: 'Hold a body where you put it; let go and it keeps the speed.',
    params: [
      // `eq_solref[0]`, a time constant: below ~4 ms the constraint fights the solver.
      { name: 'softness', label: 'Give', unit: 's', min: 0.004, max: 0.2, step: 0.002, default: 0.02 },
      // `eq_data[10]`. At 0 the body hangs from the grab point instead of holding its pose.
      { name: 'torqueScale', label: 'Hold rotation', unit: '', min: 0, max: 1, step: 0.05, default: 1 },
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

/** The value a parameter will actually take: clamped to its own range, never NaN. */
export function clampParam(spec: InteractionParamSpec, value: number): number {
  if (!Number.isFinite(value)) return spec.default;
  return Math.min(spec.max, Math.max(spec.min, value));
}
