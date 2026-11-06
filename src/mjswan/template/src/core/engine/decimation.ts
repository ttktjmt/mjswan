/** Physics substeps per control step. Real tasks use tens; one step past this cannot keep pace. */
export const MAX_DECIMATION = 1000;

/**
 * Physics steps per control step for a `control_dt` and a model timestep, both from the
 * document. Refused rather than clamped past MAX_DECIMATION: every substep runs before the
 * browser gets the thread back, and a clamped rate would run a policy off-speed.
 */
export function decimationFor(controlDt: number, timestep: number): number {
  const decimation = Math.round(controlDt / timestep);
  if (!(Number.isFinite(decimation) && decimation <= MAX_DECIMATION)) {
    throw new Error(
      `mjswan: control_dt ${controlDt} over timestep ${timestep} needs ${decimation} ` +
        `physics substeps per control step; at most ${MAX_DECIMATION} are allowed.`,
    );
  }
  return Math.max(1, decimation);
}
