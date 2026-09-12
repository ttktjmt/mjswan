/**
 * mjlab's "Generalized forces". `joint_torques` is the one property without a reader:
 * mjlab raises on it too, naming `qfrc_actuator` / `qfrc_external` instead.
 */
import { gather, type FieldReader } from './util';

export const FORCE_READERS: Record<string, FieldReader> = {
  actuator_force: (index, d) => gather(d.actuator_force, index.ctrlIds),
  qfrc_actuator: (index, d) => gather(d.qfrc_actuator, index.qvelAdr),
  // MuJoCo folds J^T xfrc_applied into qfrc_smooth without storing it; mjlab recovers
  // it from the qfrc_smooth identity, per DoF.
  qfrc_external: (index, d) => {
    const smooth = gather(d.qfrc_smooth, index.qvelAdr);
    const actuator = gather(d.qfrc_actuator, index.qvelAdr);
    const applied = gather(d.qfrc_applied, index.qvelAdr);
    const passive = gather(d.qfrc_passive, index.qvelAdr);
    const bias = gather(d.qfrc_bias, index.qvelAdr);
    const out = new Float32Array(smooth.length);
    for (let i = 0; i < out.length; i++) {
      out[i] = smooth[i] - actuator[i] - applied[i] - passive[i] + bias[i];
    }
    return out;
  },
};
