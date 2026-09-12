/** mjlab's "Joint properties": the entity's non-free joints, by qpos / dof address. */
import { gather, type FieldReader } from './util';

export const JOINT_READERS: Record<string, FieldReader> = {
  joint_pos: (index, d) => gather(d.qpos, index.qposAdr),
  joint_pos_biased: (index, d) => {
    const out = gather(d.qpos, index.qposAdr);
    for (let i = 0; i < out.length; i++) out[i] += index.jointBias[i] ?? 0;
    return out;
  },
  joint_vel: (index, d) => gather(d.qvel, index.qvelAdr),
  joint_acc: (index, d) => gather(d.qacc, index.qvelAdr),
};
