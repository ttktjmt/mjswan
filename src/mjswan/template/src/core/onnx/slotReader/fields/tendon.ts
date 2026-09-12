/** mjlab's "Tendon properties". */
import { gather, type FieldReader } from './util';

export const TENDON_READERS: Record<string, FieldReader> = {
  tendon_len: (index, d) => gather(d.ten_length, index.tendonIds),
  tendon_vel: (index, d) => gather(d.ten_velocity, index.tendonIds),
};
