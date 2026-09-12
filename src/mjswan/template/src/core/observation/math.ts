export function normalizeQuat(quat: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const n = Math.hypot(w, x, y, z);
  if (n < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / n;
  return [w * inv, x * inv, y * inv, z * inv];
}

export function quatMultiply(a: ArrayLike<number>, b: ArrayLike<number>): number[] {
  const aw = a[0] ?? 1;
  const ax = a[1] ?? 0;
  const ay = a[2] ?? 0;
  const az = a[3] ?? 0;
  const bw = b[0] ?? 1;
  const bx = b[1] ?? 0;
  const by = b[2] ?? 0;
  const bz = b[3] ?? 0;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

export function quatInverse(quat: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const normSq = w * w + x * x + y * y + z * z;
  if (normSq < 1e-9) {
    return [1, 0, 0, 0];
  }
  const inv = 1.0 / normSq;
  return [w * inv, -x * inv, -y * inv, -z * inv];
}

/** Rotate `vec` by `quat` (w, x, y, z) — mjlab's `quat_apply`. */
export function quatApply(quat: ArrayLike<number>, vec: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const vx = vec[0] ?? 0;
  const vy = vec[1] ?? 0;
  const vz = vec[2] ?? 0;
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  const cx = y * tz - z * ty;
  const cy = z * tx - x * tz;
  const cz = x * ty - y * tx;
  return [vx + w * tx + cx, vy + w * ty + cy, vz + w * tz + cz];
}

export function quatApplyInv(quat: ArrayLike<number>, vec: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const vx = vec[0] ?? 0;
  const vy = vec[1] ?? 0;
  const vz = vec[2] ?? 0;
  const tx = 2.0 * (y * vz - z * vy);
  const ty = 2.0 * (z * vx - x * vz);
  const tz = 2.0 * (x * vy - y * vx);
  const cx = y * tz - z * ty;
  const cy = z * tx - x * tz;
  const cz = x * ty - y * tx;
  return [vx - w * tx + cx, vy - w * ty + cy, vz - w * tz + cz];
}

/** The yaw-only part of `quat` (w, x, y, z) — mjlab's `yaw_quat`. */
export function yawQuat(quat: ArrayLike<number>): number[] {
  const w = quat[0] ?? 1;
  const x = quat[1] ?? 0;
  const y = quat[2] ?? 0;
  const z = quat[3] ?? 0;
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  return [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
}

export function quatToRot6d(quat: ArrayLike<number>): number[] {
  const [w, x, y, z] = normalizeQuat(quat);
  const xx = x * x;
  const yy = y * y;
  const zz = z * z;
  const xy = x * y;
  const xz = x * z;
  const yz = y * z;
  const wx = w * x;
  const wy = w * y;
  const wz = w * z;

  const r00 = 1.0 - 2.0 * (yy + zz);
  const r01 = 2.0 * (xy - wz);
  const r10 = 2.0 * (xy + wz);
  const r11 = 1.0 - 2.0 * (xx + zz);
  const r20 = 2.0 * (xz - wy);
  const r21 = 2.0 * (yz + wx);

  // mjlab's `matrix_from_quat(q)[..., :2]`: the first two columns, row-major.
  return [r00, r01, r10, r11, r20, r21];
}

export function clampFutureIndices(
  base: number,
  steps: number[],
  length: number
): number[] {
  return steps.map((step) => {
    const idx = base + step;
    if (idx < 0) return 0;
    if (idx >= length) return Math.max(0, length - 1);
    return idx;
  });
}

/** `sqrt(max(0, x))` — mjlab's `_sqrt_positive_part`. */
function sqrtPositivePart(x: number): number {
  return x > 0 ? Math.sqrt(x) : 0;
}

/**
 * A rotation matrix (row-major, 9 values) as a quaternion (w, x, y, z) — mjlab's
 * `quat_from_matrix`, pytorch3d's algorithm: four candidates, the best-conditioned one
 * kept, so the sign convention is mjlab's and not merely equivalent up to sign.
 */
export function quatFromMatrix(matrix: ArrayLike<number>): number[] {
  const m = (i: number): number => matrix[i] ?? 0;
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = [
    m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8),
  ];
  const qAbs = [
    sqrtPositivePart(1 + m00 + m11 + m22),
    sqrtPositivePart(1 + m00 - m11 - m22),
    sqrtPositivePart(1 - m00 + m11 - m22),
    sqrtPositivePart(1 - m00 - m11 + m22),
  ];
  const candidates = [
    [qAbs[0] * qAbs[0], m21 - m12, m02 - m20, m10 - m01],
    [m21 - m12, qAbs[1] * qAbs[1], m10 + m01, m02 + m20],
    [m02 - m20, m10 + m01, qAbs[2] * qAbs[2], m12 + m21],
    [m10 - m01, m20 + m02, m21 + m12, qAbs[3] * qAbs[3]],
  ];
  // The first of equal maxima, as `torch.argmax` picks.
  let best = 0;
  for (let i = 1; i < 4; i++) if (qAbs[i] > qAbs[best]) best = i;
  // Floor of 0.1 keeps the divisor away from zero for the candidates not chosen.
  const scale = 2 * Math.max(qAbs[best], 0.1);
  return candidates[best].map(v => v / scale);
}
