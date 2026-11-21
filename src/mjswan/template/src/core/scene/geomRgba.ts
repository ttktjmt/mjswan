/** The `mjModel` fields a geom's colour is resolved from. */
export interface GeomRgbaModel {
  geom_rgba: ArrayLike<number>;
  geom_matid: ArrayLike<number>;
  mat_rgba: ArrayLike<number>;
}

export type Rgba = [number, number, number, number];

/**
 * The colour MuJoCo draws geom `g` in, as its `setMaterial` resolves it: the material's,
 * unless the geom's own `rgba` differs from the default (0.5, 0.5, 0.5, 1), which then wins.
 * MuJoCo draws nothing for a result whose alpha is 0.
 */
export function geomRgba(model: GeomRgbaModel, g: number): Rgba {
  const own: Rgba = [
    model.geom_rgba[g * 4],
    model.geom_rgba[g * 4 + 1],
    model.geom_rgba[g * 4 + 2],
    model.geom_rgba[g * 4 + 3],
  ];
  const material = model.geom_matid[g];
  const isDefault = own[0] === 0.5 && own[1] === 0.5 && own[2] === 0.5 && own[3] === 1;
  if (material < 0 || !isDefault) return own;
  return [
    model.mat_rgba[material * 4],
    model.mat_rgba[material * 4 + 1],
    model.mat_rgba[material * 4 + 2],
    model.mat_rgba[material * 4 + 3],
  ];
}
