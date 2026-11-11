/**
 * three.js layers a camera sensor selects what it draws with. The viewer's camera stays on
 * layer 0, where everything the viewer draws sits; a geom is also on its group's layer,
 * and a light on the sensor light layer, so a sensor sees geoms and lights and nothing
 * else (no skin, splat or debug drawing), as MuJoCo's renderer does.
 */

/** Groups 0 to 2 are drawn by the viewer; the rest only for a camera sensor asking. */
export const VIEWER_GEOM_GROUPS = 3;

export const SENSOR_LIGHT_LAYER = 1;

/** Geom groups 0 to 5 on layers 2 to 7. */
export function geomGroupLayer(group: number): number {
  return 2 + group;
}
