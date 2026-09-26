/**
 * The pointer's grab, declared with the model: one mocap anchor and one inactive weld,
 * adding no degrees of freedom. Names stay flat (no `/`, see `scene/mjcfInject`).
 */
import { appendToMjcf } from '../scene/mjcfInject';

/** The pointer's handle: a mocap body the grabbed thing is welded to. */
export const POINTER_ANCHOR_BODY = 'mjswan_ptr_anchor';
export const POINTER_WELD = 'mjswan_ptr_hold';

/** Where the anchor waits between grabs, above any scene. */
export const POINTER_PARK_Z = 100;

/**
 * `solref` / `solimp` are the hand's suspension values (`WELD` in `xr/handMocap`), soft so
 * a carried body stops at a surface rather than punching through. `softness` overrides
 * `solref[0]` per grab.
 */
const GRAB_BLOCK =
  `  <worldbody>\n` +
  `    <body name="${POINTER_ANCHOR_BODY}" mocap="true" pos="0 0 ${POINTER_PARK_Z}"/>\n` +
  `  </worldbody>\n` +
  `  <equality>\n` +
  `    <weld name="${POINTER_WELD}" body1="${POINTER_ANCHOR_BODY}" body2="world"` +
  ` active="false" torquescale="1" solref="0.02 1" solimp="0.95 0.99 0.001"/>\n` +
  `  </equality>\n`;

export function injectPointerGrabXml(xml: string): string {
  return appendToMjcf(xml, GRAB_BLOCK);
}
