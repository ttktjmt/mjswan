/**
 * The bodies and constraints the pointer needs, declared with the model.
 *
 * One mocap body to hold onto and one inactive weld to hold with, with no degrees of
 * freedom between them, so a scene that never grabs anything pays nothing for carrying it. Both
 * names are flat: a `/` in an injected name changes how `slotReader/indexing.ts` reads
 * the *whole* model (see `scene/mjcfInject`).
 */
import { appendToMjcf } from '../scene/mjcfInject';

/** The pointer's handle: a mocap body the grabbed thing is welded to. */
export const POINTER_ANCHOR_BODY = 'mjswan_ptr_anchor';
export const POINTER_WELD = 'mjswan_ptr_hold';

/** Where the anchor waits: above any scene, clear of a floor plane that is solid downward. */
export const POINTER_PARK_Z = 100;

/**
 * `solref` and `solimp` are the hand's values, which were tuned so a grabbed object stops
 * at a surface instead of punching through it and so a reacquired grip does not arrive as
 * one impulse. `softness` retunes the first number per grab.
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
