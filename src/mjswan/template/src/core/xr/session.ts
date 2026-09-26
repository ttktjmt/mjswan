/**
 * What a WebXR session asks for, and what the host is told it can start. The engine draws
 * no button: a host renders these reports and calls `engine.xr.enter` from its own click,
 * which is the gesture `requestSession` requires.
 */
import type { ModelFormat } from '../scene/scene';

export type XrSessionId = 'vr' | 'ar';

export const XR_SESSION_IDS: readonly XrSessionId[] = ['vr', 'ar'];

export const XR_SESSION_MODES: Readonly<Record<XrSessionId, XRSessionMode>> = {
  vr: 'immersive-vr',
  ar: 'immersive-ar',
};

/** What a host prints for each session, to enter it and while it runs. */
const LABELS: Readonly<Record<XrSessionId, { enter: string; exit: string }>> = {
  vr: { enter: 'Enter VR', exit: 'Exit VR' },
  ar: { enter: 'Start AR', exit: 'Stop AR' },
};

export interface XrSessionReport {
  id: XrSessionId;
  /** The action a press takes now: entering the session, or leaving it while it runs. */
  label: string;
  active: boolean;
}

/** Every session this device can start, in a fixed order, with the one running marked. */
export function xrSessionReports(
  supported: Readonly<Record<XrSessionId, boolean>>,
  active: XrSessionId | null,
): XrSessionReport[] {
  return XR_SESSION_IDS.filter((id) => supported[id]).map((id) => ({
    id,
    label: active === id ? LABELS[id].exit : LABELS[id].enter,
    active: active === id,
  }));
}

/**
 * VR asks for what three's `VRButton` did. AR asks for `local-floor` so MuJoCo's z = 0
 * lands on the real floor rather than at eye height, where `local` would put it.
 */
export function xrSessionInit(id: XrSessionId, hands: boolean): XRSessionInit {
  const features = id === 'vr' ? ['local-floor', 'bounded-floor', 'layers'] : ['local-floor'];
  // A Quest leaves hands untracked unless the session asks, in either mode.
  if (hands) features.push('hand-tracking');
  return { optionalFeatures: features };
}

/**
 * Why the loaded scene cannot take the hand bodies, or null when it can. The same rule
 * keeps the pointer's grab anchor out of such a model (`injectViewerBodies`): an
 * unprefixed model's bodies all count as its entity's, so a policy's traced graphs would
 * be fed a wider input.
 */
export function handTrackingBlocker(scene: {
  format: ModelFormat;
  prefixed: boolean;
  hasPolicy: boolean;
}): string | null {
  if (scene.format === 'mjb') return 'This scene was built as a compiled model, so hands cannot be added.';
  if (scene.hasPolicy && !scene.prefixed) return "Hands would change what this scene's policy reads.";
  return null;
}
