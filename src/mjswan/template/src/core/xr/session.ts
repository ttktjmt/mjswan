/** WebXR session options, and the reports a host draws its own XR buttons from. */
import type { ModelFormat } from '../scene/scene';

export type XrSessionId = 'vr' | 'ar';

export const XR_SESSION_IDS: readonly XrSessionId[] = ['vr', 'ar'];

export const XR_SESSION_MODES: Readonly<Record<XrSessionId, XRSessionMode>> = {
  vr: 'immersive-vr',
  ar: 'immersive-ar',
};

const LABELS: Readonly<Record<XrSessionId, { enter: string; exit: string }>> = {
  vr: { enter: 'Enter VR', exit: 'Exit VR' },
  ar: { enter: 'Start AR', exit: 'Stop AR' },
};

export interface XrSessionReport {
  id: XrSessionId;
  label: string;
  active: boolean;
}

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
 * VR asks for three's `VRButton` defaults. AR asks for `local-floor` so MuJoCo's z = 0
 * lands on the real floor rather than at eye height, where `local` would put it.
 */
export function xrSessionInit(id: XrSessionId, hands: boolean): XRSessionInit {
  const features = id === 'vr' ? ['local-floor', 'bounded-floor', 'layers'] : ['local-floor'];
  // A Quest leaves hands untracked unless the session asks, in either mode.
  if (hands) features.push('hand-tracking');
  return { optionalFeatures: features };
}

/**
 * Why the loaded scene cannot take the hand bodies, or null when it can. Same rule as the
 * grab anchor in `injectViewerBodies`.
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
