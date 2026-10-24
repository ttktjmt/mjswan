/** Query-parameter view state, so a copied link reopens the same view (guides/embedding.md). */

export const PROJECT_PARAM = 'project';
export const SCENE_PARAM = 'scene';
export const POLICY_PARAM = 'policy';
export const SKIN_PARAM = 'skin';
export const PANEL_PARAM = 'panel';
export const DEBUG_PARAM = 'debug';
export const XRAY_PARAM = 'xray';

export interface UrlState {
  /** Id of the active entry, or null to leave the parameter off. */
  project: string | null;
  scene: string | null;
  policy: string | null;
  /** As above, or `''` while the Skin menu is set to None. */
  skin: string | null;
  panel: boolean;
  /** Debug Viz "Enabled". */
  debug: boolean;
  /** Debug Viz "Xray": the robot drawn see-through over its skin. */
  xray: boolean;
}

/** Rewrite a query string to carry `state`, leaving unrelated parameters alone. */
export function applyUrlState(search: string, state: UrlState): string {
  const params = new URLSearchParams(search);
  const set = (key: string, value: string | null) =>
    value === null ? params.delete(key) : params.set(key, value);
  set(PROJECT_PARAM, state.project);
  set(SCENE_PARAM, state.scene);
  set(POLICY_PARAM, state.policy);
  set(SKIN_PARAM, state.skin);
  // Booleans are written only off their default, so a default view keeps a bare URL.
  set(PANEL_PARAM, state.panel ? null : '0');
  set(DEBUG_PARAM, state.debug ? null : '0');
  set(XRAY_PARAM, state.xray ? '1' : null);
  return params.toString();
}
