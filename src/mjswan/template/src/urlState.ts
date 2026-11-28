/** Query-parameter view state, so a copied link reopens the same view (guides/embedding.md). */

export const PROJECT_PARAM = 'project';
export const SCENE_PARAM = 'scene';
export const POLICY_PARAM = 'policy';
export const SKIN_PARAM = 'skin';
export const PANEL_PARAM = 'panel';
export const DEBUG_PARAM = 'debug';
export const XRAY_PARAM = 'xray';
export const MANIFEST_PARAM = 'manifest';

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

/**
 * The manifest a `?manifest=` link names, resolved against the page, or null without one.
 * One on another origin is refused: the document's files load from the page's own base
 * wherever its manifest is, so another site's could only put its text on this one's files.
 */
export function manifestOverrideUrl(search: string, page: string): string | null {
  const value = new URLSearchParams(search).get(MANIFEST_PARAM);
  if (!value) return null;
  const url = new URL(value, page);
  const origin = new URL(page).origin;
  if (url.origin !== origin) {
    throw new Error(`?manifest= names a manifest outside ${origin}, which this page does not open.`);
  }
  return url.href;
}

/** Where a build writes the custom-TS plugins module, beside the engine. */
export const PLUGINS_PATH = 'assets/plugins.js';

/**
 * Where the build's plugins module loads from, or null when it must not load. A manifest
 * named by `?manifest=` belongs to whoever wrote the link, so its plugins never run, and
 * the module must be the one a build writes, on the page's origin: `assets/plugins.js`.
 */
export function pluginModuleUrl(
  base: string,
  pluginsPath: string | undefined,
  search: string,
  page: string,
): string | null {
  if (pluginsPath !== PLUGINS_PATH || new URLSearchParams(search).has(MANIFEST_PARAM)) return null;
  const url = new URL(pluginsPath, new URL(base, page));
  return url.origin === new URL(page).origin ? url.href : null;
}
