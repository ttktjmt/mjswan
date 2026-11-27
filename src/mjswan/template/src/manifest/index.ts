/**
 * `mjswan/manifest`: parse a build's `manifest.json` + a byte source into a typed catalog
 * of loadable scenes/policies/splats/skins. Framework-free; shared by the in-repo React
 * app and mjswan Cloud. The engine itself knows nothing about the manifest; the app owns
 * the catalog and calls engine verbs.
 *
 * The manifest is the one descriptor of a document: every key `snake_case`, every path
 * under a scene entry relative to `<project-id>/<scene-id>/`, under a project's `work`
 * relative to `<project-id>/`, the top-level `plugins` relative to the document root.
 * Ids and paths are checked at parse (`path.ts`); a URL the document names is fetched only
 * through the host's `remote`.
 *
 * See context/document.md.
 */
import { resolveBytes, type Bytes } from '../core/utils/bytes';
import type { ViewerConfig } from '../core/engine/viewer_config';
import type { EventConfig, TerrainData } from '../core/event/EventBase';
import type { SplatTransform } from '../core/scene/splat';
import type { SkinConfig } from '../core/skin/config';
import { policyGraphRefs } from '../core/onnx/graphRefs';
import {
  DEFAULT_GRAPHICS,
  mergeGraphics,
  type ColorSpaceName,
  type GraphicsSettings,
  type ColorManagement,
  type ToneMappingName,
} from '../core/scene/graphicsSettings';
import type { PolicyInput, SceneInput, SkinInput, SplatInput } from '../engine/types';
import type { OnFire, TermDefaults } from '../core/mdp/operator';
import { documentName, documentPath, isUrl, remoteUrl } from './path';
import { isReleaseVersion } from './version';

export { documentUrl } from './path';
export { compareVersions, ENGINE_CDN, engineUrl, isReleaseVersion } from './version';

/** Maps a checked document-relative path (e.g. `demo/humanoid/scene.mjz`) to bytes. */
export type ByteSource = (relPath: string) => Bytes;

export interface ParseOptions {
  /**
   * Bytes for a splat `url` or `collider_url`. Pass it only where the document's author may
   * choose the hosts viewers fetch from; without it such a splat is left out and such a
   * collider dropped.
   */
  remote?: (url: URL) => Bytes;
}

// ── manifest.json shape (Python Builder ↔ consumer contract) ─────────────────
export interface ManifestCamera {
  lookat?: [number, number, number];
  distance?: number;
  fovy?: number;
  elevation?: number;
  azimuth?: number;
  origin_type?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  entity_name?: string;
  body_name?: string;
  height?: number;
  width?: number;
}
/** How a scene is drawn (format 9): `ViewerConfig`'s graphics settings, resolved at build time. */
export interface ManifestGraphics {
  color_management?: ColorManagement;
  tone_mapping?: ToneMappingName;
  /** In stops. */
  exposure?: number;
  output_color_space?: ColorSpaceName;
  base_color_type?: ColorSpaceName;
  enable_shadows?: boolean;
  enable_reflections?: boolean;
  enable_haze?: boolean;
}
export interface ManifestSplat {
  id: string;
  name: string;
  /** Scene-relative path of a bundled `.spz` (`assets/<id>.spz`). */
  path?: string;
  /** `http(s)` URL of an unbundled `.spz`, fetched through `ParseOptions.remote`; set instead of `path`. */
  url?: string;
  scale?: number;
  x_offset?: number;
  y_offset?: number;
  z_offset?: number;
  roll?: number;
  pitch?: number;
  yaw?: number;
  /** A scene-relative `.glb` path, or an `http(s)` URL fetched as `url` is. */
  collider_url?: string;
  control?: boolean;
}
export interface ManifestSkin extends SkinConfig {
  id: string;
  name: string;
  /** Scene-relative path of the bundled file (`assets/<id>.vrm` or `assets/<id>.glb`). */
  path: string;
}
export interface ManifestMotion {
  name: string;
  /** Scene-relative path of the bundled `.npz` (`assets/<name>.npz`). */
  path: string;
  default?: boolean;
  /** Arrays of the `.npz`, one row per frame, the tracking command serves as state fields. */
  frame_fields?: string[];
  /** Heading the motion selector lists it under. */
  group?: string;
  [key: string]: unknown;
}
/**
 * One MDP: the five term sets a policy runs against, traced. Every `onnx`/`fused` ref is
 * scene-relative.
 */
export interface ManifestMdp {
  id: string;
  observations?: Record<string, unknown>;
  actions?: Record<string, unknown>;
  terminations?: Record<string, unknown>;
  commands?: Record<string, unknown>;
  events?: EventConfig[];
}
export interface ManifestPolicy {
  id: string;
  name: string;
  default?: boolean;
  /** The `ManifestMdp.id` this policy runs against; always written. */
  mdp: string;
  /** Scene-relative path of the network (`policy/<id>.onnx`). */
  onnx: string;
  /** ONNX input slot table; absent means the single default slot. */
  in_keys?: string[];
  out_keys?: (string | string[])[];
  motions?: ManifestMotion[];
  /** The checkpoint's own metadata: `policy_joint_names`, `default_joint_pos`, … */
  [key: string]: unknown;
}
/** A license file the build wrote, named and identified at build time. */
export interface ManifestLicense {
  /** Path of the `LICENSE` file. */
  license?: string;
  /** Its SPDX id; `LicenseRef-custom` for a text that could not be identified. */
  spdx?: string;
  /** `Non-commercial`, `Share-alike`, `No derivatives` or `Copyleft`, when restricted. */
  restriction?: string;
  /** Its first line naming a holder, when it has one. */
  copyright?: string;
  /** Path of the `NOTICE` file. */
  notice?: string;
}
/** One third-party component of a scene: `LICENSE.<component>` / `NOTICE.<component>`. */
export interface ManifestCredit extends ManifestLicense {
  component: string;
  /** The skin it belongs to, by id. */
  skin?: string;
  /** A VRM skin's terms as its own `meta` states them. */
  vrm?: { name?: string; authors: string[]; license?: string; other_license_url?: string };
}
export interface ManifestScene {
  id: string;
  name: string;
  /**
   * Scene-relative model path, named for the argument the scene was built from:
   * `scene.mjz` for `add_scene(spec=...)`, `scene.mjb` for `add_scene(model=...)`.
   */
  scene: string;
  /** Seconds per control step, from the task (mjlab's `timestep * decimation`). */
  control_dt?: number;
  camera?: ManifestCamera;
  /** Absent before format 9: drawn as MuJoCo's classic renderer with its defaults. */
  graphics?: ManifestGraphics;
  terrain_data?: TerrainData;
  splat_section?: boolean;
  splats?: ManifestSplat[];
  skins?: ManifestSkin[];
  credits?: ManifestCredit[];
  /** What each term does when the viewer opens, by name (format 11); absent, mjlab's. */
  term_defaults?: ManifestTermDefaults;
  mdps: ManifestMdp[];
  policies: ManifestPolicy[];
}

/** `SceneHandle.set_term_defaults`'s, the engine's override in snake_case. */
export interface ManifestTermDefaults {
  terminations?: Record<string, { enabled?: boolean; on_fire?: unknown }>;
  events?: Record<string, { enabled?: boolean }>;
}
export interface ManifestProject {
  /** `name2id(name)`, unique in the document: the project's directory and `?project=` value. */
  id: string;
  name: string;
  /** At most one project sets it; none set means the first in document order. */
  default?: boolean;
  /** The work's own `LICENSE` / `NOTICE`. */
  work?: ManifestLicense;
  scenes: ManifestScene[];
}
/** A node of mjlab's viewer GUI as recorded at build time; the app fills a `slot` from engine state. */
export type ViewerGuiNode =
  | { type: 'folder'; label: string; children: ViewerGuiNode[] }
  | { type: 'checkbox'; label: string; default?: boolean; hint?: string }
  | {
      type: 'slider';
      label: string;
      min?: number;
      max?: number;
      step?: number;
      default?: number;
      hint?: string;
    }
  | { type: 'slot'; name: string };

export interface Manifest {
  /**
   * Document format: the structure of the build, bumped only for a break an older reader
   * would misread. Distinct from `version`, the mjswan release, which is never a gate
   * here.
   */
  format: number;
  /** The release that wrote the document: `X.Y.Z` or `X.Y.Z-<prerelease>`. */
  version: string;
  uses_custom_js?: boolean;
  /** Document-root-relative path to the runtime custom-MDP plugin ESM (custom-TS builds). */
  plugins?: string;
  /** mjlab's viewer Scene section; absent from a build made without mjlab. */
  viewer_gui?: ViewerGuiNode[];
  projects: ManifestProject[];
}

// ── catalog (what the app holds) ─────────────────────────────────────────────
export interface PolicyEntry {
  id: string;
  name: string;
  default: boolean;
  motions: Array<{ name: string; default: boolean; group?: string }>;
  /** Assemble the engine PolicyInput: its manifest entry merged with its MDP, plus bytes. */
  build(): Promise<PolicyInput>;
}
export interface SplatEntry {
  id: string;
  name: string;
  /** Whether the UI should show calibration controls. */
  control: boolean;
  /** Initial placement, for seeding a calibration UI (engine.calibrateSplat). */
  transform: SplatTransform;
  build(): Promise<SplatInput>;
}
export interface SkinEntry {
  id: string;
  name: string;
  build(): Promise<SkinInput>;
}
/** A license, its texts fetched on demand. */
export interface LicenseEntry {
  spdx?: string;
  restriction?: string;
  copyright?: string;
  license?: () => Promise<string>;
  notice?: () => Promise<string>;
}
export interface CreditEntry extends LicenseEntry {
  component: string;
  skin?: string;
  vrm?: { name?: string; authors: string[]; license?: string; otherLicenseUrl?: string };
}
export interface SceneEntry {
  id: string;
  name: string;
  camera?: ViewerConfig;
  /** The scene's own graphics settings, every one resolved: what `graphics.reset()` returns to. */
  graphics: GraphicsSettings;
  splatSection: boolean;
  policies: PolicyEntry[];
  splats: SplatEntry[];
  skins: SkinEntry[];
  /** Its third-party components, every skin's included, in manifest order. */
  credits: CreditEntry[];
  /**
   * Assemble a full SceneInput for a chosen policy/splat/skin (ids). An omitted policy is
   * the default one; an omitted splat or skin is none.
   */
  buildScene(opts?: {
    policy?: string | null;
    splat?: string | null;
    skin?: string | null;
  }): Promise<SceneInput>;
}
export interface ProjectCatalog {
  id: string;
  name: string;
  /** The project the app opens on. Exactly one entry in a catalog has it set. */
  default: boolean;
  /** The work's own license. */
  work?: LicenseEntry;
  scenes: SceneEntry[];
}
export interface Catalog {
  /** All projects in the build; the default first, so `projects[0]` is the one to open. */
  projects: ProjectCatalog[];
  /**
   * Document-root-relative path to the author custom-MDP plugin ESM, for a document with
   * `uses_custom_js: true` only. A trusted app imports it and passes the exports as
   * {@link EnginePlugins}; mjswan Cloud ignores it.
   */
  pluginsPath?: string;
  /** {@link Manifest.viewer_gui}. */
  viewerGui?: ViewerGuiNode[];
}

/**
 * The id a name sanitizes to: Python's `name2id`, character for character, pinned by
 * `name2id_cases.json`, which both test suites read. A case they disagree on is a link
 * that opens the wrong scene.
 */
export function sanitizeName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * The newest document format this reader understands. A ceiling: every earlier format
 * still parses. 2 added `sim` input slots and their `rows`; 3 made `camera.azimuth`
 * MuJoCo's, the direction the camera looks; 4 added the `env` input slot that `time_out`
 * is traced through; 5 the `action` input slot that `last_action` is traced through, the
 * `model` input slot and write target that model-field randomization is traced through,
 * and a command graph's `reset_mask`, gated writes and `root_velocity_b` write; 6 an
 * action term's traced `process` and `reset` graphs, its `interpolate`, `hold_s` and
 * `delay`, the `joint_position_relative` type, and the `action_term` input slot; 7 a
 * scene's `skins`; 8 a motion's `frame_fields`; 9 a camera sensor's descriptor, an
 * image-shaped policy input, a skin modelled on the robot (`fit: 'modelled'`), with
 * `align`, `rest_qpos`, a solid material and `hide_tendons`, and a scene's `graphics`;
 * 10 an action term's DC motor `saturation_effort` and `velocity_limit`; 11 a scene's
 * `term_defaults`. Bump it with `DOCUMENT_FORMAT` (`document/manifest.py`) and add a row
 * to the format table in context/document.md.
 */
export const MAX_DOCUMENT_FORMAT = 11;

/**
 * Refuse a document this reader cannot read: one without a `format`, one newer than it
 * knows, or one whose `version` (what a host picks an engine by) no release has.
 */
function checkFormat(parsed: Manifest): void {
  const format = parsed.format;
  if (format === undefined) {
    throw new Error(
      'mjswan/manifest: this build predates document format 1 (a root assets/config.json ' +
        'rather than manifest.json). Rebuild it with a current mjswan.',
    );
  }
  if (!isReleaseVersion(parsed.version)) {
    throw new Error(
      `mjswan/manifest: version ${JSON.stringify(parsed.version)} is not one a release ` +
        'has (X.Y.Z or X.Y.Z-<prerelease>), so no mjswan build wrote this document.',
    );
  }
  if (!Number.isInteger(format) || format < 1) {
    throw new Error(`mjswan/manifest: document format ${JSON.stringify(format)} is not a format number.`);
  }
  if (format > MAX_DOCUMENT_FORMAT) {
    throw new Error(
      `mjswan/manifest: this document is format ${format} (written by mjswan ${parsed.version}), ` +
        `but this engine reads up to format ${MAX_DOCUMENT_FORMAT}. Update the engine.`,
    );
  }
}

type Remote = NonNullable<ParseOptions['remote']>;

function checkLicense(dir: string, entry: ManifestLicense | undefined): void {
  if (entry?.license !== undefined) documentPath(dir, entry.license, 'license');
  if (entry?.notice !== undefined) documentPath(dir, entry.notice, 'notice');
}

/** Refuse a document naming anything outside itself (an id, a path, a URL), before any fetch. */
function checkDocument(parsed: Manifest): void {
  if (parsed.plugins !== undefined) documentPath('', parsed.plugins, 'plugins');
  for (const project of parsed.projects) {
    const projectDir = `${documentName(project.id, 'project id')}/`;
    checkLicense(projectDir, project.work);
    for (const scene of project.scenes ?? []) {
      const dir = `${projectDir}${documentName(scene.id, 'scene id')}/`;
      documentPath(dir, scene.scene, 'scene');
      for (const owner of [...(scene.mdps ?? []), ...(scene.policies ?? [])]) {
        for (const ref of policyGraphRefs(owner as unknown as Record<string, unknown>)) {
          documentPath(dir, ref, 'graph');
        }
      }
      for (const policy of scene.policies ?? []) {
        documentPath(dir, policy.onnx, 'policy');
        for (const motion of policy.motions ?? []) documentPath(dir, motion.path, 'motion');
      }
      for (const splat of scene.splats ?? []) {
        if (splat.path !== undefined) documentPath(dir, splat.path, 'splat');
        else remoteUrl(splat.url, 'splat url');
        const collider = splat.collider_url;
        if (collider !== undefined) {
          if (isUrl(collider)) remoteUrl(collider, 'collider_url');
          else documentPath(dir, collider, 'collider_url');
        }
      }
      for (const skin of scene.skins ?? []) documentPath(dir, skin.path, 'skin');
      for (const credit of scene.credits ?? []) checkLicense(dir, credit);
    }
  }
}

function toLicenseEntry(dir: string, entry: ManifestLicense, source: ByteSource): LicenseEntry {
  const text = (rel: string) => async () =>
    new TextDecoder().decode(await resolveBytes(source(documentPath(dir, rel, 'license'))));
  return {
    spdx: entry.spdx,
    restriction: entry.restriction,
    copyright: entry.copyright,
    license: entry.license ? text(entry.license) : undefined,
    notice: entry.notice ? text(entry.notice) : undefined,
  };
}

function toCreditEntry(dir: string, credit: ManifestCredit, source: ByteSource): CreditEntry {
  const vrm = credit.vrm && {
    name: credit.vrm.name,
    authors: credit.vrm.authors,
    license: credit.vrm.license,
    otherLicenseUrl: credit.vrm.other_license_url,
  };
  return { ...toLicenseEntry(dir, credit, source), component: credit.component, skin: credit.skin, vrm };
}

/** The manifest's snake_case camera entry as the engine's `ViewerConfig`. */
function toViewerConfig(camera: ManifestCamera | undefined, format: number): ViewerConfig | undefined {
  if (!camera) return undefined;
  const view: ViewerConfig = {};
  if (camera.lookat !== undefined) view.lookat = camera.lookat;
  if (camera.distance !== undefined) view.distance = camera.distance;
  if (camera.fovy !== undefined) view.fovy = camera.fovy;
  if (camera.elevation !== undefined) view.elevation = camera.elevation;
  // Before format 3 the azimuth pointed at the camera rather than the way it looks.
  if (camera.azimuth !== undefined) view.azimuth = format < 3 ? camera.azimuth + 180 : camera.azimuth;
  if (camera.origin_type !== undefined) view.originType = camera.origin_type;
  if (camera.entity_name !== undefined) view.entityName = camera.entity_name;
  if (camera.body_name !== undefined) view.bodyName = camera.body_name;
  if (camera.height !== undefined) view.height = camera.height;
  if (camera.width !== undefined) view.width = camera.width;
  return view;
}

/** The manifest's snake_case `graphics` as the engine's settings; without it (format 8 and below), MuJoCo's defaults. */
function toGraphics(graphics: ManifestGraphics | undefined): GraphicsSettings {
  return mergeGraphics(DEFAULT_GRAPHICS, graphics && {
    colorManagement: graphics.color_management,
    toneMapping: graphics.tone_mapping,
    exposure: graphics.exposure,
    outputColorSpace: graphics.output_color_space,
    baseColorType: graphics.base_color_type,
    shadows: graphics.enable_shadows,
    reflections: graphics.enable_reflections,
    haze: graphics.enable_haze,
  });
}

function splatTransform(splat: ManifestSplat): SplatTransform {
  return {
    scale: splat.scale,
    xOffset: splat.x_offset,
    yOffset: splat.y_offset,
    zOffset: splat.z_offset,
    roll: splat.roll,
    pitch: splat.pitch,
    yaw: splat.yaw,
  };
}

function buildSplat(dir: string, splat: ManifestSplat, source: ByteSource, remote?: Remote): SplatInput {
  const data =
    splat.path !== undefined
      ? source(documentPath(dir, splat.path, 'splat'))
      : remote!(remoteUrl(splat.url, 'splat url'));
  const named = splat.collider_url;
  const collider =
    named === undefined
      ? undefined
      : !isUrl(named)
        ? source(documentPath(dir, named, 'collider_url'))
        : remote?.(remoteUrl(named, 'collider_url'));
  return { data, collider, transform: splatTransform(splat) };
}

/**
 * Byte sources for a config's traced term graphs, keyed scene-relative, which is how the
 * runtime looks a session up. The source is asked for the document-relative path instead.
 */
function graphBytes(refs: string[], dir: string, source: ByteSource): Record<string, Bytes> {
  const graphs: Record<string, Bytes> = {};
  for (const ref of refs) graphs[ref] = source(documentPath(dir, ref, 'graph'));
  return graphs;
}

/** The engine's policy config: the policy entry's own fields plus its MDP's term sets. */
function policyConfig(scene: ManifestScene, policy: ManifestPolicy): Record<string, unknown> {
  const mdp = scene.mdps.find((m) => m.id === policy.mdp);
  if (!mdp) {
    throw new Error(
      `mjswan/manifest: policy "${policy.name}" refers to mdp "${policy.mdp}", ` +
        `which scene "${scene.name}" does not declare.`,
    );
  }
  // Bookkeeping keys stay out of what the engine interprets; the network arrives as bytes.
  const own = Object.fromEntries(
    Object.entries(policy).filter(([key]) => !['id', 'mdp', 'onnx'].includes(key)),
  );
  const sections = Object.fromEntries(Object.entries(mdp).filter(([key]) => key !== 'id'));
  return { ...own, ...sections };
}

function buildPolicy(
  dir: string,
  scene: ManifestScene,
  policy: ManifestPolicy,
  source: ByteSource,
): PolicyInput {
  const config = policyConfig(scene, policy);
  const motions = (policy.motions ?? []).map((m) => ({
    name: m.name,
    data: source(documentPath(dir, m.path, 'motion')),
    default: m.default,
  }));
  return {
    config,
    onnx: source(documentPath(dir, policy.onnx, 'policy')),
    graphs: graphBytes(policyGraphRefs(config), dir, source),
    motions,
  };
}

function toSceneEntry(
  project: ManifestProject,
  scene: ManifestScene,
  source: ByteSource,
  format: number,
  remote: Remote | undefined,
): SceneEntry {
  // `<project-id>/<scene-id>/`: the base every path under a scene entry resolves against.
  const dir = `${project.id}/${scene.id}/`;

  const policies: PolicyEntry[] = scene.policies.map((p) => ({
    id: p.id,
    name: p.name,
    default: p.default ?? false,
    motions: (p.motions ?? []).map((m) => ({ name: m.name, default: m.default ?? false, group: m.group })),
    build: async () => buildPolicy(dir, scene, p, source),
  }));

  // A splat named by URL loads only through the host's `remote`.
  const reachable = (scene.splats ?? []).filter((s) => {
    const fetchable = s.path !== undefined || remote !== undefined;
    if (!fetchable) console.warn(`mjswan/manifest: splat "${s.name}" is left out: it loads from ${s.url}.`);
    else if (!remote && s.collider_url !== undefined && isUrl(s.collider_url)) {
      console.warn(`mjswan/manifest: splat "${s.name}" has no collider: it loads from ${s.collider_url}.`);
    }
    return fetchable;
  });
  const splats: SplatEntry[] = reachable.map((s) => ({
    id: s.id,
    name: s.name,
    control: s.control ?? false,
    transform: splatTransform(s),
    build: async () => buildSplat(dir, s, source, remote),
  }));

  const buildSkin = (skin: ManifestSkin): SkinInput => ({
    data: source(documentPath(dir, skin.path, 'skin')),
    config: skin,
  });
  const skins: SkinEntry[] = (scene.skins ?? []).map((s) => ({
    id: s.id,
    name: s.name,
    build: async () => buildSkin(s),
  }));

  const graphics = toGraphics(scene.graphics);
  return {
    id: scene.id,
    name: scene.name,
    camera: toViewerConfig(scene.camera, format),
    graphics,
    splatSection: scene.splat_section ?? false,
    policies,
    splats,
    skins,
    credits: (scene.credits ?? []).map((c) => toCreditEntry(dir, c, source)),
    buildScene: async (opts) => {
      const policyId = opts?.policy;
      const policy =
        policyId == null
          ? (scene.policies.find((p) => p.default) ?? scene.policies[0])
          : scene.policies.find((p) => p.id === policyId);
      const splatId = opts?.splat;
      const splat = splatId == null ? undefined : reachable.find((s) => s.id === splatId);
      const skinId = opts?.skin;
      const skin = skinId == null ? undefined : scene.skins?.find((s) => s.id === skinId);
      // Events travel with the policy, inside its MDP, so the scene carries only
      // `terrainData`, which every MDP on it may draw from.
      return {
        model: source(documentPath(dir, scene.scene, 'scene')),
        modelFormat: scene.scene.endsWith('.mjb') ? 'mjb' : 'mjz',
        policy: policy ? buildPolicy(dir, scene, policy, source) : null,
        splat: splat ? buildSplat(dir, splat, source, remote) : null,
        skin: skin ? buildSkin(skin) : null,
        viewer: toViewerConfig(scene.camera, format),
        graphics,
        terrainData: scene.terrain_data,
        controlDt: scene.control_dt,
        termDefaults: toTermDefaults(scene.term_defaults),
      };
    },
  };
}

/** The manifest's `term_defaults` as the engine takes them; the engine refuses an `on_fire` it cannot keep. */
function toTermDefaults(defaults: ManifestTermDefaults | undefined): TermDefaults | undefined {
  if (!defaults) return undefined;
  const map = <T, U>(record: Record<string, T> | undefined, f: (value: T) => U): Record<string, U> =>
    Object.fromEntries(Object.entries(record ?? {}).map(([name, value]) => [name, f(value)]));
  return {
    terminations: map(defaults.terminations, (t) => ({ enabled: t.enabled, onFire: t.on_fire as OnFire | undefined })),
    events: map(defaults.events, (e) => ({ enabled: e.enabled })),
  };
}

/**
 * Parse a build's `manifest.json` (object or JSON string) into a {@link Catalog}; throws
 * for a document this reader cannot read or one naming anything outside itself.
 */
export function parseManifest(
  manifest: Manifest | string,
  source: ByteSource,
  options: ParseOptions = {},
): Catalog {
  const parsed: Manifest = typeof manifest === 'string' ? JSON.parse(manifest) : manifest;
  checkFormat(parsed);
  if (!parsed.projects?.length) {
    throw new Error('mjswan/manifest: manifest.json has no projects.');
  }
  checkDocument(parsed);
  // The default project first, so `projects[0]` is always the one to open on. The build
  // refuses two defaults; with none, document order already puts the right one first.
  const flagged = parsed.projects.find((p) => p.default) ?? parsed.projects[0];
  const ordered = [flagged, ...parsed.projects.filter((p) => p !== flagged)];
  return {
    projects: ordered.map((project) => ({
      id: project.id,
      name: project.name,
      default: project === flagged,
      work: project.work && toLicenseEntry(`${project.id}/`, project.work, source),
      scenes: project.scenes.map((scene) => toSceneEntry(project, scene, source, parsed.format, options.remote)),
    })),
    // A custom-TS build's module only; the host decides whether to load it (urlState.ts).
    pluginsPath: parsed.uses_custom_js === true ? parsed.plugins : undefined,
    viewerGui: parsed.viewer_gui,
  };
}
