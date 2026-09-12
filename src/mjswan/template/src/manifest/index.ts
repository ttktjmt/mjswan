/**
 * `mjswan/manifest`: parse a build's `manifest.json` + a byte source into a typed
 * catalog of loadable scenes/policies/splats (ADR 0004 §1, §9; ADR 0006). Framework-free;
 * shared by the in-repo React app and mjswan Cloud. The engine itself knows nothing
 * about the manifest; the app owns the catalog and calls engine verbs.
 *
 * The manifest is the one descriptor of a document: every key `snake_case`, every path
 * under a scene entry relative to `<project-id>/<scene-id>/`, the top-level `plugins`
 * relative to the document root.
 */
import type { Bytes } from '../core/utils/bytes';
import type { ViewerConfig } from '../core/engine/viewer_config';
import type { EventConfig, TerrainData } from '../core/event/EventBase';
import type { SplatTransform } from '../core/scene/splat';
import { policyGraphRefs } from '../core/onnx/graphRefs';
import type { PolicyInput, SceneInput, SplatInput } from '../engine/types';

/** Maps a document-relative path (e.g. `demo/humanoid/scene.mjz`) to bytes. */
export type ByteSource = (relPath: string) => Bytes;

// ── manifest.json shape (Python Builder ↔ consumer contract, ADR 0006) ───────
export interface ManifestCamera {
  lookat?: [number, number, number];
  distance?: number;
  fovy?: number;
  elevation?: number;
  azimuth?: number;
  origin_type?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  entity_name?: string;
  body_name?: string;
  enable_reflections?: boolean;
  enable_shadows?: boolean;
  height?: number;
  width?: number;
}
export interface ManifestSplat {
  id: string;
  name: string;
  /** Scene-relative path of a bundled `.spz` (`assets/<id>.spz`). */
  path?: string;
  /** External URL of an unbundled `.spz`. Exactly one of `path`/`url` is set. */
  url?: string;
  scale?: number;
  x_offset?: number;
  y_offset?: number;
  z_offset?: number;
  roll?: number;
  pitch?: number;
  yaw?: number;
  collider_url?: string;
  control?: boolean;
}
export interface ManifestMotion {
  name: string;
  /** Scene-relative path of the bundled `.npz` (`assets/<name>.npz`). */
  path: string;
  default?: boolean;
  [key: string]: unknown;
}
/**
 * One MDP: the five term sets a policy runs against, traced. Term entries keep the
 * shape ADR 0005 §1 gave them; every `onnx`/`fused` ref is scene-relative.
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
  /** ONNX input slot table; absent means the single default slot (ADR 0006 §5). */
  in_keys?: string[];
  out_keys?: (string | string[])[];
  motions?: ManifestMotion[];
  /** The checkpoint's own metadata: `policy_joint_names`, `default_joint_pos`, … */
  [key: string]: unknown;
}
export interface ManifestScene {
  id: string;
  name: string;
  /** Scene-relative model path: `scene.mjz` or `scene.mjb`. */
  scene: string;
  /** Seconds per control step, from the task (mjlab's `timestep * decimation`). */
  control_dt?: number;
  camera?: ManifestCamera;
  terrain_data?: TerrainData;
  splat_section?: boolean;
  splats?: ManifestSplat[];
  mdps: ManifestMdp[];
  policies: ManifestPolicy[];
}
export interface ManifestProject {
  /** `name2id(name)`, unique in the document: the project's directory and `?project=` value. */
  id: string;
  name: string;
  /** At most one project sets it; none set means the first in document order. */
  default?: boolean;
  scenes: ManifestScene[];
}
export interface Manifest {
  /**
   * Document format (ADR 0006 §7): the structure of the build, bumped only for a break an
   * older reader would misread. Distinct from `version`, the mjswan release, which is
   * never a gate here.
   */
  format: number;
  version: string;
  uses_custom_js?: boolean;
  /** Document-root-relative path to the runtime custom-MDP plugin ESM (custom-JS builds). */
  plugins?: string;
  projects: ManifestProject[];
}

// ── catalog (what the app holds) ─────────────────────────────────────────────
export interface PolicyEntry {
  id: string;
  name: string;
  default: boolean;
  motions: Array<{ name: string; default: boolean }>;
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
export interface SceneEntry {
  id: string;
  name: string;
  camera?: ViewerConfig;
  splatSection: boolean;
  policies: PolicyEntry[];
  splats: SplatEntry[];
  /** Assemble a full SceneInput for a chosen policy/splat (ids; defaults if omitted). */
  buildScene(opts?: { policy?: string | null; splat?: string | null }): Promise<SceneInput>;
}
export interface ProjectCatalog {
  id: string;
  name: string;
  /** The project the app opens on. Exactly one entry in a catalog has it set. */
  default: boolean;
  scenes: SceneEntry[];
}
export interface Catalog {
  /** All projects in the build; the default first, so `projects[0]` is the one to open. */
  projects: ProjectCatalog[];
  /**
   * Document-root-relative path to the author custom-MDP plugin ESM, if any. A trusted
   * app imports it and passes the exports as {@link EnginePlugins}; mjswan Cloud ignores it.
   */
  pluginsPath?: string;
}

/**
 * The id a name sanitizes to: Python's `name2id`, character for character, pinned by
 * `name2id_cases.json`, which both test suites read. A case they disagree on is a link
 * that opens the wrong scene (ADR 0006 §4).
 */
export function sanitizeName(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * The newest document format this reader understands (ADR 0006 §7). A ceiling: every
 * earlier format still parses. 2 added `sim` input slots and their `rows`.
 */
export const MAX_DOCUMENT_FORMAT = 2;

/**
 * Refuse a document this reader cannot read: one without a `format`, or one newer than it
 * knows. `version` is provenance a host may pick an engine by, never a gate here.
 */
function checkFormat(parsed: Manifest): void {
  const format = parsed.format;
  if (format === undefined) {
    throw new Error(
      'mjswan/manifest: this build predates document format 1 (a root assets/config.json ' +
        'rather than manifest.json). Rebuild it with a current mjswan.',
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

function inScene(dir: string, rel: string): string {
  return `${dir}${rel}`.replace(/\/+/g, '/');
}

/** Lazy-fetch an absolute/external URL as bytes (for splats hosted off-build). */
function urlBytes(url: string): Bytes {
  return async () => {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`mjswan/manifest: failed to fetch ${url}: ${res.status}`);
    }
    return res.arrayBuffer();
  };
}

/** The manifest's snake_case camera entry as the engine's `ViewerConfig`. */
function toViewerConfig(camera: ManifestCamera | undefined): ViewerConfig | undefined {
  if (!camera) return undefined;
  const view: ViewerConfig = {};
  if (camera.lookat !== undefined) view.lookat = camera.lookat;
  if (camera.distance !== undefined) view.distance = camera.distance;
  if (camera.fovy !== undefined) view.fovy = camera.fovy;
  if (camera.elevation !== undefined) view.elevation = camera.elevation;
  if (camera.azimuth !== undefined) view.azimuth = camera.azimuth;
  if (camera.origin_type !== undefined) view.originType = camera.origin_type;
  if (camera.entity_name !== undefined) view.entityName = camera.entity_name;
  if (camera.body_name !== undefined) view.bodyName = camera.body_name;
  if (camera.enable_reflections !== undefined) view.enableReflections = camera.enable_reflections;
  if (camera.enable_shadows !== undefined) view.enableShadows = camera.enable_shadows;
  if (camera.height !== undefined) view.height = camera.height;
  if (camera.width !== undefined) view.width = camera.width;
  return view;
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

function buildSplat(dir: string, splat: ManifestSplat, source: ByteSource): SplatInput {
  const data = splat.path ? source(inScene(dir, splat.path)) : urlBytes(splat.url!);
  const collider = splat.collider_url
    ? (/^[a-z]+:\/\//i.test(splat.collider_url)
        ? urlBytes(splat.collider_url)
        : source(inScene(dir, splat.collider_url)))
    : undefined;
  return { data, collider, transform: splatTransform(splat) };
}

/**
 * Byte sources for a config's traced term graphs, keyed scene-relative, which is how the
 * runtime looks a session up (ADR 0005 §4). The source is asked for the document-relative
 * path instead.
 */
function graphBytes(refs: string[], dir: string, source: ByteSource): Record<string, Bytes> {
  const graphs: Record<string, Bytes> = {};
  for (const ref of refs) graphs[ref] = source(inScene(dir, ref));
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
    data: source(inScene(dir, m.path)),
    default: m.default,
  }));
  return {
    config,
    onnx: source(inScene(dir, policy.onnx)),
    graphs: graphBytes(policyGraphRefs(config), dir, source),
    motions,
  };
}

function toSceneEntry(project: ManifestProject, scene: ManifestScene, source: ByteSource): SceneEntry {
  // `<project-id>/<scene-id>/`: the base every path under a scene entry resolves against.
  const dir = `${project.id}/${scene.id}/`;

  const policies: PolicyEntry[] = scene.policies.map((p) => ({
    id: p.id,
    name: p.name,
    default: p.default ?? false,
    motions: (p.motions ?? []).map((m) => ({ name: m.name, default: m.default ?? false })),
    build: async () => buildPolicy(dir, scene, p, source),
  }));

  const splats: SplatEntry[] = (scene.splats ?? []).map((s) => ({
    id: s.id,
    name: s.name,
    control: s.control ?? false,
    transform: splatTransform(s),
    build: async () => buildSplat(dir, s, source),
  }));

  return {
    id: scene.id,
    name: scene.name,
    camera: toViewerConfig(scene.camera),
    splatSection: scene.splat_section ?? false,
    policies,
    splats,
    buildScene: async (opts) => {
      const policyId = opts?.policy;
      const policy =
        policyId == null
          ? (scene.policies.find((p) => p.default) ?? scene.policies[0])
          : scene.policies.find((p) => p.id === policyId);
      const splatId = opts?.splat;
      const splat = splatId == null ? undefined : scene.splats?.find((s) => s.id === splatId);
      // Events travel with the policy, inside its MDP (ADR 0006 §3), so the scene carries
      // only `terrainData`, which every MDP on it may draw from.
      return {
        model: source(inScene(dir, scene.scene)),
        policy: policy ? buildPolicy(dir, scene, policy, source) : null,
        splat: splat ? buildSplat(dir, splat, source) : null,
        viewer: toViewerConfig(scene.camera),
        terrainData: scene.terrain_data,
        controlDt: scene.control_dt,
      };
    },
  };
}

/** Parse a build's `manifest.json` (object or JSON string) into a {@link Catalog}. */
export function parseManifest(manifest: Manifest | string, source: ByteSource): Catalog {
  const parsed: Manifest = typeof manifest === 'string' ? JSON.parse(manifest) : manifest;
  checkFormat(parsed);
  if (!parsed.projects?.length) {
    throw new Error('mjswan/manifest: manifest.json has no projects.');
  }
  // The default project first, so `projects[0]` is always the one to open on. The build
  // refuses two defaults; with none, document order already puts the right one first.
  const flagged = parsed.projects.find((p) => p.default) ?? parsed.projects[0];
  const ordered = [flagged, ...parsed.projects.filter((p) => p !== flagged)];
  return {
    projects: ordered.map((project) => ({
      id: project.id,
      name: project.name,
      default: project === flagged,
      scenes: project.scenes.map((scene) => toSceneEntry(project, scene, source)),
    })),
    pluginsPath: parsed.plugins,
  };
}
