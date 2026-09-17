import * as THREE from 'three';
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { MjData, MjModel } from 'mujoco';
import { mjcToThreeCoordinate } from '../scene/coordinate';

export type ViewerConfig = {
  /** Look-at point in MuJoCo coordinates [x forward, y left, z up]. */
  lookat?: [number, number, number];
  /** Distance from the look-at point to the camera. */
  distance?: number;
  /** Vertical field of view in degrees. */
  fovy?: number;
  /** Camera elevation in degrees (negative = camera above the look-at point). */
  elevation?: number;
  /** Camera azimuth in degrees measured from the x-axis (forward) CCW. */
  azimuth?: number;
  /** Origin type for camera tracking. */
  originType?: 'AUTO' | 'WORLD' | 'ASSET_ROOT' | 'ASSET_BODY';
  /** Entity/asset name (currently unused in single-entity scenes). */
  entityName?: string;
  /** Body name to track when originType is ASSET_BODY. */
  bodyName?: string;
  /** Whether to enable reflections. */
  enableReflections?: boolean;
  /** Whether to enable shadows. */
  enableShadows?: boolean;
  /** Viewer canvas height in pixels. */
  height?: number;
  /** Viewer canvas width in pixels. */
  width?: number;
  /**
   * How many throwable boxes the `throw` interaction mode gets, 0 to turn it off. They are
   * declared with the model, so this is fixed for the life of a scene; absent leaves it to
   * the engine's own default.
   */
  spawnPool?: number;
};

/**
 * Fallback view for a document that carries no `camera`. The build writes a complete
 * one for every scene, so nothing mjswan produces lands here; a hand-written or
 * third-party manifest can, and gets a usable view rather than a black screen.
 */
const FALLBACK_VIEW = {
  lookat: [0, 0, 0] as [number, number, number],
  distance: 4.0,
  elevation: -30.0,
  azimuth: 45.0,
  fovy: 45,
} as const;

export type ViewerState = {
  /** Body index to track each frame, or null. */
  trackBodyId: number | null;
  /** Previous body world position used to compute per-frame delta for parallel tracking. */
  prevBodyPos: THREE.Vector3 | null;
};

/** Camera pose in spherical MuJoCo coordinates (x forward, y left, z up). */
export type CameraView = {
  lookat: [number, number, number];
  distance: number;
  azimuth: number;
  elevation: number;
  fovy: number;
};


export function computeCameraPosition(
  lookat: [number, number, number],
  distance: number,
  elevation: number,
  azimuth: number
): THREE.Vector3 {
  const el = (elevation * Math.PI) / 180;
  const az = (azimuth * Math.PI) / 180;
  const camX = lookat[0] + distance * Math.cos(el) * Math.cos(az);
  const camY = lookat[1] + distance * Math.cos(el) * Math.sin(az);
  const camZ = lookat[2] - distance * Math.sin(el);
  return mjcToThreeCoordinate([camX, camY, camZ]);
}

/**
 * Apply a ViewerConfig after a scene loads.
 *
 * Computes camera position from lookat + distance + elevation + azimuth in
 * MuJoCo coordinates (x forward, y left, z up), then converts to Three.js.
 * Returns the ViewerState that runtime.ts must keep to drive per-frame updates.
 *
 * Mirrors the pattern of createLights() in lights.ts.
 */
export function applyViewerConfig(
  config: ViewerConfig | null,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  mjModel: MjModel | null,
  mjData: MjData | null,
  /** Bodies the viewer put in the model itself; never what a scene means by "the robot". */
  injected: ReadonlySet<number> = new Set()
): ViewerState {
  const state: ViewerState = { trackBodyId: null, prevBodyPos: null };
  controls.enabled = true;

  if (!config) {
    console.warn(
      '[Camera] this scene carries no `camera`; falling back to the built-in view. ' +
        'A manifest written by mjswan always carries one.'
    );
  }

  const lookat = config?.lookat ?? FALLBACK_VIEW.lookat;
  const distance = config?.distance ?? FALLBACK_VIEW.distance;
  const elevation = config?.elevation ?? FALLBACK_VIEW.elevation;
  const azimuth = config?.azimuth ?? FALLBACK_VIEW.azimuth;

  camera.fov = config?.fovy ?? FALLBACK_VIEW.fovy;
  camera.updateProjectionMatrix();
  camera.position.copy(computeCameraPosition(lookat, distance, elevation, azimuth));
  controls.target.copy(mjcToThreeCoordinate(lookat));
  controls.update();

  if (!config) return state;

  const originType = config.originType ?? 'AUTO';

  if (originType === 'ASSET_BODY' && config.bodyName && mjModel) {
    const requestedName = config.bodyName;
    const entityName = config.entityName;
    const prefixedName = entityName ? `${entityName}/${requestedName}` : null;
    for (let b = 0; b < mjModel.nbody; b++) {
      const bodyName = mjModel.body(b).name;
      if (
        bodyName === requestedName ||
        bodyName === prefixedName ||
        bodyName.endsWith(`/${requestedName}`)
      ) {
        state.trackBodyId = b;
        break;
      }
    }
    if (state.trackBodyId === null) {
      console.warn(`[Camera] bodyName: body "${config.bodyName}" not found.`);
    }
  } else if ((originType === 'AUTO' || originType === 'ASSET_ROOT') && mjModel) {
    // The first non-world body the *scene* owns (typically its floating base). Injected
    // bodies are appended, so this is body 1 for every scene that has one of its own — and
    // for a scene that has none, tracking the viewer's own parked crate 120 m up would
    // carry the camera off with it.
    for (let bodyId = 1; bodyId < mjModel.nbody; bodyId++) {
      if (injected.has(bodyId)) continue;
      state.trackBodyId = bodyId;
      break;
    }
  }
  // WORLD: no tracking.

  if (state.trackBodyId !== null && mjData) {
    const bodyPos = mjcToThreeCoordinate(
      mjData.xpos.slice(state.trackBodyId * 3, state.trackBodyId * 3 + 3)
    );
    camera.position.add(bodyPos);
    controls.target.add(bodyPos);
    state.prevBodyPos = bodyPos;
    controls.update();
  }

  return state;
}

/**
 * Update the Three.js camera each frame for body tracking.
 *
 * Must be called before controls.update().
 * Mirrors the pattern of updateLightsFromData() in lights.ts.
 */
export function updateCameraFromData(
  mjData: MjData,
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  state: ViewerState,
  /**
   * A viewer standing in the scene is not carried around by the body it is watching, so
   * only the orbit target tracks: the desktop view still points at the body on the way out.
   */
  presenting = false
): void {
  if (state.trackBodyId !== null) {
    // Parallel tracking: the orbit target takes the body's delta, and off a headset the
    // camera with it, so the camera angle and zoom level survive.
    const b = state.trackBodyId;
    const bodyPos = mjcToThreeCoordinate(mjData.xpos.slice(b * 3, b * 3 + 3));
    if (state.prevBodyPos !== null) {
      const delta = bodyPos.clone().sub(state.prevBodyPos);
      controls.target.add(delta);
      if (!presenting) {
        camera.position.add(delta);
      }
    }
    state.prevBodyPos = bodyPos;
  }
}
