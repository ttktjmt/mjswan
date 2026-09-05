import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  getPosition,
  getQuaternion,
  loadSceneFromURL,
} from '../scene/scene';
import { loadMjzFile } from '../utils/mjzLoader';
import { type Bytes } from '../utils/bytes';
import type { EnginePlugins } from '../plugins';
import { type SplatTransform, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { DragStateManager } from '../utils/dragStateManager';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { mjcToThreeCoordinate, threeToMjcCoordinate } from '../scene/coordinate';
import { HandMocap, injectHandMocapFile } from '../xr/handMocap';
import { updateXrLocomotion } from '../xr/locomotion';
import { updateRigGrounding } from '../xr/grounding';
import { createArButton } from '../xr/arButton';
import { Passthrough } from '../xr/passthrough';
import {
  type CameraView,
  type ViewerConfig,
  type ViewerState,
  applyViewerConfig,
  computeCameraPosition,
  updateCameraFromData,
} from './viewer_config';
import {
  advanceActionSmoothing,
  clampActions,
  readClipActions,
  resetActionSmoothing,
  resolveActionClip,
  stepPhysics,
  type ResolvedActionTerm,
} from '../action/applyAction';
import { applyResetTerms } from './resetChain';
import { TerminationManager } from '../termination/TerminationManager';
import * as ort from 'onnxruntime-web';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { CommandManager, type CommandTermContext, type CommandsConfig } from '../command';
import { EventManager, type EventControl } from '../event/EventManager';
import { ModelFieldDefaults } from '../event/modelFieldDr';
import type { EventContext, TerrainData } from '../event/EventBase';
import { OnnxSessionCache, type SlotReader } from '../onnx/session';
import { ContactSensorSet, type ContactSensorDescriptor } from '../onnx/contact';
import type { RaycastSensorDescriptor } from '../onnx/raycast';
import { createSlotReader, sensorWindow } from '../onnx/slotReader';
import { SeededRng } from '../rng';

/** Fixed rather than time-derived, so a plain page load replays identically. */
const DEFAULT_TERM_SEED = 0x5eed;

const EMPTY_ACTIONS = new Float32Array(0);

/** Only for a policy-less scene; a policy carries its own rate in `controlDt`. */
const DEFAULT_VIEWER_CONTROL_DT = 0.02;

/** The frame time every XR rate is scaled by, capped so a slept tab is not a teleport. */
const MAX_XR_FRAME_SECONDS = 0.1;

/** Keyed by joint name: the slot reader needs entity-joint order, not action order. */
function buildJointBias(config: PolicyConfig): Map<string, number> {
  const bias = new Map<string, number>();
  const values = config.encoder_bias;
  const names = config.policy_joint_names;
  if (!Array.isArray(values) || !Array.isArray(names)) return bias;
  for (let i = 0; i < names.length && i < values.length; i++) {
    if (values[i]) bias.set(names[i], values[i]);
  }
  return bias;
}

type StructuredSensorDescriptor = RaycastSensorDescriptor | ContactSensorDescriptor;

/**
 * Descriptors for the structured sensors this policy's graphs read, by kind. Terminations
 * carry them too, not just observation groups (`ee_ground_collision` is termination-only).
 */
function collectStructuredSensors(config: PolicyConfig): {
  raycast: Record<string, RaycastSensorDescriptor>;
  contact: Record<string, ContactSensorDescriptor>;
} {
  const raycast: Record<string, RaycastSensorDescriptor> = {};
  const contact: Record<string, ContactSensorDescriptor> = {};
  const owners: Array<{ sensors?: Record<string, StructuredSensorDescriptor> }> = [
    ...Object.values(config.observations ?? {}),
    ...Object.values(config.terminations ?? {}),
  ] as Array<{ sensors?: Record<string, StructuredSensorDescriptor> }>;
  for (const owner of owners) {
    for (const [name, descriptor] of Object.entries(owner?.sensors ?? {})) {
      if (descriptor.kind === 'contact') contact[name] = descriptor;
      else raycast[name] = descriptor;
    }
  }
  return { raycast, contact };
}

/** A policy with its ONNX weights resolved to bytes; motion data stays lazy. */
export type ResolvedPolicy = {
  config: PolicyConfig;
  onnx: ArrayBuffer;
  /** Traced term graphs, keyed by the config-relative path. */
  graphs?: Array<{ name: string; data: ArrayBuffer }>;
  motions: Array<{ name: string; data: Bytes; default?: boolean }>;
  /** Policy-scoped custom terms (observations / terminations / commands). */
  plugins?: EnginePlugins;
};

/** A splat with its bytes resolved. */
export type ResolvedSplat = {
  data: ArrayBuffer;
  collider?: ArrayBuffer | null;
  transform?: SplatTransform;
};

/** A full scene ready to build: model bytes plus resolved policy/splat/config. */
export type ResolvedScene = {
  model: ArrayBuffer;
  policy?: ResolvedPolicy | null;
  splat?: ResolvedSplat | null;
  viewer?: ViewerConfig | null;
  /** Spawn positions the event terms of any policy on this scene may draw from. */
  terrainData?: TerrainData | null;
  /** Seconds per control step (mjlab's `step_dt`). */
  controlDt?: number | null;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
};

type MotionCommandTerm = {
  setSelectedMotion(name: string | null): Promise<boolean> | boolean;
  setReferenceVisible?(visible: boolean): void;
  getSelectedMotionName?(): string | null;
};

function isMotionCommandTerm(term: unknown): term is MotionCommandTerm {
  return (
    typeof term === 'object' &&
    term !== null &&
    typeof (term as MotionCommandTerm).setSelectedMotion === 'function'
  );
}

/** Thrown when a scene exceeds the browser's 2 GB WebAssembly memory limit. */
export class WasmMemoryLimitError extends Error {
  constructor() {
    super(
      "This scene cannot be loaded because it exceeds the browser's " +
        'WebAssembly 2 GB memory limit. ' +
        'Try closing other browser tabs or reloading the page to free memory.'
    );
    this.name = 'WasmMemoryLimitError';
  }
}

// mj_loadXML surfaces OOM via four distinct paths depending on which internal
// allocator hits the 2 GB ceiling first (null return, MuJoCo error string,
// lodepng error string, or raw bad_alloc).
function isWasmOom(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    msg.includes('MjModel loading returned null') ||
    msg.includes('Could not allocate memory') ||
    msg.includes('memory allocation failed') ||
    msg.includes('bad_alloc')
  );
}

type BodyState = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

export class mjswanRuntime {
  private mujoco: MainModule;
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;
  private mjModel: MjModel | null;
  private mjData: MjData | null;
  private bodies: Record<number, THREE.Group> | null;
  private lights: THREE.Light[];
  private mujocoRoot: THREE.Group | null;
  private lastSimState: {
    bodies: Map<number, BodyState>;
    tendons: ReturnType<typeof createTendonState>;
  };
  private dynamicBodyIds: Set<number> | null;
  private loopPromise: Promise<void> | null;
  private running: boolean;
  private timestep: number;
  private decimation: number;
  /** Null for a policy-less scene. */
  private controlDt: number | null;
  private loadingScene: Promise<void> | null;
  private resizeObserver: ResizeObserver | null;
  private dragStateManager: DragStateManager | null;
  private dragForceScale: number;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private initialQpos: number[] | null;
  private initialQvel: number[] | null;
  private policyControl: ResolvedActionTerm[] | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private onnxTimeStep: number;
  private terminationManager: TerminationManager | null;
  private eventManager: EventManager | null;
  private terrainData: TerrainData | null;
  private vrButton: HTMLElement | null;
  private arButton: HTMLElement | null;
  /** Both XR support checks are async: a late one must not add a button after teardown. */
  private disposed = false;
  private readonly passthrough: Passthrough;
  private handMocap: HandMocap | null;
  /** Parent of the camera and hands: what XR locomotion moves. Identity outside a session. */
  private readonly xrRig: THREE.Group;
  private readonly xrClock = new THREE.Clock(false);
  /** three has not written a head pose yet on a session's first frame. */
  private xrFirstFrame = true;
  /** Only an opaque session stands the viewer on the terrain; passthrough keeps the room's floor. */
  private groundsRig = false;
  /** Camera-to-target offset from before a session. */
  private preXrCameraOffset: THREE.Vector3 | null;
  private splatMesh: SplatMesh | null;
  private colliderMesh: THREE.Group | null;
  private currentSplatTransform: SplatTransform;
  private cameraState: ViewerState;
  private commandManager: CommandManager;
  /** `joint_position_reference` terms → the command name publishing their reference. */
  private readonly referenceActionCommands = new Map<ResolvedActionTerm, string>();
  private scenePlugins: EnginePlugins;
  private policyPlugins: EnginePlugins;
  /** Every traced graph of the current MDP: observations, terminations, commands, events. */
  private policyGraphs: OnnxSessionCache;
  /**
   * The compiled values of every `mjModel` field a startup randomization has touched, for
   * the life of the model (ADR 0006 §9). Restored before the next MDP's startup pass, so
   * randomizations never compound across a policy switch.
   */
  private modelFieldDefaults: ModelFieldDefaults | null;
  /**
   * Shared by every ONNX term's `rand` input. Reseeded at every scene load *and* every
   * policy load, so randomization is a function of (seed, MDP) rather than of playback
   * so far.
   */
  private termRng: SeededRng;
  /** Held so an app recording a session can persist the seed the run used. */
  private readonly termSeed: number;
  private readonly readOnnxSlot: SlotReader;
  private jointBias = new Map<string, number>();
  private clipActions: number | null = null;
  private raycastSensors: Record<string, RaycastSensorDescriptor> = {};
  /** Owned here, not by the slot reader: the history advances per substep. */
  private contactSensors = new ContactSensorSet();

  constructor(
    mujoco: MainModule,
    container: HTMLElement,
    termSeed = DEFAULT_TERM_SEED,
    handTracking = false,
  ) {
    this.mujoco = mujoco;
    this.container = container;
    this.termSeed = termSeed;
    this.commandManager = new CommandManager();
    this.scenePlugins = {};
    this.policyPlugins = {};
    this.policyGraphs = new OnnxSessionCache();
    this.modelFieldDefaults = null;
    this.termRng = new SeededRng(termSeed);
    // Re-reads mjModel/mjData per call so a scene rebuild needs no rewiring.
    this.readOnnxSlot = createSlotReader(
      () => ({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        commandManager: this.commandManager,
      }),
      {
        jointBias: (name) => this.jointBias.get(name) ?? 0,
        raycastSensors: () => this.raycastSensors,
        contactSensors: () => this.contactSensors,
      },
    );

    const workingPath = '/working';
    try {
      this.mujoco.FS.mkdir(workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST') {
        console.warn('Failed to create /working directory:', error);
      }
    }
    try {
      this.mujoco.FS.mount(this.mujoco.MEMFS, { root: '.' }, workingPath);
    } catch (error: unknown) {
      if (error && typeof error === 'object' && 'code' in error && error.code !== 'EEXIST' && error.code !== 'EBUSY') {
        console.warn('Failed to mount MEMFS at /working:', error);
      }
    }

    const { width, height } = this.getSize();

    this.scene = new THREE.Scene();
    this.scene.name = 'scene';

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
    this.camera.name = 'PerspectiveCamera';
    this.camera.position.set(2.0, 1.7, 1.7);
    this.xrRig = new THREE.Group();
    this.xrRig.name = 'XR Rig';
    this.xrRig.add(this.camera);
    this.scene.add(this.xrRig);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.xr.enabled = true;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.container.appendChild(this.renderer.domElement);

    this.handMocap = null;
    if (handTracking) {
      const hands = [0, 1].map((i) => this.renderer.xr.getHand(i));
      // Spheres, not the `mesh` profile: that one fetches a glTF from a CDN, and a built
      // mjswan app is self-contained.
      const handModels = new XRHandModelFactory();
      for (const hand of hands) {
        hand.add(handModels.createHandModel(hand, 'spheres'));
        this.xrRig.add(hand);
      }
      this.handMocap = new HandMocap(hands);
    }

    // Asked for in both modes: a Quest leaves hands untracked without it.
    const sessionInit: XRSessionInit = handTracking ? { optionalFeatures: ['hand-tracking'] } : {};

    this.vrButton = null;
    navigator.xr?.isSessionSupported('immersive-vr').then((supported) => {
      if (supported && !this.disposed) {
        this.vrButton = VRButton.createButton(this.renderer, sessionInit);
        document.body.appendChild(this.vrButton);
        this.layoutArButton();
      }
    });

    this.arButton = null;
    createArButton(this.renderer, sessionInit).then((button) => {
      if (button && !this.disposed) {
        this.arButton = button;
        document.body.appendChild(this.arButton);
        this.layoutArButton();
      }
    });

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.2, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.passthrough = new Passthrough(this.scene);
    this.preXrCameraOffset = null;
    this.renderer.xr.addEventListener('sessionstart', this.onXrSessionStart);
    this.renderer.xr.addEventListener('sessionend', this.onXrSessionEnd);

    this.renderer.setAnimationLoop(this.render);
    window.addEventListener('resize', this.onWindowResize);

    if ('ResizeObserver' in window) {
      this.resizeObserver = new ResizeObserver(() => this.onWindowResize());
      this.resizeObserver.observe(this.container);
    } else {
      this.resizeObserver = null;
    }

    this.lastSimState = {
      bodies: new Map(),
      tendons: createTendonState(),
    };
    this.dynamicBodyIds = null;

    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.loopPromise = null;
    this.running = false;
    this.timestep = 0.001;
    this.decimation = 1;
    this.controlDt = null;
    this.loadingScene = null;
    this.dragStateManager = null;
    this.dragForceScale = 100.0;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.initialQpos = null;
    this.initialQvel = null;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.clipActions = null;
    this.terminationManager = null;
    this.eventManager = null;
    this.terrainData = null;
    this.splatMesh = null;
    this.colliderMesh = null;
    this.currentSplatTransform = {};
    this.cameraState = { trackBodyId: null, prevBodyPos: null };
  }

  async loadEnvironment(scene: ResolvedScene): Promise<void> {
    // Before the graph swap below, or a running term steps a session it just released.
    await this.stop();
    this.scenePlugins = scene.plugins ?? {};
    this.terrainData = scene.terrainData ?? null;
    // Needed before `buildSceneFromMjz`, which derives `decimation` from it.
    this.controlDt = scene.controlDt && scene.controlDt > 0 ? scene.controlDt : null;
    // Reseed so two loads of the same scene draw the same randomness.
    this.termRng = new SeededRng(this.termSeed);
    // Events belong to the policy's MDP; `loadPolicyConfig` builds their manager.
    this.eventManager = null;
    this.modelFieldDefaults = null;

    // Dispose previous splat/collider before switching scenes
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Clear current references before loading the new scene.
    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;

    await this.buildSceneFromMjz(scene.model);
    // A fresh model: nothing snapshotted yet, nothing to restore.
    if (this.mjModel) this.modelFieldDefaults = new ModelFieldDefaults(this.mjModel);

    if (scene.splat) {
      await this.applySplat(scene.splat);
    }

    // Builds the MDP's event manager and fires its startup pass before the reset.
    await this.loadPolicyConfig(scene.policy ?? null);

    this.applyViewerConfig(scene.viewer ?? null);

    this.running = true;
    void this.startLoop();
  }

  /** Context every event term writes through; re-read per call (scene reloads). */
  private eventContext(): EventContext {
    return {
      mujoco: this.mujoco,
      mjModel: this.mjModel,
      mjData: this.mjData,
      terrainData: this.terrainData,
    };
  }

  /** Clear only; terms are registered from the policy config in `loadPolicyConfig`. */
  private initializeCommands(): void {
    this.commandManager.clear();
  }

  /** Initialize commands from policy config */
  private initializeCommandsFromConfig(
    commands: CommandsConfig,
    context: CommandTermContext
  ): void {
    const commandManager = this.commandManager;
    commandManager.initialize(commands, context, this.policyPlugins.commands);
    console.log('[mjswanRuntime] Commands loaded from policy config:', Object.keys(commands));
  }

  /** Reset the simulation state; callable from the UI via the CommandManager. */
  resetSimulation(): void {
    // Sync by contract, so the reset terms land a microtask late; the step loop awaits.
    void this.resetSimulationState().catch((error) => {
      console.warn('[mjswanRuntime] reset terms failed:', error);
    });
    console.log('[mjswanRuntime] Simulation reset');
  }

  async setSelectedMotion(motionName: string | null): Promise<boolean> {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return false;
    }
    const accepted = await term.setSelectedMotion(motionName);
    if (accepted) {
      this.resetSimulation();
    }
    return accepted;
  }

  getSelectedMotionName(): string | null {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term)) {
      return null;
    }
    return term.getSelectedMotionName?.() ?? null;
  }

  setReferenceVisible(visible: boolean): void {
    const term = this.commandManager.getTerm('motion');
    if (!isMotionCommandTerm(term) || typeof term.setReferenceVisible !== 'function') {
      return;
    }
    term.setReferenceVisible(visible);
  }

  private async buildScene(xmlPath: string): Promise<void> {
    if (this.loadingScene) {
      await this.loadingScene;
    }

    this.loadingScene = (async () => {
      const existingRoot = this.scene.getObjectByName('MuJoCo Root');
      if (existingRoot) {
        this.scene.remove(existingRoot);
      }

      const parent = {
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
      };

      [this.mjModel, this.mjData, this.bodies, this.lights] = await loadSceneFromURL(
        this.mujoco,
        xmlPath,
        parent
      );

      if (!this.mjModel || !this.mjData) {
        throw new Error('Failed to load MuJoCo model.');
      }

      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root') as THREE.Group | null;
      this.passthrough.refresh();

      this.mujoco.mj_forward(this.mjModel, this.mjData);
      updateLightsFromData(this.mujoco, this.mjData, this.lights);
      updateHeadlightFromCamera(this.camera, this.lights);
      this.dynamicBodyIds = this.computeDynamicBodyIds(this.mjModel);
      this.syncStaticBodiesFromData();

      this.handMocap?.bind(this.mujoco, this.mjModel);
      // A scene with no policy never reaches `resetSimulationState`, and a keyframe
      // written before injection zero-pads the appended free joints: without this the
      // fingertips spawn at the world origin, inside the scene.
      this.handMocap?.park(this.mjData);
      // Tagged so `frameCamera` can leave them out: a parked hand waits 100 m up, and
      // with DEBUG_DRAW_BONES on it is drawn there.
      for (const bodyId of this.handMocap?.bodyIds() ?? []) {
        if (this.bodies[bodyId]) {
          this.bodies[bodyId].userData.xrHand = true;
        }
      }

      this.timestep = this.mjModel.opt.timestep || 0.001;
      // Never inferred: a wrong control rate runs the policy off-speed silently.
      this.decimation = Math.max(
        1,
        Math.round((this.controlDt ?? DEFAULT_VIEWER_CONTROL_DT) / this.timestep),
      );

      this.lastSimState.bodies.clear();
      this.updateCachedState();

      // Initialize DragStateManager
      if (!this.dragStateManager) {
        this.dragStateManager = new DragStateManager({
          scene: this.scene,
          renderer: this.renderer,
          camera: this.camera,
          container: this.container,
          controls: this.controls,
          draggableBodyIds: this.dynamicBodyIds,
        });
      } else {
        this.dragStateManager.setDraggableBodyIds(this.dynamicBodyIds);
      }

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  // No scene cache to reclaim on OOM, so surface it as WasmMemoryLimitError directly.
  private async buildSceneFromMjz(model: ArrayBuffer): Promise<void> {
    try {
      const xmlPath = await loadMjzFile(this.mujoco, model);
      if (this.handMocap) {
        injectHandMocapFile(this.mujoco, `/working/${xmlPath}`);
      }
      await this.buildScene(xmlPath);
    } catch (error) {
      this.loadingScene = null;
      if (isWasmOom(error)) {
        throw new WasmMemoryLimitError();
      }
      throw error;
    }
  }

  async startLoop(): Promise<void> {
    if (this.loopPromise) {
      return this.loopPromise;
    }
    this.running = true;
    this.loopPromise = this.mainLoop();
    return this.loopPromise;
  }

  async setSplat(splat: ResolvedSplat | null): Promise<void> {
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }
    if (splat) {
      await this.applySplat(splat);
    }
  }

  private async applySplat(splat: ResolvedSplat): Promise<void> {
    this.currentSplatTransform = splat.transform ?? {};
    this.splatMesh = loadSplat(splat.data, this.currentSplatTransform, this.scene);
    if (splat.collider) {
      this.colliderMesh = await loadCollider(splat.collider, this.scene);
    }
  }

  /** Update transform of the existing splat without disposing/reloading (dev calibration). */
  calibrateSplat(transform: SplatTransform): void {
    this.currentSplatTransform = transform;
    if (this.splatMesh) {
      applySplatTransform(this.splatMesh, transform);
    }
  }

  setSplatVisible(visible: boolean): void {
    if (this.splatMesh) {
      this.splatMesh.visible = visible;
    }
  }

  // ── playback + commands (driven by the engine's verbs) ──────────────────
  /** Instance-scoped command manager the engine reads/writes and subscribes to. */
  get commands(): CommandManager {
    return this.commandManager;
  }

  /** The seed this instance's traced terms draw from, so an app can persist it. */
  get seed(): number {
    return this.termSeed;
  }

  /** The event controls this scene offers: a button per manual term, a checkbox per interval. */
  eventControls(): EventControl[] {
    return this.eventManager?.controls() ?? [];
  }

  /** Fire one `mode="manual"` event term. */
  async fireEvent(name: string): Promise<void> {
    if (!this.eventManager || !this.mjModel || !this.mjData) return;
    try {
      await this.eventManager.fire(name, this.eventContext());
    } catch (error) {
      console.warn(`[mjswanRuntime] manual event "${name}" failed:`, error);
      return;
    }
    // Publish the write even while paused, so the operator sees what they asked for.
    if (!this.running && this.mjModel && this.mjData) {
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();
    }
  }

  /** Start or stop one `mode="interval"` term's schedule. */
  setEventArmed(name: string, armed: boolean): void {
    this.eventManager?.setArmed(name, armed);
  }

  /** Resume the physics loop (rendering runs continuously regardless). */
  play(): void {
    this.running = true;
    void this.startLoop();
  }

  /** Halt physics; rendering continues so a frozen frame can be orbited. */
  pause(): void {
    void this.stop();
  }

  get isRunning(): boolean {
    return this.running;
  }

  // ── camera (spherical, MuJoCo coordinates) ──────────────────────────────
  /** Read the current camera pose back in spherical MuJoCo coordinates. */
  getCameraView(): CameraView {
    const offsetThree = this.camera.position.clone().sub(this.controls.target);
    const distance = offsetThree.length();
    const offset = threeToMjcCoordinate(offsetThree);
    const target = threeToMjcCoordinate(this.controls.target.clone());
    const RAD2DEG = 180 / Math.PI;
    const elevation = distance > 1e-9 ? Math.asin(-offset.z / distance) * RAD2DEG : 0;
    const azimuth = Math.atan2(offset.y, offset.x) * RAD2DEG;
    return {
      lookat: [target.x, target.y, target.z],
      distance,
      azimuth,
      elevation,
      fovy: this.camera.fov,
    };
  }

  /** Overwrite the camera pose; body tracking and OrbitControls both stay live. */
  setCameraView(view: Partial<CameraView>): void {
    const cur = this.getCameraView();
    const lookat = view.lookat ?? cur.lookat;
    const distance = view.distance ?? cur.distance;
    const elevation = view.elevation ?? cur.elevation;
    const azimuth = view.azimuth ?? cur.azimuth;
    this.camera.fov = view.fovy ?? cur.fovy;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(computeCameraPosition(lookat, distance, elevation, azimuth));
    this.controls.target.copy(mjcToThreeCoordinate(lookat));
    this.controls.update();
  }

  /** Re-fit the camera to the current scene bounds, keeping azimuth/elevation. */
  frameCamera(): void {
    if (!this.mujocoRoot) {
      return;
    }
    const box = new THREE.Box3();
    for (const child of this.mujocoRoot.children) {
      if (!child.userData.xrHand) {
        box.expandByObject(child);
      }
    }
    if (box.isEmpty()) {
      return;
    }
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    const fov = (this.camera.fov * Math.PI) / 180;
    const fitDistance = size / (2 * Math.tan(fov / 2));
    const cur = this.getCameraView();
    const lookat = threeToMjcCoordinate(center);
    this.setCameraView({
      lookat: [lookat.x, lookat.y, lookat.z],
      distance: fitDistance > 0 ? fitDistance : cur.distance,
    });
  }

  async stop(): Promise<void> {
    this.running = false;
    const pending = this.loopPromise;
    if (pending) {
      await pending;
    }
    this.loopPromise = null;
  }

  /**
   * Caught, or an escaping throw leaves `loopPromise` rejected for good — `startLoop`
   * returns it instead of restarting and `stop()` re-throws it, wedging `loadScene` too.
   * Stops rather than skips: what reaches here is config, identical next frame.
   */
  private async mainLoop(): Promise<void> {
    try {
      await this.runLoop();
    } catch (error) {
      this.running = false;
      console.error('[mjswanRuntime] simulation loop stopped:', error);
    } finally {
      this.loopPromise = null;
    }
  }

  private async runLoop(): Promise<void> {
    while (this.running) {
      const loopStart = performance.now();
      const target = this.timestep * this.decimation;

      if (this.mjModel && this.mjData) {
        // Mirrors mjlab's `ManagerBasedRlEnv.step()`, one forward per step:
        // forward → command → event → obs → action → physics → term → reset.
        if (this.policyRunner && this.policyStateBuilder) {
          const state = this.policyStateBuilder.build();
          const obs = await this.policyRunner.collectObservationsByKey(state);
          await this.runOnnxInference(obs);
        }
        this.executeSimulationSteps();

        // Pre-forward, as in mjlab: derived state lags by one substep, consistently.
        if (this.terminationManager && this.policyStateBuilder) {
          const postState = this.policyStateBuilder.build();
          const result = this.terminationManager.evaluate(postState, target);
          if (result.done) {
            // Awaited so the writes precede the forward; caught so a failure costs a reset.
            try {
              await this.resetSimulationState({ forward: false });
            } catch (error) {
              console.warn('[mjswanRuntime] reset terms failed:', error);
            }
            // Last, as in mjlab's `_reset_idx`.
            this.terminationManager.reset();
          }
        }

        // The one forward: clears the loop's staleness and publishes a reset's writes.
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();

        this.commandManager.update(target);
        this.commandManager.updateDebugVisuals();
        // Awaited so `mode="interval"` terms resolve in config order.
        try {
          await this.eventManager?.tick(target, this.eventContext());
        } catch (error) {
          console.warn('[mjswanRuntime] interval events failed:', error);
        }
      }

      const elapsed = (performance.now() - loopStart) / 1000;
      const sleepTime = Math.max(0, target - elapsed);
      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
      }
    }
  }

  async loadPolicyConfig(policy: ResolvedPolicy | null): Promise<void> {
    this.policyPlugins = policy?.plugins ?? {};
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.terminationManager = null;
    // The outgoing MDP's events go with it; `terrainData` stays, it is the scene's.
    this.eventManager = null;
    // Before the release below — `setPolicy` runs live.
    this.commandManager.clear();
    await this.policyGraphs.clear();
    this.jointBias.clear();
    this.clipActions = null;
    this.raycastSensors = {};
    this.contactSensors = new ContactSensorSet();

    // An MDP switch, in order (ADR 0006 §9): restore every model field the previous
    // startup pass touched, reseed, then apply the incoming MDP's startup events over the
    // compiled values. Restoring runs even with no policy following, so clearing the
    // policy leaves the scene as compiled.
    const restored = this.modelFieldDefaults?.restore() ?? false;
    if (restored && this.mjModel && this.mjData) {
      this.mujoco.mj_setConst(this.mjModel, this.mjData);
    }
    this.termRng = new SeededRng(this.termSeed);

    if (!policy) {
      if (restored && this.mjModel && this.mjData) {
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();
      }
      return;
    }

    if (!this.mjModel || !this.mjData) {
      // A caller-ordering mistake, not a bad bundle — unreachable from the app.
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    try {
      const config = policy.config;
      await this.policyGraphs.load(policy.graphs ?? []);
      if (config.events && config.events.length > 0) {
        this.eventManager = new EventManager(
          config.events,
          { ...this.scenePlugins.events, ...this.policyPlugins.events },
          {
            sessions: this.policyGraphs,
            rng: this.termRng,
            readSlot: this.readOnnxSlot,
          },
        );
        console.log(
          `[EventManager] ${this.eventManager.size} event term(s) loaded ` +
            `(${this.policyGraphs.size} traced graph(s) in this MDP)`,
        );
        // `mode="startup"` fires once per MDP before its first reset, as mjlab fires it
        // at env construction, over the model `restore()` just put back to compiled.
        await this.eventManager.startup(this.eventContext(), this.modelFieldDefaults ?? undefined);
        this.mujoco.mj_forward(this.mjModel, this.mjData);
      }
      this.jointBias = buildJointBias(config);
      this.clipActions = readClipActions(config.clip_actions);
      const structured = collectStructuredSensors(config);
      this.raycastSensors = structured.raycast;
      this.contactSensors = new ContactSensorSet(structured.contact);
      // Metadata comes from policy.json, bytes from the app; merge them by name.
      if (Array.isArray(config.motions)) {
        const dataByName = new Map(policy.motions.map((m) => [m.name, m.data]));
        config.motions = config.motions.map((motion) => ({
          ...motion,
          data: dataByName.get(motion.name),
        }));
      }
      if (config.commands?.motion && Array.isArray(config.motions)) {
        config.commands.motion = {
          ...config.commands.motion,
          motions: config.motions,
        };
      }
      this.initialQpos = Array.isArray(config.initial_qpos) ? (config.initial_qpos as number[]) : null;
      this.initialQvel = Array.isArray(config.initial_qvel) ? (config.initial_qvel as number[]) : null;
      // Does its own forward; awaited so the reset terms precede the rebuild below.
      await this.resetSimulationState();

      // Initialize commands from policy config if present
      if (config.commands && typeof config.commands === 'object') {
        this.initializeCommandsFromConfig(config.commands as CommandsConfig, {
          mujoco: this.mujoco,
          mjModel: this.mjModel,
          mjData: this.mjData,
          scene: this.scene,
          bodies: this.bodies,
          mujocoRoot: this.mujocoRoot,
          requestReset: () => this.resetSimulation(),
          // Traced commands run a graph per frame off these.
          rng: this.termRng,
          onnxSessions: this.policyGraphs,
          readOnnxSlot: this.readOnnxSlot,
        });
        await this.commandManager.resetTerms();
        const motionTerm = this.commandManager.getTerm('motion');
        if (isMotionCommandTerm(motionTerm)) {
          await motionTerm.setSelectedMotion(
            config.motions?.find((motion) => motion.default)?.name
              ?? config.motions?.[0]?.name
              ?? null
          );
          motionTerm.setReferenceVisible?.(true);
        }
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();
      }

      if (
        !(config.policy_num_actions as number | undefined) &&
        (!config.policy_joint_names || config.policy_joint_names.length === 0)
      ) {
        throw new Error('Policy config missing policy_joint_names.');
      }

      const runner = new PolicyRunner(config, {
        policyModules: {
          tracking: TrackingPolicy,
          locomotion: LocomotionPolicy,
        },
        observations: { ...this.policyPlugins.observations },
        // Custom terms read clip bytes via `runner.getMotionData` rather than fetching.
        motions: policy.motions,
        onnxSessions: this.policyGraphs,
        readOnnxSlot: this.readOnnxSlot,
      });

      await runner.init({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        scene: this.scene,
        commandManager: this.commandManager,
      });

      // Await observation preloads so clip-based obs don't return zeros on the first step.
      await runner.preloadAll();

      this.policyRunner = runner;
      this.policyStateBuilder = new PolicyStateBuilder(
        this.mujoco,
        this.mjModel,
        this.mjData,
        runner.getPolicyJointNames()
      );

      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
      this.policyControl = this.buildPolicyControl(config, runner, this.policyStateBuilder);

      // Initialize termination manager if termination config is present
      if (config.terminations && Object.keys(config.terminations).length > 0) {
        this.terminationManager = new TerminationManager(
          config.terminations,
          { ...this.policyPlugins.terminations },
          runner,
          { onnxSessions: this.policyGraphs, readOnnxSlot: this.readOnnxSlot }
        );
        console.log(`[TerminationManager] ${this.terminationManager.size} termination term(s) loaded`);
      }

      const module = new OnnxModule(policy.onnx, { in_keys: config.in_keys, out_keys: config.out_keys });
      await module.init();
      this.onnxModule = module;
      this.onnxInputDict = module.initInput();

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.policyControl !== null,
      });
    } catch (error) {
      // Everything above is load-bearing, so rethrow rather than report success, and
      // clear the partially-assigned fields so a failure leaves no policy, not half of one.
      this.policyRunner = null;
      this.policyStateBuilder = null;
      this.policyControl = null;
      this.onnxModule = null;
      this.onnxInputDict = null;
      this.terminationManager = null;
      this.eventManager = null;
      throw error;
    }
  }

  private buildPolicyControl(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ): ResolvedActionTerm[] | null {
    const jointNames = runner.getPolicyJointNames();
    const affineBiasValue = this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1;

    const buildEntry = (
      termKey: string,
      controlType: string,
      mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] },
      configScale: number[] | number | Record<string, number> | undefined,
      configOffset: number[] | number | Record<string, number> | undefined,
      configStiffness: number[] | number | Record<string, number> | undefined,
      configDamping: number[] | number | Record<string, number> | undefined,
      useDefaultOffset: boolean,
      configClip?: Record<string, readonly number[]>,
      configEmaAlpha?: number,
      configWarmupTimeS?: number
    ) => {
      const n = mapping.qposAdr.length;
      const subsetJointNames = mapping.actionIndices.map((i) => jointNames[i]);

      const actionScale = this.normalizeControlArray(configScale, n, 1.0, subsetJointNames);
      const actionOffset = this.normalizeControlArray(configOffset, n, 0.0, subsetJointNames);

      const allDefaultJointPos = useDefaultOffset
        ? runner.getDefaultJointPos()
        : new Float32Array(jointNames.length);
      const defaultJointPos = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        defaultJointPos[i] = allDefaultJointPos[mapping.actionIndices[i]];
      }

      const allEncoderBias = Array.isArray(config.encoder_bias)
        ? Float32Array.from(config.encoder_bias)
        : new Float32Array(jointNames.length);
      const encoderBias = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        encoderBias[i] = allEncoderBias[mapping.actionIndices[i]] ?? 0.0;
      }

      const kp = this.normalizeControlArray(configStiffness, n, 0.0, subsetJointNames);
      const kd = this.normalizeControlArray(configDamping, n, 0.0, subsetJointNames);
      // Pattern-keyed, unlike the exact-name siblings above.
      const { clipLo, clipHi } = resolveActionClip(configClip, subsetJointNames, n);

      // Position actuators (biastype=affine) take a target and run their PD in MuJoCo;
      // motor actuators (biastype=none) take a torque and need the PD computed here.
      const positionActuator: boolean[] = mapping.ctrlAdr.map((adr) => {
        if (adr < 0 || !this.mjModel) return false;
        return this.mjModel.actuator_biastype[adr] === affineBiasValue;
      });

      const isPosition = positionActuator.some(Boolean);
      const isMotor = positionActuator.some((v) => !v);
      if (isPosition && isMotor) {
        console.warn(`[PolicyRunner] Action term "${termKey}": mixed actuator types detected.`);
      }
      if (isMotor && controlType !== 'torque' && kp.every((v) => v === 0)) {
        console.error(
          `[PolicyRunner] Action term "${termKey}": motor actuators with no stiffness — ` +
          'every ctrl will be zero. Set `stiffness`/`damping` on the action term.'
        );
      }
      console.log(
        `[PolicyRunner] Action term "${termKey}" (${controlType}): ${n} joint(s), ` +
        `mode: ${isPosition ? 'position (ctrl=target_pos)' : 'motor (ctrl=torque, external PD)'}`
      );

      return {
        controlType,
        ctrlAdr: mapping.ctrlAdr,
        qposAdr: mapping.qposAdr,
        qvelAdr: mapping.qvelAdr,
        actionIndices: mapping.actionIndices,
        actionScale,
        actionOffset,
        defaultJointPos,
        encoderBias,
        positionActuator,
        kp,
        kd,
        muscleNormalize: false,
        clipLo,
        clipHi,
        emaAlpha: configEmaAlpha ?? 1,
        // Whole control steps, as mjlab's `episode_length_buf * step_dt < warmup_time_s`.
        warmupSteps: configWarmupTimeS
          ? Math.ceil(configWarmupTimeS / (this.controlDt ?? DEFAULT_VIEWER_CONTROL_DT))
          : 0,
      };
    };

    // ── Legacy path: no `actions` block, use flat top-level fields ──────────
    const actionsConfig = config.actions;
    if (!actionsConfig || Object.keys(actionsConfig).length === 0) {
      const controlType = config.control_type ?? 'joint_position';
      if (controlType !== 'joint_position' && controlType !== 'torque') {
        console.warn(`[PolicyRunner] Unsupported control_type: ${controlType}`);
        return null;
      }
      const baseMapping = stateBuilder.getControlMapping();
      if (!baseMapping) {
        console.warn('[PolicyRunner] Failed to build control mapping.');
        return null;
      }
      const mapping = {
        ...baseMapping,
        actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
      };
      return [buildEntry(
        'legacy',
        controlType,
        mapping,
        config.action_scale,
        undefined,
        config.stiffness,
        config.damping,
        true
      )];
    }

    // ── Multi-term path: iterate every entry in the `actions` block ─────────
    const results: Array<ReturnType<typeof buildEntry>> = [];

    for (const [termKey, actionTerm] of Object.entries(actionsConfig)) {
      const controlType = actionTerm.type ?? 'joint_position';
      if (
        controlType !== 'joint_position' &&
        controlType !== 'joint_position_reference' &&
        controlType !== 'torque' &&
        controlType !== 'muscle_activation'
      ) {
        console.warn(`[PolicyRunner] Action term "${termKey}": unsupported type "${controlType}", skipping.`);
        continue;
      }

      const patterns = actionTerm.actuator_names ?? ['.*'];

      if (controlType === 'muscle_activation') {
        const muscleMapping = stateBuilder.getCtrlMappingByActuatorNames(patterns);
        if (!muscleMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no actuators matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
        const n = muscleMapping.ctrlAdr.length;
        const actionScale = this.normalizeControlArray(
          actionTerm.scale as number[] | number | Record<string, number> | undefined,
          n,
          1.0
        );
        const actionOffset = this.normalizeControlArray(
          actionTerm.offset as number[] | number | Record<string, number> | undefined,
          n,
          0.0
        );
        const muscleNormalize = (actionTerm as { normalize?: boolean }).normalize ?? true;
        console.log(
          `[PolicyRunner] Action term "${termKey}" (muscle_activation): ${n} actuator(s), normalize=${muscleNormalize}`
        );
        results.push({
          controlType,
          ctrlAdr: muscleMapping.ctrlAdr,
          qposAdr: [],
          qvelAdr: [],
          actionIndices: muscleMapping.actionIndices,
          actionScale,
          actionOffset,
          defaultJointPos: new Float32Array(n),
          encoderBias: new Float32Array(n),
          positionActuator: new Array(n).fill(false),
          kp: new Float32Array(n),
          kd: new Float32Array(n),
          muscleNormalize,
          emaAlpha: 1,
          warmupSteps: 0,
          ...resolveActionClip(
            actionTerm.clip as Record<string, readonly number[]> | undefined,
            muscleMapping.ctrlAdr.map((_, i) => patterns[i] ?? ''),
            n
          ),
        });
        continue;
      }

      // If actuator_names is absent or [".*"], match all joints (backward-compatible).
      const isMatchAll = patterns.length === 1 && patterns[0] === '.*';

      let mapping: { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[]; actionIndices: number[] } | null;

      if (isMatchAll) {
        const baseMapping = stateBuilder.getControlMapping();
        if (!baseMapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": failed to build control mapping, skipping.`);
          continue;
        }
        mapping = {
          ...baseMapping,
          actionIndices: Array.from({ length: baseMapping.qposAdr.length }, (_, i) => i),
        };
      } else {
        mapping = stateBuilder.getControlMappingFor(patterns, jointNames);
        if (!mapping) {
          console.warn(`[PolicyRunner] Action term "${termKey}": no joints matched patterns [${patterns.join(', ')}], skipping.`);
          continue;
        }
      }

      const useDefaultOffset = actionTerm.use_default_offset !== undefined
        ? actionTerm.use_default_offset
        : controlType === 'joint_position';

      const entry = buildEntry(
        termKey,
        controlType,
        mapping,
        actionTerm.scale as number[] | number | Record<string, number> | undefined,
        actionTerm.offset as number[] | number | Record<string, number> | undefined,
        actionTerm.stiffness as number[] | number | Record<string, number> | undefined,
        actionTerm.damping as number[] | number | Record<string, number> | undefined,
        useDefaultOffset,
        actionTerm.clip as Record<string, readonly number[]> | undefined,
        actionTerm.ema_alpha as number | undefined,
        actionTerm.warmup_time_s as number | undefined
      );
      if (controlType === 'joint_position_reference') {
        this.referenceActionCommands.set(entry, String(actionTerm.command_name ?? 'motion'));
      }
      results.push(entry);
    }

    if (results.length === 0) {
      console.warn('[PolicyRunner] No valid action terms found in config.actions.');
      return null;
    }
    return results;
  }

  private normalizeControlArray(
    values: number[] | number | Record<string, number> | undefined,
    length: number,
    fallback: number,
    jointNames?: string[]
  ): Float32Array {
    const output = new Float32Array(length);
    output.fill(fallback);
    if (typeof values === 'number') {
      output.fill(values);
      return output;
    }
    if (Array.isArray(values)) {
      for (let i = 0; i < length; i++) {
        output[i] = typeof values[i] === 'number' ? values[i] : fallback;
      }
      return output;
    }
    if (values !== null && typeof values === 'object' && jointNames) {
      for (const [name, val] of Object.entries(values)) {
        const idx = jointNames.indexOf(name);
        if (idx >= 0 && idx < length) {
          output[idx] = val;
        } else {
          console.warn(`[PolicyRunner] Joint name "${name}" not found in policy_joint_names; skipping.`);
        }
      }
      return output;
    }
    return output;
  }

  /**
   * Put the sim back to its initial state, fire `mode="reset"` events, then reset
   * the command terms — mjlab's order (see `resetChain`).
   *
   * The sim state resets synchronously, before the first `await`, so a caller that
   * cannot await still gets an immediately-reset sim. Pass `forward: false` when the
   * caller runs its own `mj_forward` right after.
   */
  private async resetSimulationState({ forward = true }: { forward?: boolean } = {}): Promise<void> {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    if (this.mjModel.nkey > 0) {
      this.mujoco.mj_resetDataKeyframe(this.mjModel, this.mjData, 0);
    } else {
      this.mujoco.mj_resetData(this.mjModel, this.mjData);
    }
    if (this.initialQpos) {
      const qpos = this.mjData.qpos;
      for (let i = 0; i < Math.min(this.initialQpos.length, this.mjModel.nq); i++) {
        qpos[i] = this.initialQpos[i];
      }
    }
    if (this.initialQvel) {
      const qvel = this.mjData.qvel;
      for (let i = 0; i < Math.min(this.initialQvel.length, this.mjModel.nv); i++) {
        qvel[i] = this.initialQvel[i];
      }
    }
    // After the qpos writes above, which do not cover the injected hand bodies.
    this.handMocap?.park(this.mjData);
    // With the sim state, as mjlab does: a force from before the reset would otherwise
    // keep an `illegal_contact` term firing.
    this.contactSensors.reset();
    // Reset with the sim state, not the terms: it reads nothing from the scene.
    if (this.onnxModule) {
      this.onnxInputDict = this.onnxModule.initInput();
    }
    this.onnxTimeStep = 0;
    this.lastSimState.bodies.clear();
    resetActionSmoothing(this.policyControl ?? []);

    await applyResetTerms({
      events: this.eventManager,
      policy: this.policyRunner,
      commands: this.commandManager,
      context: this.eventContext(),
      // A thunk so it reads state after the events land.
      buildState: () => this.policyStateBuilder?.build(),
    });

    // Re-checked: the scene can be disposed while the reset events are in flight.
    if (forward && this.mjModel && this.mjData) {
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();
    }
  }

  private executeSimulationSteps(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    // Viewer-only: mouse-drag forces and tracked hands, not part of the MDP.
    this.applyDragForces();
    this.handMocap?.update(this.mjModel, this.mjData);

    this.refreshActionReferences();
    advanceActionSmoothing(
      this.policyControl ?? [],
      this.policyRunner?.getLastActions() ?? EMPTY_ACTIONS,
    );
    stepPhysics(
      this.mujoco,
      this.mjModel,
      this.mjData,
      this.policyControl ?? [],
      this.policyRunner?.getLastActions() ?? EMPTY_ACTIONS,
      this.decimation,
      undefined,
      // Per substep, not per control step, as mjlab rolls it from
      // `scene.update(dt=physics_dt)` inside its own decimation loop.
      this.contactSensors.size > 0 ? () => this.advanceContactSensors() : undefined,
    );
  }

  /**
   * Point each reference-residual action term at this step's reference pose. Refreshed
   * here so `applyAction` stays a pure function of what it is handed — the rollout-parity
   * harness drives it in Node with no command manager.
   */
  private refreshActionReferences(): void {
    if (this.referenceActionCommands.size === 0) return;
    for (const [term, commandName] of this.referenceActionCommands) {
      const command = this.commandManager.getTerm(commandName);
      term.referenceJointPos = command?.getStateField?.('tracked_joint_pos') ?? null;
    }
  }

  private advanceContactSensors(): void {
    if (!this.mjModel || !this.mjData) return;
    const mjModel = this.mjModel;
    this.contactSensors.advance(mjModel, this.mjData, (sensor) =>
      sensorWindow(mjModel, sensor),
    );
  }

  private async runOnnxInference(obs: Record<string, Float32Array>): Promise<void> {
    if (!this.onnxModule || !this.policyRunner || this.onnxInferencing) {
      return;
    }

    this.onnxInferencing = true;
    try {
      if (!this.onnxInputDict) {
        this.onnxInputDict = this.onnxModule.initInput();
      }
      const input: Record<string, ort.Tensor> = { ...this.onnxInputDict };
      if (this.onnxModule.inKeys.includes('time_step')) {
        input.time_step = new ort.Tensor('float32', new Float32Array([this.onnxTimeStep]), [1, 1]);
      }
      for (const [key, value] of Object.entries(obs)) {
        input[key] = new ort.Tensor('float32', value, [1, value.length]);
      }
      for (const key of this.onnxModule.inKeys) {
        if (!input[key]) {
          console.warn('[PolicyRunner] Missing ONNX input:', {
            key,
            available: Object.keys(input),
          });
          return;
        }
      }

      const [result, carry] = await this.onnxModule.runInference(input);
      if (Object.keys(carry).length > 0) {
        this.onnxInputDict = { ...this.onnxInputDict, ...carry };
      }
      if (this.onnxModule.inKeys.includes('time_step')) {
        this.onnxTimeStep += 1;
      }

      const outKey = this.onnxModule.outKeys[0];
      const actionTensor = result.action ?? (outKey ? result[outKey] : null) ?? result.policy ?? null;
      if (!actionTensor) {
        return;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      const expectedActionCount = this.policyRunner?.getNumActions() ?? 0;
      if (this.policyControl && action.length !== expectedActionCount) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: expectedActionCount,
          got: action.length,
        });
        return;
      }
      // Before `setLastActions`, which is what both the action terms and the
      // `prev_action` observation slot read — mirroring rsl-rl, where the clamp lands
      // ahead of `env.step` and so ahead of the action manager recording the action.
      clampActions(action, this.clipActions);
      this.policyRunner.setLastActions(action);
    } catch (error) {
      console.warn('[PolicyRunner] ONNX inference failed:', error);
    } finally {
      this.onnxInferencing = false;
    }
  }

  private applyDragForces(): void {
    if (!this.dragStateManager || !this.mjModel || !this.mjData || !this.bodies) {
      return;
    }

    // Clear xfrc_applied (reset to zero at each step)
    for (let i = 0; i < this.mjData.xfrc_applied.length; i++) {
      this.mjData.xfrc_applied[i] = 0.0;
    }

    const dragged = this.dragStateManager.physicsObject;
    if (!dragged || !('bodyID' in dragged) || typeof dragged.bodyID !== 'number' || dragged.bodyID <= 0) {
      return;
    }

    const bodyId = dragged.bodyID as number;
    if (this.dynamicBodyIds && !this.dynamicBodyIds.has(bodyId)) {
      return;
    }

    // Update body positions (for drag calculation)
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (this.bodies[b]) {
        getPosition(this.mjData.xpos, b, this.bodies[b].position);
        getQuaternion(this.mjData.xquat, b, this.bodies[b].quaternion);
        this.bodies[b].updateWorldMatrix(true, false);
      }
    }

    // Update offset
    this.dragStateManager.update();

    // Calculate force (Three.js coordinate system → MuJoCo coordinate system)
    const forceThree = this.dragStateManager.offset
      .clone()
      .multiplyScalar(this.dragForceScale);
    const force = threeToMjcCoordinate(forceThree);

    // Point where force is applied (world coordinates)
    const pointThree = this.dragStateManager.worldHit.clone();
    const point = threeToMjcCoordinate(pointThree);
    // Body position
    const bodyPos = new THREE.Vector3(
      this.mjData.xpos[bodyId * 3 + 0],
      this.mjData.xpos[bodyId * 3 + 1],
      this.mjData.xpos[bodyId * 3 + 2]
    );

    // Calculate torque: τ = r × F
    const r = new THREE.Vector3(
      point.x - bodyPos.x,
      point.y - bodyPos.y,
      point.z - bodyPos.z
    );
    const f = new THREE.Vector3(force.x, force.y, force.z);
    const torque = new THREE.Vector3().crossVectors(r, f);

    // Set xfrc_applied xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz] for each body
    const offset = bodyId * 6;
    this.mjData.xfrc_applied[offset + 0] = force.x;
    this.mjData.xfrc_applied[offset + 1] = force.y;
    this.mjData.xfrc_applied[offset + 2] = force.z;
    this.mjData.xfrc_applied[offset + 3] = torque.x;
    this.mjData.xfrc_applied[offset + 4] = torque.y;
    this.mjData.xfrc_applied[offset + 5] = torque.z;
  }

  private updateCachedState(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let b = 0; b < this.mjModel.nbody; b++) {
      if (dynamicBodyIds && !dynamicBodyIds.has(b)) {
        continue;
      }
      if (this.bodies[b]) {
        if (!this.lastSimState.bodies.has(b)) {
          this.lastSimState.bodies.set(b, {
            position: new THREE.Vector3(),
            quaternion: new THREE.Quaternion(),
          });
        }
        const state = this.lastSimState.bodies.get(b) as BodyState;
        getPosition(this.mjData.xpos, b, state.position);
        getQuaternion(this.mjData.xquat, b, state.quaternion);
      }
    }

    if (this.mujocoRoot && this.mujocoRoot.cylinders) {
      updateTendonGeometry(
        this.mjModel,
        this.mjData,
        {
          cylinders: this.mujocoRoot.cylinders,
          spheres: this.mujocoRoot.spheres!,
        },
        this.lastSimState.tendons
      );
    }
  }

  private applyViewerConfig(config: ViewerConfig | null): void {
    this.cameraState = applyViewerConfig(config, this.camera, this.controls, this.mjModel, this.mjData);
  }

  private computeDynamicBodyIds(mjModel: MjModel): Set<number> {
    const dynamic = new Set<number>();
    for (let bodyId = 1; bodyId < mjModel.nbody; bodyId++) {
      let current = bodyId;
      while (current > 0) {
        if (mjModel.body_jntnum[current] > 0) {
          dynamic.add(bodyId);
          break;
        }
        current = mjModel.body_parentid[current];
      }
    }
    return dynamic;
  }

  /**
   * A hand bone is resized every frame, but `CapsuleGeometry` is built once at load, so
   * the drawn capsule has to be remade to keep its ends on the joints. A no-op unless
   * `handMocap.ts`'s debug switch is drawing the bones: with it off they have no mesh.
   */
  private resizeHandBoneMeshes(): void {
    if (!this.mjModel || !this.bodies) {
      return;
    }
    for (const bodyId of this.handMocap?.bodyIds() ?? []) {
      const mesh = this.bodies[bodyId]?.children[0] as THREE.Mesh | undefined;
      const geomId = mesh?.userData.geomId as number | undefined;
      if (!mesh || geomId === undefined) {
        continue;
      }
      const half = this.mjModel.geom_size[geomId * 3 + 1];
      // Half a millimetre is under what a headset resolves, and it keeps tracking noise
      // from rebuilding every geometry every frame.
      if (Math.abs(half - ((mesh.userData.drawnHalf as number) ?? 0)) < 0.0005) {
        continue;
      }
      mesh.userData.drawnHalf = half;
      mesh.geometry.dispose();
      mesh.geometry = new THREE.CapsuleGeometry(
        this.mjModel.geom_size[geomId * 3],
        half * 2.0,
        20,
        20
      );
    }
  }

  private syncStaticBodiesFromData(): void {
    if (!this.mjModel || !this.mjData || !this.bodies) {
      return;
    }
    const dynamicBodyIds = this.dynamicBodyIds;
    for (let bodyId = 0; bodyId < this.mjModel.nbody; bodyId++) {
      if (dynamicBodyIds?.has(bodyId)) {
        continue;
      }
      const body = this.bodies[bodyId];
      if (!body) {
        continue;
      }
      getPosition(this.mjData.xpos, bodyId, body.position);
      getQuaternion(this.mjData.xquat, bodyId, body.quaternion);
    }
  }

  /**
   * Right of the VR button, or centred when there is none. Only the AR button moves:
   * `VRButton` re-centres itself when its own check resolves, after this runs.
   */
  private layoutArButton(): void {
    if (!this.arButton) {
      return;
    }
    this.arButton.style.left = this.vrButton ? 'calc(50% + 60px)' : 'calc(50% - 50px)';
  }

  /** Kept as an offset: the orbit target goes on tracking a moving body through a session. */
  private onXrSessionStart = (): void => {
    this.preXrCameraOffset = this.camera.position.clone().sub(this.controls.target);
    this.xrClock.start();
    this.xrFirstFrame = true;
    // Blend mode rather than which button was pressed: anything but `opaque` composites
    // the room in behind the scene, so the scene has to stop covering it.
    const passthrough = this.renderer.xr.getEnvironmentBlendMode() !== 'opaque';
    if (passthrough) {
      this.passthrough.enter();
    }
    this.groundsRig = !passthrough;
    if (this.groundsRig) {
      // Where the viewer config points the desktop camera, beside the tracked body: the
      // world origin is a terrain generator's base plane, not a place to stand.
      this.xrRig.position.set(this.camera.position.x, 0, this.camera.position.z);
    }
  };

  private onXrSessionEnd = (): void => {
    this.passthrough.exit();
    this.groundsRig = false;
    this.xrClock.stop();
    // Back to the origin, so the desktop camera is handed back where it was.
    this.xrRig.position.set(0, 0, 0);
    this.xrRig.quaternion.identity();
    if (this.preXrCameraOffset) {
      this.camera.position.copy(this.controls.target).add(this.preXrCameraOffset);
      this.preXrCameraOffset = null;
    }
    this.controls.update();
  };

  private render = (): void => {
    this.commandManager.updateDebugVisuals();

    // In a session the head pose owns the camera, and OrbitControls stays out of it, or its
    // next update rebuilds the orbit from the head.
    const presenting = this.renderer.xr.isPresenting;
    if (this.mjData) {
      updateCameraFromData(this.mjData, this.camera, this.controls, this.cameraState, presenting);
    }
    if (presenting) {
      const seconds = Math.min(this.xrClock.getDelta(), MAX_XR_FRAME_SECONDS);
      if (this.xrFirstFrame) {
        this.xrFirstFrame = false;
      } else {
        updateXrLocomotion(this.xrRig, this.camera, this.renderer.xr.getSession(), seconds);
        if (this.groundsRig) {
          updateRigGrounding(
            this.xrRig,
            this.camera,
            this.mujoco,
            this.mjModel,
            this.mjData,
            seconds
          );
        }
      }
    } else {
      this.controls.update();
    }

    if (this.mjModel && this.mjData && this.bodies) {
      updateHeadlightFromCamera(this.camera, this.lights);

      for (const [b, state] of this.lastSimState.bodies) {
        const body = this.bodies[b];
        if (body) {
          body.position.copy(state.position);
          body.quaternion.copy(state.quaternion);
        }
      }

      this.resizeHandBoneMeshes();
      updateLightsFromData(this.mujoco, this.mjData, this.lights);

      if (this.mujocoRoot && this.mujocoRoot.cylinders) {
        updateTendonRendering(
          {
            cylinders: this.mujocoRoot.cylinders,
            spheres: this.mujocoRoot.spheres!,
          },
          this.lastSimState.tendons
        );
      }
    }

    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Capture the current view as a JPEG Blob. Renders one frame and immediately
   * copies it into a 2D canvas in the same synchronous turn, so it works without
   * `preserveDrawingBuffer` — the WebGL drawing buffer is still intact before the
   * browser composites. Used by the upload preview's "Scan thumbnail" (ADR 0005).
   */
  async captureThumbnail(options: { maxDim?: number; quality?: number } = {}): Promise<Blob> {
    const { maxDim = 1280, quality = 0.85 } = options;
    const src = this.renderer.domElement;

    this.renderer.render(this.scene, this.camera);

    const sw = src.width;
    const sh = src.height;
    if (!sw || !sh) {
      throw new Error('mjswan: canvas has zero size; nothing to capture.');
    }

    const scale = Math.min(1, maxDim / Math.max(sw, sh));
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));

    const out = document.createElement('canvas');
    out.width = dw;
    out.height = dh;
    const ctx = out.getContext('2d');
    if (!ctx) {
      throw new Error('mjswan: failed to get a 2D context for thumbnail capture.');
    }
    ctx.drawImage(src, 0, 0, dw, dh);

    return await new Promise<Blob>((resolve, reject) => {
      out.toBlob(
        (blob) => (blob ? resolve(blob) : reject(new Error('mjswan: toBlob returned null.'))),
        'image/jpeg',
        quality
      );
    });
  }

  private onWindowResize = (): void => {
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  async dispose(): Promise<void> {
    this.disposed = true;
    // Await stop so the loop halts before we free the state it reads.
    await this.stop();
    this.policyRunner = null;
    this.policyStateBuilder = null;

    // Not cache-managed, so they leak across navigations unless freed here.
    if (this.onnxModule) {
      this.onnxModule.dispose();
      this.onnxModule = null;
    }
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    await this.policyGraphs.clear();
    this.modelFieldDefaults = null;
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    if (this.dragStateManager) {
      this.dragStateManager.dispose();
      this.dragStateManager = null;
    }

    this.mjData = null;
    this.mjModel = null;

    // NOTE: Do NOT dispose Three.js resources here as they may be cached The cache manager
    // will handle their disposal when evicting Just clear references
    // this.disposeThreeJSResources();

    window.removeEventListener('resize', this.onWindowResize);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

    this.renderer.xr.removeEventListener('sessionstart', this.onXrSessionStart);
    this.renderer.xr.removeEventListener('sessionend', this.onXrSessionEnd);
    this.controls.dispose();
    this.renderer.setAnimationLoop(null);
    this.renderer.dispose();

    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }

    if (this.vrButton?.parentElement) {
      this.vrButton.parentElement.removeChild(this.vrButton);
      this.vrButton = null;
    }

    if (this.arButton?.parentElement) {
      this.arButton.parentElement.removeChild(this.arButton);
      this.arButton = null;
    }

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;
    this.lastSimState.bodies.clear();
    this.commandManager.dispose();
  }

  private disposeThreeJSResources(): void {
    if (!this.scene) {
      return;
    }

    this.scene.traverse((object) => {
      if ('geometry' in object && object.geometry) {
        (object.geometry as THREE.BufferGeometry).dispose();
      }
      if ('material' in object && object.material) {
        if (Array.isArray(object.material)) {
          object.material.forEach((material) => this.disposeMaterial(material));
        } else {
          this.disposeMaterial(object.material as THREE.Material);
        }
      }
    });

    while (this.scene.children.length > 0) {
      this.scene.remove(this.scene.children[0]);
    }
  }

  private disposeMaterial(material: THREE.Material): void {
    const anyMaterial = material as THREE.MeshStandardMaterial & {
      map?: THREE.Texture;
      aoMap?: THREE.Texture;
      emissiveMap?: THREE.Texture;
      metalnessMap?: THREE.Texture;
      normalMap?: THREE.Texture;
      roughnessMap?: THREE.Texture;
    };

    if (anyMaterial.map) {
      anyMaterial.map.dispose();
    }
    if (anyMaterial.aoMap) {
      anyMaterial.aoMap.dispose();
    }
    if (anyMaterial.emissiveMap) {
      anyMaterial.emissiveMap.dispose();
    }
    if (anyMaterial.metalnessMap) {
      anyMaterial.metalnessMap.dispose();
    }
    if (anyMaterial.normalMap) {
      anyMaterial.normalMap.dispose();
    }
    if (anyMaterial.roughnessMap) {
      anyMaterial.roughnessMap.dispose();
    }
    material.dispose();
  }

  private getSize(): { width: number; height: number } {
    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;
    return {
      width: Math.max(1, width),
      height: Math.max(1, height),
    };
  }
}
