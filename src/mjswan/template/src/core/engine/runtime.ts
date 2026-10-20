import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  getPosition,
  getQuaternion,
  loadSceneFromURL,
  type ModelFormat,
} from '../scene/scene';
import { loadMjzFile } from '../utils/mjzLoader';
import { type Bytes } from '../utils/bytes';
import type { EnginePlugins } from '../plugins';
import { type SplatTransform, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { Avatar, type AvatarConfig } from '../avatar/avatar';
import { InteractionManager, type InteractionSim } from '../interaction/InteractionManager';
import type { InteractionModeId } from '../interaction/params';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { mjcToThreeCoordinate, threeToMjcCoordinate } from '../scene/coordinate';
import { HandMocap, handWeldNames, injectHandMocapXml } from '../xr/handMocap';
import { WeldHold } from '../grab/weldHold';
import { isEntityPrefixed, readMjcfFile } from '../scene/mjcfInject';
import {
  POINTER_ANCHOR_BODY,
  POINTER_WELD,
  injectPointerGrabXml,
} from '../interaction/grabInject';
import { updateXrLocomotion } from '../xr/locomotion';
import { updateRigGrounding } from '../xr/grounding';
import {
  XR_SESSION_IDS,
  XR_SESSION_MODES,
  handTrackingBlocker,
  xrSessionInit,
  xrSessionReports,
  type XrSessionId,
  type XrSessionReport,
} from '../xr/session';
import { Passthrough } from '../xr/passthrough';
import {
  type CameraView,
  type ViewerConfig,
  type ViewerState,
  applyViewerConfig,
  cameraAngles,
  computeCameraPosition,
  updateCameraFromData,
} from './viewer_config';
import {
  clampActions,
  isTargetActuator,
  readClipActions,
  resolveActionClip,
  stepPhysics,
  substepOptions,
  type ControlType,
  type MuscleMode,
  type ResolvedActionTerm,
} from '../action/applyAction';
import { ActionManager, type ManagedActionTerm } from '../action/ActionManager';
import {
  OnnxActionProcess,
  type OnnxActionGraphConfig,
  type OnnxActionProcessConfig,
} from '../action/OnnxActionProcess';
import { applyResetTerms } from './resetChain';
import { yieldToBrowser } from './yieldToBrowser';
import { TerminationManager } from '../termination/TerminationManager';
import * as ort from 'onnxruntime-web/wasm';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { ActionConfigEntry, PolicyConfig } from '../policy/types';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { CommandManager, type CommandTermContext, type CommandsConfig } from '../command';
import { DebugViz } from '../debugViz/DebugViz';
import { EventManager, type EventControl } from '../event/EventManager';
import { ModelFieldDefaults } from '../event/modelWrite';
import type { EventContext, TerrainData } from '../event/EventBase';
import { OnnxSessionCache, type SlotReader } from '../onnx/session';
import { ContactSensorSet, type ContactSensorDescriptor } from '../onnx/contact';
import type { RaycastSensorDescriptor } from '../onnx/raycast';
import { createSlotReader, sensorWindow } from '../onnx/slotReader';
import { SeededRng } from '../rng';

/** Fixed rather than time-derived, so a plain page load replays identically. */
const DEFAULT_TERM_SEED = 0x5eed;


/**
 * The scene's drawn bounds without the viewer's injected bodies. Recursive because every
 * body hangs under body 0, so a flag checked on the root's children alone never matches.
 */
function expandSceneBounds(box: THREE.Box3, object: THREE.Object3D): void {
  if (object.userData.injected) return;
  if ((object as THREE.Mesh).isMesh) box.expandByObject(object);
  for (const child of object.children) expandSceneBounds(box, child);
}

const EMPTY_ACTIONS = new Float32Array(0);
/** Only the `direct` muscle mode reads an actuator range; the rest carry none. */
const EMPTY_CTRL_RANGE = new Float32Array(0);
/** What `applyAction` implements; a term of any other type is skipped with a warning. */
const SUPPORTED_CONTROL_TYPES: ReadonlySet<string> = new Set<ControlType>([
  'joint_position',
  'joint_position_reference',
  'joint_position_command',
  'joint_position_relative',
  'joint_velocity',
  'torque',
  'muscle_activation',
]);

/** A muscle term's mode, falling back to the pre-`mode` `normalize` flag. */
function readMuscleMode(term: Record<string, unknown>): MuscleMode {
  const mode = term.mode;
  if (mode === 'sigmoid' || mode === 'direct' || mode === 'excitation') return mode;
  return term.normalize === false ? 'excitation' : 'sigmoid';
}

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

/** An avatar with its bytes resolved. */
export type ResolvedAvatar = {
  data: ArrayBuffer;
  config: AvatarConfig;
};

/** A full scene ready to build: model bytes plus resolved policy/splat/config. */
export type ResolvedScene = {
  model: ArrayBuffer;
  /** `mjz` unless the build wrote the scene from `add_scene(model=...)`. */
  modelFormat?: ModelFormat;
  policy?: ResolvedPolicy | null;
  splat?: ResolvedSplat | null;
  avatar?: ResolvedAvatar | null;
  viewer?: ViewerConfig | null;
  /** Spawn positions the event terms of any policy on this scene may draw from. */
  terrainData?: TerrainData | null;
  /** Seconds per control step (mjlab's `step_dt`). */
  controlDt?: number | null;
  /** Scene-scoped custom terms (events). */
  plugins?: EnginePlugins;
};

/** A session granted from the host's click, waiting for `startXr`. */
export type XrRequest = {
  id: XrSessionId;
  session: XRSession;
  /** Whether it asked for `hand-tracking`: a session granted without it tracks none. */
  hands: boolean;
};

type MotionCommandTerm = {
  setSelectedMotion(name: string | null): Promise<boolean> | boolean;
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
  private readonly interaction: InteractionManager;
  /** The weld slots a scene carries, shared by the pointer and the tracked hands. */
  private readonly weldHold = new WeldHold();
  /** Whether this scene got the grab anchor (see `injectViewerBodies`). */
  private grabInjected = false;
  /** The anchor went into an unprefixed model only because no policy came with it. */
  private injectedWithoutPolicy = false;
  /** Every body the viewer added: never part of the scene's own bounds or camera target. */
  private injectedBodyIds: ReadonlySet<number> = new Set();
  /** A policy is loaded, or will load onto the model being built. */
  private sceneHasPolicy = false;
  /**
   * The scene came with a policy. Stays true once that policy is cleared: its policies are
   * sized to the model as loaded, so a rebuild adds nothing that would widen it.
   */
  private sceneLoadedWithPolicy = false;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private initialQpos: number[] | null;
  private initialQvel: number[] | null;
  private actionManager: ActionManager | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private onnxTimeStep: number;
  /** The last inference error logged: one that repeats every frame is logged once. */
  private lastInferenceError: string | null = null;
  /** Control steps since the last reset: mjlab's `episode_length_buf`, which graphs read. */
  private episodeLength = 0;
  private terminationManager: TerminationManager | null;
  private eventManager: EventManager | null;
  private terrainData: TerrainData | null;
  /** A late async XR support check must not report after teardown. */
  private disposed = false;
  private readonly passthrough: Passthrough;
  private readonly handMocap: HandMocap;
  /** The hand-tracking switch, applied at the next model build. */
  private handTrackingEnabled: boolean;
  private modelHasHands = false;
  private xrSupported: Record<XrSessionId, boolean> = { vr: false, ar: false };
  /** Counts support checks, so a slower earlier one cannot overwrite a later answer. */
  private xrCheck = 0;
  /** The session `requestXr` was granted, until it ends. */
  private xrSession: XRSession | null = null;
  private activeXr: XrSessionId | null = null;
  private enteringXr = false;
  /** Called when `xrSessions()` or `handTrackingReport()` may have changed. */
  onXrChange: (() => void) | null = null;
  /** The scene MJCF as shipped, so a rebuild can add or drop the hands. Null for `.mjb`. */
  private sceneXml: { file: string; text: string } | null = null;
  private modelFormat: ModelFormat | null = null;
  private scenePrefixed = false;
  /** Loaded again onto the new model by a rebuild. */
  private currentPolicy: ResolvedPolicy | null = null;
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
  /** Drawn beside the robot, posed from its bodies; outside the model. */
  private avatar: Avatar | null = null;
  private cameraState: ViewerState;
  /** mjviser's "Track camera": whether the view follows the tracked body. */
  private cameraTracking = true;
  private commandManager: CommandManager;
  private readonly debugVizSet: DebugViz;
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
    this.debugVizSet = new DebugViz(this.commandManager, () => ({
      mjModel: this.mjModel,
      mjData: this.mjData,
    }));
    this.scenePlugins = {};
    this.policyPlugins = {};
    this.policyGraphs = new OnnxSessionCache();
    this.modelFieldDefaults = null;
    this.termRng = new SeededRng(termSeed);
    // Lazy: the context is rebuilt on every slot read, and only an `action` slot needs it.
    const lastActions = () => this.policyRunner?.getLastActions() ?? null;
    // Re-reads mjModel/mjData per call so a scene rebuild needs no rewiring.
    this.readOnnxSlot = createSlotReader(
      () => ({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
        commandManager: this.commandManager,
        episodeLength: this.episodeLength,
        lastActions,
        actionTermField: (term, field) => this.actionManager?.getStateField(term, field) ?? null,
        modelDefaults: this.modelFieldDefaults,
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

    // Always created: the hand spaces stay empty until a session is granted `hand-tracking`.
    const hands = [0, 1].map((i) => this.renderer.xr.getHand(i));
    // Spheres, not the `mesh` profile: that one fetches a glTF from a CDN, and a built
    // mjswan app is self-contained.
    const handModels = new XRHandModelFactory();
    for (const hand of hands) {
      hand.add(handModels.createHandModel(hand, 'spheres'));
      this.xrRig.add(hand);
    }
    this.handMocap = new HandMocap(hands, this.weldHold);
    this.handTrackingEnabled = handTracking;

    void this.checkXrSupport();
    navigator.xr?.addEventListener('devicechange', this.onXrDeviceChange);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.2, 0);
    this.controls.panSpeed = 2;
    this.controls.zoomSpeed = 1;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.screenSpacePanning = true;
    this.controls.update();

    this.interaction = new InteractionManager({
      scene: this.scene,
      renderer: this.renderer,
      camera: this.camera,
      container: this.container,
      controls: this.controls,
      sim: () => this.interactionSim(),
      weld: this.weldHold,
    });

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
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.initialQpos = null;
    this.initialQvel = null;
    this.actionManager = null;
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
    // Before the model is built: it decides whether the grab anchor is injected.
    this.sceneHasPolicy = !!scene.policy;
    this.sceneLoadedWithPolicy = !!scene.policy;
    // Needed before `buildSceneFromModel`, which derives `decimation` from it.
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
    this.avatar?.dispose();
    this.avatar = null;

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Clear current references before loading the new scene.
    this.releaseModel();
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;

    await this.buildSceneFromModel(scene.model, scene.modelFormat ?? 'mjz');
    // A fresh model: nothing snapshotted yet, nothing to restore.
    if (this.mjModel) this.modelFieldDefaults = new ModelFieldDefaults(this.mjModel);

    if (scene.splat) {
      await this.applySplat(scene.splat);
    }
    if (scene.avatar) {
      await this.applyAvatar(scene.avatar);
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
      modelDefaults: this.modelFieldDefaults,
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

  private async buildScene(modelPath: string): Promise<void> {
    if (this.loadingScene) {
      await this.loadingScene;
    }

    this.loadingScene = (async () => {
      const existingRoot = this.scene.getObjectByName('MuJoCo Root');
      if (existingRoot) {
        this.scene.remove(existingRoot);
      }

      const parent = { scene: this.scene };

      [this.mjModel, this.mjData, this.bodies, this.lights] = await loadSceneFromURL(
        this.mujoco,
        modelPath,
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

      if (this.modelHasHands) {
        this.handMocap.bind(this.mujoco, this.mjModel);
      } else {
        this.handMocap.clear();
      }
      this.weldHold.bind(this.mujoco, this.mjModel, [
        ...(this.grabInjected ? [POINTER_WELD] : []),
        ...(this.modelHasHands ? handWeldNames() : []),
      ]);
      // A scene with no policy never reaches `resetSimulationState`, and a keyframe
      // written before injection zero-pads the appended free joints: without this the
      // fingertips spawn at the world origin, inside the scene.
      if (this.modelHasHands) this.handMocap.park(this.mjData);
      // Tagged so `frameCamera` can leave them out: a parked hand waits 100 m up, and
      // with DEBUG_DRAW_BONES on it is drawn there.
      this.injectedBodyIds = this.collectInjectedBodyIds();
      for (const bodyId of this.injectedBodyIds) {
        if (this.bodies[bodyId]) {
          this.bodies[bodyId].userData.injected = true;
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

      this.interaction.onSceneLoaded();

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  /** Frees the model on the WASM heap, which nothing else does. Call with the loop stopped. */
  private releaseModel(): void {
    this.mjData?.delete();
    this.mjModel?.delete();
    this.mjData = null;
    this.mjModel = null;
  }

  // No scene cache to reclaim on OOM, so surface it as WasmMemoryLimitError directly.
  private async buildSceneFromModel(model: ArrayBuffer, format: ModelFormat): Promise<void> {
    try {
      let modelPath: string;
      this.modelFormat = format;
      if (format === 'mjb') {
        // `loadSceneFromURL` reads a `.mjb` path with `mj_loadModel`. A compiled model has
        // no XML to inject into, so the XR hands and Grab stay off.
        modelPath = 'scene.mjb';
        this.mujoco.FS.writeFile(`/working/${modelPath}`, new Uint8Array(model));
        this.sceneXml = null;
        this.scenePrefixed = false;
        this.grabInjected = false;
        this.injectedWithoutPolicy = false;
        this.modelHasHands = false;
      } else {
        modelPath = await loadMjzFile(this.mujoco, model);
        const text = readMjcfFile(this.mujoco, `/working/${modelPath}`);
        this.sceneXml = { file: modelPath, text };
        this.scenePrefixed = isEntityPrefixed(text);
        this.injectViewerBodies(this.wantsHands());
      }
      await this.buildScene(modelPath);
    } catch (error) {
      this.loadingScene = null;
      if (isWasmOom(error)) {
        throw new WasmMemoryLimitError();
      }
      throw error;
    }
  }

  /**
   * Write the scene MJCF to the VFS with the viewer's bodies added. Always starts from the
   * shipped text, so a rebuild can drop the hands as well as add them.
   *
   * For an unprefixed model of a policy scene, the grab anchor is skipped here and the hands
   * by `handTrackingBlocker`: `buildEntityIndex` counts every body of such a model as the
   * entity's, so extra bodies would widen a traced graph's input.
   */
  private injectViewerBodies(hands: boolean): void {
    if (!this.sceneXml) return;
    const { file, text } = this.sceneXml;
    let injected = text;
    this.modelHasHands = hands;
    if (hands) injected = injectHandMocapXml(injected);
    this.grabInjected = this.scenePrefixed || !this.policyScene;
    this.injectedWithoutPolicy = (this.grabInjected || hands) && !this.scenePrefixed;
    if (this.grabInjected) {
      injected = injectPointerGrabXml(injected);
    } else {
      console.warn(
        '[mjswan] grab is off for this scene: it has a policy and the model is not ' +
          'entity-prefixed, so an extra body would change a traced graph\'s input width.',
      );
    }
    this.mujoco.FS.writeFile(`/working/${file}`, injected);
  }

  async startLoop(): Promise<void> {
    // A load or rebuild finishing after dispose must not restart the loop.
    if (this.disposed) {
      this.running = false;
      return;
    }
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

  /** Render only, so a file that does not load costs the avatar and nothing else. */
  private async applyAvatar(input: ResolvedAvatar): Promise<void> {
    try {
      const avatar = await Avatar.load(input.data, input.config);
      if (this.disposed || !this.mjModel) {
        avatar.dispose();
        return;
      }
      avatar.bind(this.mujoco, this.mjModel);
      this.scene.add(avatar.root);
      this.avatar = avatar;
      this.hideRobotUnderAvatar();
    } catch (error) {
      console.warn('[mjswanRuntime] avatar not loaded; the scene runs without it:', error);
    }
  }

  /**
   * The avatar as the robot's skin. Hidden by material, each geom's own: neither drawn
   * nor shadowed, still hit by the pointer's raycast, and the tracking ghost, a clone
   * with materials of its own, still shows.
   */
  private hideRobotUnderAvatar(): void {
    if (!this.avatar || !this.bodies) return;
    for (const bodyId of this.avatar.hiddenBodies) {
      this.bodies[bodyId]?.traverse((object) => {
        if (!(object as THREE.Mesh).isMesh) return;
        const material = (object as THREE.Mesh).material;
        for (const m of Array.isArray(material) ? material : [material]) m.visible = false;
      });
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

  get debugViz(): DebugViz {
    return this.debugVizSet;
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
    // A held body would otherwise keep its force the moment play resumes.
    this.interaction.cancel();
    void this.stop();
  }

  // ── pointer interaction ───────────────────────────────────────────────

  getInteractionMode(): InteractionModeId {
    return this.interaction.getMode();
  }

  setInteractionMode(mode: InteractionModeId): void {
    this.interaction.setMode(mode);
  }

  getInteractionParams(mode: InteractionModeId): Readonly<Record<string, number>> {
    return this.interaction.getParams(mode);
  }

  setInteractionParam(mode: InteractionModeId, name: string, value: number): void {
    this.interaction.setParam(mode, name, value);
  }

  interactionModes(): ReturnType<InteractionManager['report']> {
    return this.interaction.report();
  }

  cancelInteraction(): void {
    this.interaction.cancel();
  }

  private interactionSim(): InteractionSim {
    return {
      mujoco: this.mujoco,
      mjModel: this.mjModel,
      mjData: this.mjData,
      dynamicBodyIds: this.dynamicBodyIds,
      controlDt: this.timestep * this.decimation,
    };
  }

  // ── WebXR ─────────────────────────────────────────────────────────────

  /** The sessions this device can start. */
  xrSessions(): XrSessionReport[] {
    return xrSessionReports(this.xrSupported, this.activeXr);
  }

  /** Null on a device without VR, where no session could track hands. */
  handTrackingReport(): { available: boolean; reason?: string; enabled: boolean } | null {
    if (!this.xrSupported.vr) return null;
    const reason = this.handTrackingBlockerNow();
    return reason === null
      ? { available: true, enabled: this.handTrackingEnabled }
      : { available: false, reason, enabled: this.handTrackingEnabled };
  }

  setHandTracking(enabled: boolean): void {
    if (enabled === this.handTrackingEnabled) return;
    this.handTrackingEnabled = enabled;
    this.onXrChange?.();
  }

  /**
   * Asks for the session from the host's click: the browser grants one only inside that
   * gesture, so nothing is awaited before the request. `hand-tracking` is decided here,
   * since it has to be in the request. Null when no entry is possible now.
   */
  async requestXr(id: XrSessionId): Promise<XrRequest | null> {
    const xr = navigator.xr;
    if (this.disposed || !xr || !this.xrSupported[id] || this.enteringXr || this.xrSession) return null;
    this.enteringXr = true;
    const hands = this.wantsHands();
    try {
      const session = await xr.requestSession(XR_SESSION_MODES[id], xrSessionInit(id, hands));
      this.xrSession = session;
      session.addEventListener('end', () => this.onXrSessionClosed(session), { once: true });
      return { id, session, hands };
    } catch (error) {
      this.enteringXr = false;
      throw error;
    }
  }

  /**
   * Rebuilds the model when it disagrees with the hand switch, then hands the session to
   * three, which must not render a model about to be replaced. Runs once no other load
   * is, so the scene may have changed since the request, and the hands are decided again.
   */
  async startXr(request: XrRequest, onRebuild?: (addingHands: boolean) => void): Promise<void> {
    const { id, session } = request;
    try {
      // Disposed, or ended from the headset, while it waited.
      if (this.disposed || this.xrSession !== session) {
        await session.end().catch(() => {});
        return;
      }
      const hands = request.hands && this.wantsHands();
      try {
        // No model means an earlier rebuild failed; this retries it.
        if ((hands !== this.modelHasHands || !this.mjModel) && this.sceneXml) {
          onRebuild?.(hands);
          await this.rebuildModel(hands);
        }
        if (this.xrSession !== session) return;
        this.renderer.xr.setReferenceSpaceType('local-floor');
        await this.renderer.xr.setSession(session);
      } catch (error) {
        await session.end().catch(() => {});
        throw error;
      }
      this.activeXr = id;
      this.onXrChange?.();
    } finally {
      this.enteringXr = false;
    }
  }

  async exitXr(): Promise<void> {
    await this.xrSession?.end();
  }

  private onXrSessionClosed(session: XRSession): void {
    if (this.xrSession !== session) return;
    this.xrSession = null;
    this.activeXr = null;
    this.onXrChange?.();
  }

  private wantsHands(): boolean {
    return this.handTrackingEnabled && this.xrSupported.vr && this.handTrackingBlockerNow() === null;
  }

  private handTrackingBlockerNow(): string | null {
    if (this.modelFormat === null) return null;
    return handTrackingBlocker({
      format: this.modelFormat,
      prefixed: this.scenePrefixed,
      hasPolicy: this.policyScene,
    });
  }

  /** A policy runs on this model, or came with the scene and may run on it again. */
  private get policyScene(): boolean {
    return this.sceneHasPolicy || this.sceneLoadedWithPolicy;
  }

  /** Asked again on `devicechange`, since a PC headset can be plugged in after load. */
  private async checkXrSupport(): Promise<void> {
    const xr = navigator.xr;
    if (!xr) return;
    const check = ++this.xrCheck;
    const answers = await Promise.all(
      XR_SESSION_IDS.map(async (id) => {
        try {
          return await xr.isSessionSupported(XR_SESSION_MODES[id]);
        } catch (error: unknown) {
          // Throws in a frame not granted `xr-spatial-tracking`.
          console.warn(`[mjswan] isSessionSupported('${XR_SESSION_MODES[id]}') failed:`, error);
          return false;
        }
      }),
    );
    if (this.disposed || check !== this.xrCheck) return;
    const supported = { ...this.xrSupported };
    XR_SESSION_IDS.forEach((id, i) => {
      supported[id] = answers[i];
    });
    const changed = XR_SESSION_IDS.some((id) => supported[id] !== this.xrSupported[id]);
    this.xrSupported = supported;
    if (changed) this.onXrChange?.();
  }

  private onXrDeviceChange = (): void => {
    void this.checkXrSupport();
  };

  /**
   * Recompile the loaded scene with or without the hand bones, keeping the policy, motion,
   * command values, debug drawings, event schedules, splat and camera. The simulation
   * restarts as after a reset. The hands are appended, so original body ids (and a camera
   * tracking one) stay valid.
   */
  private async rebuildModel(hands: boolean): Promise<void> {
    const wasRunning = this.running;
    await this.stop();
    this.interaction.cancel();
    const motion = this.getSelectedMotionName();
    const values = this.commandManager.getValues();
    const drawings = this.debugVizSet.entries();
    const schedules = this.eventManager?.controls().filter((c) => c.kind === 'interval') ?? [];
    const kept = { module: this.onnxModule };

    // Drop everything holding the old model before freeing it: command terms read it every frame.
    this.commandManager.clear();
    this.eventManager = null;
    this.onnxModule = null;
    this.modelFieldDefaults = null;
    this.releaseModel();
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;

    this.injectViewerBodies(hands);
    try {
      await this.buildScene(this.sceneXml!.file);
    } catch (error) {
      // As `buildSceneFromModel` does, or the next scene load waits on this failure.
      this.loadingScene = null;
      // Free any half-built model, so the next entry finds none and retries.
      this.releaseModel();
      await kept.module?.dispose();
      throw isWasmOom(error) ? new WasmMemoryLimitError() : error;
    }
    if (this.mjModel) this.modelFieldDefaults = new ModelFieldDefaults(this.mjModel);
    // New body groups, drawn again; the avatar's body ids hold, as the hands are appended.
    this.hideRobotUnderAvatar();
    await this.loadPolicyConfig(this.currentPolicy, kept);
    if (motion !== null) await this.setSelectedMotion(motion);
    for (const [id, value] of Object.entries(values)) this.commandManager.setValue(id, value);
    for (const { id, enabled } of drawings) this.debugVizSet.set(id, enabled);
    for (const { name, armed } of schedules) this.setEventArmed(name, armed);

    if (wasRunning) {
      this.running = true;
      void this.startLoop();
    }
  }

  get isRunning(): boolean {
    return this.running;
  }

  // ── camera (spherical, MuJoCo coordinates) ──────────────────────────────
  /** Read the current camera pose back in spherical MuJoCo coordinates. */
  getCameraView(): CameraView {
    const offset = threeToMjcCoordinate(this.camera.position.clone().sub(this.controls.target));
    const target = threeToMjcCoordinate(this.controls.target.clone());
    return {
      lookat: [target.x, target.y, target.z],
      ...cameraAngles(offset),
      fovy: this.camera.fov,
    };
  }

  get isCameraTracking(): boolean {
    return this.cameraTracking;
  }

  /** Turned back on, the view jumps to the body keeping its angle and zoom, as in mjviser. */
  setCameraTracking(enabled: boolean): void {
    if (enabled && !this.cameraTracking) {
      const body = this.cameraState.trackBodyId;
      if (body !== null && this.mjData) {
        const bodyPos = mjcToThreeCoordinate(this.mjData.xpos.slice(body * 3, body * 3 + 3));
        this.camera.position.add(bodyPos.clone().sub(this.controls.target));
        this.controls.target.copy(bodyPos);
        this.controls.update();
      }
      this.cameraState.prevBodyPos = null;
    }
    this.cameraTracking = enabled;
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
    expandSceneBounds(box, this.mujocoRoot);
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
   * Caught, or an escaping throw leaves `loopPromise` rejected for good: `startLoop`
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
          const fresh = await this.runOnnxInference(obs);
          await this.processActions(fresh);
        }
        this.executeSimulationSteps();
        this.episodeLength += 1;

        // Pre-forward, as in mjlab: derived state lags by one substep, consistently.
        let autoReset = false;
        // Held across the awaits below: `setPolicy` can null the field mid-step.
        const terminations = this.terminationManager;
        if (terminations && this.policyStateBuilder) {
          const postState = this.policyStateBuilder.build();
          const result = await terminations.evaluate(postState);
          if (result.done) {
            autoReset = true;
            // Awaited so the writes precede the forward; caught so a failure costs a reset.
            try {
              await this.resetSimulationState({ forward: false });
            } catch (error) {
              console.warn('[mjswanRuntime] reset terms failed:', error);
            }
            // Last, as in mjlab's `_reset_idx`.
            terminations.reset();
          }
        }

        // The one forward: clears the loop's staleness and publishes a reset's writes.
        this.mujoco.mj_forward(this.mjModel, this.mjData);
        this.updateCachedState();

        this.commandManager.update(target, autoReset);
        this.debugVizSet.update();
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
      } else {
        // Behind schedule: yield anyway, or rAF stops firing and the tab hangs at 100% CPU
        // (issue #103).
        await yieldToBrowser();
      }
    }
  }

  /** `kept` holds a rebuild's ONNX sessions: bound to no model, they are reused, not rebuilt. */
  async loadPolicyConfig(
    policy: ResolvedPolicy | null,
    kept?: { module: OnnxModule | null },
  ): Promise<void> {
    const outgoing = this.onnxModule;
    this.currentPolicy = policy;
    this.sceneHasPolicy = policy !== null;
    this.policyPlugins = policy?.plugins ?? {};
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.actionManager = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.onnxTimeStep = 0;
    this.lastInferenceError = null;
    this.terminationManager = null;
    // The outgoing MDP's events go with it; `terrainData` stays, it is the scene's.
    this.eventManager = null;
    // Before the releases below: `setPolicy` runs live.
    this.commandManager.clear();
    await outgoing?.dispose();
    if (!kept) await this.policyGraphs.clear();
    this.jointBias.clear();
    this.clipActions = null;
    this.raycastSensors = {};
    this.contactSensors = new ContactSensorSet();
    this.debugVizSet.clear();

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
      // A caller-ordering mistake, not a bad bundle: unreachable from the app.
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    if (this.injectedWithoutPolicy) {
      // `injectViewerBodies` only saw the policy the model was built for, not this one.
      console.warn(
        '[mjswan] this scene was loaded without a policy, so the viewer added its grab ' +
          'anchor or tracked hands to a model that does not namespace its elements. A ' +
          'traced graph indexed by body or geom will be fed the wrong width. Load the ' +
          'scene with the policy selected to get the model without them.',
      );
    }

    try {
      const config = policy.config;
      if (!kept) await this.policyGraphs.load(policy.graphs ?? []);
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
        await this.eventManager.startup(this.eventContext());
        this.mujoco.mj_forward(this.mjModel, this.mjData);
      }
      this.jointBias = buildJointBias(config);
      this.clipActions = readClipActions(config.clip_actions);
      const structured = collectStructuredSensors(config);
      this.raycastSensors = structured.raycast;
      this.contactSensors = new ContactSensorSet(structured.contact);
      this.debugVizSet.load(config.debug_vis, this.mujoco, this.mujocoRoot ?? this.scene);
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
      // The first reset's sim reset and events; its term resets follow once the terms exist.
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
        const motionTerm = this.commandManager.getTerm('motion');
        if (isMotionCommandTerm(motionTerm)) {
          await motionTerm.setSelectedMotion(
            config.motions?.find((motion) => motion.default)?.name
              ?? config.motions?.[0]?.name
              ?? null
          );
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

      // The rest of the first reset, in mjlab's order: observations, actions, then commands.
      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
      const actions = this.buildActionManager(config, runner, this.policyStateBuilder);
      await actions?.reset();
      // Before the commands' reset, which can read an action term.
      this.actionManager = actions;
      await this.commandManager.resetTerms();
      this.mujoco.mj_forward(this.mjModel, this.mjData);
      this.updateCachedState();

      // Initialize termination manager if termination config is present
      if (config.terminations && Object.keys(config.terminations).length > 0) {
        this.terminationManager = new TerminationManager(
          config.terminations,
          { ...this.policyPlugins.terminations },
          runner,
          {
            onnxSessions: this.policyGraphs,
            readOnnxSlot: this.readOnnxSlot,
            episodeLength: () => this.episodeLength,
            stepDt: this.timestep * this.decimation,
          }
        );
        console.log(`[TerminationManager] ${this.terminationManager.size} termination term(s) loaded`);
      }

      const module =
        kept?.module ?? new OnnxModule(policy.onnx, { in_keys: config.in_keys, out_keys: config.out_keys });
      if (module !== kept?.module) await module.init();
      this.onnxModule = module;
      this.onnxInputDict = module.initInput();

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.actionManager !== null,
      });
    } catch (error) {
      // Everything above is load-bearing, so rethrow rather than report success, and
      // clear the partially-assigned fields so a failure leaves no policy, not half of one.
      await kept?.module?.dispose();
      this.policyRunner = null;
      this.policyStateBuilder = null;
      this.actionManager = null;
      this.onnxModule = null;
      this.onnxInputDict = null;
      this.terminationManager = null;
      this.eventManager = null;
      this.currentPolicy = null;
      this.sceneHasPolicy = false;
      throw error;
    }
  }

  private buildActionManager(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ): ActionManager | null {
    const jointNames = runner.getPolicyJointNames();
    const targetActuator = this.targetActuatorTest();

    const buildEntry = (
      termKey: string,
      controlType: string,
      mapping: {
        ctrlAdr: number[];
        qposAdr: number[];
        qvelAdr: number[];
        actionIndices: number[];
        jointNames?: string[];
      },
      actionTerm: ActionConfigEntry,
      useDefaultOffset: boolean,
    ): ResolvedActionTerm => {
      const n = mapping.qposAdr.length;
      const subsetJointNames = mapping.jointNames ?? mapping.actionIndices.map((i) => jointNames[i]);

      const actionScale = this.normalizeControlArray(actionTerm.scale, n, 1.0, subsetJointNames);
      const actionOffset = this.normalizeControlArray(actionTerm.offset, n, 0.0, subsetJointNames);

      // Default pose and encoder bias are indexed by policy joint, not command joint.
      const defaultJointPos = new Float32Array(n);
      const encoderBias = new Float32Array(n);
      if (!mapping.jointNames) {
        const allDefaultJointPos = useDefaultOffset
          ? runner.getDefaultJointPos()
          : new Float32Array(jointNames.length);
        const allEncoderBias = Array.isArray(config.encoder_bias)
          ? Float32Array.from(config.encoder_bias)
          : new Float32Array(jointNames.length);
        for (let i = 0; i < n; i++) {
          defaultJointPos[i] = allDefaultJointPos[mapping.actionIndices[i]];
          encoderBias[i] = allEncoderBias[mapping.actionIndices[i]] ?? 0.0;
        }
      }

      const kp = this.normalizeControlArray(actionTerm.stiffness, n, 0.0, subsetJointNames);
      const kd = this.normalizeControlArray(actionTerm.damping, n, 0.0, subsetJointNames);
      // Pattern-keyed, unlike the exact-name siblings above.
      const { clipLo, clipHi } = resolveActionClip(actionTerm.clip, subsetJointNames, n);

      const positionActuator = mapping.ctrlAdr.map(targetActuator);
      const isPosition = positionActuator.some(Boolean);
      const isMotor = positionActuator.some((v) => !v);
      if (isPosition && isMotor) {
        console.warn(`[PolicyRunner] Action term "${termKey}": mixed actuator types detected.`);
      }
      const gain = controlType === 'joint_velocity' ? kd : kp;
      if (isMotor && controlType !== 'torque' && gain.every((v) => v === 0)) {
        console.error(
          `[PolicyRunner] Action term "${termKey}": motor actuators with no ` +
          `${controlType === 'joint_velocity' ? 'damping' : 'stiffness'}, so every ctrl ` +
          'will be zero. Set `stiffness`/`damping` on the action term.'
        );
      }
      console.log(
        `[PolicyRunner] Action term "${termKey}" (${controlType}): ${n} joint(s), ` +
        `mode: ${isPosition ? 'actuator (ctrl=target)' : 'motor (ctrl=torque, external PD)'}`
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
        muscleMode: 'sigmoid' as MuscleMode,
        muscleCtrlLo: EMPTY_CTRL_RANGE,
        muscleCtrlHi: EMPTY_CTRL_RANGE,
        clipLo,
        clipHi,
        processed: new Float32Array(n),
        active: false,
        ...substepOptions(actionTerm, subsetJointNames, termKey, this.termRng),
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
      const legacy: ActionConfigEntry = {
        type: controlType,
        scale: config.action_scale,
        stiffness: config.stiffness,
        damping: config.damping,
      };
      return new ActionManager([
        { name: 'legacy', term: buildEntry('legacy', controlType, mapping, legacy, true) },
      ]);
    }

    // ── Multi-term path: iterate every entry in the `actions` block ─────────
    const results: ManagedActionTerm[] = [];

    for (const [termKey, actionTerm] of Object.entries(actionsConfig)) {
      const controlType = actionTerm.type ?? 'joint_position';
      if (!SUPPORTED_CONTROL_TYPES.has(controlType)) {
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
        const actionScale = this.normalizeControlArray(actionTerm.scale, n, 1.0);
        const actionOffset = this.normalizeControlArray(actionTerm.offset, n, 0.0);
        const muscleMode = readMuscleMode(actionTerm as Record<string, unknown>);
        // `direct` writes the raw action, so it needs each actuator's own range.
        const ctrlRange = this.mjModel?.actuator_ctrlrange;
        const ctrlLoRange = new Float32Array(n);
        const ctrlHiRange = new Float32Array(n);
        for (let i = 0; i < n; i++) {
          const adr = muscleMapping.ctrlAdr[i];
          ctrlLoRange[i] = ctrlRange?.[2 * adr] ?? -1;
          ctrlHiRange[i] = ctrlRange?.[2 * adr + 1] ?? 1;
        }
        console.log(
          `[PolicyRunner] Action term "${termKey}" (muscle_activation): ${n} actuator(s), mode=${muscleMode}`
        );
        results.push({
          name: termKey,
          term: {
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
            muscleMode,
            muscleCtrlLo: ctrlLoRange,
            muscleCtrlHi: ctrlHiRange,
            ...resolveActionClip(
              actionTerm.clip,
              muscleMapping.ctrlAdr.map((_, i) => patterns[i] ?? ''),
              n
            ),
            processed: new Float32Array(n),
            active: false,
            rampFrom: null,
            holdS: null,
            holding: false,
            delays: [],
          },
        });
        continue;
      }

      // If actuator_names is absent or [".*"], match all joints (backward-compatible).
      const isMatchAll = patterns.length === 1 && patterns[0] === '.*';

      let mapping: Parameters<typeof buildEntry>[2] | null;

      if (controlType === 'joint_position_command') {
        mapping = stateBuilder.getModelControlMappingFor(patterns);
        const commandName = String(actionTerm.command_name);
        const width = this.commandManager.getCommand(commandName).length;
        if (mapping && width !== mapping.qposAdr.length) {
          console.error(
            `[PolicyRunner] Action term "${termKey}": command "${commandName}" is ${width} ` +
            `wide but the term drives ${mapping.qposAdr.length} joint(s), skipping.`
          );
          continue;
        }
      } else if (isMatchAll) {
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
      }
      if (!mapping) {
        console.warn(`[PolicyRunner] Action term "${termKey}": no joints matched patterns [${patterns.join(', ')}], skipping.`);
        continue;
      }

      // A velocity term's default offset is a velocity, already folded into `offset`; a
      // reference term holds the default pose only until its clip loads.
      const useDefaultOffset = controlType === 'joint_velocity'
        ? false
        : controlType === 'joint_position_reference'
          ? true
          : actionTerm.use_default_offset !== undefined
            ? actionTerm.use_default_offset
            : controlType === 'joint_position';

      const entry = buildEntry(termKey, controlType, mapping, actionTerm, useDefaultOffset);
      if (controlType === 'joint_position_reference' || controlType === 'joint_position_command') {
        entry.commandName = String(actionTerm.command_name ?? 'motion');
      }
      const traced = actionTerm.process ? this.buildTracedProcess(termKey, actionTerm) : null;
      if (actionTerm.process && !traced) continue;
      results.push({ name: termKey, term: entry, traced });
    }

    if (results.length === 0) {
      console.warn('[PolicyRunner] No valid action terms found in config.actions.');
      return null;
    }
    return new ActionManager(results);
  }

  /** `isTargetActuator` against the loaded model, with this MuJoCo build's bias values. */
  private targetActuatorTest(): (adr: number) => boolean {
    const bias = {
      affine: this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1,
      dcMotor: this.mujoco.mjtBias?.mjBIAS_DCMOTOR?.value ?? 3,
    };
    return (adr) => (this.mjModel ? isTargetActuator(this.mjModel, adr, bias) : false);
  }

  /** A traced term's process (and reset) graph, or `null` with an error when one is missing. */
  private buildTracedProcess(termKey: string, actionTerm: ActionConfigEntry): OnnxActionProcess | null {
    const process = actionTerm.process as OnnxActionProcessConfig;
    const reset = (actionTerm.reset as OnnxActionGraphConfig | undefined) ?? null;
    const processSession = this.policyGraphs.get(process.onnx);
    const resetSession = reset ? this.policyGraphs.get(reset.onnx) : null;
    if (!processSession || (reset && !resetSession)) {
      console.error(`[PolicyRunner] Action term "${termKey}": its traced graph did not load, skipping.`);
      return null;
    }
    return new OnnxActionProcess(termKey, process, reset, {
      process: processSession,
      reset: resetSession,
      rng: this.termRng,
      readSlot: this.readOnnxSlot,
    });
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
          console.warn(`[PolicyRunner] Joint name "${name}" is not one of the term's joints; skipping.`);
        }
      }
      return output;
    }
    return output;
  }

  /**
   * Put the sim back to its initial state, fire `mode="reset"` events, then reset the
   * observations, the action terms and the command terms, in mjlab's order (see
   * `resetChain`).
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
    if (this.modelHasHands) this.handMocap.park(this.mjData);
    this.interaction.onReset();
    // With the sim state, as mjlab does: a force from before the reset would otherwise
    // keep an `illegal_contact` term firing.
    this.contactSensors.reset();
    // mjlab's `scene.reset` resets the actuators' command delays with the entity.
    this.actionManager?.resetActuators();
    // Reset with the sim state, not the terms: it reads nothing from the scene.
    if (this.onnxModule) {
      this.onnxInputDict = this.onnxModule.initInput();
    }
    this.onnxTimeStep = 0;
    // Before the reset terms, not after as in mjlab: the loop may step while a UI reset's
    // terms run, and zeroing after them would drop that step. No reset term reads it: the
    // build refuses a traced action reset that does.
    this.episodeLength = 0;
    this.lastSimState.bodies.clear();

    await applyResetTerms({
      events: this.eventManager,
      policy: this.policyRunner,
      actions: this.actionManager,
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
    // Viewer-only: pointer modes and tracked hands, not part of the MDP.
    this.interaction.preStep();
    if (this.modelHasHands) this.handMocap.update(this.mjModel, this.mjData);

    stepPhysics(
      this.mujoco,
      this.mjModel,
      this.mjData,
      this.actionManager?.terms ?? [],
      this.decimation,
      undefined,
      // Per substep, not per control step, as mjlab rolls it from
      // `scene.update(dt=physics_dt)` inside its own decimation loop.
      this.contactSensors.size > 0 ? () => this.advanceContactSensors() : undefined,
    );
  }

  /**
   * mjlab's `action_manager.process_action`: every term's targets for this control step,
   * from the action just stored, before the substeps apply them. `fresh` is false when
   * inference stored no new action this step.
   */
  private async processActions(fresh: boolean): Promise<void> {
    // Held across the await: `setPolicy` can swap the manager mid-step.
    const actions = this.actionManager;
    if (!actions) return;
    this.refreshActionCommands(actions);
    try {
      await actions.process(
        this.policyRunner?.getLastActions() ?? EMPTY_ACTIONS,
        { episodeLength: this.episodeLength, stepDt: this.timestep * this.decimation },
        { fresh },
      );
    } catch (error) {
      console.error('[ActionManager] processing the action failed:', error);
    }
  }

  /** Hands each command-reading term this step's value, so `processAction` needs no command manager. */
  private refreshActionCommands(actions: ActionManager): void {
    for (const term of actions.terms) {
      if (term.commandName === undefined) continue;
      const command = this.commandManager.getTerm(term.commandName);
      if (term.controlType === 'joint_position_command') {
        term.commandValue = command?.getCommand() ?? null;
      } else {
        term.referenceJointPos = command?.getStateField?.('tracked_joint_pos') ?? null;
      }
    }
  }

  private advanceContactSensors(): void {
    if (!this.mjModel || !this.mjData) return;
    const mjModel = this.mjModel;
    this.contactSensors.advance(mjModel, this.mjData, (sensor) =>
      sensorWindow(mjModel, sensor),
    );
  }

  /** Run the policy and store its action; true only when this step stored a new one. */
  private async runOnnxInference(obs: Record<string, Float32Array>): Promise<boolean> {
    const module = this.onnxModule;
    if (!module || !this.policyRunner || this.onnxInferencing) {
      return false;
    }

    this.onnxInferencing = true;
    try {
      if (!this.onnxInputDict) {
        this.onnxInputDict = module.initInput();
      }
      const input: Record<string, ort.Tensor> = { ...this.onnxInputDict };
      if (module.inKeys.includes('time_step')) {
        input.time_step = new ort.Tensor('float32', new Float32Array([this.onnxTimeStep]), [1, 1]);
      }
      for (const [key, value] of Object.entries(obs)) {
        input[key] = new ort.Tensor('float32', value, [1, value.length]);
      }
      for (const key of module.inKeys) {
        if (!input[key]) {
          console.warn('[PolicyRunner] Missing ONNX input:', {
            key,
            available: Object.keys(input),
          });
          return false;
        }
      }

      const [result, carry] = await module.runInference(input);
      // `setPolicy` swapped the policy mid-run: this output is the old one's.
      if (this.onnxModule !== module) {
        return false;
      }
      if (Object.keys(carry).length > 0) {
        this.onnxInputDict = { ...this.onnxInputDict, ...carry };
      }
      if (module.inKeys.includes('time_step')) {
        this.onnxTimeStep += 1;
      }

      const outKey = module.outKeys[0];
      const actionTensor = result.action ?? (outKey ? result[outKey] : null) ?? result.policy ?? null;
      if (!actionTensor) {
        return false;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      const expectedActionCount = this.policyRunner?.getNumActions() ?? 0;
      if (this.actionManager && action.length !== expectedActionCount) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: expectedActionCount,
          got: action.length,
        });
        return false;
      }
      // Before `setLastActions`, which is what both the action terms and the
      // `action` observation slot read, mirroring rsl-rl, where the clamp lands
      // ahead of `env.step` and so ahead of the action manager recording the action.
      clampActions(action, this.clipActions);
      this.policyRunner.setLastActions(action);
      // So the same error, recurring after a recovery, is logged again.
      this.lastInferenceError = null;
      return true;
    } catch (error) {
      // An error, not a warning: until it clears, the policy acts no more.
      const message = String(error);
      if (message !== this.lastInferenceError) {
        this.lastInferenceError = message;
        console.error('[PolicyRunner] ONNX inference failed; the last action holds:', error);
      }
      return false;
    } finally {
      this.onnxInferencing = false;
    }
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
    this.cameraState = applyViewerConfig(
      config,
      this.camera,
      this.controls,
      this.mjModel,
      this.mjData,
      this.injectedBodyIds,
    );
  }

  /** Hand bones and the pointer's grab anchor, once the model is built. */
  private collectInjectedBodyIds(): Set<number> {
    const ids = new Set<number>(this.handMocap.bodyIds());
    if (this.mjModel && this.grabInjected) {
      const anchor = this.mujoco.mj_name2id(
        this.mjModel,
        this.mujoco.mjtObj.mjOBJ_BODY.value,
        POINTER_ANCHOR_BODY,
      );
      if (anchor > 0) ids.add(anchor);
    }
    return ids;
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
    for (const bodyId of this.handMocap.bodyIds()) {
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

  /** Kept as an offset: the orbit target goes on tracking a moving body through a session. */
  private onXrSessionStart = (): void => {
    // A gesture held into the session would keep pulling with no cursor to show it.
    this.interaction.cancel();
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
    this.debugVizSet.update();

    // In a session the head pose owns the camera, and OrbitControls stays out of it, or its
    // next update rebuilds the orbit from the head.
    const presenting = this.renderer.xr.isPresenting;
    if (this.mjData) {
      if (this.cameraTracking) {
        updateCameraFromData(this.mjData, this.camera, this.controls, this.cameraState, presenting);
      }
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
      // From the poses just drawn, so the avatar is never a frame behind the robot.
      this.avatar?.update(this.bodies, performance.now());

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
   * `preserveDrawingBuffer`: the WebGL drawing buffer is still intact before the
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
    navigator.xr?.removeEventListener('devicechange', this.onXrDeviceChange);
    // The headset would otherwise stay on a scene nothing renders any more.
    void this.xrSession?.end().catch(() => {});
    // Await stop so the loop halts before we free the state it reads.
    await this.stop();
    this.policyRunner = null;
    this.policyStateBuilder = null;

    // Not cache-managed, so they leak across navigations unless freed here.
    const module = this.onnxModule;
    this.onnxModule = null;
    await module?.dispose();
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
    this.avatar?.dispose();
    this.avatar = null;

    this.interaction.dispose();

    this.releaseModel();

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

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.dynamicBodyIds = null;
    this.lastSimState.bodies.clear();
    this.debugVizSet.clear();
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
