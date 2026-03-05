import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import type { MainModule, MjData, MjModel } from 'mujoco';
import {
  downloadExampleScenesFolder,
  getPosition,
  getQuaternion,
  loadSceneFromURL,
} from '../scene/scene';
import { type SplatConfig, type SplatMesh, loadSplat, disposeSplat, applySplatTransform } from '../scene/splat';
import { loadCollider, disposeCollider } from '../scene/collider';
import { DragStateManager } from '../utils/dragStateManager';
import { createTendonState, updateTendonGeometry, updateTendonRendering } from '../scene/tendons';
import { updateHeadlightFromCamera, updateLightsFromData } from '../scene/lights';
import { threeToMjcCoordinate } from '../scene/coordinate';
import { SceneCacheManager } from '../cache/sceneCacheManager';
import { SceneResourceTracker } from '../cache/resourceTracker';
import { MemoryMonitor } from '../cache/memoryMonitor';
import { Observations } from '../observation/observations';
import * as ort from 'onnxruntime-web';
import { PolicyRunner } from '../policy/PolicyRunner';
import { OnnxModule } from '../policy/OnnxModule';
import { PolicyStateBuilder } from '../policy/PolicyStateBuilder';
import type { PolicyConfig } from '../policy/types';
import { TrackingPolicy } from '../policy/modules/TrackingPolicy';
import { LocomotionPolicy } from '../policy/modules/LocomotionPolicy';
import { getCommandManager, type CommandsConfig } from '../command';

type RuntimeOptions = {
  baseUrl?: string;
};

type BodyState = {
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
};

export class mjswanRuntime {
  private mujoco: MainModule;
  private container: HTMLElement;
  private baseUrl: string;
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
  private loopPromise: Promise<void> | null;
  private running: boolean;
  private timestep: number;
  private decimation: number;
  private loadingScene: Promise<void> | null;
  private resizeObserver: ResizeObserver | null;
  private dragStateManager: DragStateManager | null;
  private dragForceScale: number;
  private sceneCacheManager: SceneCacheManager;
  private resourceTracker: SceneResourceTracker;
  private memoryMonitor: MemoryMonitor;
  private policyRunner: PolicyRunner | null;
  private policyStateBuilder: PolicyStateBuilder | null;
  private policyConfigPath: string | null;
  private policyDebugCounter: number;
  private policyControl:
    | {
      controlType: string;
      ctrlAdr: number[];
      qposAdr: number[];
      qvelAdr: number[];
      actionScale: Float32Array;
      defaultJointPos: Float32Array;
      // Per-actuator flag: true = position actuator (ctrl=target_pos, PD internal),
      // false = motor actuator (ctrl=torque, PD computed in browser from kp/kd).
      positionActuator: boolean[];
      kp: Float32Array;
      kd: Float32Array;
    }
    | null;
  private onnxModule: OnnxModule | null;
  private onnxInputDict: Record<string, ort.Tensor> | null;
  private onnxInferencing: boolean;
  private vrButton: HTMLElement | null;
  private splatMesh: SplatMesh | null;
  private colliderMesh: THREE.Group | null;

  constructor(mujoco: MainModule, container: HTMLElement, options: RuntimeOptions = {}) {
    this.mujoco = mujoco;
    this.container = container;
    this.baseUrl = options.baseUrl || '/';

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
    this.scene.add(this.camera);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.xr.enabled = true;
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(width, height);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    this.vrButton = null;
    navigator.xr?.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        this.vrButton = VRButton.createButton(this.renderer);
        document.body.appendChild(this.vrButton);
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

    this.mjModel = null;
    this.mjData = null;
    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.loopPromise = null;
    this.running = false;
    this.timestep = 0.001;
    this.decimation = 1;
    this.loadingScene = null;
    this.dragStateManager = null;
    this.dragForceScale = 100.0;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;
    this.splatMesh = null;
    this.colliderMesh = null;

    // Initialize cache system (singleton shared across runtime instances)
    this.sceneCacheManager = SceneCacheManager.getInstance(this.mujoco);
    this.resourceTracker = new SceneResourceTracker();
    this.memoryMonitor = new MemoryMonitor();
  }

  async loadEnvironment(
    scenePath: string,
    policyConfigPath: string | null = null,
    splatConfig: SplatConfig | null = null
  ): Promise<void> {
    await this.stop();

    // Dispose previous splat/collider before switching scenes
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }

    const startTime = performance.now();

    // Initialize CommandManager with default velocity commands
    this.initializeCommands();

    // Check cache first
    if (this.sceneCacheManager.has(scenePath)) {
      await this.restoreFromCache(scenePath);
      const elapsed = performance.now() - startTime;
      this.memoryMonitor.logCacheOperation('hit', scenePath, { elapsedMs: elapsed });
    } else {
      this.memoryMonitor.logCacheOperation('miss', scenePath);

      // Prepare cache for new scene (may trigger eviction)
      await this.sceneCacheManager.prepareForNewScene();

      // Clear current references before loading new scene
      // This prevents loadSceneFromURL from deleting cached objects
      this.mjModel = null;
      this.mjData = null;
      this.bodies = null;
      this.lights = [];
      this.mujocoRoot = null;

      // Start tracking resources
      this.resourceTracker.startTracking(this.mujoco);

      // Normal load
      await downloadExampleScenesFolder(this.mujoco, scenePath, this.baseUrl);
      await this.loadScene(scenePath);

      // Capture and cache resources
      await this.captureAndCacheResources(scenePath);
    }

    // Load splat and optional collider
    if (splatConfig) {
      this.splatMesh = loadSplat(splatConfig, this.scene);
      if (splatConfig.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(splatConfig.colliderUrl),
          this.scene
        );
      }
    }

    await this.loadPolicyConfig(policyConfigPath);

    this.running = true;
    void this.startLoop();
  }

  /**
   * Initialize the CommandManager (clear and set up reset callback)
   * Commands are registered from policy config in loadPolicyConfig()
   */
  private initializeCommands(): void {
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());
  }

  /**
   * Initialize commands from policy config
   */
  private initializeCommandsFromConfig(commands: CommandsConfig): void {
    const commandManager = getCommandManager();
    commandManager.registerCommandsFromConfig(commands);
    console.log('[mjswanRuntime] Commands loaded from policy config:', Object.keys(commands));
  }

  /**
   * Public method to reset the simulation state
   * Can be called from UI components via the CommandManager
   */
  resetSimulation(): void {
    this.resetSimulationState();
    if (this.policyRunner && this.policyStateBuilder) {
      const state = this.policyStateBuilder.build();
      this.policyRunner.reset(state);
    }
    console.log('[mjswanRuntime] Simulation reset');
  }

  async loadScene(scenePath: string): Promise<void> {
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
        scenePath,
        parent
      );

      if (!this.mjModel || !this.mjData) {
        throw new Error('Failed to load MuJoCo model.');
      }

      this.mujocoRoot = this.scene.getObjectByName('MuJoCo Root') as THREE.Group | null;

      this.mujoco.mj_forward(this.mjModel, this.mjData);

      this.timestep = this.mjModel.opt.timestep || 0.001;
      this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

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
        });
      }

      this.loadingScene = null;
    })();

    await this.loadingScene;
  }

  async startLoop(): Promise<void> {
    if (this.loopPromise) {
      return this.loopPromise;
    }
    this.running = true;
    this.loopPromise = this.mainLoop();
    return this.loopPromise;
  }

  async setSplat(config: SplatConfig | null): Promise<void> {
    if (this.splatMesh) {
      disposeSplat(this.splatMesh, this.scene);
      this.splatMesh = null;
    }
    if (this.colliderMesh) {
      disposeCollider(this.colliderMesh, this.scene);
      this.colliderMesh = null;
    }
    if (config) {
      this.splatMesh = loadSplat(config, this.scene);
      if (config.colliderUrl) {
        this.colliderMesh = await loadCollider(
          this.resolveAssetUrl(config.colliderUrl),
          this.scene
        );
      }
    }
  }

  /** Update transform of the existing splat without disposing/reloading. */
  calibrateSplat(config: SplatConfig): void {
    if (this.splatMesh) {
      applySplatTransform(this.splatMesh, config);
    }
  }

  setSplatVisible(visible: boolean): void {
    if (this.splatMesh) {
      this.splatMesh.visible = visible;
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    const pending = this.loopPromise;
    if (pending) {
      await pending;
    }
    this.loopPromise = null;
  }

  private async mainLoop(): Promise<void> {
    while (this.running) {
      const loopStart = performance.now();

      if (this.mjModel && this.mjData) {
        if (this.policyRunner && this.policyStateBuilder) {
          const state = this.policyStateBuilder.build();
          const obs = this.policyRunner.collectObservationsByKey(state);
          await this.runOnnxInference(obs);
          if (this.policyDebugCounter % 60 === 0) {
            const debugKey =
              'policy' in obs
                ? 'policy'
                : 'observation' in obs
                  ? 'observation'
                  : Object.keys(obs)[0];
            const debugObs = debugKey ? obs[debugKey] : null;
            const preview = debugObs ? Array.from(debugObs.slice(0, 8)) : [];
            console.log('[PolicyRunner] obs', {
              key: debugKey,
              size: debugObs ? debugObs.length : 0,
              sample: preview,
            });
          }
          this.policyDebugCounter += 1;
        }
        this.executeSimulationSteps();
        this.updateCachedState();
      }

      const elapsed = (performance.now() - loopStart) / 1000;
      const target = this.timestep * this.decimation;
      const sleepTime = Math.max(0, target - elapsed);
      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime * 1000));
      }
    }
    this.loopPromise = null;
  }

  private async loadPolicyConfig(policyConfigPath: string | null): Promise<void> {
    const previousPolicyConfigPath = this.policyConfigPath;
    this.policyConfigPath = policyConfigPath;
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyDebugCounter = 0;
    this.policyControl = null;
    this.onnxModule = null;
    this.onnxInputDict = null;
    this.onnxInferencing = false;

    // Clear existing commands when switching policies
    const commandManager = getCommandManager();
    commandManager.clear();
    commandManager.setResetCallback(() => this.resetSimulation());

    if (!policyConfigPath) {
      return;
    }

    if (!this.mjModel || !this.mjData) {
      console.warn('Policy config loaded before MuJoCo model is ready.');
      return;
    }

    if (policyConfigPath !== previousPolicyConfigPath) {
      this.resetSimulationState();
    }

    try {
      const { config } = await this.fetchPolicyConfig(policyConfigPath);

      // Initialize commands from policy config if present
      if (config.commands && typeof config.commands === 'object') {
        this.initializeCommandsFromConfig(config.commands as CommandsConfig);
      }

      if (!config.policy_joint_names || config.policy_joint_names.length === 0) {
        throw new Error('Policy config missing policy_joint_names.');
      }

      const runner = new PolicyRunner(config, {
        policyModules: {
          tracking: TrackingPolicy,
          locomotion: LocomotionPolicy,
        },
        observations: Observations,
      });

      await runner.init({
        mujoco: this.mujoco,
        mjModel: this.mjModel,
        mjData: this.mjData,
      });

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

      if (config.onnx?.path) {
        const onnxPath = this.resolvePolicyAssetPath(policyConfigPath, config.onnx.path);
        const onnxUrl = this.resolveAssetUrl(onnxPath);
        const onnxConfig = { ...config.onnx, path: onnxUrl };
        const module = new OnnxModule(onnxConfig);
        await module.init();
        this.onnxModule = module;
        this.onnxInputDict = module.initInput();
      }

      console.log('[PolicyRunner] config loaded', {
        obsSize: runner.getObservationSize(),
        obsLayout: runner.getObservationLayout(),
        pdEnabled: this.policyControl !== null,
      });
    } catch (error) {
      console.warn('Failed to load policy config:', error);
    }
  }

  private async fetchPolicyConfig(
    policyConfigPath: string
  ): Promise<{ config: PolicyConfig; resolvedUrl: string }> {
    const resolved = this.resolveAssetUrl(policyConfigPath);
    const response = await fetch(resolved, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Failed to fetch policy config: ${response.status}`);
    }
    const payload = await response.json();
    return { config: payload as PolicyConfig, resolvedUrl: resolved };
  }

  private resolveAssetUrl(assetPath: string): string {
    if (/^[a-z]+:\/\//i.test(assetPath)) {
      return assetPath;
    }
    const base = (this.baseUrl || '/').replace(/\/+$/, '/');
    const baseUrl = new URL(base, window.location.origin + '/').toString();
    return new URL(assetPath.replace(/^\/+/, ''), baseUrl).toString();
  }

  private resolvePolicyAssetPath(configPath: string, assetPath: string): string {
    const normalizedConfig = configPath.replace(/\\/g, '/');
    const lastSlash = normalizedConfig.lastIndexOf('/');
    if (lastSlash >= 0) {
      const dir = normalizedConfig.slice(0, lastSlash + 1);
      return `${dir}${assetPath}`.replace(/\/+/g, '/');
    }
    return assetPath;
  }

  private buildPolicyControl(
    config: PolicyConfig,
    runner: PolicyRunner,
    stateBuilder: PolicyStateBuilder
  ):
    | {
      controlType: string;
      ctrlAdr: number[];
      qposAdr: number[];
      qvelAdr: number[];
      actionScale: Float32Array;
      defaultJointPos: Float32Array;
      positionActuator: boolean[];
      kp: Float32Array;
      kd: Float32Array;
    }
    | null {
    const controlType = config.control_type ?? 'joint_position';
    if (controlType !== 'joint_position' && controlType !== 'torque') {
      console.warn(`[PolicyRunner] Unsupported control_type: ${controlType}`);
      return null;
    }

    const mapping = stateBuilder.getControlMapping();
    if (!mapping) {
      console.warn('[PolicyRunner] Failed to build control mapping.');
      return null;
    }

    const numActions = mapping.qposAdr.length;
    const actionScale = this.normalizeControlArray(config.action_scale, numActions, 1.0);
    const defaultJointPos = runner.getDefaultJointPos();
    const kp = this.normalizeControlArray(config.stiffness, numActions, 0.0);
    const kd = this.normalizeControlArray(config.damping, numActions, 0.0);

    // Detect per-actuator whether the scene uses position actuators (biastype=affine,
    // ctrl=target_pos, PD handled internally by MuJoCo) or motor actuators
    // (biastype=none, ctrl=torque, PD must be computed externally from kp/kd).
    const affineBiasValue = this.mujoco.mjtBias?.mjBIAS_AFFINE?.value ?? 1;
    const positionActuator: boolean[] = mapping.ctrlAdr.map((adr) => {
      if (adr < 0 || !this.mjModel) return false;
      return this.mjModel.actuator_biastype[adr] === affineBiasValue;
    });

    const isPosition = positionActuator.some(Boolean);
    const isMotor = positionActuator.some((v) => !v);
    if (isPosition && isMotor) {
      console.warn('[PolicyRunner] Mixed actuator types detected; behavior may be incorrect.');
    }
    console.log(
      `[PolicyRunner] Actuator mode: ${isPosition ? 'position (ctrl=target_pos)' : 'motor (ctrl=torque, external PD)'}`
    );

    return {
      controlType,
      ...mapping,
      actionScale,
      defaultJointPos,
      positionActuator,
      kp,
      kd,
    };
  }

  private normalizeControlArray(
    values: number[] | number | undefined,
    length: number,
    fallback: number
  ): Float32Array {
    const output = new Float32Array(length);
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
    output.fill(fallback);
    return output;
  }

  private resetSimulationState(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    this.mujoco.mj_resetData(this.mjModel, this.mjData);
    this.mujoco.mj_forward(this.mjModel, this.mjData);
    this.lastSimState.bodies.clear();
    this.updateCachedState();
  }

  private executeSimulationSteps(): void {
    if (!this.mjModel || !this.mjData) {
      return;
    }
    // Apply drag forces
    this.applyDragForces();

    for (let substep = 0; substep < this.decimation; substep++) {
      this.applyPolicyControl();
      this.mujoco.mj_step(this.mjModel, this.mjData);
    }
  }

  private applyPolicyControl(): void {
    if (!this.policyControl || !this.mjData) {
      return;
    }

    const { controlType, ctrlAdr, qposAdr, qvelAdr, actionScale, defaultJointPos, positionActuator, kp, kd } =
      this.policyControl;
    const numActions = ctrlAdr.length;
    const actions = this.policyRunner?.getLastActions() ?? new Float32Array(numActions);
    const ctrl = this.mjData.ctrl;
    ctrl.fill(0.0);

    if (controlType === 'joint_position') {
      for (let i = 0; i < numActions; i++) {
        const target = defaultJointPos[i] + actionScale[i] * actions[i];
        const ctrlIndex = ctrlAdr[i];
        if (ctrlIndex < 0) continue;

        if (positionActuator[i]) {
          // Position actuator (biastype=affine): ctrl = target joint position.
          // MuJoCo computes force = kp*(ctrl - qpos) - kd*qvel internally.
          ctrl[ctrlIndex] = target;
        } else {
          // Motor actuator (biastype=none): ctrl = torque.
          // PD must be computed externally using kp/kd from the policy config.
          const qpos = this.mjData.qpos[qposAdr[i]];
          const qvel = this.mjData.qvel[qvelAdr[i]];
          ctrl[ctrlIndex] = kp[i] * (target - qpos) + kd[i] * (0 - qvel);
        }
      }
    } else if (controlType === 'torque') {
      for (let i = 0; i < numActions; i++) {
        const ctrlIndex = ctrlAdr[i];
        if (ctrlIndex >= 0) {
          ctrl[ctrlIndex] = actionScale[i] * actions[i];
        }
      }
    }
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

      const outKey = this.onnxModule.outKeys[0];
      const actionTensor = (outKey ? result[outKey] : null) ?? result.action ?? result.policy ?? null;
      if (!actionTensor) {
        return;
      }

      const raw = actionTensor.data as Float32Array | number[];
      const action = ArrayBuffer.isView(raw) ? new Float32Array(raw) : Float32Array.from(raw);
      if (this.policyControl && action.length !== this.policyControl.ctrlAdr.length) {
        console.warn('[PolicyRunner] Action size mismatch:', {
          expected: this.policyControl.ctrlAdr.length,
          got: action.length,
        });
        return;
      }
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

    // Set xfrc_applied
    // xfrc_applied: (nbody, 6) = [fx, fy, fz, tx, ty, tz] for each body
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
    for (let b = 0; b < this.mjModel.nbody; b++) {
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

  private render = (): void => {
    this.controls.update();

    if (this.mjModel && this.mjData && this.bodies) {
      updateHeadlightFromCamera(this.camera, this.lights);

      for (const [b, state] of this.lastSimState.bodies) {
        const body = this.bodies[b];
        if (body) {
          body.position.copy(state.position);
          body.quaternion.copy(state.quaternion);
          body.updateWorldMatrix(true, false);
        }
      }

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

  private onWindowResize = (): void => {
    const { width, height } = this.getSize();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  };

  dispose(): void {
    this.stop();
    this.policyRunner = null;
    this.policyStateBuilder = null;
    this.policyConfigPath = null;

    if (this.dragStateManager) {
      this.dragStateManager.dispose();
      this.dragStateManager = null;
    }

    // NOTE: Do NOT delete mjData/mjModel here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    this.mjData = null;
    this.mjModel = null;

    // NOTE: Do NOT dispose Three.js resources here as they may be cached
    // The cache manager will handle their disposal when evicting
    // Just clear references
    // this.disposeThreeJSResources();

    window.removeEventListener('resize', this.onWindowResize);
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;

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

    this.bodies = null;
    this.lights = [];
    this.mujocoRoot = null;
    this.lastSimState.bodies.clear();
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

  /**
   * Restore scene from cache
   */
  private async restoreFromCache(scenePath: string): Promise<void> {
    const resources = this.sceneCacheManager.get(scenePath);
    if (!resources) {
      throw new Error(`Scene ${scenePath} not found in cache`);
    }

    // Remove existing root if present
    const existingRoot = this.scene.getObjectByName('MuJoCo Root');
    if (existingRoot) {
      this.scene.remove(existingRoot);
    }

    // Restore MuJoCo objects
    this.mjModel = resources.mjModel;
    this.mjData = resources.mjData;

    // Restore Three.js resources
    this.bodies = resources.bodies;
    this.lights = resources.lights;
    this.mujocoRoot = resources.mujocoRoot;

    // Re-add root to scene
    this.scene.add(this.mujocoRoot);

    // Restore skybox background
    this.scene.background = resources.skybox;

    // Run forward dynamics
    this.mujoco.mj_forward(this.mjModel, this.mjData);

    // Update runtime parameters
    this.timestep = this.mjModel.opt.timestep || 0.001;
    this.decimation = Math.max(1, Math.round(0.02 / this.timestep));

    // Clear and update cached state
    this.lastSimState.bodies.clear();
    this.updateCachedState();

    // Initialize DragStateManager if needed
    if (!this.dragStateManager) {
      this.dragStateManager = new DragStateManager({
        scene: this.scene,
        renderer: this.renderer,
        camera: this.camera,
        container: this.container,
        controls: this.controls,
      });
    }
  }

  /**
   * Capture resources and add to cache
   */
  private async captureAndCacheResources(scenePath: string): Promise<void> {
    // Stop tracking and get FS files
    const fsFiles = this.resourceTracker.stopTracking(this.mujoco);

    if (!this.mjModel || !this.mjData || !this.bodies || !this.mujocoRoot) {
      console.warn('[SceneCache] Cannot cache scene: missing resources');
      return;
    }

    // Estimate memory usage
    const estimatedMemoryBytes = this.resourceTracker.estimateSceneMemory({
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      meshes: {}, // Meshes are part of the scene
      mujocoRoot: this.mujocoRoot,
    });

    // Create cache entry
    await this.sceneCacheManager.set(scenePath, {
      scenePath,
      lastAccessed: Date.now(),
      loadedAt: Date.now(),
      mjModel: this.mjModel,
      mjData: this.mjData,
      bodies: this.bodies,
      lights: this.lights,
      meshes: {},
      mujocoRoot: this.mujocoRoot,
      skybox: this.scene.background instanceof THREE.CubeTexture ? this.scene.background : null,
      fsFiles,
      estimatedMemoryBytes,
    });

    // Log the cache operation
    const metrics = this.sceneCacheManager.getMetrics();
    this.memoryMonitor.logCacheOperation('load', scenePath, {
      memoryMB: estimatedMemoryBytes / 1048576,
      totalScenes: metrics.totalScenes,
      totalMemoryMB: metrics.totalMemoryBytes / 1048576,
    });
  }
}
