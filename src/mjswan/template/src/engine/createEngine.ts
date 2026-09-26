/**
 * `createEngine` — the headless, instance-scoped mjswan engine (ADR 0004).
 *
 * Wraps the React-free {@link mjswanRuntime}: loads the MuJoCo WASM module up
 * front, resolves app-supplied {@link Bytes} to concrete buffers, drives the
 * switch verbs, and folds the runtime's command events into one immutable state
 * snapshot for `subscribe`. No React, no catalog, no config.json, no fetch.
 */
import { mjswanRuntime, type ResolvedPolicy, type ResolvedScene, type ResolvedSplat } from '../core/engine/runtime';
import { type Bytes, resolveBytes } from '../core/utils/bytes';
import type { CommandDefinition, CommandEventListener } from '../core/command';
import type { PolicyConfig } from '../core/policy/types';
import { INTERACTION_MODES, INTERACTION_MODE_IDS } from '../core/interaction/params';
import { createSerial } from './serial';
import type {
  CameraControls,
  CommandControls,
  CommandDescriptor,
  CreateEngineOptions,
  DebugVisControls,
  EventControls,
  InteractionControls,
  MjswanEngine,
  InteractionModeId,
  MjswanEngineState,
  PolicyInput,
  SceneInput,
  SplatInput,
  SplatTransform,
  XrControls,
  XrSessionId,
} from './types';

function toDescriptor(def: CommandDefinition): CommandDescriptor {
  const config = def.config;
  const base = { id: def.id, group: def.groupName, type: config.type, label: config.label };
  if (config.type === 'slider') {
    return {
      ...base,
      min: config.min,
      max: config.max,
      step: config.step,
      enabledWhen: config.enabled_when,
      adjustableRange: config.adjustable_range,
    };
  }
  return config.type === 'button' ? { ...base, icon: config.icon } : base;
}

/**
 * Resolve traced term graphs to bytes, in parallel. Eager unlike motions: a graph is
 * needed the moment its manager is constructed, and they are small.
 */
async function resolveGraphs(
  graphs: Record<string, Bytes> | undefined,
): Promise<Array<{ name: string; data: ArrayBuffer }>> {
  const entries = Object.entries(graphs ?? {});
  return Promise.all(
    entries.map(async ([name, bytes]) => ({ name, data: await resolveBytes(bytes) })),
  );
}

async function resolvePolicy(input: PolicyInput): Promise<ResolvedPolicy> {
  return {
    config: input.config as PolicyConfig,
    onnx: await resolveBytes(input.onnx),
    graphs: await resolveGraphs(input.graphs),
    // Motion bytes stay lazy — loaded on demand when a motion is selected.
    motions: (input.motions ?? []).map((m) => ({ name: m.name, data: m.data, default: m.default })),
    plugins: input.plugins,
  };
}

async function resolveSplat(input: SplatInput): Promise<ResolvedSplat> {
  return {
    data: await resolveBytes(input.data),
    collider: input.collider ? await resolveBytes(input.collider) : null,
    transform: input.transform,
  };
}

function asError(err: unknown): Error {
  return err instanceof Error ? err : new Error(String(err));
}

class Engine implements MjswanEngine {
  private readonly runtime: mjswanRuntime;
  private phase: 'running' | 'paused' = 'paused';
  private loading = false;
  private loadingMessage: string | null = null;
  /** Loads in flight: a scene load can wait behind an XR rebuild, and each clears only its own. */
  private loads = 0;
  private error: Error | null = null;
  /**
   * Scene and policy loads, motion switches and the XR rebuild each replace what the
   * others bind to the model, so they run one at a time, and `dispose` after them.
   */
  private readonly serial = createSerial();
  private disposed = false;
  private state: MjswanEngineState;
  private readonly listeners = new Set<(state: MjswanEngineState) => void>();

  readonly camera: CameraControls;
  readonly commands: CommandControls;
  readonly debugVis: DebugVisControls;
  readonly events: EventControls;
  readonly interaction: InteractionControls;
  readonly xr: XrControls;

  constructor(runtime: mjswanRuntime) {
    this.runtime = runtime;
    this.state = this.buildState();
    // The CommandManager outlives individual loads, so one listener covers every change.
    this.runtime.commands.addEventListener(this.onCommandEvent);
    // Device support resolves late, and sessions start and end outside any verb.
    this.runtime.onXrChange = () => this.refresh();

    this.camera = {
      set: (view) => this.runtime.setCameraView(view),
      get: () => this.runtime.getCameraView(),
      frame: () => this.runtime.frameCamera(),
    };
    this.commands = {
      set: (id, value) => this.runtime.commands.setValue(id, value),
      trigger: (id) => this.runtime.commands.triggerButton(id),
    };
    this.debugVis = {
      set: (term, enabled) => this.runtime.commands.setDebugVisEnabled(term, enabled),
    };
    this.events = {
      // Only the arm toggle changes the snapshot; the panel reads its checkbox from it.
      fire: (name) => {
        void this.runtime.fireEvent(name);
      },
      setArmed: (name, armed) => {
        this.runtime.setEventArmed(name, armed);
        this.refresh();
      },
    };
    this.interaction = {
      // The setters refresh the snapshot themselves: the runtime emits no event for them.
      setMode: (id) => {
        this.runtime.setInteractionMode(id);
        this.refresh();
      },
      getMode: () => this.runtime.getInteractionMode(),
      setParam: (mode, name, value) => {
        this.runtime.setInteractionParam(mode, name, value);
        this.refresh();
      },
      getParams: (mode) => this.runtime.getInteractionParams(mode),
      cancel: () => this.runtime.cancelInteraction(),
    };
    this.xr = {
      enter: (id) => this.enterXr(id),
      exit: () => this.runtime.exitXr(),
      setHandTracking: (enabled) => this.runtime.setHandTracking(enabled),
    };
  }

  private onCommandEvent: CommandEventListener = () => this.refresh();

  private buildState(): MjswanEngineState {
    const cm = this.runtime.commands;
    return {
      phase: this.phase,
      loading: this.loading,
      loadingMessage: this.loadingMessage,
      error: this.error,
      commands: cm.getCommands().map(toDescriptor),
      commandValues: cm.getValues(),
      debugVis: cm.getDebugVisTerms().map(({ name, enabled }) => ({ term: name, enabled })),
      events: this.runtime.eventControls(),
      interactions: this.runtime.interactionModes().map((mode) => ({
        ...mode,
        params: INTERACTION_MODES.find((spec) => spec.id === mode.id)?.params ?? [],
      })),
      interactionMode: this.runtime.getInteractionMode(),
      interactionParams: INTERACTION_MODE_IDS.reduce(
        (all, id) => ({ ...all, [id]: this.runtime.getInteractionParams(id) }),
        {} as Record<InteractionModeId, Readonly<Record<string, number>>>,
      ),
      xrSessions: this.runtime.xrSessions(),
      handTracking: this.runtime.handTrackingReport(),
      termSeed: this.runtime.seed,
    };
  }

  private beginLoad(message: string): void {
    this.loads += 1;
    this.loading = true;
    this.loadingMessage = message;
    this.refresh();
  }

  private endLoad(): void {
    this.loads -= 1;
    if (this.loads === 0) {
      this.loading = false;
      this.loadingMessage = null;
    }
    this.refresh();
  }

  /** Model work in turn; none starts once the engine is disposed. */
  private exclusive<T>(work: () => Promise<T>, whenDisposed: T): Promise<T> {
    return this.serial(() => (this.disposed ? Promise.resolve(whenDisposed) : work()));
  }

  private refresh(): void {
    // Nobody is listening, and the runtime may already be freed.
    if (this.disposed) return;
    this.state = this.buildState();
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.warn('[mjswan] subscribe listener error:', err);
      }
    }
  }

  async loadScene(input: SceneInput): Promise<void> {
    this.error = null;
    this.beginLoad('Loading scene…');
    try {
      // Resolved before taking a turn: a slow fetch must not hold up the others.
      const scene: ResolvedScene = {
        model: await resolveBytes(input.model),
        modelFormat: input.modelFormat,
        policy: input.policy ? await resolvePolicy(input.policy) : null,
        splat: input.splat ? await resolveSplat(input.splat) : null,
        viewer: input.viewer ?? null,
        terrainData: input.terrainData ?? null,
        controlDt: input.controlDt ?? null,
        plugins: input.plugins,
      };
      await this.exclusive(() => this.runtime.loadEnvironment(scene), undefined);
      this.phase = this.runtime.isRunning ? 'running' : 'paused';
    } catch (err) {
      this.error = asError(err);
      throw err;
    } finally {
      this.endLoad();
    }
  }

  /**
   * Asks for the session at once, inside the host's click, and waits its turn only for
   * the rebuild, so a load in flight delays entry instead of refusing it.
   */
  private async enterXr(id: XrSessionId): Promise<void> {
    if (this.disposed) return;
    const request = await this.runtime.requestXr(id);
    if (!request) return;
    let rebuilt = false;
    try {
      // Not `exclusive`: after `dispose` this still has to end the granted session.
      await this.serial(() =>
        this.runtime.startXr(request, (addingHands) => {
          rebuilt = true;
          this.beginLoad(addingHands ? 'Adding tracked hands…' : 'Removing tracked hands…');
        }),
      );
    } catch (err) {
      // A failed rebuild leaves no model behind, which the host must hear about as it
      // would a failed load.
      if (rebuilt) this.error = asError(err);
      throw err;
    } finally {
      if (rebuilt) {
        this.phase = this.runtime.isRunning ? 'running' : 'paused';
        this.endLoad();
      }
    }
  }

  async setPolicy(input: PolicyInput | null): Promise<void> {
    // Records and rethrows like `loadScene`. `refresh()` runs either way, so a rejected
    // `setPolicy` still leaves the snapshot describing what is loaded — on failure, nothing.
    try {
      const policy = input ? await resolvePolicy(input) : null;
      await this.exclusive(() => this.runtime.loadPolicyConfig(policy), undefined);
    } catch (err) {
      this.error = asError(err);
      throw err;
    } finally {
      this.refresh();
    }
  }

  async setSplat(input: SplatInput | null): Promise<void> {
    await this.runtime.setSplat(input ? await resolveSplat(input) : null);
  }

  setMotion(name: string | null): Promise<boolean> {
    return this.exclusive(() => this.runtime.setSelectedMotion(name), false);
  }

  setReferenceVisible(visible: boolean): void {
    this.runtime.setReferenceVisible(visible);
  }

  calibrateSplat(transform: SplatTransform): void {
    this.runtime.calibrateSplat(transform);
  }

  play(): void {
    this.runtime.play();
    this.phase = 'running';
    this.refresh();
  }

  pause(): void {
    this.runtime.pause();
    this.phase = 'paused';
    this.refresh();
  }

  reset(): void {
    this.runtime.resetSimulation();
  }

  getState(): MjswanEngineState {
    return this.state;
  }

  subscribe(listener: (state: MjswanEngineState) => void): () => void {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  captureThumbnail(options?: { maxDim?: number; quality?: number }): Promise<Blob> {
    return this.runtime.captureThumbnail(options);
  }

  dispose(): void {
    this.disposed = true;
    this.runtime.onXrChange = null;
    this.runtime.commands.removeEventListener(this.onCommandEvent);
    this.listeners.clear();
    // Behind the model work in flight, which would otherwise rebuild what dispose frees.
    void this.serial(() => this.runtime.dispose());
  }
}

/**
 * Prepare an engine (MuJoCo WASM + WebGL) in `element`, then `loadScene(...)`.
 * `multithreaded` lazily loads `mujoco/mt`, which needs COOP/COEP — the app's call.
 */
export async function createEngine(
  element: HTMLElement,
  options: CreateEngineOptions = {},
): Promise<MjswanEngine> {
  const mujocoModule = options.multithreaded ? await import('mujoco/mt') : await import('mujoco');
  const mujoco = await mujocoModule.default();
  return new Engine(
    new mjswanRuntime(mujoco, element, options.termSeed, options.handTracking),
  );
}
