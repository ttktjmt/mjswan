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
import type {
  CameraControls,
  CommandControls,
  CommandDescriptor,
  CreateEngineOptions,
  DebugVisControls,
  EventControls,
  MjswanEngine,
  MjswanEngineState,
  PolicyInput,
  SceneInput,
  SplatInput,
  SplatTransform,
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

class Engine implements MjswanEngine {
  private readonly runtime: mjswanRuntime;
  private phase: 'running' | 'paused' = 'paused';
  private loading = false;
  private loadingMessage: string | null = null;
  private error: Error | null = null;
  private state: MjswanEngineState;
  private readonly listeners = new Set<(state: MjswanEngineState) => void>();

  readonly camera: CameraControls;
  readonly commands: CommandControls;
  readonly debugVis: DebugVisControls;
  readonly events: EventControls;

  constructor(runtime: mjswanRuntime) {
    this.runtime = runtime;
    this.state = this.buildState();
    // The CommandManager outlives individual loads, so one listener covers every change.
    this.runtime.commands.addEventListener(this.onCommandEvent);

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
      termSeed: this.runtime.seed,
    };
  }

  private refresh(): void {
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
    this.loading = true;
    this.loadingMessage = 'Loading scene…';
    this.error = null;
    this.refresh();
    try {
      const scene: ResolvedScene = {
        model: await resolveBytes(input.model),
        policy: input.policy ? await resolvePolicy(input.policy) : null,
        splat: input.splat ? await resolveSplat(input.splat) : null,
        viewer: input.viewer ?? null,
        terrainData: input.terrainData ?? null,
        controlDt: input.controlDt ?? null,
        plugins: input.plugins,
      };
      await this.runtime.loadEnvironment(scene);
      this.phase = this.runtime.isRunning ? 'running' : 'paused';
    } catch (err) {
      this.error = err instanceof Error ? err : new Error(String(err));
      throw err;
    } finally {
      this.loading = false;
      this.loadingMessage = null;
      this.refresh();
    }
  }

  async setPolicy(input: PolicyInput | null): Promise<void> {
    // Records and rethrows like `loadScene`. `refresh()` runs either way, so a rejected
    // `setPolicy` still leaves the snapshot describing what is loaded — on failure, nothing.
    try {
      await this.runtime.loadPolicyConfig(input ? await resolvePolicy(input) : null);
    } catch (err) {
      this.error = err instanceof Error ? err : new Error(String(err));
      throw err;
    } finally {
      this.refresh();
    }
  }

  async setSplat(input: SplatInput | null): Promise<void> {
    await this.runtime.setSplat(input ? await resolveSplat(input) : null);
  }

  setMotion(name: string | null): Promise<boolean> {
    return this.runtime.setSelectedMotion(name);
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
    this.runtime.commands.removeEventListener(this.onCommandEvent);
    this.listeners.clear();
    void this.runtime.dispose();
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
  return new Engine(new mjswanRuntime(mujoco, element, options.termSeed, options.handTracking));
}
