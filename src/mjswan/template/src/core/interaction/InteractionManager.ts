/**
 * Pointer interaction for the viewer: which mode the pointer drives, what each mode's
 * numbers are, and the one hook the step loop calls.
 *
 * This sits beside `xr/`, not inside the MDP layer: nothing here is part of a task's
 * definition, and nothing here reaches Python, the manifest or the `.swn` document. The
 * XR hand keeps its own driver (contact-driven pinching is a different gesture from a
 * raycast) and shares only the machinery under it.
 */
import * as THREE from 'three';
import type { MjvPerturb } from 'mujoco';

import { DragArrow, PushRing } from './gizmos';
import {
  DEFAULT_INTERACTION_MODE,
  INTERACTION_MODES,
  clampParam,
  defaultParams,
  modeSpec,
  type InteractionModeId,
} from './params';
import { PointerTracker, type PointerClaim, type PointerGesture } from './pointer';
import { InteractionWrench } from './wrench';
import type { InteractionMode, InteractionSim, ModeContext } from './modes/mode';
import { PullMode } from './modes/pull';
import { PushMode } from './modes/push';
import { ViewMode } from './modes/view';
import { WeldMode } from './modes/weld';
import type { WeldHold } from '../grab/weldHold';

export type { InteractionSim } from './modes/mode';

/** One mode as the app sees it. */
export interface InteractionModeReport {
  id: InteractionModeId;
  label: string;
  /** False when this scene cannot run it; `reason` says why. */
  available: boolean;
  reason?: string;
}

export interface InteractionManagerOptions {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.Camera;
  container: HTMLElement;
  controls: { enabled: boolean };
  sim: () => InteractionSim;
  /** Shared with the XR hand, so one body is never held by two constraints. */
  weld: WeldHold;
}

export class InteractionManager {
  private readonly pointer: PointerTracker;
  private readonly arrow: DragArrow;
  private readonly ring: PushRing;
  private readonly wrench = new InteractionWrench();
  private readonly readSim: () => InteractionSim;
  private readonly weld: WeldHold;
  private readonly canvas: HTMLElement;
  private readonly modes = new Map<InteractionModeId, InteractionMode>();
  private readonly values = defaultParams();
  private active: InteractionModeId = DEFAULT_INTERACTION_MODE;
  private perturbHandle: MjvPerturb | null = null;

  constructor(options: InteractionManagerOptions) {
    this.readSim = options.sim;
    this.weld = options.weld;
    this.canvas = options.renderer.domElement;
    this.arrow = new DragArrow(options.scene);
    this.ring = new PushRing(options.scene);
    for (const mode of [new ViewMode(), new PullMode(), new PushMode(), new WeldMode()]) {
      this.modes.set(mode.id, mode);
    }

    this.pointer = new PointerTracker({
      scene: options.scene,
      renderer: options.renderer,
      camera: options.camera,
      container: options.container,
      controls: options.controls,
    });
    this.pointer.setHandlers({
      onDown: (gesture) => this.onDown(gesture),
      onMove: (gesture) => this.current()?.onMove?.(gesture, this.context()),
      onUp: (gesture) => this.current()?.onUp?.(gesture, this.context()),
      onCancel: () => this.current()?.onCancel(this.context()),
    });
    this.refreshPickable();
  }

  getMode(): InteractionModeId {
    return this.active;
  }

  setMode(id: InteractionModeId): void {
    if (!this.modes.has(id) || id === this.active) return;
    this.cancel();
    this.active = id;
    this.refreshPickable();
  }

  getParams(id: InteractionModeId): Readonly<Record<string, number>> {
    return this.values[id] ?? {};
  }

  /** Out-of-range is clamped rather than refused, as the command sliders are. */
  setParam(id: InteractionModeId, name: string, value: number): void {
    const spec = modeSpec(id).params.find((p) => p.name === name);
    if (!spec) {
      console.warn(`[interaction] ${id} has no parameter "${name}"`);
      return;
    }
    this.values[id][name] = clampParam(spec, value);
  }

  /** Every mode, with whether the loaded scene can run it. */
  report(): InteractionModeReport[] {
    return INTERACTION_MODES.map((spec) => {
      const reason = this.modes.get(spec.id)?.unavailable?.(this.context()) ?? null;
      const known = this.modes.has(spec.id);
      return {
        id: spec.id,
        label: spec.label,
        available: known && reason === null,
        reason: known ? (reason ?? undefined) : 'Not implemented yet.',
      };
    });
  }

  /** Drop any held gesture: pausing, switching scene, entering XR, tearing down. */
  cancel(): void {
    this.pointer.cancel();
    this.current()?.onCancel(this.context());
    this.wrench.clear(this.readSim().mjData);
    this.arrow.hide();
  }

  /**
   * The sim went back to its initial state. Welds are dropped here rather than left to
   * `mj_resetData`, which restores `eq_active` but not the model-side retargeting.
   */
  onReset(): void {
    const { mjModel, mjData } = this.readSim();
    this.current()?.onCancel(this.context());
    if (mjModel) this.weld.restore(mjModel, mjData);
    this.wrench.clear(mjData);
    this.pointer.release();
    this.arrow.hide();
  }

  /**
   * A new model: body ids from the old one mean nothing now.
   *
   * The mode is cancelled rather than just released, because a gesture can outlive the
   * scene it started in: a queued shove fired on the new scene's first step would land
   * `impulse / controlDt` newtons on whatever body inherited that id.
   */
  onSceneLoaded(): void {
    this.pointer.release();
    this.current()?.onCancel(this.context());
    this.wrench.forget();
    this.arrow.hide();
    this.refreshPickable();
  }

  /**
   * One control step's worth of interaction, run before the physics substeps: the slot
   * the mouse drag has always occupied.
   */
  preStep(): void {
    const sim = this.readSim();
    if (!sim.mjData) return;
    this.wrench.begin(sim.mjData);
    this.current()?.preStep(this.pointer.current(), this.context());
  }

  dispose(): void {
    this.canvas.style.cursor = '';
    this.pointer.dispose();
    this.arrow.dispose();
    this.ring.dispose();
    // An embind handle, not a view: it has to be released or the WASM heap grows.
    this.perturbHandle?.delete();
    this.perturbHandle = null;
  }

  private onDown(gesture: PointerGesture): PointerClaim {
    const mode = this.current();
    if (!mode) return 'none';
    const context = this.context();
    // A mode the scene cannot run stays selected (the panel shows it greyed with its
    // reason), but it must not keep taking presses, or the camera stops orbiting in a
    // scene where the mode does nothing at all.
    if (mode.unavailable?.(context)) return 'none';
    return mode.onDown(gesture, context);
  }

  private current(): InteractionMode | undefined {
    return this.modes.get(this.active);
  }

  /** Every mode acts on a body, so a press that misses one is the camera's. */
  private refreshPickable(): void {
    this.pointer.setActableBodyIds(this.readSim().dynamicBodyIds);
  }

  private context(): ModeContext {
    return {
      sim: this.readSim,
      params: () => this.values[this.active],
      wrench: this.wrench,
      perturb: () => this.getPerturb(),
      arrow: this.arrow,
      ring: this.ring,
      weld: this.weld,
      setCursor: (cursor) => {
        this.canvas.style.cursor = cursor;
      },
    };
  }

  private getPerturb(): MjvPerturb {
    if (!this.perturbHandle) {
      this.perturbHandle = new (this.readSim().mujoco as unknown as {
        MjvPerturb: new () => MjvPerturb;
      }).MjvPerturb();
    }
    return this.perturbHandle;
  }
}
