import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { TENDON_RGB, type TendonPaint } from '../scene/tendons';
import {
  bellyLengths,
  branchCount,
  MUSCLE_UNIFORM,
  muscleActuators,
  muscleColour,
  muscleShapes,
  muscleTube,
  type MuscleShape,
  tubeIndices,
  tubeVertices,
} from './tubes';

/** How muscles are drawn, in the terms of MyoSuite's `RenderConfig` (ADR 0012). */
export interface MuscleView {
  /** MyoSuite's `muscles`: a volume-preserving tube along each tendon path, or the path coloured. */
  style: 'volumetric' | 'paths';
  /** MyoSuite's `muscle_color`: by activation, or one anatomical red. */
  color: 'activation' | 'uniform';
  /** MyoSuite's `muscle_scale`: multiplies every tube's radius. */
  scale: number;
  /** Points per tube ring: `full` 12, as MyoSuite; `light` 6, for half the vertices. */
  detail: 'full' | 'light';
}

export const DEFAULT_MUSCLE_VIEW: MuscleView = Object.freeze({ style: 'volumetric', color: 'activation', scale: 1, detail: 'full' });

const STYLES: readonly unknown[] = ['volumetric', 'paths'];
const COLORS: readonly unknown[] = ['activation', 'uniform'];
const DETAILS: readonly unknown[] = ['full', 'light'];

/** `view` over `base`, frozen as the state hands it out; a value `view` gets wrong keeps `base`'s, with a warning. */
export function mergeMuscleView(base: MuscleView, view: Partial<MuscleView>): MuscleView {
  const pick = <K extends keyof MuscleView>(key: K, valid: (value: unknown) => boolean): MuscleView[K] => {
    const value = view[key];
    if (value === undefined) return base[key];
    if (valid(value)) return value as MuscleView[K];
    console.warn(`[muscles] ignored ${key}: ${String(value)}`);
    return base[key];
  };
  return Object.freeze({
    style: pick('style', (v) => STYLES.includes(v)),
    color: pick('color', (v) => COLORS.includes(v)),
    scale: pick('scale', (v) => typeof v === 'number' && Number.isFinite(v) && v > 0),
    detail: pick('detail', (v) => DETAILS.includes(v)),
  });
}

/** Every muscle's tubes as one mesh, for one ring size; muscle `i` owns vertices `starts[i]` to `starts[i + 1]`. */
interface TubeMesh {
  k: number;
  group: THREE.Group;
  mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhongMaterial>;
  positions: THREE.BufferAttribute;
  normals: THREE.BufferAttribute;
  colors: THREE.BufferAttribute;
  starts: Int32Array;
}

/** A model's muscles, drawn as the view says; built with the model and disposed with it. */
export class Muscles {
  /** Read by the tendon drawing: each muscle's tendon in the view's colour, or hidden under its tube. */
  readonly paint: TendonPaint;
  private readonly actuators: Int32Array;
  private readonly actadr: Int32Array;
  private readonly tendons: Int32Array;
  private readonly branches: Int32Array;
  private view: MuscleView | null = null;
  private shapes: MuscleShape[] | null = null;
  private tubes: TubeMesh | null = null;
  private readonly activations: Float64Array;
  // The last forward's arrays, copied with the bodies' poses: the loop forwards far more
  // often than frames are drawn, so the tubes are built from these once per frame.
  private wrapXpos = new Float64Array(0);
  private wrapObj = new Int32Array(0);
  private wrapadr = new Int32Array(0);
  private wrapnum = new Int32Array(0);
  private stale = false;
  private readonly colour = new Float32Array(3);

  /** Null for a model without a muscle MyoSuite would draw. */
  static create(mujoco: MainModule, model: MjModel, root: THREE.Object3D): Muscles | null {
    const actuators = muscleActuators(mujoco, model);
    return actuators.length ? new Muscles(mujoco, model, root, actuators) : null;
  }

  private constructor(
    private readonly mujoco: MainModule,
    private readonly model: MjModel,
    private readonly root: THREE.Object3D,
    actuators: number[],
  ) {
    const trnid = model.actuator_trnid;
    const actadr = model.actuator_actadr;
    this.actuators = Int32Array.from(actuators);
    this.tendons = Int32Array.from(actuators, (a) => trnid[2 * a]);
    this.actadr = Int32Array.from(actuators, (a) => actadr[a]);
    this.branches = this.tendons.map((t) => branchCount(mujoco, model, t));
    this.activations = new Float64Array(actuators.length);
    const rgb = new Float32Array(3 * model.ntendon);
    for (let t = 0; t < model.ntendon; t++) rgb.set(TENDON_RGB, 3 * t);
    this.paint = { rgb, hidden: new Uint8Array(model.ntendon) };
  }

  get count(): number {
    return this.actuators.length;
  }

  setView(view: MuscleView | null): void {
    this.view = view;
    const rgb = view?.style === 'paths' && view.color === 'uniform' ? MUSCLE_UNIFORM : TENDON_RGB;
    for (const t of this.tendons) {
      this.paint.hidden[t] = view?.style === 'volumetric' ? 1 : 0;
      this.paint.rgb.set(rgb, 3 * t);
    }
    if (view?.style !== 'volumetric') {
      if (this.tubes) this.tubes.group.visible = false;
      return;
    }
    const k = view.detail === 'light' ? 6 : 12;
    if (this.tubes?.k !== k) this.buildTubes(k);
    this.tubes!.group.visible = true;
    if (view.color === 'uniform') this.fillColours(() => MUSCLE_UNIFORM);
  }

  /** After each forward: the tendons' colours, and what the next frame's tubes are built from. */
  update(data: MjData): void {
    const view = this.view;
    if (!view) return;
    if (view.color === 'activation') {
      const act = data.act;
      const ctrl = data.ctrl;
      for (let i = 0; i < this.count; i++) {
        // Muscle gain without muscle dynamics keeps no activation: the control stands in.
        const adr = this.actadr[i];
        this.activations[i] = adr >= 0 ? act[adr] : ctrl[this.actuators[i]];
        if (view.style === 'paths') muscleColour(this.activations[i], true, this.paint.rgb, 3 * this.tendons[i]);
      }
    }
    if (view.style !== 'volumetric') return;
    this.wrapXpos = copy(data.wrap_xpos, this.wrapXpos);
    this.wrapObj = copy(data.wrap_obj, this.wrapObj);
    this.wrapadr = copy(data.ten_wrapadr, this.wrapadr);
    this.wrapnum = copy(data.ten_wrapnum, this.wrapnum);
    this.stale = true;
  }

  /** Before each render: the tubes, from the last forward `update` saw. */
  commit(): void {
    const view = this.view;
    if (!this.stale || !this.tubes || view?.style !== 'volumetric') return;
    this.stale = false;
    this.buildFrame(this.tubes, view);
  }

  dispose(): void {
    this.disposeTubes();
  }

  private buildFrame(tubes: TubeMesh, view: MuscleView): void {
    const shapes = (this.shapes ??= this.referenceShapes());
    const positions = tubes.positions.array as Float32Array;
    const normals = tubes.normals.array as Float32Array;
    for (let i = 0; i < this.count; i++) {
      const t = this.tendons[i];
      muscleTube(this.wrapXpos, this.wrapObj, this.wrapadr[t], this.wrapnum[t], this.branches[i], shapes[i], view.scale, tubes.k, positions, normals, tubes.starts[i]);
    }
    tubes.positions.needsUpdate = true;
    tubes.normals.needsUpdate = true;
    if (view.color !== 'activation') return;
    this.fillColours((i) => {
      muscleColour(this.activations[i], false, this.colour, 0);
      return this.colour;
    });
  }

  /** The muscles' bellies at keyframe 0, or `qpos0` without one, so every load draws them alike. */
  private referenceShapes(): MuscleShape[] {
    const mujoco = this.mujoco;
    const model = this.model;
    const data = new mujoco.MjData(model);
    try {
      if (model.nkey > 0) mujoco.mj_resetDataKeyframe(model, data, 0);
      mujoco.mj_forward(model, data);
      const lengths = bellyLengths(data.wrap_xpos, data.wrap_obj, data.ten_wrapadr, data.ten_wrapnum, this.tendons);
      return muscleShapes(model, Array.from(this.actuators), lengths);
    } finally {
      data.delete();
    }
  }

  private buildTubes(k: number): void {
    this.disposeTubes();
    const starts = new Int32Array(this.count + 1);
    for (let i = 0; i < this.count; i++) starts[i + 1] = starts[i] + this.branches[i] * tubeVertices(k);
    const vertices = starts[this.count];
    const geometry = new THREE.BufferGeometry();
    const positions = new THREE.BufferAttribute(new Float32Array(3 * vertices), 3).setUsage(THREE.DynamicDrawUsage);
    const normals = new THREE.BufferAttribute(new Float32Array(3 * vertices), 3).setUsage(THREE.DynamicDrawUsage);
    const colors = new THREE.BufferAttribute(new Uint8Array(3 * vertices), 3, true).setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute('position', positions);
    geometry.setAttribute('normal', normals);
    geometry.setAttribute('color', colors);
    geometry.setIndex(new THREE.BufferAttribute(tubeIndices(starts[this.count] / tubeVertices(k), k), 1));
    const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ vertexColors: true }));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    // Their bounds move every step; recomputing them would cost more than drawing.
    mesh.frustumCulled = false;
    // The tubes are in MuJoCo's frame (z up), which this turns into the scene's.
    const group = new THREE.Group();
    group.rotation.x = -Math.PI / 2;
    group.add(mesh);
    this.root.add(group);
    this.tubes = { k, group, mesh, positions, normals, colors, starts };
  }

  private disposeTubes(): void {
    if (!this.tubes) return;
    this.tubes.group.removeFromParent();
    this.tubes.mesh.geometry.dispose();
    this.tubes.mesh.material.dispose();
    this.tubes = null;
  }

  private fillColours(colourOf: (i: number) => ArrayLike<number>): void {
    const tubes = this.tubes!;
    const colors = tubes.colors.array as Uint8Array;
    for (let i = 0; i < this.count; i++) {
      const rgb = colourOf(i);
      const r = Math.round(rgb[0] * 255), g = Math.round(rgb[1] * 255), b = Math.round(rgb[2] * 255);
      for (let v = tubes.starts[i]; v < tubes.starts[i + 1]; v++) {
        colors[3 * v] = r;
        colors[3 * v + 1] = g;
        colors[3 * v + 2] = b;
      }
    }
    tubes.colors.needsUpdate = true;
  }
}

/** `source` into `target`, reallocated only when the length changes. */
function copy<T extends Float64Array | Int32Array>(source: ArrayLike<number>, target: T): T {
  if (target.length !== source.length) target = new (target.constructor as new (n: number) => T)(source.length);
  target.set(source);
  return target;
}
