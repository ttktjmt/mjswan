/**
 * A geom drawn over the scene and never simulated: what a host shows where a geom would
 * go before it adds one (ADR 0011), and how it edits one (ADR 0012). Parented to the
 * three.js scene, as the mode gizmos are, so a scene load keeps it, `frameCamera` never
 * frames it, and the pointer's raycast skips it.
 */
import * as THREE from 'three';

import { mjcToThreeCoordinate, threeToMjcCoordinate } from './coordinate';
import { primitiveGeometry, primitiveScale, type PrimitiveType } from './primitive';
import {
  PreviewHandles,
  axisOf,
  extent,
  resized,
  turned,
  type Axis,
  type Handle,
  type HandleKind,
  type PreviewGeom,
} from './previewHandles';
import { getQuaternion, reflectanceParams } from './scene';

export type { PreviewGeom, HandleKind };

/** One primitive geom, in MuJoCo's world axes and units. */
export interface PreviewInput {
  type: PrimitiveType;
  /** MuJoCo's `size` for the type. */
  size: ArrayLike<number>;
  pos: ArrayLike<number>;
  /** (w, x, y, z); the identity when absent. */
  quat?: ArrayLike<number>;
  /** 0 to 1 each; MuJoCo's geom grey when absent. */
  rgba?: ArrayLike<number>;
  /** Handles the pointer can drag: arrows that resize it, or rings that turn it. */
  handles?: HandleKind;
  /** A geom of the loaded model it stands in for, hidden while it shows. A scene load ends it. */
  replaces?: number;
}

export type PreviewListener = (geom: PreviewGeom, done: boolean) => void;

const GEOM_GREY = [0.5, 0.5, 0.5, 1];
const IDENTITY = [1, 0, 0, 0];
/** Snapped to this with Shift held, radians. */
const TURN_STEP = Math.PI / 12;

type Drag = { pointerId: number; handle: Handle; start: PreviewGeom } & (
  | { kind: 'size'; axis: Axis; index: number; base: THREE.Vector3; along: THREE.Vector3; from: number }
  /** `x`, `y`: the centre on screen; `swept`: the angle the pointer has gone round it. */
  | { kind: 'turn'; axis: Axis; x: number; y: number; last: number; swept: number; sign: number }
);

export class GeomPreview {
  private mesh: THREE.Mesh<THREE.BufferGeometry, THREE.MeshPhysicalMaterial> | null = null;
  /** The type and size its geometry was built for: moving it alone rebuilds nothing. */
  private shape = '';
  private geom: PreviewGeom | null = null;
  private handles: PreviewHandles | null = null;
  private replaced: { geom: number; meshes: THREE.Object3D[] } | null = null;
  private drag: Drag | null = null;
  private hovered: Handle | null = null;
  private readonly listeners = new Set<PreviewListener>();
  private readonly raycaster = new THREE.Raycaster();
  private readonly window: Window;

  constructor(
    private readonly scene: THREE.Scene,
    private readonly canvas: HTMLElement,
    private readonly camera: THREE.PerspectiveCamera
  ) {
    this.window = canvas.ownerDocument.defaultView as Window;
    // Capture on the window, ahead of the pointer modes and the camera, which listen lower down.
    this.window.addEventListener('pointerdown', this.onPointerDown, true);
    this.canvas.addEventListener('pointermove', this.onHover);
    this.canvas.addEventListener('pointerleave', this.onLeave);
  }

  /** `specularRatio` is the loaded scene's (`lightSpecularRatio`), so it lights as a geom there. */
  show(input: PreviewInput, specularRatio = 1): void {
    if (!this.mesh) {
      const material = new THREE.MeshPhysicalMaterial(reflectanceParams(null, -1, specularRatio));
      this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
      this.mesh.castShadow = true;
      this.mesh.userData.interactionGizmo = true;
      this.scene.add(this.mesh);
    }
    this.geom = {
      type: input.type,
      size: Array.from(input.size),
      pos: Array.from(input.pos),
      quat: Array.from(input.quat ?? IDENTITY),
      rgba: Array.from(input.rgba ?? GEOM_GREY),
    };
    this.draw();
    this.setHandles(input.handles);
    this.replace(input.replaces);
    this.update();
  }

  hide(): void {
    this.setHandles(undefined);
    this.replace(undefined);
    this.geom = null;
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh = null;
    this.shape = '';
  }

  /** Whether the preview or one of its handles is under a point of the canvas. */
  hit(clientX: number, clientY: number): boolean {
    if (!this.mesh) return false;
    if (this.handleAt(clientX, clientY)) return true;
    this.aim(clientX, clientY);
    this.mesh.updateMatrixWorld();
    return this.raycaster.intersectObject(this.mesh, false).length > 0;
  }

  onChange(listener: PreviewListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Once a frame, before the render: the handles keep their size on screen. */
  update(): void {
    if (!this.handles || !this.geom) return;
    const centre = mjcToThreeCoordinate(this.geom.pos);
    const eye = this.camera.getWorldPosition(new THREE.Vector3());
    const depth = centre.clone().sub(eye).dot(this.camera.getWorldDirection(new THREE.Vector3()));
    const height = this.canvas.clientHeight;
    this.handles.group.visible = depth > 0 && height > 0;
    if (!this.handles.group.visible) return;
    const worldPerPx = (2 * depth * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)) / height;
    this.handles.layout(this.geom, worldPerPx, eye);
  }

  /** The ids it stood in for belong to the model just replaced, whose meshes went with it. */
  onSceneLoaded(): void {
    this.replaced = null;
  }

  dispose(): void {
    this.hide();
    this.window.removeEventListener('pointerdown', this.onPointerDown, true);
    this.canvas.removeEventListener('pointermove', this.onHover);
    this.canvas.removeEventListener('pointerleave', this.onLeave);
    this.listeners.clear();
  }

  private draw(): void {
    const mesh = this.mesh;
    const geom = this.geom;
    if (!mesh || !geom) return;
    const shape = `${geom.type} ${geom.size.join(' ')}`;
    if (shape !== this.shape) {
      mesh.geometry.dispose();
      mesh.geometry = primitiveGeometry(geom.type, geom.size);
      this.shape = shape;
    }
    const [r, g, b, a] = geom.rgba;
    const material = mesh.material;
    material.color.setRGB(r, g, b);
    material.opacity = a;
    if (material.transparent !== a < 1) {
      material.transparent = a < 1;
      material.needsUpdate = true;
    }
    mesh.position.copy(mjcToThreeCoordinate(geom.pos));
    getQuaternion(Float32Array.from(geom.quat), 0, mesh.quaternion);
    primitiveScale(geom.type, geom.size, mesh.scale);
  }

  private setHandles(kind: HandleKind | undefined): void {
    const type = this.geom?.type;
    if (this.handles && this.handles.kind === kind && this.handles.type === type) return;
    this.endDrag();
    this.handles?.dispose();
    this.handles = null;
    this.setHovered(null);
    if (!kind || !type) return;
    this.handles = new PreviewHandles(kind, type);
    this.handles.highlight(null);
    this.scene.add(this.handles.group);
    this.update();
  }

  private replace(geom: number | undefined): void {
    if (this.replaced?.geom === geom) return;
    for (const mesh of this.replaced?.meshes ?? []) mesh.visible = true;
    this.replaced = null;
    if (geom === undefined) return;
    const meshes: THREE.Object3D[] = [];
    // Only what is on show: a floor hidden for AR passthrough stays hidden after.
    this.scene.traverse((object) => {
      if (object.userData.geomId === geom && object.visible && !object.userData.interactionGizmo) meshes.push(object);
    });
    for (const mesh of meshes) mesh.visible = false;
    this.replaced = { geom, meshes };
  }

  private aim(clientX: number, clientY: number): void {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1
    );
    this.raycaster.setFromCamera(ndc, this.camera);
  }

  /** Where the pointer's ray passes closest to the line through `base` along `along`, as a distance along it. */
  private along(base: THREE.Vector3, along: THREE.Vector3, clientX: number, clientY: number): number | null {
    this.aim(clientX, clientY);
    const origin = threeToMjcCoordinate(this.raycaster.ray.origin);
    const direction = threeToMjcCoordinate(this.raycaster.ray.direction);
    const b = along.dot(direction);
    const denominator = 1 - b * b;
    // Looking straight down the arrow, no point on it is closer than another.
    if (denominator < 1e-6) return null;
    const toBase = base.clone().sub(origin);
    return (b * direction.dot(toBase) - along.dot(toBase)) / denominator;
  }

  private screenCentre(geom: PreviewGeom): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = mjcToThreeCoordinate(geom.pos).project(this.camera);
    return { x: rect.left + ((ndc.x + 1) / 2) * rect.width, y: rect.top + ((1 - ndc.y) / 2) * rect.height };
  }

  private startDrag(handle: Handle, event: PointerEvent): Drag | null {
    const start = this.geom as PreviewGeom;
    const { pointerId } = event;
    if (handle.kind === 'size') {
      const along = axisOf(start.quat, handle.axis);
      const base = new THREE.Vector3(...start.pos).addScaledVector(along, extent(start.type, start.size, handle.axis));
      const from = this.along(base, along, event.clientX, event.clientY);
      if (from === null) return null;
      return { pointerId, handle, start, kind: 'size', axis: handle.axis, index: handle.index, base, along, from };
    }
    const { x, y } = this.screenCentre(start);
    const centre = mjcToThreeCoordinate(start.pos);
    const axis = mjcToThreeCoordinate(axisOf(start.quat, handle.axis).toArray());
    const towardEye = this.camera.getWorldPosition(new THREE.Vector3()).sub(centre).dot(axis) > 0;
    const last = Math.atan2(event.clientY - y, event.clientX - x);
    return { pointerId, handle, start, kind: 'turn', axis: handle.axis, x, y, last, swept: 0, sign: towardEye ? 1 : -1 };
  }

  private dragTo(drag: Drag, event: PointerEvent): PreviewGeom | null {
    if (drag.kind === 'size') {
      const to = this.along(drag.base, drag.along, event.clientX, event.clientY);
      return to === null ? null : resized(drag.start, drag.axis, drag.index, to - drag.from);
    }
    const angle = Math.atan2(event.clientY - drag.y, event.clientX - drag.x);
    // Unwrapped step by step, so a drag past half a turn keeps going.
    drag.swept += Math.atan2(Math.sin(angle - drag.last), Math.cos(angle - drag.last));
    drag.last = angle;
    // Screen y points down, so a sweep that looks anticlockwise lowers the angle: positive
    // about an axis pointing at the viewer, negative about one pointing away.
    let turn = -drag.swept * drag.sign;
    if (event.shiftKey) turn = Math.round(turn / TURN_STEP) * TURN_STEP;
    return turned(drag.start, drag.axis, turn);
  }

  private emit(done: boolean): void {
    const geom = this.geom;
    if (!geom) return;
    for (const listener of this.listeners) {
      listener({ ...geom, size: [...geom.size], pos: [...geom.pos], quat: [...geom.quat], rgba: [...geom.rgba] }, done);
    }
  }

  private setHovered(handle: Handle | null): void {
    const was = this.hovered;
    if (was?.kind === handle?.kind && was?.axis === handle?.axis) return;
    this.hovered = handle;
    this.handles?.highlight(handle);
    if (handle) this.canvas.style.cursor = 'grab';
    // Only its own: a cursor a pointer mode set stays.
    else if (was) this.canvas.style.cursor = '';
  }

  private endDrag(): void {
    if (!this.drag) return;
    this.drag = null;
    this.window.removeEventListener('pointermove', this.onDragMove, true);
    this.window.removeEventListener('pointerup', this.onDragEnd, true);
    this.window.removeEventListener('pointercancel', this.onDragEnd, true);
  }

  private handleAt(clientX: number, clientY: number): Handle | null {
    if (!this.handles?.group.visible) return null;
    this.aim(clientX, clientY);
    return this.handles.hit(this.raycaster);
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.target !== this.canvas || this.drag || !this.geom) return;
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    const handle = this.handleAt(event.clientX, event.clientY);
    const drag = handle && this.startDrag(handle, event);
    if (!drag) return;
    // The press is the handle's, through to its release: no mode and no orbit sees it.
    event.stopImmediatePropagation();
    event.preventDefault();
    this.drag = drag;
    this.setHovered(drag.handle);
    this.canvas.style.cursor = 'grabbing';
    this.window.addEventListener('pointermove', this.onDragMove, true);
    this.window.addEventListener('pointerup', this.onDragEnd, true);
    this.window.addEventListener('pointercancel', this.onDragEnd, true);
  };

  private onDragMove = (event: PointerEvent): void => {
    const drag = this.drag;
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.stopImmediatePropagation();
    const next = this.dragTo(drag, event);
    if (!next) return;
    this.geom = next;
    this.draw();
    this.update();
    this.emit(false);
  };

  private onDragEnd = (event: PointerEvent): void => {
    if (!this.drag || event.pointerId !== this.drag.pointerId) return;
    event.stopImmediatePropagation();
    this.endDrag();
    this.canvas.style.cursor = 'grab';
    this.emit(true);
  };

  private onHover = (event: PointerEvent): void => {
    if (!this.drag) this.setHovered(this.handleAt(event.clientX, event.clientY));
  };

  private onLeave = (): void => {
    if (!this.drag) this.setHovered(null);
  };
}
