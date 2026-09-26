/**
 * The one place pointer events reach the simulation.
 *
 * - A press on a body a mode can act on belongs to the active mode; any other press
 *   belongs to the camera. No modifier key, no hover.
 * - Only the first pointer is tracked. Later touches stay with `OrbitControls`, so
 *   pinch-zoom and two-finger orbit keep working mid-gesture.
 */
import * as THREE from 'three';

/** Where a press landed. Positions are three.js world axes. */
export interface PointerHit {
  /** The MuJoCo body under the pointer. */
  bodyId: number;
  object: THREE.Object3D;
  point: THREE.Vector3;
  /** Outward surface normal, turned to face the camera; null when the hit carried none. */
  normal: THREE.Vector3 | null;
  distance: number;
}

/** The live state of a claimed gesture. */
export interface PointerGesture {
  hit: PointerHit;
  /** The point on the current ray at the press's depth: where the pointer "is" in 3D. */
  ray: THREE.Vector3;
  /** Unit direction of the current ray: what a mode falls back to with no surface normal. */
  direction: THREE.Vector3;
  /** Farthest the pointer has moved from the press, CSS px: what tells a tap from a drag. */
  travel: number;
}

/**
 * Who owns a press. `shared` reports it to the mode but leaves the camera enabled, for a
 * tap: `OrbitControls` cannot pick up a drag whose start it missed.
 */
export type PointerClaim = 'none' | 'exclusive' | 'shared';

export interface PointerHandlers {
  onDown(gesture: PointerGesture): PointerClaim;
  onMove(gesture: PointerGesture): void;
  onUp(gesture: PointerGesture): void;
  onCancel(): void;
}

export interface PointerTrackerOptions {
  scene: THREE.Scene;
  renderer: THREE.WebGLRenderer;
  camera: THREE.Camera;
  container: HTMLElement;
  controls: { enabled: boolean };
  /** Bodies a mode may act on; everything else reads as a miss. Null means "any body". */
  actableBodyIds?: Set<number> | null;
}

/** Below this the press is a tap, above it a drag. CSS px, so it is the same on a phone. */
export const TAP_SLOP_PX = 8;

export class PointerTracker {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene: THREE.Scene;
  private readonly container: HTMLElement;
  private readonly controls: { enabled: boolean };
  private camera: THREE.Camera;
  private actableBodyIds: Set<number> | null;
  private handlers: PointerHandlers | null = null;

  private readonly raycaster = new THREE.Raycaster();
  /** The pointer in NDC, kept so the ray can be re-cast as the camera moves. */
  private readonly ndc = new THREE.Vector2();
  private readonly ray = new THREE.Vector3();
  private readonly direction = new THREE.Vector3();

  private pointerId: number | null = null;
  private hit: PointerHit | null = null;
  private downX = 0;
  private downY = 0;
  private travel = 0;

  constructor(options: PointerTrackerOptions) {
    this.scene = options.scene;
    this.renderer = options.renderer;
    this.container = options.container;
    this.controls = options.controls;
    this.camera = options.camera;
    this.actableBodyIds = options.actableBodyIds ?? null;
    this.raycaster.params.Line!.threshold = 0.1;

    // Capture phase on the container, which is `OrbitControls`' parent: this decides who
    // owns the press before the camera's own listener sees it.
    this.container.addEventListener('pointerdown', this.onPointerDown, true);
    document.addEventListener('pointermove', this.onPointerMove, true);
    document.addEventListener('pointerup', this.onPointerUp, true);
    document.addEventListener('pointercancel', this.onPointerCancel, true);
  }

  setHandlers(handlers: PointerHandlers | null): void {
    if (this.pointerId !== null) this.release();
    this.handlers = handlers;
  }

  setActableBodyIds(bodyIds: Set<number> | null): void {
    this.actableBodyIds = bodyIds;
  }

  setCamera(camera: THREE.Camera): void {
    this.camera = camera;
  }

  /** Whether a gesture is live. */
  get active(): boolean {
    return this.pointerId !== null;
  }

  /**
   * The live gesture, re-cast from the stored screen position through the current camera,
   * so the target keeps up with a moving camera while the pointer is still.
   */
  current(): PointerGesture | null {
    if (!this.hit) return null;
    this.raycaster.setFromCamera(this.ndc, this.camera);
    this.direction.copy(this.raycaster.ray.direction);
    this.ray.copy(this.raycaster.ray.origin).addScaledVector(this.direction, this.hit.distance);
    return { hit: this.hit, ray: this.ray, direction: this.direction, travel: this.travel };
  }

  /** Drop the gesture without telling the mode. */
  release(): void {
    this.pointerId = null;
    this.hit = null;
    this.travel = 0;
    this.controls.enabled = true;
  }

  /** Drop the gesture and tell the mode. */
  cancel(): void {
    if (this.pointerId === null) return;
    this.release();
    this.handlers?.onCancel();
  }

  dispose(): void {
    this.container.removeEventListener('pointerdown', this.onPointerDown, true);
    document.removeEventListener('pointermove', this.onPointerMove, true);
    document.removeEventListener('pointerup', this.onPointerUp, true);
    document.removeEventListener('pointercancel', this.onPointerCancel, true);
    this.release();
    this.handlers = null;
  }

  private updateNdc(x: number, y: number): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.ndc.x = ((x - rect.left) / rect.width) * 2 - 1;
    this.ndc.y = -((y - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.ndc, this.camera);
  }

  /** The first intersection that belongs to a body a mode may act on. */
  private pick(): PointerHit | null {
    for (const intersect of this.raycaster.intersectObjects(this.scene.children, true)) {
      const object = intersect.object;
      if (isIgnored(object)) continue;
      const bodyId = bodyIdOf(object);
      if (bodyId === null) continue;
      if (this.actableBodyIds && !this.actableBodyIds.has(bodyId)) continue;
      return {
        bodyId,
        object,
        point: intersect.point.clone(),
        normal: worldNormal(intersect, this.raycaster.ray.direction),
        distance: intersect.distance,
      };
    }
    return null;
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (this.pointerId !== null || !this.handlers) return;
    // A right- or middle-press is the camera's (pan and dolly), and taking it would also
    // leave `controls.enabled` false under the context menu the browser then opens.
    if (event.pointerType === 'mouse' && event.button !== 0) return;
    this.updateNdc(event.clientX, event.clientY);
    const hit = this.pick();
    if (!hit) return; // A miss is the camera's press.

    this.hit = hit;
    this.downX = event.clientX;
    this.downY = event.clientY;
    this.travel = 0;
    const gesture = this.current();
    const claim = gesture ? this.handlers.onDown(gesture) : 'none';
    if (claim === 'none') {
      this.hit = null;
      return;
    }
    this.pointerId = event.pointerId;
    if (claim === 'exclusive') this.controls.enabled = false;
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) return;
    this.updateNdc(event.clientX, event.clientY);
    this.travel = Math.max(this.travel, Math.hypot(event.clientX - this.downX, event.clientY - this.downY));
    const gesture = this.current();
    if (gesture) this.handlers?.onMove(gesture);
  };

  private onPointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) return;
    this.updateNdc(event.clientX, event.clientY);
    this.travel = Math.max(this.travel, Math.hypot(event.clientX - this.downX, event.clientY - this.downY));
    const gesture = this.current();
    this.release();
    if (gesture) this.handlers?.onUp(gesture);
  };

  private onPointerCancel = (event: PointerEvent): void => {
    if (event.pointerId !== this.pointerId) return;
    this.cancel();
  };
}

/** The MuJoCo body an object draws, or null for untagged objects and the worldbody. */
export function bodyIdOf(object: THREE.Object3D): number | null {
  if (!('bodyID' in object) || typeof object.bodyID !== 'number') return null;
  return object.bodyID > 0 ? object.bodyID : null;
}

/** Opt-outs: the mode gizmos, and anything tagged `ignoreDragForce`. */
function isIgnored(object: THREE.Object3D): boolean {
  let current: THREE.Object3D | null = object;
  while (current) {
    if (current.userData?.ignoreDragForce === true || current.userData?.interactionGizmo === true) {
      return true;
    }
    current = current.parent;
  }
  return false;
}

/** The hit face's world normal, flipped toward the viewer; null when the hit has no face. */
function worldNormal(
  intersect: THREE.Intersection,
  rayDirection: THREE.Vector3,
): THREE.Vector3 | null {
  const local = intersect.face?.normal;
  if (!local) return null;
  const normal = local
    .clone()
    .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(intersect.object.matrixWorld))
    .normalize();
  if (normal.lengthSq() < 1e-12) return null;
  // A back face points away from the camera; the surface it describes does not.
  return normal.dot(rayDirection) > 0 ? normal.negate() : normal;
}
