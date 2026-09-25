/**
 * What a mode draws while it is doing something.
 *
 * All of it hangs off the three.js scene rather than the MuJoCo root, so it is outside
 * `frameCamera`'s bounds by construction, and every object carries `interactionGizmo` so
 * the pointer's own raycast skips it: otherwise the arrow you are dragging becomes the
 * thing you grab next.
 */
import * as THREE from 'three';

const GIZMO_COLOR = 0xff6347;

function tag(object: THREE.Object3D): void {
  object.userData.interactionGizmo = true;
  object.traverse((child) => {
    child.userData.interactionGizmo = true;
  });
}

/**
 * The pull arrow: from the grab point to the pointer, thicker the harder it pulls. The
 * tip stays on the pointer, so the length still says where the body is being pulled to
 * and the girth is left to say how hard.
 */
export class DragArrow {
  private readonly scene: THREE.Scene;
  private readonly group = new THREE.Group();
  private readonly shaft: THREE.Mesh;
  private readonly head: THREE.Mesh;
  private readonly material: THREE.MeshStandardMaterial;
  private static readonly SHAFT_RADIUS = 0.008;
  private static readonly HEAD_RADIUS = 0.03;
  private static readonly HEAD_HEIGHT = 0.1;
  /** How much thicker the arrow can get than at rest; the head grows by half as much. */
  private static readonly MAX_THICKENING = 4;
  /** The pull, newtons, at which the arrow is halfway to its thickest. */
  private static readonly HALF_FORCE = 100;
  private static readonly UP = new THREE.Vector3(0, 1, 0);

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.material = new THREE.MeshStandardMaterial({
      color: GIZMO_COLOR,
      transparent: true,
      opacity: 0.5,
      metalness: 0,
      roughness: 0.2,
    });
    this.shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(DragArrow.SHAFT_RADIUS, DragArrow.SHAFT_RADIUS, 1),
      this.material,
    );
    this.head = new THREE.Mesh(
      new THREE.ConeGeometry(DragArrow.HEAD_RADIUS, DragArrow.HEAD_HEIGHT),
      this.material,
    );
    this.group.add(this.shaft, this.head);
    this.group.visible = false;
    tag(this.group);
    scene.add(this.group);
  }

  /** `force` is the pull's magnitude, newtons. */
  show(from: THREE.Vector3, to: THREE.Vector3, force: number): void {
    const offset = to.clone().sub(from);
    const length = offset.length();
    if (length <= 0.001) {
      this.hide();
      return;
    }
    this.group.visible = true;
    this.group.position.copy(from);
    this.group.quaternion.setFromUnitVectors(DragArrow.UP, offset.normalize());

    // Saturating rather than linear: there is no clamp to scale against, and a heavy robot
    // takes forces a block never sees, so the girth has to stay readable across both.
    const f = Number.isFinite(force) && force > 0 ? force : 0;
    const t = f / (f + DragArrow.HALF_FORCE);
    const girth = 1 + (DragArrow.MAX_THICKENING - 1) * t;
    const headScale = 1 + ((DragArrow.MAX_THICKENING - 1) / 2) * t;
    // A pull shorter than the head squashes the head rather than pushing it past the pointer.
    const shaftLength = Math.max(0.001, length - DragArrow.HEAD_HEIGHT * headScale);
    const headHeight = length - shaftLength;

    this.shaft.scale.set(girth, shaftLength, girth);
    this.shaft.position.y = shaftLength / 2;
    this.head.scale.set(headScale, headHeight / DragArrow.HEAD_HEIGHT, headScale);
    this.head.position.y = shaftLength + headHeight / 2;
  }

  hide(): void {
    this.group.visible = false;
  }

  dispose(): void {
    this.scene.remove(this.group);
    this.shaft.geometry.dispose();
    this.head.geometry.dispose();
    this.material.dispose();
  }
}

/** A ring that opens where a push landed, so a tap that did something looks different from one that missed. */
export class PushRing {
  private static readonly LIFETIME = 0.25;
  private static readonly RADIUS = 0.12;
  private readonly scene: THREE.Scene;
  private readonly mesh: THREE.Mesh;
  private readonly material: THREE.MeshBasicMaterial;
  private age = PushRing.LIFETIME;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.material = new THREE.MeshBasicMaterial({
      color: GIZMO_COLOR,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.mesh = new THREE.Mesh(new THREE.RingGeometry(0.75, 1, 32), this.material);
    this.mesh.visible = false;
    tag(this.mesh);
    scene.add(this.mesh);
  }

  /** `normal` orients the ring against the surface; null lays it in the view plane. */
  fire(at: THREE.Vector3, normal: THREE.Vector3 | null): void {
    this.mesh.position.copy(at);
    if (normal) this.mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal.clone().normalize());
    this.age = 0;
    this.mesh.visible = true;
  }

  /** For a mode switch or a pause, which stop the clock that would have faded it out. */
  hide(): void {
    this.age = PushRing.LIFETIME;
    this.mesh.visible = false;
  }

  update(dt: number): void {
    if (this.age >= PushRing.LIFETIME) return;
    this.age = Math.min(PushRing.LIFETIME, this.age + dt);
    const t = this.age / PushRing.LIFETIME;
    const scale = PushRing.RADIUS * (0.3 + t);
    this.mesh.scale.setScalar(scale);
    this.material.opacity = 0.7 * (1 - t);
    if (this.age >= PushRing.LIFETIME) this.mesh.visible = false;
  }

  dispose(): void {
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}
