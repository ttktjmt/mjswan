/**
 * What a mode draws while it is doing something.
 *
 * All of it hangs off the three.js scene rather than the MuJoCo root, so it is outside
 * `frameCamera`'s bounds by construction, and every object carries `interactionGizmo` so
 * the pointer's own raycast skips it — otherwise the arrow you are dragging becomes the
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
 * The pull arrow: from the grab point toward the pointer. Same shape and colour the
 * pre-mode drag had, since it is the one piece of this people already read at a glance.
 */
export class DragArrow {
  private readonly scene: THREE.Scene;
  private readonly group = new THREE.Group();
  private readonly shaft: THREE.Mesh;
  private readonly head: THREE.Mesh;
  private readonly material: THREE.MeshStandardMaterial;
  private static readonly HEAD_HEIGHT = 0.1;
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
    this.shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 1), this.material);
    this.shaft.position.y = 0.5;
    this.head = new THREE.Mesh(new THREE.ConeGeometry(0.03, DragArrow.HEAD_HEIGHT), this.material);
    this.head.position.y = 1;
    this.group.add(this.shaft, this.head);
    this.group.visible = false;
    tag(this.group);
    scene.add(this.group);
  }

  /** `saturated` recolours the arrow while the force is riding its clamp. */
  show(from: THREE.Vector3, to: THREE.Vector3, saturated = false): void {
    const offset = to.clone().sub(from);
    const length = offset.length();
    if (length <= 0.001) {
      this.hide();
      return;
    }
    this.group.visible = true;
    this.group.position.copy(from);
    this.group.quaternion.setFromUnitVectors(DragArrow.UP, offset.normalize());
    const shaftLength = Math.max(0.01, length - DragArrow.HEAD_HEIGHT);
    this.shaft.scale.y = shaftLength;
    this.shaft.position.y = shaftLength / 2;
    this.head.position.y = shaftLength + DragArrow.HEAD_HEIGHT / 2;
    this.material.color.setHex(saturated ? 0xffd166 : GIZMO_COLOR);
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
