/**
 * A skin's spring bones (hair, skirt, ...) simulated by MuJoCo, when the skin asks for it
 * (`mujoco_springs`), in a model of their own: the scene's model, and so the robot and its
 * policy, never see them.
 *
 * Each spring joint is a ball-jointed body with its mass at the bone's tail, hung from
 * the node the chain hangs from. That node, and each node carrying a VRM collider, is a
 * free body pushed onto the posed avatar every substep, so a chain feels the avatar's
 * motion as inertia rather than being carried. The model is in three.js's world frame
 * (y up), so poses go across unconverted.
 */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import type {
  VRMSpringBoneCollider,
  VRMSpringBoneColliderGroup,
  VRMSpringBoneJoint,
  VRMSpringBoneManager,
} from '@pixiv/three-vrm';

/** The frame step VRM's per-frame spring settings are tuned at (UniVRM, three-vrm). */
const AUTHORED_STEP = 1 / 60;
/** No substep is longer: contacts and short, stiff segments need it. */
export const PHYSICS_STEP = 1 / 240;
/** Capped so the stiffest joint still resolves at PHYSICS_STEP. */
const MAX_ANGULAR_FREQUENCY = 0.8 / PHYSICS_STEP;
/** Kilograms per metre of bone: the scale drops out of the dynamics, not of the contacts. */
const LINEAR_DENSITY = 0.02;
const MIN_LENGTH = 0.01;
const MIN_RADIUS = 0.002;
/** Per second; a drag of 0 would otherwise ring forever. */
const MIN_DAMPING = 2;
/** Spin about the bone has little inertia of its own; this keeps it from out-running the step. */
const ARMATURE_RATIO = 0.25;
/** Heavy enough that a chain's pull on it is noise next to the push onto the pose. */
const ANCHOR_MASS = 1000;

/** A node a chain hangs from or a collider rides on, driven onto the avatar's pose. */
type Anchor = {
  node: THREE.Object3D;
  body: number;
  qposAdr: number;
  dofAdr: number;
  fromPos: THREE.Vector3;
  fromQuat: THREE.Quaternion;
  toPos: THREE.Vector3;
  toQuat: THREE.Quaternion;
};

/** A spring bone and the ball joint it takes its rotation from. */
type Link = { bone: THREE.Object3D; qposAdr: number; rest: THREE.Quaternion };

/** A body of the model being written, before it is compiled. */
type BodyNode = {
  node: THREE.Object3D;
  joint?: VRMSpringBoneJoint;
  children: BodyNode[];
  geoms: string[];
  /** Rest pose, relative to the skin's scene, scale left out. */
  pos: THREE.Vector3;
  quat: THREE.Quaternion;
  /** Joint bodies: the tail in the body's frame, and the point mass there. */
  tail?: THREE.Vector3;
  mass?: number;
};

const num = (x: number): string => String(Number(x.toPrecision(7)));
const vec = (v: THREE.Vector3): string => `${num(v.x)} ${num(v.y)} ${num(v.z)}`;
const wxyz = (q: THREE.Quaternion): string => `${num(q.w)} ${num(q.x)} ${num(q.y)} ${num(q.z)}`;

function isCollider(object: THREE.Object3D | null): boolean {
  return !!object && 'shape' in object && 'colliderMatrix' in object;
}

export class SpringPhysics {
  private readonly position = new THREE.Vector3();
  private readonly quaternion = new THREE.Quaternion();
  private readonly scale = new THREE.Vector3();
  private readonly target = new THREE.Quaternion();
  private readonly turn = new THREE.Quaternion();
  private readonly ball = new THREE.Quaternion();
  private readonly spin = new THREE.Vector3();

  private constructor(
    private readonly mujoco: MainModule,
    private readonly model: MjModel,
    private readonly data: MjData,
    private readonly anchors: Anchor[],
    private readonly links: Link[],
  ) {}

  /** Joints simulated; zero means the skin has no springs. */
  get jointCount(): number {
    return this.links.length;
  }

  /**
   * The model for `manager`'s springs, from the avatar at rest: call before it is first
   * posed. Null when it has none.
   */
  static build(mujoco: MainModule, manager: VRMSpringBoneManager, scene: THREE.Object3D): SpringPhysics | null {
    const { xml, anchors, links } = SpringPhysics.describe(manager, scene);
    if (!links.length) return null;
    const model = (mujoco as unknown as { MjModel: { from_xml_string(xml: string): MjModel } }).MjModel.from_xml_string(
      xml,
    );
    const data = new mujoco.MjData(model);
    const qposOf = (body: number) => model.jnt_qposadr[model.body_jntadr[body]];
    return new SpringPhysics(
      mujoco,
      model,
      data,
      anchors.map(({ node, body }) => ({
        node,
        body,
        qposAdr: qposOf(body),
        dofAdr: model.jnt_dofadr[model.body_jntadr[body]],
        fromPos: new THREE.Vector3(),
        fromQuat: new THREE.Quaternion(),
        toPos: new THREE.Vector3(),
        toQuat: new THREE.Quaternion(),
      })),
      links.map(({ bone, body }) => ({ bone, qposAdr: qposOf(body), rest: bone.quaternion.clone() })),
    );
  }

  /** The MJCF for the springs, and the body ids its anchors and joints compile to. */
  static describe(
    manager: VRMSpringBoneManager,
    scene: THREE.Object3D,
  ): {
    xml: string;
    anchors: Array<{ node: THREE.Object3D; body: number }>;
    links: Array<{ bone: THREE.Object3D; body: number }>;
  } {
    scene.updateMatrixWorld(true);
    const toScene = new THREE.Matrix4().copy(scene.matrixWorld).invert();
    const restOf = (node: THREE.Object3D) => {
      const pos = new THREE.Vector3();
      const quat = new THREE.Quaternion();
      new THREE.Matrix4().multiplyMatrices(toScene, node.matrixWorld).decompose(pos, quat, new THREE.Vector3());
      return { pos, quat };
    };
    const inScene = (node: THREE.Object3D, local: THREE.Vector3) =>
      node.localToWorld(local.clone()).applyMatrix4(toScene);

    // three-vrm makes a joint of every node under a VRM 0.x spring root, colliders included.
    const joints = new Map<THREE.Object3D, VRMSpringBoneJoint>();
    for (const joint of manager.joints) {
      if (!isCollider(joint.bone) && !joints.has(joint.bone)) joints.set(joint.bone, joint);
    }

    const bodies = new Map<THREE.Object3D, BodyNode>();
    const roots: BodyNode[] = [];
    const bodyOf = (node: THREE.Object3D): BodyNode => {
      let body = bodies.get(node);
      if (body) return body;
      const joint = joints.get(node);
      body = { node, joint, children: [], geoms: [], ...restOf(node) };
      bodies.set(node, body);
      if (joint && node.parent) bodyOf(node.parent).children.push(body);
      else roots.push(body);
      return body;
    };

    // One contype bit per collider group; a chain's spheres meet only its own groups.
    const groupBits = new Map<VRMSpringBoneColliderGroup, number>();
    for (const joint of joints.values()) {
      for (const group of joint.colliderGroups) {
        if (!groupBits.has(group)) groupBits.set(group, 1 << groupBits.size % 30);
      }
    }
    const colliderBits = new Map<VRMSpringBoneCollider, number>();
    for (const [group, bit] of groupBits) {
      for (const collider of group.colliders) colliderBits.set(collider, (colliderBits.get(collider) ?? 0) | bit);
    }

    let gravity = 0;
    for (const [bone, joint] of joints) {
      const body = bodyOf(bone);
      const tailWorld = inScene(bone, joint.initialLocalChildPosition);
      body.tail = tailWorld.sub(body.pos).applyQuaternion(body.quat.clone().invert());
      body.mass = LINEAR_DENSITY * Math.max(body.tail.length(), MIN_LENGTH);
      gravity = Math.max(gravity, SpringPhysics.gravityOf(joint));
    }

    for (const [collider, bits] of colliderBits) {
      const node = collider.parent;
      const shape = collider.shape as VRMSpringBoneCollider['shape'] & {
        type: string;
        offset?: THREE.Vector3;
        tail?: THREE.Vector3;
        radius?: number;
        inside?: boolean;
      };
      // Keeping a chain inside a shape, or above a plane, has no MuJoCo geom to say it.
      if (!node || shape.inside || (shape.type !== 'sphere' && shape.type !== 'capsule')) continue;
      const body = bodyOf(node);
      const inBody = (local: THREE.Vector3) =>
        inScene(node, local).sub(body.pos).applyQuaternion(body.quat.clone().invert());
      const radius = num(Math.max(shape.radius ?? 0, MIN_RADIUS));
      const offset = inBody(shape.offset ?? new THREE.Vector3());
      const type = `contype="${bits}" conaffinity="0"`;
      body.geoms.push(
        shape.type === 'sphere'
          ? `<geom type="sphere" pos="${vec(offset)}" size="${radius}" ${type}/>`
          : `<geom type="capsule" fromto="${vec(offset)} ${vec(inBody(shape.tail ?? new THREE.Vector3()))}" size="${radius}" ${type}/>`,
      );
    }

    const anchors: Array<{ node: THREE.Object3D; body: number }> = [];
    const links: Array<{ bone: THREE.Object3D; body: number }> = [];
    let nextBody = 1; // MuJoCo numbers bodies depth first, as written; 0 is the world.
    const write = (body: BodyNode, parent: BodyNode | null): string => {
      const id = nextBody++;
      const joint = body.joint;
      let pos = body.pos;
      let quat = body.quat;
      if (parent) {
        const inverse = parent.quat.clone().invert();
        pos = body.pos.clone().sub(parent.pos).applyQuaternion(inverse);
        quat = inverse.multiply(body.quat);
      }
      const lines: string[] = [];
      if (joint) {
        links.push({ bone: body.node, body: id });
        lines.push(...SpringPhysics.jointXml(body, joint, groupBits));
      } else {
        anchors.push({ node: body.node, body: id });
        lines.push(
          '<freejoint/>',
          `<inertial pos="0 0 0" mass="${ANCHOR_MASS}" diaginertia="${ANCHOR_MASS} ${ANCHOR_MASS} ${ANCHOR_MASS}"/>`,
        );
      }
      lines.push(...body.geoms);
      for (const child of body.children) lines.push(write(child, body));
      // One gravity for the model, the strongest; each joint cancels what it does not take.
      const gravcomp = gravity > 0 ? ` gravcomp="${num(joint ? 1 - SpringPhysics.gravityOf(joint) / gravity : 1)}"` : '';
      return `<body pos="${vec(pos)}" quat="${wxyz(quat)}"${gravcomp}>\n${lines.join('\n')}\n</body>`;
    };
    // Every root is an anchor: a joint hangs from its parent's body, simulated or not.
    const worldbody = roots.map((body) => write(body, null)).join('\n');

    const contacts = colliderBits.size > 0;
    const xml = `<mujoco model="skin springs">
<option timestep="${num(PHYSICS_STEP)}" gravity="0 ${num(-gravity)} 0" integrator="implicitfast" jacobian="sparse">
<flag filterparent="disable"${contacts ? '' : ' contact="disable"'}/>
</option>
<default><geom condim="1" solref="${num(2 * PHYSICS_STEP)} 1" solimp="0.95 0.99 0.001"/></default>
<worldbody>
${worldbody}
</worldbody>
</mujoco>`;
    return { xml, anchors, links };
  }

  /** m/s² downward: VRM adds `gravityPower` of displacement per second each frame. */
  private static gravityOf(joint: VRMSpringBoneJoint): number {
    const down = Math.max(0, -joint.settings.gravityDir.y);
    return (joint.settings.gravityPower * down) / AUTHORED_STEP;
  }

  /**
   * VRM pulls a tail `stiffness` metres per second per frame toward rest and keeps
   * `1 - dragForce` of its velocity each frame. At the authored step that is an angular
   * frequency `sqrt(stiffness / (step * length))` and a decay `-ln(1 - drag) / step`,
   * here given against the inertia of everything the joint carries.
   */
  private static jointXml(
    body: BodyNode,
    joint: VRMSpringBoneJoint,
    groupBits: Map<VRMSpringBoneColliderGroup, number>,
  ): string[] {
    const tail = body.tail!;
    const length = Math.max(tail.length(), MIN_LENGTH);
    const carried = SpringPhysics.carriedInertia(body);
    const armature = ARMATURE_RATIO * carried;
    const inertia = carried + armature;
    const omega = Math.min(Math.sqrt(joint.settings.stiffness / (AUTHORED_STEP * length)), MAX_ANGULAR_FREQUENCY);
    const decay = Math.max(-Math.log(1 - Math.min(Math.max(joint.settings.dragForce, 0), 0.95)) / AUTHORED_STEP, MIN_DAMPING);
    let mask = 0;
    for (const group of joint.colliderGroups) mask |= groupBits.get(group) ?? 0;
    const mass = body.mass!;
    const own = 0.1 * mass * length * length;
    return [
      `<joint type="ball" stiffness="${num(inertia * omega * omega)}" damping="${num(inertia * decay)}" armature="${num(armature)}"/>`,
      `<inertial pos="${vec(tail)}" mass="${num(mass)}" diaginertia="${num(own)} ${num(own)} ${num(own)}"/>`,
      `<geom type="sphere" pos="${vec(tail)}" size="${num(Math.max(joint.settings.hitRadius, MIN_RADIUS))}" contype="0" conaffinity="${mask}"/>`,
    ];
  }

  /** About the joint, at rest: the tails of this joint and every joint below it. */
  private static carriedInertia(body: BodyNode): number {
    const origin = body.pos;
    let total = 0;
    const visit = (b: BodyNode) => {
      if (b.tail && b.mass) {
        const tail = b.tail.clone().applyQuaternion(b.quat).add(b.pos);
        const length = Math.max(b.tail.length(), MIN_LENGTH);
        total += b.mass * tail.distanceToSquared(origin) + 0.1 * b.mass * length * length;
      }
      for (const child of b.children) if (child.joint) visit(child);
    };
    visit(body);
    return Math.max(total, 1e-9);
  }

  /** Start over: every anchor on its node, every chain at rest, nothing moving. */
  reset(): void {
    this.mujoco.mj_resetData(this.model, this.data);
    const qpos = this.data.qpos;
    for (const anchor of this.anchors) {
      this.readPose(anchor.node, anchor.toPos, anchor.toQuat);
      anchor.fromPos.copy(anchor.toPos);
      anchor.fromQuat.copy(anchor.toQuat);
      qpos.set([anchor.toPos.x, anchor.toPos.y, anchor.toPos.z], anchor.qposAdr);
      qpos.set([anchor.toQuat.w, anchor.toQuat.x, anchor.toQuat.y, anchor.toQuat.z], anchor.qposAdr + 3);
    }
    this.writeBones();
  }

  /**
   * Advance `dt` seconds, the anchors moving from where they were to where their nodes are
   * now (the avatar posed and its world matrices updated), then turn the spring bones.
   */
  step(dt: number): void {
    for (const anchor of this.anchors) this.readPose(anchor.node, anchor.toPos, anchor.toQuat);
    const steps = Math.max(1, Math.ceil(dt / PHYSICS_STEP - 1e-6));
    const h = dt / steps;
    this.model.opt.timestep = h;
    for (let step = 1; step <= steps; step++) {
      const alpha = step / steps;
      for (const anchor of this.anchors) this.drive(anchor, alpha, h);
      this.mujoco.mj_step(this.model, this.data);
    }
    for (const anchor of this.anchors) {
      anchor.fromPos.copy(anchor.toPos);
      anchor.fromQuat.copy(anchor.toQuat);
    }
    this.writeBones();
  }

  /**
   * The wrench that lands the anchor on its target at the end of this substep: its
   * velocity set to cover the gap in one step, as a force, so the chains feel it.
   */
  private drive(anchor: Anchor, alpha: number, h: number): void {
    const qpos = this.data.qpos;
    const qvel = this.data.qvel;
    const wrench = this.data.xfrc_applied;
    const p = this.position.lerpVectors(anchor.fromPos, anchor.toPos, alpha);
    const q = this.quaternion.set(
      qpos[anchor.qposAdr + 4],
      qpos[anchor.qposAdr + 5],
      qpos[anchor.qposAdr + 6],
      qpos[anchor.qposAdr + 3],
    );
    const row = anchor.body * 6;
    for (let i = 0; i < 3; i++) {
      const v = (p.getComponent(i) - qpos[anchor.qposAdr + i]) / h;
      wrench[row + i] = (ANCHOR_MASS * (v - qvel[anchor.dofAdr + i])) / h;
    }
    // A free joint's angular velocity is in its own frame; the wrench's torque in the world's.
    this.target.slerpQuaternions(anchor.fromQuat, anchor.toQuat, alpha);
    const turn = this.turn.copy(q).invert().multiply(this.target);
    if (turn.w < 0) turn.set(-turn.x, -turn.y, -turn.z, -turn.w);
    const sin = Math.hypot(turn.x, turn.y, turn.z);
    const angle = 2 * Math.atan2(sin, turn.w);
    const spin = sin > 1e-12 ? this.spin.set(turn.x, turn.y, turn.z).multiplyScalar(angle / sin / h) : this.spin.set(0, 0, 0);
    spin.set(
      (ANCHOR_MASS * (spin.x - qvel[anchor.dofAdr + 3])) / h,
      (ANCHOR_MASS * (spin.y - qvel[anchor.dofAdr + 4])) / h,
      (ANCHOR_MASS * (spin.z - qvel[anchor.dofAdr + 5])) / h,
    ).applyQuaternion(q);
    wrench[row + 3] = spin.x;
    wrench[row + 4] = spin.y;
    wrench[row + 5] = spin.z;
  }

  private readPose(node: THREE.Object3D, pos: THREE.Vector3, quat: THREE.Quaternion): void {
    node.matrixWorld.decompose(pos, quat, this.scale);
  }

  /** Each spring bone at its rest rotation, turned by its ball joint. */
  private writeBones(): void {
    const qpos = this.data.qpos;
    for (const link of this.links) {
      const a = link.qposAdr;
      this.ball.set(qpos[a + 1], qpos[a + 2], qpos[a + 3], qpos[a]);
      link.bone.quaternion.copy(link.rest).multiply(this.ball);
      // three-vrm leaves a spring bone's matrix to whoever turns it.
      link.bone.updateMatrix();
    }
  }

  dispose(): void {
    this.data.delete();
    this.model.delete();
  }
}
