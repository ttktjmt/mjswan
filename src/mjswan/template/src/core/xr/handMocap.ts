/**
 * WebXR hand tracking as bodies inside the simulation, so a hand can be touched back
 * rather than only shoving forces in.
 *
 * A hand enters as capsules, one per bone between two adjacent WebXR joints. A capsule
 * is a sphere swept along a cylinder, so a finger is continuous rather than a row of
 * balls with notches between them, and its length is fixed because the bone between two
 * adjacent joints does not change length when the finger bends.
 *
 * How a bone enters depends on what it has to do:
 *
 * - `grip` bones — the palm and the five fingertips — carry load, so each is a mocap
 *   target plus a dynamic capsule welded to it. MuJoCo takes contact velocity from body
 *   velocity and a teleported mocap body has none, so a bare mocap bone can push but
 *   never hold: a squeezed object slides straight out of it.
 * - `wall` bones only ever push, so they stay plain mocap. Carrying no degrees of
 *   freedom is what makes the ten bones a hand needs for coverage nearly free.
 *
 * Two stiffness dials pull in opposite directions, so they are set apart from each
 * other: the geom's `solref` is the grip and wants to be stiff, or a held object slips;
 * the weld's `solref` is the hand's own suspension and wants to be soft, or the hand
 * punches through whatever it leans on.
 */
import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';

import { threeToMjcCoordinate } from '../scene/coordinate';
import { normalizeQuat, quatApplyInv, quatInverse, quatMultiply } from '../observation/math';

type Quat = readonly [number, number, number, number];

/** One bone: a capsule spanning two adjacent WebXR joints. */
type Segment = {
  from: XRHandJoint;
  /** Also the segment's name: every joint is the far end of at most one bone. */
  to: XRHandJoint;
  radius: number;
  /** Nominal bone length, metres. The geom is rigid, and WebXR reports no bone lengths. */
  length: number;
  role: 'grip' | 'wall';
};

const FINGERS = ['index', 'middle', 'ring', 'pinky'] as const;

/** Adult-hand proximal / intermediate / distal bone lengths, metres. */
const FINGER_BONES: Record<(typeof FINGERS)[number], readonly [number, number, number]> = {
  index: [0.04, 0.024, 0.019],
  middle: [0.045, 0.027, 0.02],
  ring: [0.041, 0.026, 0.02],
  pinky: [0.032, 0.018, 0.018],
};
const FINGER_RADII = [0.01, 0.009, 0.008] as const;

export const HAND_SEGMENTS: readonly Segment[] = [
  // The palm's two edges, along the index and pinky metacarpals. One capsule down the
  // middle would be a cylinder and anything round would roll straight off it; two make
  // the shallow V a real palm has, which an object can sit in. The first is also the
  // frame a grabbed object is held in.
  { from: 'wrist', to: 'index-finger-metacarpal', radius: 0.02, length: 0.095, role: 'grip' },
  { from: 'wrist', to: 'pinky-finger-metacarpal', radius: 0.02, length: 0.088, role: 'grip' },
  { from: 'thumb-metacarpal', to: 'thumb-phalanx-proximal', radius: 0.014, length: 0.046, role: 'wall' },
  { from: 'thumb-phalanx-proximal', to: 'thumb-phalanx-distal', radius: 0.012, length: 0.032, role: 'wall' },
  { from: 'thumb-phalanx-distal', to: 'thumb-tip', radius: 0.011, length: 0.026, role: 'grip' },
  ...FINGERS.flatMap((finger): Segment[] => {
    const [proximal, intermediate, distal] = FINGER_BONES[finger];
    return [
      {
        from: `${finger}-finger-phalanx-proximal`,
        to: `${finger}-finger-phalanx-intermediate`,
        radius: FINGER_RADII[0],
        length: proximal,
        role: 'wall',
      },
      {
        from: `${finger}-finger-phalanx-intermediate`,
        to: `${finger}-finger-phalanx-distal`,
        radius: FINGER_RADII[1],
        length: intermediate,
        role: 'wall',
      },
      {
        from: `${finger}-finger-phalanx-distal`,
        to: `${finger}-finger-tip`,
        radius: FINGER_RADII[2],
        length: distal,
        role: 'grip',
      },
    ];
  }),
];

/** Grip mass, kg: how hard a fingertip can shove something heavy. */
const GRIP_MASS = 0.05;
/** The palm is what a whole arm leans through. */
const PALM_MASS = 0.15;

/**
 * Dial one: contact. Stiff, and `priority` so the hand's value wins over the object's.
 * At MuJoCo's default the reaction force saturates far below a grip, and a pinched
 * 50 g box slides out however hard it is squeezed.
 */
const CONTACT = 'condim="6" friction="2 0.05 0.001" priority="1" solref="0.004 1" solimp="0.99 0.999 0.001"';
/** A wall never has to hold anything, so it keeps MuJoCo's ordinary contact. */
const WALL_CONTACT = 'condim="4" friction="1.5 0.02 0.001"';
/**
 * Dial two: the hand's own suspension. Soft, so a hand driven into something immovable
 * stops at its surface instead of punching 63 mm through it, and so a reacquired hand
 * does not arrive as one enormous impulse.
 */
const WELD = 'solref="0.02 1" solimp="0.95 0.99 0.001"';

const RGBA = '0.35 0.75 1 0.55';

/** Where an untracked hand waits: above any scene, and outside the floor plane, which
 * is solid all the way down. */
const PARKED_Z = 100;

/** Spread along the parking row, so parked bones do not contact each other. */
const PARKED_SPACING = 0.05;

const HAND_COUNT = 2;

const IDENTITY_QUAT: Quat = [1, 0, 0, 0];

/** Squeezed, not merely touched: two of the hand's own geoms pushing against each other. */
const OPPOSING = -0.5;

function bodyName(hand: number, segment: Segment, kind: 'target' | 'body'): string {
  return `mjswan_xr${hand}_${segment.to}_${kind}`;
}

function weldName(hand: number): string {
  return `mjswan_xr${hand}_grab`;
}

/** Derived from the indices in both places, so the XML and the writes agree. */
function parkedPosition(hand: number, index: number): THREE.Vector3 {
  return new THREE.Vector3((hand * HAND_SEGMENTS.length + index) * PARKED_SPACING, 0, PARKED_Z);
}

/** Rotation taking the capsule's local +z onto `d`, in MuJoCo's `(w, x, y, z)` order. */
export function quatFromZ(d: THREE.Vector3): Quat {
  const w = 1 + d.z;
  // Antiparallel: the axis is undefined, and any half turn about one perpendicular to z
  // maps +z to -z. A capsule is symmetric, so which one does not matter.
  if (w < 1e-6) return [0, 1, 0, 0];
  const q = normalizeQuat([w, -d.y, d.x, 0]);
  return [q[0], q[1], q[2], q[3]];
}

/**
 * Append the hand bones as a second `<worldbody>` and `<equality>`. MJCF merges repeated
 * sections, so this needs nothing from the model it edits: the bodies may live entirely
 * in `<include>`d files. Appended rather than inserted, so the robot's own free joint
 * stays at `qpos[0]`, where `PolicyStateBuilder` reads it.
 */
export function injectHandMocapXml(xml: string): string {
  const bodies: string[] = [];
  const equalities: string[] = [];
  for (let hand = 0; hand < HAND_COUNT; hand++) {
    for (const [index, segment] of HAND_SEGMENTS.entries()) {
      const { x, y, z } = parkedPosition(hand, index);
      const pos = `${x} ${y} ${z}`;
      const size = `size="${segment.radius} ${segment.length / 2}"`;
      const geom = `<geom type="capsule" ${size} rgba="${RGBA}"`;
      if (segment.role === 'wall') {
        bodies.push(
          `    <body name="${bodyName(hand, segment, 'body')}" mocap="true" pos="${pos}">\n` +
            `      ${geom} ${WALL_CONTACT}/>\n` +
            `    </body>`,
        );
        continue;
      }
      const mass = segment.to.endsWith('-metacarpal') ? PALM_MASS : GRIP_MASS;
      bodies.push(`    <body name="${bodyName(hand, segment, 'target')}" mocap="true" pos="${pos}"/>`);
      bodies.push(
        `    <body name="${bodyName(hand, segment, 'body')}" pos="${pos}">\n` +
          `      <freejoint/>\n` +
          `      ${geom} mass="${mass}" ${CONTACT}/>\n` +
          `    </body>`,
      );
      equalities.push(
        `    <weld body1="${bodyName(hand, segment, 'target')}"` +
          ` body2="${bodyName(hand, segment, 'body')}" ${WELD}/>`,
      );
    }
    // Retargeted at runtime onto whatever the hand is squeezing. Friction alone carries
    // 2 kg, so this only has to catch what friction cannot: thin plates, and loads past
    // what the fingertips can pinch.
    equalities.push(
      `    <weld name="${weldName(hand)}" body1="${bodyName(hand, HAND_SEGMENTS[0], 'body')}"` +
        ` body2="world" active="false" torquescale="1"/>`,
    );
  }

  const block =
    `  <worldbody>\n${bodies.join('\n')}\n  </worldbody>\n` +
    `  <equality>\n${equalities.join('\n')}\n  </equality>\n`;

  const close = xml.lastIndexOf('</mujoco>');
  if (close < 0) {
    throw new Error('injectHandMocapXml: scene XML has no closing </mujoco>');
  }
  return `${xml.slice(0, close)}${block}${xml.slice(close)}`;
}

/** Rewrite a scene XML in the Emscripten VFS in place, ahead of `mj_loadXML`. */
export function injectHandMocapFile(mujoco: MainModule, path: string): void {
  const xml = new TextDecoder().decode(mujoco.FS.readFile(path));
  mujoco.FS.writeFile(path, injectHandMocapXml(xml));
}

type BoundSegment = {
  segment: Segment;
  bodyId: number;
  /** Both -1 for a wall, which is driven straight through its own mocap slot. */
  qposAdr: number;
  qvelAdr: number;
  mocapId: number;
  parked: THREE.Vector3;
  tracked: boolean;
};

type BoundHand = {
  segments: BoundSegment[];
  /** The frame a grabbed object is held in; null if the model lacks the palm. */
  palm: BoundSegment | null;
  weldId: number;
  grabbed: number | null;
};

export class HandMocap {
  private readonly hands: THREE.XRHandSpace[];
  private bound: BoundHand[] = [];
  /** Every hand geom, so a contact can be attributed and a grab never targets the hand. */
  private handOfGeom = new Map<number, number>();
  private neqData = 11;
  private readonly from = new THREE.Vector3();
  private readonly to = new THREE.Vector3();

  constructor(hands: THREE.XRHandSpace[]) {
    this.hands = hands;
  }

  bind(mujoco: MainModule, mjModel: MjModel): void {
    const body = mujoco.mjtObj.mjOBJ_BODY.value;
    const equality = mujoco.mjtObj.mjOBJ_EQUALITY.value;
    this.neqData = mujoco.mjNEQDATA;
    this.bound = [];
    this.handOfGeom = new Map();
    const handOfBody = new Map<number, number>();
    for (const [hand] of this.hands.entries()) {
      const segments: BoundSegment[] = [];
      for (const [index, segment] of HAND_SEGMENTS.entries()) {
        const bodyId = mujoco.mj_name2id(mjModel, body, bodyName(hand, segment, 'body'));
        if (bodyId < 0) continue;
        const isGrip = segment.role === 'grip';
        const targetId = isGrip
          ? mujoco.mj_name2id(mjModel, body, bodyName(hand, segment, 'target'))
          : bodyId;
        if (targetId < 0) continue;
        const jntAdr = isGrip ? mjModel.body_jntadr[bodyId] : -1;
        segments.push({
          segment,
          bodyId,
          qposAdr: isGrip ? mjModel.jnt_qposadr[jntAdr] : -1,
          qvelAdr: isGrip ? mjModel.jnt_dofadr[jntAdr] : -1,
          mocapId: mjModel.body_mocapid[targetId],
          parked: parkedPosition(hand, index),
          tracked: false,
        });
        handOfBody.set(bodyId, hand);
      }
      // Pushed even when empty: the index into `bound` is the index into `hands`.
      this.bound.push({
        segments,
        palm: segments.find((s) => s.segment === HAND_SEGMENTS[0]) ?? null,
        weldId: mujoco.mj_name2id(mjModel, equality, weldName(hand)),
        grabbed: null,
      });
    }
    for (let g = 0; g < mjModel.ngeom; g++) {
      const hand = handOfBody.get(mjModel.geom_bodyid[g]);
      if (hand !== undefined) this.handOfGeom.set(g, hand);
    }
    if (this.bound.every((hand) => hand.segments.length === 0)) {
      console.warn('[HandMocap] no injected hand bodies found in the model');
    }
  }

  /** The rendered bodies, for anything measuring the scene: parked hands are far outside it. */
  bodyIds(): number[] {
    return this.bound.flatMap((hand) => hand.segments.map((s) => s.bodyId));
  }

  /**
   * Point each bone at its tracked joints, then settle grabs. A bone that has just come
   * back teleports its dynamic body instead of letting the weld drag it in from the
   * parking spot, which would arrive as one enormous impulse.
   */
  update(mjModel: MjModel, mjData: MjData): void {
    for (const [hand, boundHand] of this.bound.entries()) {
      const joints = this.hands[hand]?.joints;
      for (const bound of boundHand.segments) {
        const from = joints?.[bound.segment.from];
        const to = joints?.[bound.segment.to];
        if (!from?.visible || !to?.visible) {
          if (bound.tracked) this.parkSegment(mjData, bound);
          continue;
        }
        from.getWorldPosition(this.from);
        to.getWorldPosition(this.to);
        const a = threeToMjcCoordinate(this.from);
        const b = threeToMjcCoordinate(this.to);
        const pos = a.clone().add(b).multiplyScalar(0.5);
        const quat = quatFromZ(b.clone().sub(a).normalize());
        this.writeTarget(mjData, bound, pos, quat);
        if (!bound.tracked) {
          this.writeBody(mjData, bound, pos, quat);
          bound.tracked = true;
        }
      }
    }
    this.settleGrabs(mjModel, mjData);
  }

  /**
   * Park every hand. Called after a sim reset: a keyframe written for the model before
   * injection is zero-padded for the appended free joints, so the reset spawns the hands
   * at the world origin, in the middle of the scene.
   */
  park(mjData: MjData): void {
    for (const hand of this.bound) {
      for (const bound of hand.segments) this.parkSegment(mjData, bound);
      this.release(mjData, hand);
    }
  }

  private parkSegment(mjData: MjData, bound: BoundSegment): void {
    this.writeTarget(mjData, bound, bound.parked, IDENTITY_QUAT);
    this.writeBody(mjData, bound, bound.parked, IDENTITY_QUAT);
    bound.tracked = false;
  }

  private writeTarget(mjData: MjData, bound: BoundSegment, pos: THREE.Vector3, quat: Quat): void {
    const at = bound.mocapId * 3;
    mjData.mocap_pos[at + 0] = pos.x;
    mjData.mocap_pos[at + 1] = pos.y;
    mjData.mocap_pos[at + 2] = pos.z;
    for (let i = 0; i < 4; i++) mjData.mocap_quat[bound.mocapId * 4 + i] = quat[i];
  }

  /** The dynamic twin of a `grip` bone. A `wall` has none: its mocap slot is the body. */
  private writeBody(mjData: MjData, bound: BoundSegment, pos: THREE.Vector3, quat: Quat): void {
    if (bound.qposAdr < 0) return;
    const qpos = mjData.qpos;
    qpos[bound.qposAdr + 0] = pos.x;
    qpos[bound.qposAdr + 1] = pos.y;
    qpos[bound.qposAdr + 2] = pos.z;
    for (let i = 0; i < 4; i++) qpos[bound.qposAdr + 3 + i] = quat[i];
    for (let i = 0; i < 6; i++) mjData.qvel[bound.qvelAdr + i] = 0;
  }

  /**
   * Start and stop grabs from MuJoCo's own contacts rather than from a pinch gesture.
   * three.js only fires `pinchstart` when the thumb and index *tips* are within 15 mm of
   * each other, which cannot happen while a hand is holding anything thicker than a
   * card; contacts say what is actually being squeezed, at any thickness and any number
   * of fingers.
   */
  private settleGrabs(mjModel: MjModel, mjData: MjData): void {
    const squeezed = this.squeezedBodies(mjModel, mjData);
    for (const [hand, boundHand] of this.bound.entries()) {
      if (boundHand.weldId < 0) continue;
      const target = squeezed.get(hand) ?? null;
      if (target === null) {
        if (boundHand.grabbed !== null) this.release(mjData, boundHand);
        continue;
      }
      // The weld holds the object in the palm's frame, so a parked palm would drag it
      // out of the scene.
      if (boundHand.grabbed !== null || !boundHand.palm?.tracked) continue;
      this.grab(mjModel, mjData, boundHand, boundHand.palm.bodyId, target);
    }
  }

  /** Per hand, the body most of its geoms are squeezing: touched from opposing sides. */
  private squeezedBodies(mjModel: MjModel, mjData: MjData): Map<number, number> {
    type Touch = { hand: number; body: number; geoms: Set<number>; normals: number[][] };
    const touches = new Map<string, Touch>();
    for (let c = 0; c < mjData.ncon; c++) {
      const contact = mjData.contact.get(c);
      if (!contact) continue;
      const geom1: number = contact.geom1;
      const geom2: number = contact.geom2;
      const hand1 = this.handOfGeom.get(geom1);
      const hand2 = this.handOfGeom.get(geom2);
      const handFirst = hand1 !== undefined;
      // The contact frame's first row is the normal, pointing from geom1 into geom2.
      const frame = contact.frame as ArrayLike<number>;
      const normal = [0, 1, 2].map((i) => (handFirst ? -frame[i] : frame[i]));
      // An embind handle, not a view: it has to be released or the WASM heap grows.
      contact.delete();
      // Hand against hand, or neither: a grab needs exactly one side to be the hand.
      if (handFirst === (hand2 !== undefined)) continue;
      const hand = hand1 ?? (hand2 as number);
      const body: number = mjModel.geom_bodyid[handFirst ? geom2 : geom1];
      if (body === 0) continue;
      const key = `${hand}:${body}`;
      const entry: Touch = touches.get(key) ?? { hand, body, geoms: new Set(), normals: [] };
      entry.geoms.add(handFirst ? geom1 : geom2);
      entry.normals.push(normal);
      touches.set(key, entry);
    }

    const best = new Map<number, number>();
    const bestGeoms = new Map<number, number>();
    for (const { hand, body, geoms, normals } of touches.values()) {
      if (geoms.size < 2) continue;
      const opposed = normals.some((a) =>
        normals.some((b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2] < OPPOSING),
      );
      if (!opposed) continue;
      if (geoms.size <= (bestGeoms.get(hand) ?? 0)) continue;
      bestGeoms.set(hand, geoms.size);
      best.set(hand, body);
    }
    return best;
  }

  /**
   * Weld the squeezed body to the palm at the pose it is already in, so activating the
   * constraint holds it rather than snapping it. `eq_data` for a weld is
   * `[anchor(3), relpose pos(3), relpose quat(4), torquescale(1)]`, and its relpose is
   * body2 expressed in body1's frame — the opposite of the obvious reading.
   */
  private grab(mjModel: MjModel, mjData: MjData, hand: BoundHand, palm: number, target: number): void {
    const palmQuat = [0, 1, 2, 3].map((i) => mjData.xquat[palm * 4 + i]);
    const relPos = quatApplyInv(
      palmQuat,
      [0, 1, 2].map((i) => mjData.xpos[target * 3 + i] - mjData.xpos[palm * 3 + i]),
    );
    const relQuat = quatMultiply(
      quatInverse(palmQuat),
      [0, 1, 2, 3].map((i) => mjData.xquat[target * 4 + i]),
    );
    const at = hand.weldId * this.neqData;
    for (let i = 0; i < 3; i++) mjModel.eq_data[at + i] = 0;
    for (let i = 0; i < 3; i++) mjModel.eq_data[at + 3 + i] = relPos[i];
    for (let i = 0; i < 4; i++) mjModel.eq_data[at + 6 + i] = relQuat[i];
    mjModel.eq_data[at + 10] = 1;
    mjModel.eq_obj1id[hand.weldId] = palm;
    mjModel.eq_obj2id[hand.weldId] = target;
    mjData.eq_active[hand.weldId] = 1;
    hand.grabbed = target;
  }

  private release(mjData: MjData, hand: BoundHand): void {
    if (hand.weldId >= 0) mjData.eq_active[hand.weldId] = 0;
    hand.grabbed = null;
  }
}
