/**
 * The hand that runs without a headset: the bone table, the MJCF the loader injects, the
 * rotation that aims a capsule, and — against the real WASM — carrying a load rather
 * than only shoving it.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as THREE from 'three';

import { HAND_SEGMENTS, injectHandMocapXml, quatFromZ, HandMocap } from '../handMocap';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

const MINIMAL =
  '<mujoco>\n  <worldbody>\n    <geom type="plane" size="5 5 .1"/>\n  </worldbody>\n</mujoco>';

/** The WebXR hand-input joint names, verbatim from the spec's `XRHandJoint` enum. */
const XR_HAND_JOINTS = new Set<string>([
  'wrist',
  ...['thumb'].flatMap((f) => [
    `${f}-metacarpal`,
    `${f}-phalanx-proximal`,
    `${f}-phalanx-distal`,
    `${f}-tip`,
  ]),
  ...['index-finger', 'middle-finger', 'ring-finger', 'pinky-finger'].flatMap((f) => [
    `${f}-metacarpal`,
    `${f}-phalanx-proximal`,
    `${f}-phalanx-intermediate`,
    `${f}-phalanx-distal`,
    `${f}-tip`,
  ]),
]);

describe('HAND_SEGMENTS', () => {
  it('spans only joints the XR runtime actually reports', () => {
    for (const { from, to } of HAND_SEGMENTS) {
      expect(XR_HAND_JOINTS.has(from), from).toBe(true);
      expect(XR_HAND_JOINTS.has(to), to).toBe(true);
    }
  });

  // The far joint names the bone, so a body name is unique and readable in a dump.
  it('gives every bone a distinct far joint', () => {
    const ends = HAND_SEGMENTS.map((s) => s.to);
    expect(new Set(ends).size).toBe(ends.length);
  });

  // `*-finger-metacarpal` is the base of the metacarpal, by the wrist — aiming the palm
  // at it read the direction off a 15 mm span and hung 40 mm of capsule behind the wrist.
  it('spans the palm from the wrist to the knuckles', () => {
    for (const palm of HAND_SEGMENTS.filter((s) => s.from === 'wrist')) {
      expect(palm.to).toMatch(/-finger-phalanx-proximal$/);
      expect(palm.length).toBeGreaterThan(0.08);
    }
    expect(HAND_SEGMENTS.some((s) => s.to.endsWith('-finger-metacarpal'))).toBe(false);
  });

  // The thumb opposes every finger in a pinch, so its pinching face is one capsule
  // rather than two that hinge against each other halfway along.
  it('runs the thumb from its PIP joint straight to the tip', () => {
    const thumb = HAND_SEGMENTS.filter((s) => s.to.startsWith('thumb-'));
    expect(thumb.map((s) => `${s.from} -> ${s.to}`)).toEqual([
      'thumb-metacarpal -> thumb-phalanx-proximal',
      'thumb-phalanx-proximal -> thumb-tip',
    ]);
  });

  // Only the load-bearing bones cost degrees of freedom; the rest are near free.
  it('keeps the palm and the five fingertips as the only grips', () => {
    const grips = HAND_SEGMENTS.filter((s) => s.role === 'grip');
    expect(grips.map((s) => s.to)).toEqual([
      'index-finger-phalanx-proximal',
      'pinky-finger-phalanx-proximal',
      'thumb-tip',
      'index-finger-tip',
      'middle-finger-tip',
      'ring-finger-tip',
      'pinky-finger-tip',
    ]);
    expect(HAND_SEGMENTS.filter((s) => s.role === 'wall')).toHaveLength(9);
  });
});

describe('injectHandMocapXml', () => {
  it('gives each grip bone a target, a dynamic twin and a weld, and each wall neither', () => {
    const xml = injectHandMocapXml(MINIMAL);
    const grips = HAND_SEGMENTS.filter((s) => s.role === 'grip').length * 2;
    const walls = HAND_SEGMENTS.filter((s) => s.role === 'wall').length * 2;

    expect(xml.match(/<freejoint\/>/g)).toHaveLength(grips);
    expect(xml.match(/mocap="true"/g)).toHaveLength(grips + walls);
    // Per grip, plus the one retargeted grab weld per hand.
    expect(xml.match(/<weld /g)).toHaveLength(grips + 2);
    expect(xml.match(/type="capsule"/g)).toHaveLength(grips + walls);
    expect(xml).toContain('name="mjswan_xr0_thumb-tip_body"');
    // The palm carries an arm's worth of lean; a fingertip only has to pinch.
    expect(xml).toContain(`name="mjswan_xr0_index-finger-phalanx-proximal_body"`);
    expect(xml.match(/mass="0.15"/g)).toHaveLength(4);
    expect(xml.match(/mass="0.05"/g)).toHaveLength(10);
    expect(xml).toContain('name="mjswan_xr1_index-finger-phalanx-intermediate_body"');
    expect(xml).toContain('name="mjswan_xr1_grab"');
    // Every bone sits in one group, hidden or drawn, as DEBUG_DRAW_BONES picks: the
    // scene builder draws nothing at `group >= 3`.
    const hidden = xml.match(/group="3"/g)?.length ?? 0;
    const drawn = xml.match(/group="2" rgba="1 1 1 0\.\d+"/g)?.length ?? 0;
    expect(hidden + drawn).toBe(grips + walls);
    expect(hidden === 0 || drawn === 0).toBe(true);
  });

  // Adjacent bones share a joint, so their caps always overlap. Left colliding, a mocap
  // wall shoved the dynamic capsule beside it 78 degrees off its bone.
  it('lets the hand hit the scene but never itself', () => {
    const xml = injectHandMocapXml(MINIMAL);
    const capsules = HAND_SEGMENTS.length * 2;
    expect(xml.match(/contype="2" conaffinity="1"/g)).toHaveLength(capsules);
    // MuJoCo pairs on (contype1 & conaffinity2) || (contype2 & conaffinity1).
    expect((2 & 1) || (2 & 1)).toBe(0);
    expect((2 & 1) || (1 & 1)).toBe(1);
  });

  // Two dials that want opposite things, so a later edit cannot quietly conflate them.
  it('sets a stiff contact and a soft weld', () => {
    const xml = injectHandMocapXml(MINIMAL);
    expect(xml).toContain('priority="1" solref="0.004 1"');
    expect(xml).toContain('<weld body1="mjswan_xr0_thumb-tip_target"');
    expect(xml.match(/solref="0\.02 1"/g)).toHaveLength(
      HAND_SEGMENTS.filter((s) => s.role === 'grip').length * 2,
    );
  });

  // A mocap target adds no `qpos`, and the block is appended, so neither can move the
  // robot's own free joint off `qpos[0]`, where `PolicyStateBuilder` reads it.
  it('leaves the original model ahead of the block, and one closing tag', () => {
    const xml = injectHandMocapXml(MINIMAL);
    expect(xml.indexOf('type="plane"')).toBeLessThan(xml.indexOf('mocap="true"'));
    expect(xml.match(/<\/mujoco>/g)).toHaveLength(1);
    expect(xml.endsWith('</mujoco>')).toBe(true);
  });

  it('parks the hands above the floor, which is solid all the way down', () => {
    for (const pos of injectHandMocapXml(MINIMAL).matchAll(/pos="[-\d.]+ [-\d.]+ ([-\d.]+)"/g)) {
      expect(Number(pos[1])).toBeGreaterThan(0);
    }
  });

  it('refuses XML it cannot close', () => {
    expect(() => injectHandMocapXml('<mujoco>')).toThrow(/closing/);
  });
});

describe('quatFromZ', () => {
  const apply = (q: readonly number[], v: THREE.Vector3) =>
    v.clone().applyQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]));

  it('aims the capsule axis along the bone', () => {
    const directions = [
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(1, 0, 0),
      new THREE.Vector3(0, -1, 0),
      new THREE.Vector3(0.3, -0.5, 0.81).normalize(),
    ];
    for (const d of directions) {
      const turned = apply(quatFromZ(d), new THREE.Vector3(0, 0, 1));
      expect(turned.distanceTo(d)).toBeLessThan(1e-6);
    }
  });

  it('handles a bone pointing straight back down the axis', () => {
    const down = new THREE.Vector3(0, 0, -1);
    expect(apply(quatFromZ(down), new THREE.Vector3(0, 0, 1)).distanceTo(down)).toBeLessThan(1e-6);
  });
});

/**
 * A scripted hand, in place of `renderer.xr.getHand()`. Joints are written in MuJoCo
 * coordinates and swizzled back into three's frame, which is the direction the runtime
 * reads them in.
 */
class FakeHand {
  readonly joints: Record<string, { visible: boolean; getWorldPosition(v: THREE.Vector3): THREE.Vector3 }> = {};
  /** three.js keeps this on the hand itself, and it is what starts and stops a grab. */
  readonly inputState = { pinching: false };

  pinch(on: boolean): void {
    this.inputState.pinching = on;
  }

  set(joint: string, mjc: readonly [number, number, number]): void {
    const three = new THREE.Vector3(mjc[0], mjc[2], -mjc[1]);
    this.joints[joint] = { visible: true, getWorldPosition: (v) => v.copy(three) };
  }
}

const CUBE_HALF = 0.03;
const SUBSTEPS = 10;

/** A 6 cm box on the floor. Its mass is the load the hand has to carry. */
const cubeScene = (mass: number) => `<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="cube" pos="0 0 ${CUBE_HALF + 0.001}">
      <freejoint/>
      <geom type="box" size="${CUBE_HALF} ${CUBE_HALF} ${CUBE_HALF}" mass="${mass}"/>
    </body>
  </worldbody>
</mujoco>`;

describe('HandMocap against the real WASM', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  /**
   * Palm flat against one face of the box, four fingertips against the other, then lift.
   * `oneSided` drops the fingers, so nothing opposes the palm and the box has only
   * friction to hang from.
   */
  function clampAndLift(mass: number, { oneSided = false, pinch = true } = {}) {
    const xml = injectHandMocapXml(cubeScene(mass));
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(xml);
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(
      mjModel,
    );

    const hand = new FakeHand();
    const mocap = new HandMocap([hand as unknown as THREE.XRHandSpace]);
    mocap.bind(mujoco, mjModel);
    mocap.park(mjData);
    mujoco.mj_forward(mjModel, mjData);

    const squeeze = 0.004;
    const cubeId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, 'cube');
    const fingers = ['index-finger', 'middle-finger', 'ring-finger', 'pinky-finger'] as const;
    const bone = (name: string) => HAND_SEGMENTS.find((s) => s.to === name)!;

    /** One control step with the hand's centre at `z`. */
    const step = (z: number) => {
      // The palm's two edges lie flat on the -y face, splayed toward the knuckles as
      // they are on a real hand; their capsule surfaces have to reach the box.
      const [edgeA, edgeB] = [bone('index-finger-phalanx-proximal'), bone('pinky-finger-phalanx-proximal')];
      const palmY = -(CUBE_HALF + edgeA.radius - squeeze);
      const splay = 0.2;
      hand.set('wrist', [-0.045, palmY, z]);
      const knuckle = (b: typeof edgeA, side: number) =>
        [-0.045 + b.length * Math.cos(splay), palmY, z + side * b.length * Math.sin(splay)] as const;
      hand.set('index-finger-phalanx-proximal', knuckle(edgeA, 1));
      hand.set('pinky-finger-phalanx-proximal', knuckle(edgeB, -1));
      // Fingertips run along x on the +y face, stacked up the box.
      for (const [i, finger] of fingers.entries()) {
        const tip = bone(`${finger}-tip`);
        const tipY = CUBE_HALF + tip.radius - squeeze;
        const tipZ = z - 0.018 + i * 0.012;
        if (oneSided) continue;
        hand.set(`${finger}-phalanx-distal`, [-tip.length / 2, tipY, tipZ]);
        hand.set(`${finger}-tip`, [tip.length / 2, tipY, tipZ]);
      }
      mocap.update(mjModel, mjData);
      for (let s = 0; s < SUBSTEPS; s++) mujoco.mj_step(mjModel, mjData);
    };

    const base = CUBE_HALF + 0.001;
    for (let i = 0; i < 12; i++) step(base);
    // Pinched once the hand is already on the box, which is the order a real one goes in.
    if (pinch) hand.pinch(true);
    step(base);
    const before = mjData.xpos[cubeId * 3 + 2];
    for (let i = 1; i <= 50; i++) step(base + i * 0.005);
    const lifted = mjData.xpos[cubeId * 3 + 2] - before;
    // Half a second at the top: a grip that is merely slipping slowly shows up here.
    for (let i = 0; i < 25; i++) step(base + 0.25);
    const held = mjData.xpos[cubeId * 3 + 2] - before;

    const grabId = mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_EQUALITY.value, 'mjswan_xr0_grab');
    const weldHeld = mjData.eq_active[grabId];
    hand.pinch(false);
    step(base + 0.25);
    return { lifted, held, weldHeld, weldReleased: mjData.eq_active[grabId] };
  }

  // The whole point of the dynamic twin. A plain mocap hand scores 0 here, at any mass.
  it.each([0.15, 0.6, 2.0])('carries a %s kg box 25 cm up and holds it', (mass) => {
    const { lifted, held } = clampAndLift(mass);
    // The hand travels 25 cm; a load hangs one to two centimetres below that against the
    // soft weld, and recovers once it stops.
    expect(lifted).toBeGreaterThan(0.21);
    // Not creeping. A grip that is slowly slipping loses centimetres over the half second.
    expect(held).toBeGreaterThan(lifted - 0.002);
  });

  it('drops a box nothing is squeezing', () => {
    const { held } = clampAndLift(0.6, { oneSided: true });
    expect(held).toBeLessThan(0.01);
  });

  // The gesture, not the contact, is what arms the weld. Friction carries this box
  // either way, so the weld's own flag is the only thing that says which path ran.
  it('arms the grab weld on pinchstart and drops it on pinchend', () => {
    const pinched = clampAndLift(2.0);
    expect(pinched.weldHeld).toBe(1);
    expect(pinched.weldReleased).toBe(0);
  });

  /** A model with one scripted hand bound to it, parked. */
  function setup() {
    const xml = injectHandMocapXml(cubeScene(0.15));
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(xml);
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(mjModel);
    const hand = new FakeHand();
    const mocap = new HandMocap([hand as unknown as THREE.XRHandSpace]);
    mocap.bind(mujoco, mjModel);
    mocap.park(mjData);
    const bodyOf = (to: string) =>
      mujoco.mj_name2id(mjModel, mujoco.mjtObj.mjOBJ_BODY.value, `mjswan_xr0_${to}_body`);
    return { mjModel, mjData, hand, mocap, bodyOf };
  }

  // The bone table's length is a nominal adult hand. A wearer's is not, and a capsule
  // that keeps the nominal one has its ends somewhere other than on the joints.
  it.each([
    ['index-finger-tip', 'index-finger-phalanx-distal', 0.011],
    ['index-finger-phalanx-proximal', 'wrist', 0.081],
  ])('sizes the %s capsule from the joints in front of it', (to, from, span) => {
    const { mjModel, mjData, hand, mocap, bodyOf } = setup();
    const seg = HAND_SEGMENTS.find((s) => s.to === to)!;
    expect(span).not.toBeCloseTo(seg.length, 3);

    hand.set(from as string, [0, 0, 1]);
    hand.set(to as string, [0, 0, 1 + (span as number)]);
    mocap.update(mjModel, mjData);
    mujoco.mj_forward(mjModel, mjData);

    const id = bodyOf(to as string);
    const half = mjModel.geom_size[mjModel.body_geomadr[id] * 3 + 1];
    expect(half).toBeCloseTo((span as number) / 2, 6);
    // The capsule is centred on the body and runs along its local +z, so with the ends
    // at `centre ± half` they land on the two joints it was measured from.
    const centre = mjData.xpos[id * 3 + 2];
    expect(centre - half).toBeCloseTo(1, 5);
    expect(centre + half).toBeCloseTo(1 + (span as number), 5);
  });

  // One frame where two joints read as the same point used to snap the bone to world +z.
  it('holds the last pose when two joints collapse onto each other', () => {
    const { mjModel, mjData, hand, mocap, bodyOf } = setup();
    hand.set('index-finger-phalanx-distal', [0, 0, 1]);
    hand.set('index-finger-tip', [0.019, 0, 1]);
    mocap.update(mjModel, mjData);
    mujoco.mj_forward(mjModel, mjData);
    const id = bodyOf('index-finger-tip');
    const aimed = [0, 1, 2, 3].map((i) => mjData.xquat[id * 4 + i]);
    // Aimed along +x, which is a half turn away from the +z a lost bone falls back to.
    expect(Math.abs(mjData.xmat[id * 9 + 2])).toBeGreaterThan(0.99);

    hand.set('index-finger-tip', [0, 0, 1]);
    mocap.update(mjModel, mjData);
    mujoco.mj_forward(mjModel, mjData);
    expect([0, 1, 2, 3].map((i) => mjData.xquat[id * 4 + i])).toEqual(aimed);
  });

  /**
   * A flat adult hand, all 25 WebXR joints, in MuJoCo metres. Laid out anatomically so
   * adjacent bones meet at shared joints, which is where a hand fights itself.
   */
  const FLAT_HAND: Record<string, readonly [number, number, number]> = Object.fromEntries(
    Object.entries({
      'wrist': [50, 128], 'thumb-metacarpal': [40, 118], 'thumb-phalanx-proximal': [22, 98],
      'thumb-phalanx-distal': [12, 81], 'thumb-tip': [5, 67],
      'index-finger-metacarpal': [47, 119], 'index-finger-phalanx-proximal': [34, 70],
      'index-finger-phalanx-intermediate': [32, 45], 'index-finger-phalanx-distal': [31, 30],
      'index-finger-tip': [30, 18],
      'middle-finger-metacarpal': [50, 119], 'middle-finger-phalanx-proximal': [48, 66],
      'middle-finger-phalanx-intermediate': [48, 38], 'middle-finger-phalanx-distal': [48, 21],
      'middle-finger-tip': [48, 8],
      'ring-finger-metacarpal': [53, 119], 'ring-finger-phalanx-proximal': [61, 69],
      'ring-finger-phalanx-intermediate': [63, 43], 'ring-finger-phalanx-distal': [64, 27],
      'ring-finger-tip': [65, 14],
      'pinky-finger-metacarpal': [56, 120], 'pinky-finger-phalanx-proximal': [73, 76],
      'pinky-finger-phalanx-intermediate': [76, 57], 'pinky-finger-phalanx-distal': [78, 46],
      'pinky-finger-tip': [79, 35],
    }).map(([k, [x, y]]) => [k, [(x - 50) * 0.00158, 0, 1 + (128 - y) * 0.00158] as const])
  );

  /**
   * Every bone the hand has, held still: the dynamic grip capsules have to stay on the
   * bones their mocap targets are aimed at. Before the hand stopped colliding with
   * itself they came to rest up to 78 degrees off, since two bones meeting at a joint
   * always overlap and a mocap wall wins every such push.
   */
  it('keeps every bone on its joints with the whole hand in the model', () => {
    const { mjModel, mjData, hand, mocap, bodyOf } = setup();
    for (const [joint, at] of Object.entries(FLAT_HAND)) hand.set(joint, at);
    for (let i = 0; i < 100; i++) {
      mocap.update(mjModel, mjData);
      for (let sub = 0; sub < SUBSTEPS; sub++) mujoco.mj_step(mjModel, mjData);
    }

    // The scene's own cube resting on its floor is fine; two hand bones touching is not.
    const handBodies = new Set(HAND_SEGMENTS.map((seg) => bodyOf(seg.to)));
    const selfPairs: string[] = [];
    for (let c = 0; c < mjData.ncon; c++) {
      const contact = mjData.contact.get(c);
      if (!contact) continue;
      const [b1, b2] = [contact.geom1, contact.geom2].map((g: number) => mjModel.geom_bodyid[g]);
      contact.delete();
      if (handBodies.has(b1) && handBodies.has(b2)) selfPairs.push(`${b1}/${b2}`);
    }
    expect(selfPairs, 'the hand is colliding with itself').toEqual([]);

    for (const seg of HAND_SEGMENTS) {
      const [from, to] = [FLAT_HAND[seg.from], FLAT_HAND[seg.to]];
      const bone = new THREE.Vector3(to[0] - from[0], to[1] - from[1], to[2] - from[2]).normalize();
      const id = bodyOf(seg.to);
      // The capsule runs along the body's local +z, which is the third column of `xmat`.
      const axis = new THREE.Vector3(mjData.xmat[id * 9 + 2], mjData.xmat[id * 9 + 5], mjData.xmat[id * 9 + 8]);
      const degrees = (Math.acos(Math.min(1, Math.abs(axis.dot(bone)))) * 180) / Math.PI;
      expect(degrees, `${seg.to} axis`).toBeLessThan(1);

      const mid = [0, 1, 2].map((i) => (from[i] + to[i]) / 2);
      const off = Math.hypot(...[0, 1, 2].map((i) => mjData.xpos[id * 3 + i] - mid[i]));
      expect(off, `${seg.to} centre`).toBeLessThan(0.001);
      // Sized from the joints, so the ends are on them rather than a nominal hand's.
      const span = Math.hypot(...[0, 1, 2].map((i) => to[i] - from[i]));
      expect(mjModel.geom_size[mjModel.body_geomadr[id] * 3 + 1]).toBeCloseTo(span / 2, 6);
    }
  });

  it('never arms the weld from contact alone', () => {
    const { weldHeld } = clampAndLift(2.0, { pinch: false });
    expect(weldHeld).toBe(0);
  });
});
