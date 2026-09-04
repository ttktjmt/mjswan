/**
 * The hand that runs without a headset: the bone table, the MJCF the loader injects, the
 * rotation that aims a capsule, and — against the real WASM — the one thing the design
 * exists for, which is carrying a load rather than only shoving it.
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

  // Only the load-bearing bones cost degrees of freedom; the rest are near free.
  it('keeps the palm and the five fingertips as the only grips', () => {
    const grips = HAND_SEGMENTS.filter((s) => s.role === 'grip');
    expect(grips.map((s) => s.to)).toEqual([
      'index-finger-metacarpal',
      'pinky-finger-metacarpal',
      'thumb-tip',
      'index-finger-tip',
      'middle-finger-tip',
      'ring-finger-tip',
      'pinky-finger-tip',
    ]);
    expect(HAND_SEGMENTS.filter((s) => s.role === 'wall')).toHaveLength(10);
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
    expect(xml).toContain('name="mjswan_xr1_index-finger-phalanx-intermediate_body"');
    expect(xml).toContain('name="mjswan_xr1_grab"');
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
   * `squeeze` is how far past the surface the two sides are driven; `oneSided` drops the
   * fingers, so nothing opposes the palm and the box has only friction to hang from.
   */
  function clampAndLift(mass: number, { oneSided = false } = {}) {
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
      const [edgeA, edgeB] = [bone('index-finger-metacarpal'), bone('pinky-finger-metacarpal')];
      const palmY = -(CUBE_HALF + edgeA.radius - squeeze);
      const splay = 0.2;
      hand.set('wrist', [-0.045, palmY, z]);
      const knuckle = (b: typeof edgeA, side: number) =>
        [-0.045 + b.length * Math.cos(splay), palmY, z + side * b.length * Math.sin(splay)] as const;
      hand.set('index-finger-metacarpal', knuckle(edgeA, 1));
      hand.set('pinky-finger-metacarpal', knuckle(edgeB, -1));
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
    const before = mjData.xpos[cubeId * 3 + 2];
    for (let i = 1; i <= 50; i++) step(base + i * 0.005);
    const lifted = mjData.xpos[cubeId * 3 + 2] - before;
    // Half a second at the top: a grip that is merely slipping slowly shows up here.
    for (let i = 0; i < 25; i++) step(base + 0.25);
    return { lifted, held: mjData.xpos[cubeId * 3 + 2] - before };
  }

  // The whole point of the dynamic twin. A plain mocap hand scores 0 here, at any mass.
  it.each([0.15, 0.6, 2.0])('carries a %s kg box 25 cm up and holds it', (mass) => {
    const { lifted, held } = clampAndLift(mass);
    // The hand travels 25 cm. A load hangs one to two centimetres below that against the
    // soft weld, which is the hand's suspension yielding, and recovers once it stops.
    expect(lifted).toBeGreaterThan(0.21);
    // Not creeping. A grip that is slowly slipping loses centimetres over the half second.
    expect(held).toBeGreaterThan(lifted - 0.002);
  });

  it('drops a box nothing is squeezing', () => {
    const { held } = clampAndLift(0.6, { oneSided: true });
    expect(held).toBeLessThan(0.01);
  });
});
