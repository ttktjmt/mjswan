/**
 * What a mode draws follows the frame, while what it does to the sim stays once per control
 * step: Pull's arrow from the body as drawn, Push's ring in sim time.
 */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import { PullMode } from '../modes/pull';
import { PushMode } from '../modes/push';
import type { ModeContext } from '../modes/mode';
import type { PointerGesture } from '../pointer';
import { InteractionWrench } from '../wrench';
import { mjcToThreeCoordinate } from '../../scene/coordinate';
import { getPosition, getQuaternion } from '../../scene/scene';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;
type MjvPerturb = import('mujoco').MjvPerturb;

/** Turned about y, so a point carried into the body's frame and back has to be rotated right. */
const TILTED_BOX = `<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="box" pos="0.2 -0.1 1" quat="0.92388 0 0.38268 0">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>`;

interface Shown {
  from: THREE.Vector3;
  to: THREE.Vector3;
  force: number;
}

describe('a mode draws every frame', () => {
  let mujoco: MainModule;

  beforeAll(async () => {
    mujoco = await (await import('mujoco')).default();
  });

  it("Pull's arrow starts on the body as drawn and keeps the last control step's force", () => {
    const mjModel = (
      mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } }
    ).MjModel.from_xml_string(TILTED_BOX);
    const mjData = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(mjModel);
    mujoco.mj_forward(mjModel, mjData);
    // Posed as the render loop poses it.
    const body = new THREE.Group();
    getPosition(mjData.xpos as unknown as Float32Array, 1, body.position);
    getQuaternion(mjData.xquat as unknown as Float32Array, 1, body.quaternion);
    const perturb = new (mujoco as unknown as { MjvPerturb: new () => MjvPerturb }).MjvPerturb();
    const shown: Shown[] = [];
    const ctx = {
      sim: () => ({ mujoco, mjModel, mjData, dynamicBodyIds: new Set([1]), controlDt: 0.02, bodies: { 1: body } }),
      params: () => ({ stiffness: 100 }),
      wrench: new InteractionWrench(),
      perturb: () => perturb,
      arrow: {
        show: (from: THREE.Vector3, to: THREE.Vector3, force: number) =>
          shown.push({ from: from.clone(), to: to.clone(), force }),
        hide: () => {},
      },
    } as unknown as ModeContext;
    const hit = mjcToThreeCoordinate([mjData.xpos[3] + 0.05, mjData.xpos[4] + 0.1, mjData.xpos[5] + 0.02]);
    const gesture = {
      hit: { bodyId: 1, point: hit, normal: null, distance: 1 },
      ray: hit.clone().add(new THREE.Vector3(0.2, 0, 0)),
      direction: new THREE.Vector3(0, 0, -1),
      travel: 0,
    } as unknown as PointerGesture;

    const pull = new PullMode();
    expect(pull.onDown(gesture, ctx)).toBe('exclusive');
    pull.preStep(gesture, ctx);
    expect(shown).toEqual([]);

    pull.frame(gesture, ctx);
    expect(shown[0].from.distanceTo(hit)).toBeLessThan(1e-5);
    expect(shown[0].to.toArray()).toEqual(gesture.ray.toArray());
    expect(shown[0].force).toBeGreaterThan(0);

    // Drawn further on before the next control step: the arrow goes with it.
    body.position.x += 0.05;
    pull.frame(gesture, ctx);
    expect(shown[1].from.x - shown[0].from.x).toBeCloseTo(0.05, 6);
    expect(shown[1].force).toBe(shown[0].force);
  });

  it("Push's ring opens with the frames, not the control steps", () => {
    const updates: number[] = [];
    const ctx = {
      sim: () => ({ mujoco, mjData: null, controlDt: 0.1 }),
      ring: { update: (dt: number) => updates.push(dt) },
    } as unknown as ModeContext;
    const push = new PushMode();
    push.preStep(null, ctx);
    push.frame(null, ctx, 1 / 60);
    expect(updates).toEqual([1 / 60]);
  });
});
