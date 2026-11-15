import * as THREE from 'three';
import { beforeAll, describe, expect, it, vi } from 'vitest';

import { TENDON_RGB } from '../../scene/tendons';
import { DEFAULT_MUSCLE_VIEW, mergeMuscleView, Muscles } from '../muscles';
import { MUSCLE_UNIFORM, PATH_ACTIVE, PATH_RELAXED } from '../tubes';
import fixture from './fixtures/tubes.json';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

let mujoco: MainModule;

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
});

function compile(xml: string): { model: MjModel; data: MjData } {
  const model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(xml);
  const data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
  mujoco.mj_forward(model, data);
  return { model, data };
}

// Only `flexor` is a muscle MyoSuite draws: `ligament` is passive, `fixed` has one element
// and the last muscle acts on a joint.
const ARM = `<mujoco><worldbody>
  <body><geom size="0.05"/><site name="origin" pos="0 0 0.1"/>
    <body pos="0 0 0.3"><joint name="elbow" axis="0 1 0" range="-90 90"/><geom size="0.05"/><site name="insertion" pos="0.05 0 0"/></body>
  </body></worldbody>
  <tendon>
    <spatial name="ligament"><site site="origin"/><site site="insertion"/></spatial>
    <spatial name="flexor"><site site="origin"/><site site="insertion"/></spatial>
    <fixed name="fixed"><joint joint="elbow" coef="1"/></fixed>
  </tendon>
  <actuator><muscle tendon="flexor"/><muscle tendon="fixed"/><muscle joint="elbow"/></actuator>
</mujoco>`;
const LIGAMENT = 0;
const FLEXOR = 1;

const rgbOf = (muscles: Muscles, tendon: number) => Array.from(muscles.paint.rgb.subarray(3 * tendon, 3 * tendon + 3));
const close = (actual: number[], expected: readonly number[]) =>
  actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 6));

describe('Muscles', () => {
  it("draws only MyoSuite's muscles, coloured by activation along their path", () => {
    const { model, data } = compile(ARM);
    const muscles = Muscles.create(mujoco, model, new THREE.Group())!;
    expect(muscles.count).toBe(1);

    muscles.setView({ ...DEFAULT_MUSCLE_VIEW, style: 'paths', color: 'activation' });
    data.act[model.actuator_actadr[0]] = 0.5;
    muscles.update(data);
    close(rgbOf(muscles, FLEXOR), PATH_RELAXED.map((r, c) => r + (PATH_ACTIVE[c] - r) * 0.5 ** 0.25));
    close(rgbOf(muscles, LIGAMENT), TENDON_RGB);

    muscles.setView({ ...DEFAULT_MUSCLE_VIEW, style: 'paths', color: 'uniform' });
    muscles.update(data);
    close(rgbOf(muscles, FLEXOR), MUSCLE_UNIFORM);

    muscles.setView(null);
    muscles.update(data);
    close(rgbOf(muscles, FLEXOR), TENDON_RGB);
    model.delete();
    data.delete();
  });

  it('draws a tube per pulley branch in place of each muscle tendon', () => {
    const { model, data } = compile(fixture.xml);
    const root = new THREE.Group();
    const muscles = Muscles.create(mujoco, model, root)!;
    const tendons = fixture.actuators.map((a) => model.actuator_trnid[2 * a]);
    const vertices = () => {
      muscles.update(data);
      muscles.commit();
      const mesh = root.getObjectByProperty('type', 'Mesh') as THREE.Mesh;
      return mesh.geometry.getAttribute('position');
    };

    muscles.setView(DEFAULT_MUSCLE_VIEW);
    expect(tendons.every((t) => muscles.paint.hidden[t] === 1)).toBe(true);
    // Six tubes: the extensor's three branches and one for each other muscle.
    const full = vertices();
    expect(full.count).toBe(6 * (32 * 12 + 2));
    const nearest = Math.min(...Array.from({ length: full.count }, (_, v) => Math.hypot(full.getX(v), full.getY(v), full.getZ(v))));
    expect(nearest).toBeGreaterThan(0.03);

    muscles.setView({ ...DEFAULT_MUSCLE_VIEW, detail: 'light' });
    expect(vertices().count).toBe(6 * (32 * 6 + 2));

    muscles.setView(null);
    expect(root.children.every((child) => !child.visible)).toBe(true);
    expect(tendons.every((t) => muscles.paint.hidden[t] === 0)).toBe(true);
    muscles.dispose();
    expect(root.children).toHaveLength(0);
    model.delete();
    data.delete();
  });

  it('is none for a model without a muscle to draw', () => {
    const { model, data } = compile('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>');
    expect(Muscles.create(mujoco, model, new THREE.Group())).toBeNull();
    model.delete();
    data.delete();
  });
});

describe('mergeMuscleView', () => {
  it('keeps the base where a value is left out or unknown', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const view = mergeMuscleView(DEFAULT_MUSCLE_VIEW, { color: 'uniform', style: 'tubes' as 'paths', scale: -1 });
    expect(view).toEqual({ ...DEFAULT_MUSCLE_VIEW, color: 'uniform' });
    // The state hands the view out: a host writing to it must not change the drawing.
    expect(Object.isFrozen(view)).toBe(true);
    expect(warn).toHaveBeenCalledTimes(2);
    warn.mockRestore();
  });
});
