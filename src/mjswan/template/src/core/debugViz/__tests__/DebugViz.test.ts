/** `DebugViz` as mjlab's viser viewer builds the folder: "Enabled" over one switch per drawing. */
import * as THREE from 'three';
import { beforeAll, describe, expect, it } from 'vitest';

import type { CommandManager } from '../../command/CommandManager';
import { DebugViz, capitalize } from '../DebugViz';
import type { DebugVisConfig } from '../drawings';

type MainModule = import('mujoco').MainModule;
type MjModel = import('mujoco').MjModel;
type MjData = import('mujoco').MjData;

let mujoco: MainModule;
let model: MjModel;
let data: MjData;

beforeAll(async () => {
  mujoco = await (await import('mujoco')).default();
  model = (mujoco as unknown as { MjModel: { from_xml_string(s: string): MjModel } })
    .MjModel.from_xml_string(`<mujoco><worldbody>
      <geom type="plane" size="5 5 0.1"/>
      <body name="robot/trunk" pos="0 0 1"><joint type="free"/><geom size="0.1"/></body>
    </worldbody></mujoco>`);
  data = new (mujoco as unknown as { MjData: new (m: MjModel) => MjData }).MjData(model);
  mujoco.mj_forward(model, data);
});

/** The command side as the registry sees it: one drawing term, and what it was told. */
function commandsStub() {
  const calls = { shown: [] as boolean[], set: [] as [string, boolean][] };
  let enabled = true;
  const stub = {
    getDebugVisTerms: () => [{ name: 'base_velocity', enabled }],
    setDebugVisEnabled: (name: string, value: boolean) => {
      calls.set.push([name, value]);
      enabled = value;
    },
    updateDebugVisuals: (shown: boolean) => calls.shown.push(shown),
  };
  return { commands: stub as unknown as CommandManager, calls };
}

function sensor(debugVis: boolean) {
  return {
    kind: 'raycast' as const,
    local_offsets: [[0, 0, 0]],
    local_directions: [[0, 0, -1]],
    frames: [{ type: 'body' as const, name: 'robot/trunk' }],
    ray_alignment: 'yaw' as const,
    max_distance: 5,
    exclude_parent_body: true,
    debug_vis: debugVis,
    viz: {
      hit_color: [0, 1, 0, 0.8] as const,
      miss_color: [1, 0, 0, 0.4] as const,
      hit_sphere_color: [0, 1, 1, 1] as const,
      hit_sphere_radius: 0.5,
      show_rays: false,
      show_normals: false,
      normal_color: [1, 1, 0, 1] as const,
      normal_length: 5,
    },
  };
}

const CONFIG: DebugVisConfig = {
  meansize: 0.1,
  sensors: { foot_height_scan: sensor(true), terrain_scan: sensor(false) },
  rewards: {
    upright: { kind: 'upright', root_body: 'robot/trunk', body: null, terrain_sensors: ['terrain_scan'] },
  },
};

function hits(root: THREE.Object3D): number {
  return (root.getObjectByName('mjswan-sensor-foot_height_scan-hits') as THREE.InstancedMesh).count;
}

describe('DebugViz', () => {
  it("lists mjlab's switches in its order and with its labels", () => {
    const viz = new DebugViz(commandsStub().commands, () => ({ mjModel: model, mjData: data }));
    viz.load(CONFIG, mujoco, new THREE.Group());
    // A sensor only the reward reads is not listed, as mjlab lists `debug_vis` ones.
    expect(viz.entries()).toEqual([
      { id: 'command:base_velocity', label: 'Base_velocity', enabled: true },
      { id: 'sensor:foot_height_scan', label: 'foot_height_scan', enabled: true },
      { id: 'reward:upright', label: 'upright', enabled: true },
    ]);
  });

  it('hides every drawing under "Enabled", leaving each switch as it was', () => {
    const { commands, calls } = commandsStub();
    const viz = new DebugViz(commands, () => ({ mjModel: model, mjData: data }));
    const root = new THREE.Group();
    viz.load(CONFIG, mujoco, root);
    viz.update();
    expect(hits(root)).toBe(1);

    viz.setShown(false);
    expect(hits(root)).toBe(0);
    expect(calls.shown[calls.shown.length - 1]).toBe(false);
    expect(viz.entries().every((entry) => entry.enabled)).toBe(true);

    viz.setShown(true);
    expect(hits(root)).toBe(1);
  });

  it('toggles one drawing, a command term through its manager', () => {
    const { commands, calls } = commandsStub();
    const viz = new DebugViz(commands, () => ({ mjModel: model, mjData: data }));
    const root = new THREE.Group();
    viz.load(CONFIG, mujoco, root);

    viz.set('sensor:foot_height_scan', false);
    expect(hits(root)).toBe(0);
    viz.set('command:base_velocity', false);
    expect(calls.set).toEqual([['base_velocity', false]]);
    expect(viz.entries().map((entry) => entry.enabled)).toEqual([false, false, true]);
  });

  it('keeps "Enabled" across a policy load, and drops the old drawings', () => {
    const viz = new DebugViz(commandsStub().commands, () => ({ mjModel: model, mjData: data }));
    const root = new THREE.Group();
    viz.load(CONFIG, mujoco, root);
    viz.setShown(false);
    viz.load(undefined, mujoco, root);
    expect(viz.isShown()).toBe(false);
    expect(root.children).toHaveLength(0);
    expect(viz.entries().map((entry) => entry.id)).toEqual(['command:base_velocity']);
  });
});

describe('viewer switches', () => {
  function withSkin() {
    const viz = new DebugViz(commandsStub().commands, () => ({ mjModel: model, mjData: data }));
    const skin = { on: true, set: [] as boolean[] };
    viz.addSwitch('avatar:skin', {
      label: 'Skin',
      get: () => skin.on,
      set: (enabled) => {
        skin.set.push(enabled);
        skin.on = enabled;
      },
    });
    return { viz, skin };
  }

  it('lists a switch after the drawings, and sets it', () => {
    const { viz, skin } = withSkin();
    viz.load(CONFIG, mujoco, new THREE.Group());
    const last = () => viz.entries()[viz.entries().length - 1];
    expect(last()).toEqual({ id: 'avatar:skin', label: 'Skin', enabled: true });
    viz.set('avatar:skin', false);
    expect(skin.set).toEqual([false]);
    expect(last().enabled).toBe(false);
  });

  it('is not hidden by "Enabled", and outlives a policy load', () => {
    const { viz, skin } = withSkin();
    viz.setShown(false);
    expect(skin.set).toEqual([]);
    viz.load(undefined, mujoco, new THREE.Group());
    expect(viz.entries().map((entry) => entry.id)).toEqual(['command:base_velocity', 'avatar:skin']);
    viz.removeSwitch('avatar:skin');
    expect(viz.entries().map((entry) => entry.id)).toEqual(['command:base_velocity']);
  });
});

describe('capitalize', () => {
  it("is Python's str.capitalize", () => {
    expect(capitalize('twist')).toBe('Twist');
    expect(capitalize('base_velocity')).toBe('Base_velocity');
    expect(capitalize('liftHeight')).toBe('Liftheight');
  });
});
