/**
 * `CameraSensorSet`: which model camera a sensor's slot renders, at what size, and
 * what comes back to the graph. The render itself is the engine's, stubbed here.
 */
import { beforeAll, describe, expect, it, vi } from 'vitest';

import { CameraSensorSet, type CameraSensorDescriptor } from '../camera';
import type { SensorCameraView } from '../../scene/sensorCamera';

type MainModule = import('mujoco').MainModule;

const SCENE = `<mujoco>
  <worldbody>
    <camera name="side" pos="2 0 0.5" xyaxes="0 1 0 0 0 1"/>
    <camera name="robot/camera_d405" pos="0 0 1"/>
  </worldbody>
</mujoco>`;

let mjModel: import('mujoco').MjModel;

beforeAll(async () => {
  const load = (await import('mujoco')).default;
  const mujoco: MainModule = await load();
  mjModel = (mujoco as unknown as { MjModel: { from_xml_string(s: string): never } })
    .MjModel.from_xml_string(SCENE);
});

function descriptor(over: Partial<CameraSensorDescriptor> = {}): CameraSensorDescriptor {
  return {
    kind: 'camera',
    camera: 'robot/camera_d405',
    width: 2,
    height: 1,
    enabled_geom_groups: [0, 3],
    use_shadows: false,
    ...over,
  };
}

/** Two pixels, red then green, as one RGBA row. */
const RED_GREEN = Uint8Array.from([255, 0, 0, 255, 0, 255, 0, 255]);

describe('CameraSensorSet', () => {
  it('renders the named camera at the sensor\'s size and serves mjlab\'s [H, W, 3]', () => {
    const render = vi.fn((_view: SensorCameraView) => RED_GREEN);
    const set = new CameraSensorSet({ camera_d405: descriptor() }, render);

    const rgb = set.read('camera_d405', 'rgb', mjModel);

    expect(render).toHaveBeenCalledWith({
      cameraId: 1,
      width: 2,
      height: 1,
      enabledGeomGroups: [0, 3],
      useShadows: false,
    });
    expect(Array.from(rgb ?? [])).toEqual([255, 0, 0, 0, 255, 0]);
  });

  it('finds a camera the model names without the entity prefix', () => {
    const render = vi.fn((_view: SensorCameraView) => RED_GREEN);
    const set = new CameraSensorSet({ cam: descriptor({ camera: 'robot/side' }) }, render);

    set.read('cam', 'rgb', mjModel);

    expect(render.mock.calls[0][0].cameraId).toBe(0);
  });

  it('serves rgb only, and nothing for a camera the model lacks', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const render = vi.fn((_view: SensorCameraView) => RED_GREEN);
    const set = new CameraSensorSet(
      { cam: descriptor(), lost: descriptor({ camera: 'nowhere' }) },
      render,
    );

    expect(set.read('cam', 'depth', mjModel)).toBeNull();
    expect(set.read('lost', 'rgb', mjModel)).toBeNull();
    expect(render).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('answers only for its own sensors, and lists the groups they draw', () => {
    const set = new CameraSensorSet({
      a: descriptor({ enabled_geom_groups: [0, 3] }),
      b: descriptor({ enabled_geom_groups: [2, 3] }),
    });

    expect(set.has('a')).toBe(true);
    expect(set.has('height_scan')).toBe(false);
    expect(set.geomGroups().sort()).toEqual([0, 2, 3]);
  });
});
