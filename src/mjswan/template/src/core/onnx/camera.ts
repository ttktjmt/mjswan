/**
 * Serves mjlab's `CameraSensor` `rgb` field: the engine renders the model's camera at the
 * sensor's size (`scene/sensorCamera.ts`), and the term's own arithmetic (the scale to
 * [0, 1], the move to channels first) stays in the traced graph.
 */

import { rgbFromPixels, type SensorCameraView } from '../scene/sensorCamera';
import { decodeNames, unprefixed } from './slotReader/indexing';

type MjModel = import('mujoco').MjModel;

/** Everything the build sends about one camera sensor; the camera itself is the model's. */
export interface CameraSensorDescriptor {
  kind: 'camera';
  /** The model's camera name, which the sensor wraps or added. */
  camera: string;
  width: number;
  height: number;
  enabled_geom_groups: number[];
  use_shadows: boolean;
}

/** Draws one view: `width * height * 4` RGBA bytes, bottom row first. */
export type RenderCameraView = (view: SensorCameraView) => Uint8Array | null;

export class CameraSensorSet {
  private model: MjModel | null = null;
  private readonly cameraIds = new Map<string, number>();

  constructor(
    private readonly descriptors: Record<string, CameraSensorDescriptor> = {},
    private readonly render: RenderCameraView | null = null,
  ) {}

  get size(): number {
    return Object.keys(this.descriptors).length;
  }

  has(sensor: string): boolean {
    return sensor in this.descriptors;
  }

  /** Every geom group some sensor draws: the scene builds those the viewer skips. */
  geomGroups(): number[] {
    const groups = new Set<number>();
    for (const descriptor of Object.values(this.descriptors)) {
      for (const group of descriptor.enabled_geom_groups) groups.add(group);
    }
    return [...groups];
  }

  /** The sensor's image as mjlab's `[H, W, 3]` pixel values, or null if this set has none. */
  read(sensor: string, field: string, mjModel: MjModel): Float32Array | null {
    const descriptor = this.descriptors[sensor];
    if (!descriptor || !this.render) return null;
    if (field !== 'rgb') {
      console.warn(`[camera] sensor "${sensor}" cannot serve "${field}"; only rgb renders.`);
      return null;
    }
    const cameraId = this.cameraId(mjModel, descriptor.camera);
    if (cameraId < 0) return null;
    const { width, height } = descriptor;
    const pixels = this.render({
      cameraId,
      width,
      height,
      enabledGeomGroups: descriptor.enabled_geom_groups,
      useShadows: descriptor.use_shadows,
    });
    return pixels ? rgbFromPixels(pixels, width, height) : null;
  }

  /** The build names it as mjlab's scene does (`robot/camera_d405`); try the bare name too. */
  private cameraId(mjModel: MjModel, camera: string): number {
    if (mjModel !== this.model) {
      this.model = mjModel;
      this.cameraIds.clear();
    }
    let id = this.cameraIds.get(camera);
    if (id === undefined) {
      const names = decodeNames(mjModel, mjModel.ncam, mjModel.name_camadr);
      id = names.indexOf(camera);
      if (id < 0) {
        const bare = unprefixed(camera);
        id = names.findIndex((name) => name === bare || unprefixed(name) === bare);
      }
      if (id < 0) console.warn(`[camera] the model has no camera "${camera}".`);
      this.cameraIds.set(camera, id);
    }
    return id;
  }
}
