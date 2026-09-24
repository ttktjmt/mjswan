import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { mjcToThreeCoordinate } from './coordinate';

interface CreateLightsParams {
  mujoco: MainModule;
  mjModel: MjModel;
  mujocoRoot: THREE.Group;
  bodies: Record<number, THREE.Group>;
}

interface HeadlightSpec {
  active?: boolean;
  ambient?: number[];
  diffuse?: number[];
  specular?: number[];
}

/** The wasm build exposes `mjModel.vis`; older builds spell it `visual`. */
function headlightSpec(mjModel: MjModel): HeadlightSpec | undefined {
  const model = mjModel as unknown as {
    vis?: { headlight?: HeadlightSpec };
    visual?: { headlight?: HeadlightSpec };
  };
  return (model.vis ?? model.visual)?.headlight;
}

/**
 * A three.js light shares one colour between diffuse and specular, so it cannot
 * carry MuJoCo's separate `light_specular`. Intensity follows `light_diffuse`
 * and the ratio goes into the material's `specularIntensity` instead.
 */
export function lightSpecularRatio(mjModel: MjModel): number {
  const peak = (channels: ArrayLike<number> | undefined, l: number): number =>
    channels ? Math.max(channels[l * 3], channels[l * 3 + 1], channels[l * 3 + 2]) : 0;

  let diffuse = 0;
  let specular = 0;

  for (let l = 0; l < (mjModel.nlight ?? 0); l++) {
    if (!mjModel.light_active?.[l]) {
      continue;
    }
    diffuse += peak(mjModel.light_diffuse, l);
    specular += peak(mjModel.light_specular, l);
  }

  const headlight = headlightSpec(mjModel);
  if (headlight?.active) {
    diffuse += Math.max(...(headlight.diffuse ?? [0, 0, 0]));
    specular += Math.max(...(headlight.specular ?? [0, 0, 0]));
  }

  return diffuse > 0 ? specular / diffuse : 1;
}

export function createLights({
  mujoco,
  mjModel,
  mujocoRoot,
}: CreateLightsParams): THREE.Light[] {
  const lights: THREE.Light[] = [];
  const ambientSum = new THREE.Color(0, 0, 0);

  if (mjModel.nlight > 0) {
    for (let l = 0; l < mjModel.nlight; l++) {
      if (!mjModel.light_active[l]) {
        continue;
      }

      const lightType = mjModel.light_type[l];
      let light: THREE.DirectionalLight | THREE.PointLight | THREE.SpotLight;

      switch (lightType) {
        case mujoco.mjtLightType.mjLIGHT_DIRECTIONAL.value:
          light = new THREE.DirectionalLight();
          mujocoRoot.add((light as THREE.DirectionalLight).target);
          break;
        case mujoco.mjtLightType.mjLIGHT_POINT.value:
          light = new THREE.PointLight();
          break;
        case mujoco.mjtLightType.mjLIGHT_SPOT.value:
          light = new THREE.SpotLight();
          mujocoRoot.add((light as THREE.SpotLight).target);
          break;
        case mujoco.mjtLightType.mjLIGHT_IMAGE.value:
          console.warn(`Skipping unsupported light type: mjLIGHT_IMAGE (light index ${l})`);
          continue;
        default:
          console.warn(`Skipping unknown light type: ${lightType} (light index ${l})`);
          continue;
      }

      light.userData.mjIndex = l;
      light.userData.mjType = lightType;

      const diffuseColor = new THREE.Color().fromArray(
        mjModel.light_diffuse.slice(l * 3, l * 3 + 3)
      );
      const luminance = Math.max(diffuseColor.r, diffuseColor.g, diffuseColor.b);

      if (luminance > 0) {
        light.color = diffuseColor.clone().multiplyScalar(1 / luminance);
      } else {
        light.color = new THREE.Color(0, 0, 0);
      }

      // three.js divides irradiance by pi, so the pi cancels back to MuJoCo's diffuse.
      if ('intensity' in light && typeof (light as { intensity?: number }).intensity === 'number') {
        (light as THREE.PointLight | THREE.DirectionalLight | THREE.SpotLight).intensity =
          luminance * Math.PI;
      }

      const ambientColor = new THREE.Color().fromArray(
        mjModel.light_ambient.slice(l * 3, l * 3 + 3)
      );
      ambientSum.add(ambientColor);

      light.castShadow = mjModel.light_castshadow[l];
      if (light.castShadow) {
        light.shadow!.mapSize.width = 1024;
        light.shadow!.mapSize.height = 1024;
        light.shadow!.camera.near = 1;
        light.shadow!.camera.far = 10;
        light.shadow!.radius = mjModel.light_bulbradius[l] * 50;
      }

      const pos = mjcToThreeCoordinate(mjModel.light_pos.slice(l * 3, l * 3 + 3)).toArray();
      const dir = mjcToThreeCoordinate(mjModel.light_dir.slice(l * 3, l * 3 + 3)).normalize();

      if (lightType === mujoco.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
        const len = Math.max(1, mjModel.light_range[l] || 10);
        const dl = light as THREE.DirectionalLight;
        dl.position.set(pos[0] - dir.x * len, pos[1] - dir.y * len, pos[2] - dir.z * len);
        dl.target.position.set(pos[0], pos[1], pos[2]);
      } else if (lightType === mujoco.mjtLightType.mjLIGHT_SPOT.value) {
        const sl = light as THREE.SpotLight;
        sl.position.set(pos[0], pos[1], pos[2]);
        sl.target.position.set(pos[0] + dir.x, pos[1] + dir.y, pos[2] + dir.z);
      } else {
        (light as THREE.PointLight).position.set(pos[0], pos[1], pos[2]);
      }

      if (lightType === mujoco.mjtLightType.mjLIGHT_SPOT.value) {
        (light as THREE.SpotLight).angle = (mjModel.light_cutoff[l] * Math.PI) / 180;
        const exponent = mjModel.light_exponent[l];
        (light as THREE.SpotLight).penumbra = 1 / (1 + exponent);
      }

      if (lightType !== mujoco.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
        const att = mjModel.light_attenuation.slice(l * 3, l * 3 + 3);

        if ('distance' in light) {
          (light as THREE.PointLight | THREE.SpotLight).distance = mjModel.light_range[l] ?? 0;
        }

        if (att[2] > 0) {
          if ('decay' in light) {
            (light as THREE.PointLight | THREE.SpotLight).decay = 2;
          }
        } else if (att[1] > 0) {
          if ('decay' in light) {
            (light as THREE.PointLight | THREE.SpotLight).decay = 1;
          }
        } else {
          if ('decay' in light) {
            (light as THREE.PointLight | THREE.SpotLight).decay = 0;
          }
        }
        if ('distance' in light) {
          (light as THREE.PointLight | THREE.SpotLight).distance = mjModel.light_range[l] ?? 0;
        }
      }

      mujocoRoot.add(light);
      lights.push(light);
    }
  }

  const headlight = headlightSpec(mjModel);
  if (headlight?.active) {
    ambientSum.add(new THREE.Color().fromArray(headlight.ambient as number[]));

    const headDiffuse = new THREE.Color().fromArray(headlight.diffuse as number[]);
    const headLuminance = Math.max(headDiffuse.r, headDiffuse.g, headDiffuse.b);

    const headLight = new THREE.DirectionalLight();
    headLight.color =
      headLuminance > 0
        ? headDiffuse.clone().multiplyScalar(1 / headLuminance)
        : new THREE.Color(0, 0, 0);
    headLight.intensity = headLuminance * Math.PI;
    headLight.castShadow = false;

    mujocoRoot.add(headLight.target);
    headLight.position.set(0, 0, 2);
    headLight.target.position.set(0, 0, 0);
    headLight.userData.isHeadlight = true;
    mujocoRoot.add(headLight);
    lights.push(headLight);
  }

  if (!ambientSum.equals(new THREE.Color(0, 0, 0))) {
    // Same pi as above: intensity 1 would land pi times too dark.
    const ambientLight = new THREE.AmbientLight(ambientSum, Math.PI);
    mujocoRoot.add(ambientLight);
    lights.push(ambientLight);
  }

  return lights;
}

export function updateLightsFromData(mujoco: MainModule, mjData: MjData, lights: THREE.Light[]): void {
  if (!mjData || !mjData.light_xpos || !mjData.light_xdir) {
    return;
  }

  for (const light of lights) {
    const userData = light.userData as { mjIndex?: number; mjType?: number } | undefined;
    const idx = userData?.mjIndex;
    const type = userData?.mjType;
    if (idx == null) {
      continue;
    }

    const posMJ = mjData.light_xpos.slice(idx * 3, idx * 3 + 3);
    const dirMJ = mjData.light_xdir.slice(idx * 3, idx * 3 + 3);

    const pos = mjcToThreeCoordinate(posMJ);
    const dir = mjcToThreeCoordinate(dirMJ).normalize();

    if (type === mujoco.mjtLightType.mjLIGHT_DIRECTIONAL.value) {
      const dl = light as THREE.DirectionalLight;
      const len = Math.max(1, (dl.shadow?.camera?.far as number) || 10);
      dl.target.position.copy(pos);
      dl.position.copy(pos).addScaledVector(dir, -len);
      dl.target.updateMatrixWorld?.();
    } else if (type === mujoco.mjtLightType.mjLIGHT_SPOT.value) {
      const sl = light as THREE.SpotLight;
      sl.position.copy(pos);
      sl.target.position.copy(pos.clone().add(dir));
      sl.target.updateMatrixWorld?.();
    } else if (type === mujoco.mjtLightType.mjLIGHT_POINT.value) {
      (light as THREE.PointLight).position.copy(pos);
    }
  }
}

export function updateHeadlightFromCamera(camera: THREE.Camera, lights: THREE.Light[]): void {
  const dir = new THREE.Vector3();
  const pos = new THREE.Vector3();
  for (const light of lights) {
    const userData = light.userData as { isHeadlight?: boolean } | undefined;
    if (!userData?.isHeadlight) {
      continue;
    }
    const dl = light as THREE.DirectionalLight;
    camera.getWorldDirection(dir);
    // World, not local: in XR the camera hangs off a rig that moves.
    camera.getWorldPosition(pos);
    dl.position.copy(pos);
    dl.target.position.copy(pos).add(dir);
    dl.target.updateMatrixWorld?.();
  }
}
