import * as THREE from 'three';
import type { MjData, MjModel } from 'mujoco';
import { getPosition } from './scene';

export interface TendonMeshes {
  cylinders: THREE.InstancedMesh;
  spheres: THREE.InstancedMesh;
}

/** Every tendon's colour unless painted, linear RGB. */
export const TENDON_RGB = [0.8, 0.3, 0.3] as const;

/** Per-tendon colours, for a model whose muscles are drawn their own way (`Muscles`). */
export interface TendonPaint {
  /** Linear RGB per tendon. */
  rgb: Float32Array;
  /** 1 for a tendon drawn elsewhere, as a muscle tube. */
  hidden: Uint8Array;
}

export interface TendonState {
  numWraps: number;
  matrix: THREE.Matrix4;
  color: THREE.Color;
}

export function createTendonMeshes(mujocoRoot: THREE.Group, mjModel: MjModel): TendonMeshes {
  const tendonMat = new THREE.MeshPhongMaterial();
  tendonMat.color.setRGB(...TENDON_RGB);

  // maxCylinders: safe upper bound (actual value is nwrap - number of spatial tendons or less)
  // maxSpheres   = total number of wrap points = nwrap (this is the official correct value)
  const maxCylinders = Math.max(0, mjModel.nwrap);
  const maxSpheres   = Math.max(0, mjModel.nwrap);

  const cylinders = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1), tendonMat, maxCylinders);
  cylinders.receiveShadow = true;
  cylinders.castShadow = true;
  cylinders.count = 0;
  cylinders.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cylinders.computeBoundingSphere();
  mujocoRoot.add(cylinders);

  const spheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10), tendonMat, maxSpheres);
  spheres.receiveShadow = true;
  spheres.castShadow = true;
  spheres.count = 0;
  spheres.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  spheres.computeBoundingSphere();
  mujocoRoot.add(spheres);

  mujocoRoot.cylinders = cylinders;
  mujocoRoot.spheres = spheres;

  return { cylinders, spheres };
}

export function updateTendonGeometry(
  mjModel: MjModel,
  mjData: MjData,
  tendonMeshes: TendonMeshes,
  tendonState: TendonState,
  paint?: TendonPaint | null
): void {
  if (!tendonMeshes.cylinders) {
    return;
  }

  let numWraps = 0;
  const mat = tendonState.matrix;
  const color = tendonState.color;
  if (paint) (tendonMeshes.cylinders.material as THREE.MeshPhongMaterial).color.setRGB(1, 1, 1);

  for (let t = 0; t < mjModel.ntendon; t++) {
    if (paint?.hidden[t]) continue;
    if (paint) color.fromArray(paint.rgb, 3 * t);
    const startW = mjData.ten_wrapadr[t];
    const r = mjModel.tendon_width[t];

    for (let w = startW; w < startW + mjData.ten_wrapnum[t] - 1; w++) {
      const tendonStart = getPosition(mjData.wrap_xpos, w, new THREE.Vector3());
      const tendonEnd = getPosition(mjData.wrap_xpos, w + 1, new THREE.Vector3());
      const tendonAvg = new THREE.Vector3().addVectors(tendonStart, tendonEnd).multiplyScalar(0.5);

      const validStart = tendonStart.length() > 0.01;
      const validEnd = tendonEnd.length() > 0.01;

      if (validStart) {
        tendonMeshes.spheres.setMatrixAt(
          numWraps,
          mat.compose(tendonStart, new THREE.Quaternion(), new THREE.Vector3(r, r, r))
        );
        if (paint) tendonMeshes.spheres.setColorAt(numWraps, color);
      }

      if (validEnd) {
        tendonMeshes.spheres.setMatrixAt(
          numWraps + 1,
          mat.compose(tendonEnd, new THREE.Quaternion(), new THREE.Vector3(r, r, r))
        );
        if (paint) tendonMeshes.spheres.setColorAt(numWraps + 1, color);
      }

      if (validStart && validEnd) {
        mat.compose(
          tendonAvg,
          new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            tendonEnd.clone().sub(tendonStart).normalize()
          ),
          new THREE.Vector3(r, tendonStart.distanceTo(tendonEnd), r)
        );
        tendonMeshes.cylinders.setMatrixAt(numWraps, mat);
        if (paint) tendonMeshes.cylinders.setColorAt(numWraps, color);
        numWraps++;
      }
    }
  }

  tendonState.numWraps = numWraps;
}

export function updateTendonRendering(tendonMeshes: TendonMeshes, tendonState: TendonState): void {
  if (!tendonMeshes.cylinders) {
    return;
  }

  const numWraps = tendonState.numWraps;
  tendonMeshes.cylinders.count = numWraps;
  tendonMeshes.spheres.count = numWraps > 0 ? numWraps + 1 : 0;
  tendonMeshes.cylinders.instanceMatrix.needsUpdate = true;
  tendonMeshes.spheres.instanceMatrix.needsUpdate = true;
  if (tendonMeshes.cylinders.instanceColor) tendonMeshes.cylinders.instanceColor.needsUpdate = true;
  if (tendonMeshes.spheres.instanceColor) tendonMeshes.spheres.instanceColor.needsUpdate = true;

  if (numWraps > 0) {
    tendonMeshes.cylinders.computeBoundingSphere();
    tendonMeshes.spheres.computeBoundingSphere();
  }
}

export function createTendonState(): TendonState {
  return {
    numWraps: 0,
    matrix: new THREE.Matrix4(),
    color: new THREE.Color(),
  };
}
