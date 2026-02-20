import * as THREE from 'three';
import type { MjData, MjModel } from 'mujoco';
import { getPosition } from './scene';

export interface TendonMeshes {
  cylinders: THREE.InstancedMesh;
  spheres: THREE.InstancedMesh;
}

export interface TendonState {
  numWraps: number;
  matrix: THREE.Matrix4;
}

export function createTendonMeshes(mujocoRoot: THREE.Group): TendonMeshes {
  const tendonMat = new THREE.MeshPhongMaterial();
  tendonMat.color = new THREE.Color(0.8, 0.3, 0.3);

  const cylinders = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1), tendonMat, 1023);
  cylinders.receiveShadow = true;
  cylinders.castShadow = true;
  cylinders.count = 0;
  cylinders.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cylinders.computeBoundingSphere();
  mujocoRoot.add(cylinders);

  const spheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10), tendonMat, 1023);
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
  tendonState: TendonState
): void {
  if (!tendonMeshes.cylinders) {
    return;
  }

  let numWraps = 0;
  const mat = tendonState.matrix;

  for (let t = 0; t < mjModel.ntendon; t++) {
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
      }

      if (validEnd) {
        tendonMeshes.spheres.setMatrixAt(
          numWraps + 1,
          mat.compose(tendonEnd, new THREE.Quaternion(), new THREE.Vector3(r, r, r))
        );
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

  if (numWraps > 0) {
    tendonMeshes.cylinders.computeBoundingSphere();
    tendonMeshes.spheres.computeBoundingSphere();
  }
}

export function createTendonState(): TendonState {
  return {
    numWraps: 0,
    matrix: new THREE.Matrix4(),
  };
}
