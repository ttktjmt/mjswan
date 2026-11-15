import * as THREE from 'three';
import type { MjData, MjModel } from 'mujoco';
import { getPosition } from './scene';

export interface TendonMeshes {
  cylinders: THREE.InstancedMesh;
  spheres: THREE.InstancedMesh;
}

/** The tendon paths as the last snapshot left them. */
export interface TendonState {
  wrapXpos: Float64Array;
  wrapObj: Int32Array;
  wrapadr: Int32Array;
  wrapnum: Int32Array;
  stale: boolean;
}

/** A pulley's `wrap_obj`: its point is the origin, joining nothing. */
const PULLEY = -2;

const UP = new THREE.Vector3(0, 1, 0);
const IDENTITY = new THREE.Quaternion();
const start = new THREE.Vector3();
const end = new THREE.Vector3();
const middle = new THREE.Vector3();
const direction = new THREE.Vector3();
const scale = new THREE.Vector3();
const rotation = new THREE.Quaternion();
const matrix = new THREE.Matrix4();

export function createTendonMeshes(mujocoRoot: THREE.Group, mjModel: MjModel): TendonMeshes {
  const tendonMat = new THREE.MeshPhongMaterial();
  tendonMat.color = new THREE.Color(0.8, 0.3, 0.3);

  // A geom a tendon wraps puts two points on its path, so the paths hold up to 2 * nwrap.
  const capacity = 2 * mjModel.nwrap;

  const cylinders = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1), tendonMat, capacity);
  cylinders.receiveShadow = true;
  cylinders.castShadow = true;
  cylinders.count = 0;
  cylinders.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  cylinders.computeBoundingSphere();
  mujocoRoot.add(cylinders);

  const spheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 10), tendonMat, capacity);
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

/** Copies the paths `mjData` holds: cheap enough for every physics step. */
export function snapshotTendons(mjData: MjData, state: TendonState): void {
  state.wrapXpos = copy(mjData.wrap_xpos, state.wrapXpos);
  state.wrapObj = copy(mjData.wrap_obj, state.wrapObj);
  state.wrapadr = copy(mjData.ten_wrapadr, state.wrapadr);
  state.wrapnum = copy(mjData.ten_wrapnum, state.wrapnum);
  state.stale = true;
}

/** Poses the meshes from the last snapshot, once however many steps the loop took since. */
export function updateTendonRendering(mjModel: MjModel, tendonMeshes: TendonMeshes, state: TendonState): void {
  if (!tendonMeshes.cylinders || !state.stale) {
    return;
  }
  state.stale = false;

  const { wrapXpos, wrapObj, wrapadr, wrapnum } = state;
  const width = mjModel.tendon_width;
  let cylinders = 0;
  let spheres = 0;
  for (let t = 0; t < wrapnum.length; t++) {
    const r = width[t];
    const last = wrapadr[t] + wrapnum[t] - 1;
    for (let w = wrapadr[t]; w <= last; w++) {
      if (wrapObj[w] === PULLEY) continue;
      getPosition(wrapXpos, w, start);
      tendonMeshes.spheres.setMatrixAt(spheres++, matrix.compose(start, IDENTITY, scale.set(r, r, r)));
      if (w === last || wrapObj[w + 1] === PULLEY) continue;
      getPosition(wrapXpos, w + 1, end);
      matrix.compose(
        middle.addVectors(start, end).multiplyScalar(0.5),
        rotation.setFromUnitVectors(UP, direction.subVectors(end, start).normalize()),
        scale.set(r, start.distanceTo(end), r)
      );
      tendonMeshes.cylinders.setMatrixAt(cylinders++, matrix);
    }
  }

  tendonMeshes.cylinders.count = cylinders;
  tendonMeshes.spheres.count = spheres;
  tendonMeshes.cylinders.instanceMatrix.needsUpdate = true;
  tendonMeshes.spheres.instanceMatrix.needsUpdate = true;

  if (cylinders > 0) {
    tendonMeshes.cylinders.computeBoundingSphere();
    tendonMeshes.spheres.computeBoundingSphere();
  }
}

export function createTendonState(): TendonState {
  return {
    wrapXpos: new Float64Array(0),
    wrapObj: new Int32Array(0),
    wrapadr: new Int32Array(0),
    wrapnum: new Int32Array(0),
    stale: false,
  };
}

/** `source` into `target`, reallocated only when the length changes. */
function copy<T extends Float64Array | Int32Array>(source: ArrayLike<number>, target: T): T {
  if (target.length !== source.length) target = new (target.constructor as new (n: number) => T)(source.length);
  target.set(source);
  return target;
}
