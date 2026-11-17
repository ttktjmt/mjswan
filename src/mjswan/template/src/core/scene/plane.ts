/**
 * A plane geom as MuJoCo draws it (`render_context.c`'s `makePlane`, `render_gl3.c`'s
 * `settexture`, `engine_vis_visualize.c`): an infinite side as a sheet wider than the
 * skybox, moved under the viewer every frame by whole texture periods, so it never ends
 * and its texture never slides, and the texture placed by MuJoCo's `GL_OBJECT_PLANE`
 * mapping rather than stretched over the sheet.
 */
import * as THREE from 'three';
import type { MjModel } from 'mujoco';

/** `mjMAXPLANEGRID`: the grid squares an infinite side is divided into. */
const MAX_PLANE_GRID = 200;

/** MuJoCo's far plane, `vis.map.zfar` times `stat.extent`; read once per model, as each read of `vis` makes a handle. */
export function modelZfar(mjModel: MjModel): number {
  return mjModel.vis.map.zfar * mjModel.stat.extent;
}

/** How a plane with an infinite side follows the viewer: where it was put, and its stride per side (0: finite). */
export type PlaneRecentre = { origin: THREE.Vector3; step: [number, number] };

/** Geom `g`'s plane in three's frame (MuJoCo's x and y on three's x and -z), lit like any geom. */
export function planeGeometry(
  mjModel: MjModel,
  g: number,
  zfar: number,
): { geometry: THREE.BufferGeometry; step: [number, number] } {
  const matId = mjModel.geom_matid[g];
  const uniform = matId >= 0 && !!mjModel.mat_texuniform[matId];
  const half: number[] = [];
  const scale: number[] = [];
  const step: [number, number] = [0, 0];
  for (let k = 0; k < 2; k++) {
    const size = mjModel.geom_size[g * 3 + k];
    const repeat = matId >= 0 ? mjModel.mat_texrepeat[matId * 2 + k] : 0;
    if (size > 0) {
      half.push(size);
    } else {
      // A grid square per texture period, so moving two keeps the texture in place.
      const square = repeat > 0 ? 2 / repeat : (2.1 * zfar) / (MAX_PLANE_GRID - 2);
      half.push(1.05 * zfar + square);
      step[k] = 2 * square;
    }
    scale.push(size > 0 && !uniform ? repeat / size : repeat);
  }

  const geometry = new THREE.PlaneGeometry(half[0] * 2, half[1] * 2);
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.getAttribute('position');
  const uv = geometry.getAttribute('uv');
  for (let i = 0; i < position.count; i++) {
    const x = position.getX(i);
    const y = -position.getZ(i);
    uv.setXY(i, 0.5 * scale[0] * x - 0.5, -0.5 * scale[1] * y - 0.5);
  }
  return { geometry, step };
}

const X_AXIS = new THREE.Vector3(1, 0, 0);
/** MuJoCo's y in three's frame. */
const Y_AXIS = new THREE.Vector3(0, 0, -1);
const eyeLocal = new THREE.Vector3();
const axis = new THREE.Vector3();

/** Move a plane's infinite sides under `eye` (world), a whole stride at a time, as MuJoCo re-centres one. */
export function recentrePlane(mesh: THREE.Object3D, eye: THREE.Vector3): void {
  const recentre = mesh.userData.recentre as PlaneRecentre | undefined;
  const parent = mesh.parent;
  if (!recentre || !parent) return;
  parent.updateWorldMatrix(true, false);
  const offset = parent.worldToLocal(eyeLocal.copy(eye)).sub(recentre.origin);
  mesh.position.copy(recentre.origin);
  for (let k = 0; k < 2; k++) {
    const step = recentre.step[k];
    if (step === 0) continue;
    axis.copy(k === 0 ? X_AXIS : Y_AXIS).applyQuaternion(mesh.quaternion);
    mesh.position.addScaledVector(axis, step * Math.round(offset.dot(axis) / step));
  }
}
