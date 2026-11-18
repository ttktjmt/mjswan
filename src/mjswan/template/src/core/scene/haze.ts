/**
 * MuJoCo's haze (`mjRND_HAZE`): a cone around the viewer at skybox distance, from the
 * first infinite plane up to eye level, clear at both ends and in the model's haze
 * colour between, where the plane meets the sky. MuJoCo draws it with the skybox alone,
 * and none from below the plane.
 */
import * as THREE from 'three';

/**
 * `render_context.c`'s `haze`, in three's frame: radius 1 at eye level (y = 0), the
 * plane at y = -1. `ratio` is `vis.map.haze`, the share of the horizon it covers.
 */
export function hazeGeometry(slices: number, ratio: number, rgb: ArrayLike<number>): THREE.BufferGeometry {
  // The height the haze is thickest at.
  const alpha = Math.atan2(1, ratio);
  const peak = (Math.sqrt(0.5) * ratio * Math.sin(alpha)) / Math.sin(0.75 * Math.PI - alpha);
  const rings = [
    { h: 0, a: 0 },
    { h: peak, a: 1 },
    { h: 1, a: 0 },
  ];
  const positions: number[] = [];
  const colors: number[] = [];
  for (const { h, a } of rings) {
    const radius = 1 - ratio * (1 - h);
    for (let j = 0; j <= slices; j++) {
      const az = (2 * Math.PI * j) / slices;
      positions.push(Math.cos(az) * radius, h - 1, -Math.sin(az) * radius);
      colors.push(rgb[0], rgb[1], rgb[2], a);
    }
  }
  const indices: number[] = [];
  for (let ring = 0; ring < rings.length - 1; ring++) {
    for (let j = 0; j < slices; j++) {
      const a = ring * (slices + 1) + j;
      const b = a + slices + 1;
      indices.push(a, a + 1, b + 1, a, b + 1, b);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
  geometry.setIndex(indices);
  return geometry;
}

export function createHaze(slices: number, ratio: number, rgb: ArrayLike<number>): THREE.Mesh {
  const haze = new THREE.Mesh(
    hazeGeometry(Math.max(3, slices), ratio, rgb),
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    }),
  );
  haze.name = 'Haze';
  // With the skybox, before any transparent geom; standing at the eye, it would sort last.
  haze.renderOrder = -1;
  haze.raycast = () => {};
  haze.userData.unbounded = true;
  return haze;
}

const normal = new THREE.Vector3();
const origin = new THREE.Vector3();
const above = new THREE.Vector3();

/**
 * Stand the haze on `plane` around `eye` (world), `distance` out. False below the plane,
 * where MuJoCo draws none.
 */
export function placeHaze(
  haze: THREE.Object3D,
  plane: THREE.Object3D,
  eye: THREE.Vector3,
  distance: number,
): boolean {
  plane.updateWorldMatrix(true, false);
  normal.set(0, 1, 0).transformDirection(plane.matrixWorld);
  origin.setFromMatrixPosition(plane.matrixWorld);
  const height = above.copy(eye).sub(origin).dot(normal);
  if (height < 0) return false;
  haze.position.copy(eye);
  haze.quaternion.setFromRotationMatrix(plane.matrixWorld);
  haze.scale.set(distance, height, distance);
  return true;
}
