import * as THREE from 'three';

export function mjcToThreeCoordinate(v: ArrayLike<number>): THREE.Vector3 {
  return new THREE.Vector3(v[0], v[2], -v[1]);
}

export function threeToMjcCoordinate(v: THREE.Vector3): THREE.Vector3 {
  return new THREE.Vector3(v.x, -v.z, v.y);
}
