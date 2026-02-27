import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

type ColliderMesh = THREE.Group;

export async function loadCollider(
  url: string,
  scene: THREE.Scene
): Promise<ColliderMesh> {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(url);
  const root = gltf.scene;

  root.traverse((obj: THREE.Object3D) => {
    obj.visible = false;
  });

  scene.add(root);
  return root;
}

export function disposeCollider(collider: ColliderMesh, scene: THREE.Scene): void {
  scene.remove(collider);
  collider.traverse((obj: THREE.Object3D) => {
    const mesh = obj as THREE.Mesh;
    mesh.geometry?.dispose();
    if (Array.isArray(mesh.material)) mesh.material.forEach((m: THREE.Material) => m.dispose());
    else (mesh.material as THREE.Material | undefined)?.dispose();
  });
}
