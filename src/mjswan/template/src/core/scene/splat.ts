import * as THREE from 'three';
import { SplatMesh } from '@sparkjsdev/spark';
export type { SplatMesh };

export interface SplatConfig {
  url: string;
  scale?: number;
  groundOffset?: number;
  colliderUrl?: string;
  /** If true, shows scale and ground offset controls in the viewer control panel. */
  control?: boolean;
}

export function loadSplat(config: SplatConfig, scene: THREE.Scene): SplatMesh {
  const splat = new SplatMesh({ url: config.url });

  const scale = config.scale ?? 1.0;
  const groundOffset = config.groundOffset ?? 0.0;

  splat.scale.setScalar(scale);

  // WorldLabs splats use COLMAP/OpenCV convention (Y-down, Z-into-scene).
  // Rotating 180° around X flips to Three.js convention (Y-up, Z-towards-viewer).
  splat.rotation.x = Math.PI;

  // After the flip, shift up to align ground with Y = 0.
  splat.position.y = groundOffset * scale;

  scene.add(splat);
  return splat;
}

export function disposeSplat(splat: SplatMesh, scene: THREE.Scene): void {
  scene.remove(splat);
  splat.dispose?.();
}
