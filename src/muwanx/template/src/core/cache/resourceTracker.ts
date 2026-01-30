import * as THREE from 'three';
import type { MjData, MjModel } from '@/mujoco';
import type { Mujoco } from '../../types/mujoco';

/**
 * Snapshot of resources loaded for a scene
 */
export interface ResourceSnapshot {
  bodies: Record<number, THREE.Group>;
  lights: THREE.Light[];
  meshes: Record<number, THREE.BufferGeometry>;
  mujocoRoot: THREE.Group;
  fsFiles: string[];
  estimatedMemoryBytes: number;
}

/**
 * Tracks scene resources and estimates memory usage
 */
export class SceneResourceTracker {
  private fsWriteLog: string[] = [];
  private isTracking: boolean = false;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private originalWriteFile?: any;

  /**
   * Start tracking FS writes
   */
  public startTracking(mujoco: Mujoco): void {
    if (this.isTracking) {
      console.warn('[ResourceTracker] Already tracking FS writes');
      return;
    }

    this.fsWriteLog = [];
    this.isTracking = true;

    // Wrap FS.writeFile to track writes
    this.originalWriteFile = mujoco.FS.writeFile;
    const fsWriteLog = this.fsWriteLog;
    const originalWriteFile = this.originalWriteFile;

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    mujoco.FS.writeFile = function (path: string, data: string | ArrayBufferView, opts?: any): void {
      // Track writes to /working/
      if (path.startsWith('/working/')) {
        const relativePath = path.replace(/^\/working\//, '');
        if (!fsWriteLog.includes(relativePath)) {
          fsWriteLog.push(relativePath);
        }
      }

      // Call original function
      if (originalWriteFile) {
        return originalWriteFile.call(mujoco.FS, path, data, opts);
      }
    };
  }

  /**
   * Stop tracking and return captured files
   */
  public stopTracking(mujoco: Mujoco): string[] {
    if (!this.isTracking) {
      return [];
    }

    // Restore original function
    if (this.originalWriteFile) {
      mujoco.FS.writeFile = this.originalWriteFile;
      this.originalWriteFile = undefined;
    }

    this.isTracking = false;
    return [...this.fsWriteLog];
  }

  /**
   * Estimate memory usage for a scene
   */
  public estimateSceneMemory(resources: {
    mjModel: MjModel;
    mjData: MjData;
    bodies: Record<number, THREE.Group>;
    meshes: Record<number, THREE.BufferGeometry>;
    mujocoRoot: THREE.Group;
  }): number {
    let totalBytes = 0;

    // Estimate MuJoCo model and data separately
    totalBytes += this.estimateMuJoCoMemory(resources.mjModel);
    totalBytes += this.estimateMjDataMemory(resources.mjData);

    // Track unique instances to avoid double-counting shared resources
    const countedGeometries = new Set<THREE.BufferGeometry>();
    const countedMaterials = new Set<THREE.Material>();
    const countedTextures = new Set<THREE.Texture>();

    // Estimate geometries from meshes
    for (const geometry of Object.values(resources.meshes)) {
      if (!countedGeometries.has(geometry)) {
        totalBytes += this.estimateGeometryMemory(geometry);
        countedGeometries.add(geometry);
      }
    }

    // Traverse scene to count all geometries, materials, and textures
    resources.mujocoRoot.traverse((object) => {
      // Count geometry
      if ('geometry' in object && object.geometry) {
        const geom = object.geometry as THREE.BufferGeometry;
        if (!countedGeometries.has(geom)) {
          totalBytes += this.estimateGeometryMemory(geom);
          countedGeometries.add(geom);
        }
      }

      // Count materials and their textures
      if ('material' in object && object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];

        for (const mat of materials) {
          if (!countedMaterials.has(mat)) {
            // Count material base overhead
            totalBytes += 1024;
            countedMaterials.add(mat);

            // Count unique textures in this material
            this.collectMaterialTextures(mat as THREE.Material).forEach((texture) => {
              if (!countedTextures.has(texture)) {
                totalBytes += this.estimateTextureMemory(texture);
                countedTextures.add(texture);
              }
            });
          }
        }
      }
    });

    return totalBytes;
  }

  /**
   * Collect all textures from a material
   */
  private collectMaterialTextures(material: THREE.Material): THREE.Texture[] {
    const textures: THREE.Texture[] = [];
    const anyMaterial = material as THREE.MeshStandardMaterial & {
      map?: THREE.Texture;
      aoMap?: THREE.Texture;
      emissiveMap?: THREE.Texture;
      metalnessMap?: THREE.Texture;
      normalMap?: THREE.Texture;
      roughnessMap?: THREE.Texture;
      envMap?: THREE.Texture;
      alphaMap?: THREE.Texture;
      lightMap?: THREE.Texture;
      displacementMap?: THREE.Texture;
      bumpMap?: THREE.Texture;
    };

    if (anyMaterial.map) textures.push(anyMaterial.map);
    if (anyMaterial.aoMap) textures.push(anyMaterial.aoMap);
    if (anyMaterial.emissiveMap) textures.push(anyMaterial.emissiveMap);
    if (anyMaterial.metalnessMap) textures.push(anyMaterial.metalnessMap);
    if (anyMaterial.normalMap) textures.push(anyMaterial.normalMap);
    if (anyMaterial.roughnessMap) textures.push(anyMaterial.roughnessMap);
    if (anyMaterial.envMap) textures.push(anyMaterial.envMap);
    if (anyMaterial.alphaMap) textures.push(anyMaterial.alphaMap);
    if (anyMaterial.lightMap) textures.push(anyMaterial.lightMap);
    if (anyMaterial.displacementMap) textures.push(anyMaterial.displacementMap);
    if (anyMaterial.bumpMap) textures.push(anyMaterial.bumpMap);

    return textures;
  }

  /**
   * Estimate MuJoCo model and data memory
   */

  private estimateMuJoCoMemory(mjModel: MjModel): number {
    let bytes = 0;

    // State vectors
    if (mjModel.nq) bytes += mjModel.nq * 8; // qpos (double)
    if (mjModel.nv) bytes += mjModel.nv * 8; // qvel (double)
    if (mjModel.nbody) bytes += mjModel.nbody * 256; // Body data (approximate)

    // Mesh data
    if (mjModel.mesh_vert) bytes += mjModel.mesh_vert.length * 4;
    if (mjModel.mesh_normal) bytes += mjModel.mesh_normal.length * 4;
    if (mjModel.mesh_texcoord) bytes += mjModel.mesh_texcoord.length * 4;
    if (mjModel.mesh_face) bytes += mjModel.mesh_face.length * 4;

    // Texture data
    if (mjModel.tex_data) bytes += mjModel.tex_data.length;

    return bytes;
  }

  /**
   * Estimate MuJoCo data memory
   */
  private estimateMjDataMemory(mjData: MjData): number {
    let bytes = 0;

    // State vectors (dynamic simulation state)
    if (mjData.qpos) bytes += mjData.qpos.length * 8; // Generalized positions (double)
    if (mjData.qvel) bytes += mjData.qvel.length * 8; // Generalized velocities (double)
    if (mjData.qacc) bytes += mjData.qacc.length * 8; // Generalized accelerations (double)
    if (mjData.qacc_warmstart) bytes += mjData.qacc_warmstart.length * 8; // Acceleration warmstart (double)

    // Control and actuation
    if (mjData.ctrl) bytes += mjData.ctrl.length * 8; // Control signals (double)
    if (mjData.qfrc_applied) bytes += mjData.qfrc_applied.length * 8; // Applied forces (double)
    if (mjData.xfrc_applied) bytes += mjData.xfrc_applied.length * 8; // Applied Cartesian forces (double)

    // Computed quantities (Cartesian positions, velocities, etc.)
    if (mjData.xpos) bytes += mjData.xpos.length * 8; // Body positions (double)
    if (mjData.xquat) bytes += mjData.xquat.length * 8; // Body orientations (double)
    if (mjData.xmat) bytes += mjData.xmat.length * 8; // Body orientation matrices (double)
    if (mjData.xipos) bytes += mjData.xipos.length * 8; // Body CoM positions (double)
    if (mjData.ximat) bytes += mjData.ximat.length * 8; // Body inertia matrices (double)

    // Contact and constraint data
    if (mjData.ncon) bytes += mjData.ncon * 256; // Contact data (approximate struct size)
    if (mjData.efc_force) bytes += mjData.efc_force.length * 8; // Constraint forces (double)

    // Sensor data
    if (mjData.sensordata) bytes += mjData.sensordata.length * 8; // Sensor readings (double)

    return bytes;
  }

  /**
   * Estimate Three.js geometry memory
   */
  private estimateGeometryMemory(geometry: THREE.BufferGeometry): number {
    let bytes = 0;

    // Count all buffer attributes
    const attributes = geometry.attributes;
    for (const key in attributes) {
      const attribute = attributes[key];
      if (attribute && 'array' in attribute) {
        bytes += attribute.array.byteLength;
      }
    }

    // Index buffer
    if (geometry.index) {
      bytes += geometry.index.array.byteLength;
    }

    return bytes;
  }

  /**
   * Estimate texture memory
   */
  private estimateTextureMemory(texture: THREE.Texture): number {
    if (!texture.image) {
      return 1024; // Minimal overhead
    }

    if (texture instanceof THREE.CubeTexture) {
      // 6 faces
      if (Array.isArray(texture.image) && texture.image.length === 6) {
        let totalBytes = 0;
        for (const img of texture.image) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const imgAny = img as any;
          if (imgAny && typeof imgAny === 'object' && 'width' in imgAny && 'height' in imgAny) {
            totalBytes += (imgAny.width as number) * (imgAny.height as number) * 4; // RGBA
          }
        }
        return totalBytes;
      }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const img = texture.image as any;
    if (img && typeof img === 'object' && 'width' in img && 'height' in img) {
      return (img.width as number) * (img.height as number) * 4; // RGBA
    }

    return 1024; // Default estimate
  }

}
