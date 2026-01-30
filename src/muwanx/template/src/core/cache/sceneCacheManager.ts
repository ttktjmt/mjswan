import * as THREE from 'three';
import type { MjData, MjModel } from '@/mujoco';
import type { Mujoco } from '../../types/mujoco';
import { clearSceneDownloadCache } from '../scene/scene';
import { normalizeScenePath } from '../utils/pathUtils';

/**
 * Resources associated with a cached scene
 */
export interface CachedSceneResources {
  // Scene identification
  scenePath: string;
  lastAccessed: number;
  loadedAt: number;

  // MuJoCo resources
  mjModel: MjModel;
  mjData: MjData;

  // Three.js resources
  bodies: Record<number, THREE.Group>;
  lights: THREE.Light[];
  meshes: Record<number, THREE.BufferGeometry>;
  mujocoRoot: THREE.Group;

  // Emscripten FS paths
  fsFiles: string[];

  // ONNX model (future support)
  onnxModel?: ArrayBuffer | null;
  onnxMetadata?: Record<string, unknown>;

  // Memory estimation
  estimatedMemoryBytes: number;
}

/**
 * Cache performance metrics
 */
export interface CacheMetrics {
  totalScenes: number;
  totalMemoryBytes: number;
  oldestAccessTime: number;
  newestAccessTime: number;
  cacheHits: number;
  cacheMisses: number;
  evictions: number;
}

/**
 * LRU Cache Manager for scene resources
 *
 * Manages up to MAX_CACHE_SIZE scenes in memory, automatically evicting
 * the least recently used scene when the cache is full.
 *
 * This is implemented as a singleton to persist cache across runtime instances.
 */
export class SceneCacheManager {
  private static instance: SceneCacheManager | null = null;
  private readonly MAX_CACHE_SIZE = 5;
  private cache: Map<string, CachedSceneResources>;
  private metrics: CacheMetrics;
  private mujoco: Mujoco;

  private constructor(mujoco: Mujoco) {
    this.mujoco = mujoco;
    this.cache = new Map();
    this.metrics = {
      totalScenes: 0,
      totalMemoryBytes: 0,
      oldestAccessTime: Date.now(),
      newestAccessTime: Date.now(),
      cacheHits: 0,
      cacheMisses: 0,
      evictions: 0,
    };
  }

  /**
   * Get singleton instance
   */
  public static getInstance(mujoco: Mujoco): SceneCacheManager {
    if (!SceneCacheManager.instance) {
      SceneCacheManager.instance = new SceneCacheManager(mujoco);
      console.log('[SceneCache] Created new cache manager singleton');
    }
    return SceneCacheManager.instance;
  }

  /**
   * Reset singleton instance (for testing or cleanup)
   */
  public static resetInstance(): void {
    SceneCacheManager.instance = null;
  }

  /**
   * Normalize scene path for consistent cache keys
   */
  private normalizeScenePath(scenePath: string): string {
    if (!scenePath) {
      return '';
    }
    return normalizeScenePath(scenePath);
  }

  /**
   * Check if scene is cached
   */
  public has(scenePath: string): boolean {
    const key = this.normalizeScenePath(scenePath);
    return this.cache.has(key);
  }

  /**
   * Get cached scene (updates LRU timestamp)
   */
  public get(scenePath: string): CachedSceneResources | null {
    const key = this.normalizeScenePath(scenePath);
    const resources = this.cache.get(key);

    if (resources) {
      resources.lastAccessed = Date.now();
      this.metrics.cacheHits++;
      this.updateMetricsTimestamps();
      return resources;
    }

    this.metrics.cacheMisses++;
    return null;
  }

  /**
   * Add scene to cache (may trigger eviction)
   */
  public async set(scenePath: string, resources: CachedSceneResources): Promise<void> {
    const key = this.normalizeScenePath(scenePath);

    // Check if we need to evict
    if (!this.cache.has(key) && this.cache.size >= this.MAX_CACHE_SIZE) {
      await this.maybeEvict();
    }

    // Update resources with normalized path
    resources.scenePath = key;
    resources.lastAccessed = Date.now();

    this.cache.set(key, resources);
    this.updateMetrics();
  }

  /**
   * Manually evict specific scene
   */
  public async evict(scenePath: string): Promise<void> {
    const key = this.normalizeScenePath(scenePath);
    const resources = this.cache.get(key);

    if (resources) {
      await this.disposeSceneResources(resources);
      this.cache.delete(key);
      this.metrics.evictions++;
      this.updateMetrics();
    }
  }

  /**
   * Clear entire cache
   */
  public async clear(): Promise<void> {
    const keys = Array.from(this.cache.keys());
    for (const key of keys) {
      await this.evict(key);
    }
  }

  /**
   * Get cache statistics
   */
  public getMetrics(): CacheMetrics {
    return { ...this.metrics };
  }

  /**
   * Prepare cache for new scene (evict if needed)
   */
  public async prepareForNewScene(): Promise<void> {
    if (this.cache.size >= this.MAX_CACHE_SIZE) {
      await this.maybeEvict();
    }
  }

  /**
   * Find LRU scene
   */
  private findLRUScene(): string | null {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, resources] of this.cache.entries()) {
      if (resources.lastAccessed < oldestTime) {
        oldestTime = resources.lastAccessed;
        oldestKey = key;
      }
    }

    return oldestKey;
  }

  /**
   * Trigger eviction if needed
   */
  private async maybeEvict(): Promise<void> {
    const lruKey = this.findLRUScene();
    if (lruKey) {
      console.log(`[SceneCache] Evicting LRU scene: ${lruKey}`);
      await this.evict(lruKey);
    }
  }

  /**
   * Dispose all resources for a scene
   */
  private async disposeSceneResources(resources: CachedSceneResources): Promise<void> {
    const startTime = performance.now();

    // 1. Dispose Three.js resources
    if (resources.mujocoRoot) {
      this.disposeThreeJSObject(resources.mujocoRoot);
    }

    // 2. Delete MuJoCo objects
    if (resources.mjData) {
      try {
        resources.mjData.delete();
      } catch (error) {
        console.warn('[SceneCache] Failed to delete mjData:', error);
      }
    }

    if (resources.mjModel) {
      try {
        resources.mjModel.delete();
      } catch (error) {
        console.warn('[SceneCache] Failed to delete mjModel:', error);
      }
    }

    // 3. Clean Emscripten FS
    for (const filePath of resources.fsFiles) {
      try {
        const fullPath = `/working/${filePath}`;
        const analysis = this.mujoco.FS.analyzePath(fullPath, false);
        if (analysis.exists) {
          this.mujoco.FS.unlink(fullPath);
        }
      } catch {
        // File may not exist or already deleted, ignore
      }
    }

    // 4. Clear scene download cache
    clearSceneDownloadCache(resources.scenePath);

    const elapsed = performance.now() - startTime;
    const memoryMB = (resources.estimatedMemoryBytes / 1048576).toFixed(1);
    console.log(`[SceneCache] Evicted: ${resources.scenePath} (${memoryMB} MB) in ${elapsed.toFixed(0)}ms`);
  }

  /**
   * Dispose Three.js object and all its children recursively
   */
  private disposeThreeJSObject(object: THREE.Object3D): void {
    object.traverse((child) => {
      // Dispose geometry
      if ('geometry' in child && child.geometry) {
        (child.geometry as THREE.BufferGeometry).dispose();
      }

      // Dispose material(s)
      if ('material' in child && child.material) {
        if (Array.isArray(child.material)) {
          child.material.forEach((material) => this.disposeMaterial(material));
        } else {
          this.disposeMaterial(child.material as THREE.Material);
        }
      }
    });

    // Remove from parent
    if (object.parent) {
      object.parent.remove(object);
    }
  }

  /**
   * Dispose material and all its textures
   */
  private disposeMaterial(material: THREE.Material): void {
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

    // Dispose all texture maps
    if (anyMaterial.map) anyMaterial.map.dispose();
    if (anyMaterial.aoMap) anyMaterial.aoMap.dispose();
    if (anyMaterial.emissiveMap) anyMaterial.emissiveMap.dispose();
    if (anyMaterial.metalnessMap) anyMaterial.metalnessMap.dispose();
    if (anyMaterial.normalMap) anyMaterial.normalMap.dispose();
    if (anyMaterial.roughnessMap) anyMaterial.roughnessMap.dispose();
    if (anyMaterial.envMap) anyMaterial.envMap.dispose();
    if (anyMaterial.alphaMap) anyMaterial.alphaMap.dispose();
    if (anyMaterial.lightMap) anyMaterial.lightMap.dispose();
    if (anyMaterial.displacementMap) anyMaterial.displacementMap.dispose();
    if (anyMaterial.bumpMap) anyMaterial.bumpMap.dispose();

    material.dispose();
  }

  /**
   * Update cache metrics
   */
  private updateMetrics(): void {
    this.metrics.totalScenes = this.cache.size;
    this.metrics.totalMemoryBytes = 0;

    for (const resources of this.cache.values()) {
      this.metrics.totalMemoryBytes += resources.estimatedMemoryBytes;
    }

    this.updateMetricsTimestamps();
  }

  /**
   * Update oldest and newest access timestamps
   */
  private updateMetricsTimestamps(): void {
    let oldest = Date.now();
    let newest = 0;

    for (const resources of this.cache.values()) {
      if (resources.lastAccessed < oldest) {
        oldest = resources.lastAccessed;
      }
      if (resources.lastAccessed > newest) {
        newest = resources.lastAccessed;
      }
    }

    this.metrics.oldestAccessTime = oldest;
    this.metrics.newestAccessTime = newest;
  }
}
