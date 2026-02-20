import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { mujocoAssetCollector } from '../utils/mujocoAssetCollector';
import { normalizeScenePath } from '../utils/pathUtils';
import { loadMjzFile } from '../utils/mjzLoader';
import { createLights } from './lights';
import { createTexture } from './textures';
import { createTendonMeshes } from './tendons';

const DEFAULT_BASE_URL = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
const BINARY_EXTENSIONS = ['.png', '.stl', '.skn', '.mjb', '.mjz', '.msh', '.npy'];
const sceneDownloadPromises = new Map<string, Promise<void>>();

/**
 * Clear download promise cache for a specific scene
 * This is used by the cache manager when evicting scenes
 */
export function clearSceneDownloadCache(scenePath: string): void {
  const normalizedPath = normalizeScenePath(scenePath);
  sceneDownloadPromises.delete(normalizedPath);
}

function isBinaryAsset(path: string): boolean {
  const lower = path.toLowerCase();
  return BINARY_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

function isInlineXML(input: string): boolean {
  if (!input) {
    return false;
  }
  const trimmed = input.trim();
  return trimmed.startsWith('<mujoco') || trimmed.startsWith('<?xml');
}

function ensureWorkingDirectories(mujoco: MainModule, segments: string[]): void {
  if (!segments.length) {
    return;
  }
  let working = '/working';
  for (const segment of segments) {
    working += `/${segment}`;
    if (!mujoco.FS.analyzePath(working, false).exists) {
      mujoco.FS.mkdir(working);
    }
  }
}

function normalizePathSegments(path: string): string {
  if (!path) {
    return '';
  }
  const parts = path.split('/');
  const resolved: string[] = [];
  for (const part of parts) {
    if (!part || part === '.') {
      continue;
    }
    if (part === '..') {
      if (resolved.length) {
        resolved.pop();
      }
      continue;
    }
    resolved.push(part);
  }
  return resolved.join('/');
}

export async function loadSceneFromURL(
  mujoco: MainModule,
  filename: string,
  parent: {
    mjModel: MjModel | null;
    mjData: MjData | null;
    scene: THREE.Scene;
    bodies?: Record<number, THREE.Group>;
    lights?: THREE.Light[];
    meshes?: Record<number, THREE.BufferGeometry>;
    mujocoRoot?: THREE.Group;
  }
): Promise<[MjModel, MjData, Record<number, THREE.Group>, THREE.Light[]]> {
  if (parent.mjData != null) {
    try {
      parent.mjData.delete();
    } catch {
      // ignore
    }
    parent.mjData = null;
  }
  if (parent.mjModel != null) {
    try {
      parent.mjModel.delete();
    } catch {
      // ignore
    }
    parent.mjModel = null;
  }

  let modelPath: string;

  if (isInlineXML(filename)) {
    const xmlHash = Math.abs(
      filename.split('').reduce((a, b) => {
        a = (a << 5) - a + b.charCodeAt(0);
        return a & a;
      }, 0)
    );
    const virtualPath = `inline_model_${xmlHash}.xml`;
    modelPath = `/working/${virtualPath}`;
  } else {
    const cleanedFilename = String(filename || '')
      .trim()
      .replace(/^(\.\/)+/, '');
    const normalizedFilename = normalizePathSegments(cleanedFilename);
    modelPath = `/working/${normalizedFilename}`;
  }

  try {
    const exists = mujoco.FS.analyzePath(modelPath, false).exists;
    if (!exists) {
      throw new Error(`Scene XML not found at ${modelPath}`);
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Scene XML not accessible at ${modelPath}: ${message}`);
  }

  let newModel: MjModel | null = null;
  try {
    if (modelPath.toLowerCase().endsWith('.mjb')) {
      newModel = mujoco.MjModel.mj_loadBinary(modelPath);
    } else {
      // TODO: Remove mjzLoader after mujoco wasm mj_loadXML() supports mjz format loading.
      if (modelPath.toLowerCase().endsWith('.mjz')) {
        const xmlPath = await loadMjzFile(mujoco, modelPath);
        modelPath = xmlPath;
      }

      newModel = mujoco.MjModel.mj_loadXML(modelPath);
    }
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to load MjModel from ${modelPath}: ${message}`);
  }
  if (!newModel) {
    throw new Error(`MjModel loading returned null for ${modelPath}`);
  }

  let newData: MjData | null = null;
  try {
    newData = new mujoco.MjData(newModel);
  } catch (error: unknown) {
    try {
      newModel.delete();
    } catch {
      // ignore
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to create MjData: ${message}`);
  }
  if (!newData) {
    try {
      newModel.delete();
    } catch {
      // ignore
    }
    throw new Error(`MjData constructor returned null for model loaded from ${modelPath}`);
  }

  parent.mjModel = newModel;
  parent.mjData = newData;

  const mjModel = parent.mjModel;
  const mjData = parent.mjData;

  const textDecoder = new TextDecoder('utf-8');
  const namesArray = new Uint8Array(mjModel.names);
  const fullString = textDecoder.decode(namesArray.slice());
  const names = fullString.split(textDecoder.decode(new ArrayBuffer(1)));

  const mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';
  parent.scene.add(mujocoRoot);

  const bodies: Record<number, THREE.Group> = {};
  const meshes: Record<number, THREE.BufferGeometry> = {};

  for (let g = 0; g < mjModel.ngeom; g++) {
    if (!(mjModel.geom_group[g] < 3)) {
      continue;
    }

    const b = mjModel.geom_bodyid[g];
    const type = mjModel.geom_type[g];
    const size = [
      mjModel.geom_size[g * 3 + 0],
      mjModel.geom_size[g * 3 + 1],
      mjModel.geom_size[g * 3 + 2],
    ];

    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();

      const startIdx = mjModel.name_bodyadr[b];
      let endIdx = startIdx;
      while (endIdx < namesArray.length && namesArray[endIdx] !== 0) {
        endIdx++;
      }
      const nameBuffer = namesArray.subarray(startIdx, endIdx);
      bodies[b].name = textDecoder.decode(nameBuffer);

      bodies[b].bodyID = b;
      bodies[b].has_custom_mesh = false;
    }

    let geometry: THREE.BufferGeometry | undefined;
    switch (type) {
      case mujoco.mjtGeom.mjGEOM_PLANE.value: {
        let width: number;
        let height: number;
        if (size[0] === 0) {
          width = 100;
        } else {
          width = size[0] * 2.0;
        }
        if (size[1] === 0) {
          height = 100;
        } else {
          height = size[1] * 2.0;
        }

        geometry = new THREE.PlaneGeometry(width, height);
        geometry.rotateX(-Math.PI / 2);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_HFIELD.value:
        break;
      case mujoco.mjtGeom.mjGEOM_SPHERE.value: {
        geometry = new THREE.SphereGeometry(size[0]);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_CAPSULE.value: {
        geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_ELLIPSOID.value: {
        geometry = new THREE.SphereGeometry(1);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_CYLINDER.value: {
        geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_BOX.value: {
        geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
        break;
      }
      case mujoco.mjtGeom.mjGEOM_MESH.value: {
        const meshID = mjModel.geom_dataid[g];

        if (!(meshID in meshes)) {
          geometry = new THREE.BufferGeometry();

          const vertexBuffer = mjModel.mesh_vert.subarray(
            mjModel.mesh_vertadr[meshID] * 3,
            (mjModel.mesh_vertadr[meshID] + mjModel.mesh_vertnum[meshID]) * 3
          );
          for (let v = 0; v < vertexBuffer.length; v += 3) {
            const temp = vertexBuffer[v + 1];
            vertexBuffer[v + 1] = vertexBuffer[v + 2];
            vertexBuffer[v + 2] = -temp;
          }

          const normalBuffer = mjModel.mesh_normal.subarray(
            mjModel.mesh_normaladr[meshID] * 3,
            (mjModel.mesh_normaladr[meshID] + mjModel.mesh_normalnum[meshID]) * 3
          );
          for (let v = 0; v < normalBuffer.length; v += 3) {
            const temp = normalBuffer[v + 1];
            normalBuffer[v + 1] = normalBuffer[v + 2];
            normalBuffer[v + 2] = -temp;
          }

          const uvBuffer = mjModel.mesh_texcoord.subarray(
            mjModel.mesh_texcoordadr[meshID] * 2,
            (mjModel.mesh_texcoordadr[meshID] + mjModel.mesh_texcoordnum[meshID]) * 2
          );

          const faceToVertexBuffer = mjModel.mesh_face.subarray(
            mjModel.mesh_faceadr[meshID] * 3,
            (mjModel.mesh_faceadr[meshID] + mjModel.mesh_facenum[meshID]) * 3
          );
          const faceToUvBuffer = mjModel.mesh_facetexcoord.subarray(
            mjModel.mesh_faceadr[meshID] * 3,
            (mjModel.mesh_faceadr[meshID] + mjModel.mesh_facenum[meshID]) * 3
          );
          const faceToNormalBuffer = mjModel.mesh_facenormal.subarray(
            mjModel.mesh_faceadr[meshID] * 3,
            (mjModel.mesh_faceadr[meshID] + mjModel.mesh_facenum[meshID]) * 3
          );

          const positions: number[] = [];
          const normals: number[] = [];
          const uvs: number[] = [];
          const indices: number[] = [];
          const tupleToIndex = new Map<string, number>();

          const faceCount = faceToVertexBuffer.length / 3;
          for (let t = 0; t < faceCount; t++) {
            for (let c = 0; c < 3; c++) {
              const vi = faceToVertexBuffer[t * 3 + c];
              const nvi = faceToNormalBuffer[t * 3 + c];
              const uvi = faceToUvBuffer[t * 3 + c];
              const key = `${vi}_${nvi}_${uvi}`;
              let outIndex = tupleToIndex.get(key);
              if (outIndex === undefined) {
                outIndex = positions.length / 3;
                tupleToIndex.set(key, outIndex);
                positions.push(
                  vertexBuffer[vi * 3 + 0],
                  vertexBuffer[vi * 3 + 1],
                  vertexBuffer[vi * 3 + 2]
                );
                normals.push(
                  normalBuffer[nvi * 3 + 0],
                  normalBuffer[nvi * 3 + 1],
                  normalBuffer[nvi * 3 + 2]
                );
                uvs.push(uvBuffer[uvi * 2 + 0], uvBuffer[uvi * 2 + 1]);
              }
              indices.push(outIndex);
            }
          }

          geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
          geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
          geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
          geometry.setIndex(indices);
          geometry.computeVertexNormals();
          meshes[meshID] = geometry;
        } else {
          geometry = meshes[meshID];
        }
        bodies[b].has_custom_mesh = true;
        break;
      }
    }

    let color = [
      mjModel.geom_rgba[g * 4 + 0],
      mjModel.geom_rgba[g * 4 + 1],
      mjModel.geom_rgba[g * 4 + 2],
      mjModel.geom_rgba[g * 4 + 3],
    ];
    let texture: THREE.Texture | null = null;
    if (mjModel.geom_matid[g] !== -1) {
      const matId = mjModel.geom_matid[g];
      color = [
        mjModel.mat_rgba[matId * 4 + 0],
        mjModel.mat_rgba[matId * 4 + 1],
        mjModel.mat_rgba[matId * 4 + 2],
        mjModel.mat_rgba[matId * 4 + 3],
      ];

      const role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value;
      const texId = mjModel.mat_texid[matId * mujoco.mjtTextureRole.mjNTEXROLE.value + role];
      if (texId !== -1) {
        texture = createTexture({ mujoco, mjModel, texId });
        if (texture) {
          const repeatX = mjModel.mat_texrepeat ? mjModel.mat_texrepeat[matId * 2 + 0] : 1;
          const repeatY = mjModel.mat_texrepeat ? mjModel.mat_texrepeat[matId * 2 + 1] : 1;
          texture.repeat.set(repeatX, repeatY);
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
        }
      }
    }

    let currentMaterial: THREE.MeshPhysicalMaterial | THREE.MeshPhysicalMaterial[] =
      new THREE.MeshPhysicalMaterial({
        color: new THREE.Color(color[0], color[1], color[2]),
        transparent: color[3] < 1.0,
        opacity: color[3],
        specularIntensity:
          mjModel.geom_matid[g] !== -1 ? mjModel.mat_specular?.[mjModel.geom_matid[g]] : undefined,
        reflectivity:
          mjModel.geom_matid[g] !== -1
            ? mjModel.mat_reflectance?.[mjModel.geom_matid[g]]
            : undefined,
        roughness:
          mjModel.geom_matid[g] !== -1 && mjModel.mat_shininess
            ? 1.0 - mjModel.mat_shininess[mjModel.geom_matid[g]]
            : undefined,
        metalness:
          mjModel.geom_matid[g] !== -1 ? mjModel.mat_specular?.[mjModel.geom_matid[g]] : 0.1,
      });

    if (texture) {
      if (!(texture instanceof THREE.CubeTexture)) {
        (currentMaterial as THREE.MeshPhysicalMaterial).map = texture;
      } else {
        if (
          type === mujoco.mjtGeom.mjGEOM_BOX.value &&
          Array.isArray(texture.image) &&
          texture.image.length === 6
        ) {
          const images = texture.image as unknown as HTMLCanvasElement[];
          if (images.length === 6) {
            if (geometry && geometry.groups && geometry.groups.length !== 6) {
              geometry.clearGroups();
            }

            const materials: THREE.MeshPhysicalMaterial[] = [];
            for (let i = 0; i < 6; i++) {
              const canvas = images[i];
              const faceTex = new THREE.CanvasTexture(canvas);
              faceTex.flipY = false;
              faceTex.wrapS = THREE.ClampToEdgeWrapping;
              faceTex.wrapT = THREE.ClampToEdgeWrapping;
              faceTex.magFilter = THREE.LinearFilter;
              faceTex.minFilter = THREE.LinearFilter;

              const m = new THREE.MeshPhysicalMaterial({
                color: new THREE.Color(color[0], color[1], color[2]),
                transparent: color[3] < 1.0,
                opacity: color[3],
                specularIntensity:
                  mjModel.geom_matid[g] !== -1
                    ? mjModel.mat_specular?.[mjModel.geom_matid[g]]
                    : undefined,
                reflectivity:
                  mjModel.geom_matid[g] !== -1
                    ? mjModel.mat_reflectance?.[mjModel.geom_matid[g]]
                    : undefined,
                roughness:
                  mjModel.geom_matid[g] !== -1 && mjModel.mat_shininess
                    ? 1.0 - mjModel.mat_shininess[mjModel.geom_matid[g]]
                    : undefined,
                metalness:
                  mjModel.geom_matid[g] !== -1
                    ? mjModel.mat_specular?.[mjModel.geom_matid[g]]
                    : 0.1,
                map: faceTex,
              });
              materials.push(m);
            }
            currentMaterial = materials;
          } else {
            (currentMaterial as THREE.MeshPhysicalMaterial).envMap = texture;
            (currentMaterial as THREE.MeshPhysicalMaterial).envMapIntensity =
              mjModel.geom_matid[g] !== -1
                ? mjModel.mat_reflectance?.[mjModel.geom_matid[g]] || 0.5
                : 0.5;
          }
        } else {
          (currentMaterial as THREE.MeshPhysicalMaterial).envMap = texture;
          (currentMaterial as THREE.MeshPhysicalMaterial).envMapIntensity =
            mjModel.geom_matid[g] !== -1
              ? mjModel.mat_reflectance?.[mjModel.geom_matid[g]] || 0.5
              : 0.5;
        }
      }
    }

    if (!geometry) {
      console.warn(`Skipping geometry ${g} (type ${type}): no valid geometry created`);
      continue;
    }

    const mesh = new THREE.Mesh(geometry, currentMaterial as THREE.Material);

    mesh.castShadow = g === 0 ? false : true;
    mesh.receiveShadow = type !== mujoco.mjtGeom.mjGEOM_MESH.value;
    mesh.bodyID = b;
    bodies[b].add(mesh);
    getPosition(mjModel.geom_pos, g, mesh.position);
    if (type !== mujoco.mjtGeom.mjGEOM_PLANE.value) {
      getQuaternion(mjModel.geom_quat, g, mesh.quaternion);
    }
    if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
  }

  createTendonMeshes(mujocoRoot);

  const lights: THREE.Light[] = createLights({ mujoco, mjModel, mujocoRoot, bodies });

  for (let b = 0; b < mjModel.nbody; b++) {
    if (b === 0 || !bodies[0]) {
      mujocoRoot.add(bodies[b]);
    } else if (bodies[b]) {
      bodies[0].add(bodies[b]);
    } else {
      bodies[b] = new THREE.Group();
      bodies[b].name = names[b + 1];
      bodies[b].bodyID = b;
      bodies[b].has_custom_mesh = false;
      bodies[0].add(bodies[b]);
    }
  }

  parent.bodies = bodies;
  parent.lights = lights;
  parent.meshes = meshes;
  parent.mujocoRoot = mujocoRoot;

  if (!mjModel || 'deleted' in mjModel) {
    throw new Error('loadSceneFromURL: mjModel is invalid or already deleted');
  }

  return [mjModel, mjData, bodies, lights];
}

export function getPosition(
  buffer: Float32Array,
  index: number,
  target: THREE.Vector3,
  swizzle = true
): THREE.Vector3 {
  if (swizzle) {
    return target.set(buffer[index * 3 + 0], buffer[index * 3 + 2], -buffer[index * 3 + 1]);
  }
  return target.set(buffer[index * 3 + 0], buffer[index * 3 + 1], buffer[index * 3 + 2]);
}

export function getQuaternion(
  buffer: Float32Array,
  index: number,
  target: THREE.Quaternion,
  swizzle = true
): THREE.Quaternion {
  if (swizzle) {
    return target.set(
      -buffer[index * 4 + 1],
      -buffer[index * 4 + 3],
      buffer[index * 4 + 2],
      -buffer[index * 4 + 0]
    );
  }
  return target.set(
    buffer[index * 4 + 0],
    buffer[index * 4 + 1],
    buffer[index * 4 + 2],
    buffer[index * 4 + 3]
  );
}

export async function downloadExampleScenesFolder(
  mujoco: MainModule,
  scenePath: string,
  baseUrl: string = DEFAULT_BASE_URL
): Promise<void> {
  if (!scenePath) {
    return;
  }

  const basePrefix = baseUrl ? baseUrl.replace(/\/+$/, '') : '';

  if (isInlineXML(scenePath)) {
    const xmlHash = Math.abs(
      scenePath.split('').reduce((a, b) => {
        a = (a << 5) - a + b.charCodeAt(0);
        return a & a;
      }, 0)
    );
    const virtualPath = `inline_model_${xmlHash}.xml`;
    const fullPath = `/working/${virtualPath}`;

    try {
      mujoco.FS.writeFile(fullPath, scenePath);
    } catch (error) {
      console.error('[downloadExampleScenesFolder] Failed to write inline XML:', error);
      throw error;
    }
    return;
  }

  const normalizedPath = normalizeScenePath(scenePath);
  const pathParts = normalizedPath.split('/');

  const xmlDirectory = pathParts.slice(0, -1).join('/');
  if (!xmlDirectory) {
    return;
  }

  const cacheKey = normalizedPath;
  if (sceneDownloadPromises.has(cacheKey)) {
    return sceneDownloadPromises.get(cacheKey)!;
  }

  const downloadPromise = (async () => {
    let manifest: string[];
    try {
      manifest = await mujocoAssetCollector.analyzeScene(normalizedPath, basePrefix || '/');

      if (!Array.isArray(manifest)) {
        throw new Error(
          `Asset collector returned invalid result (not an array): ${typeof manifest}`
        );
      }

      if (manifest.length === 0) {
        throw new Error('No assets found by collector');
      }
      console.log(`[downloadExampleScenesFolder] Analyzed ${normalizedPath}: found ${manifest.length} assets`);
      console.log(`[downloadExampleScenesFolder] Asset manifest:`, manifest);
    } catch {
      try {
        const manifestResponse = await fetch(
          `${basePrefix}/${xmlDirectory}/index.json`.replace(/\/+/g, '/')
        );
        if (!manifestResponse.ok) {
          throw new Error(
            `Failed to load scene manifest for ${xmlDirectory}: ${manifestResponse.status}`
          );
        }
        manifest = await manifestResponse.json();
        if (!Array.isArray(manifest)) {
          throw new Error(`Invalid scene manifest for ${xmlDirectory}`);
        }
        console.log(`[downloadExampleScenesFolder] Loaded manifest from index.json: ${manifest.length} assets`);
      } catch (fallbackError: unknown) {
        const fallbackMsg =
          fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
        throw new Error(
          `Both asset analysis and index.json fallback failed: ${fallbackMsg}`
        );
      }
    }

    const localAssets = manifest
      .filter(
        (asset) =>
          typeof asset === 'string' && !asset.startsWith('http://') && !asset.startsWith('https://')
      )
      .map((assetPath) => {
        let assetNormalized = assetPath
          .trim()
          .replace(/^(\.\/)+/, '')
          .replace(/^public\//, '');
        if (assetNormalized.startsWith('/')) {
          assetNormalized = assetNormalized.slice(1);
        }
        if (!assetNormalized) {
          console.warn(
            `[downloadExampleScenesFolder] Skipping asset with empty path: ${assetPath}`
          );
          return null;
        }
        return { originalPath: assetPath, normalizedPath: assetNormalized };
      })
      .filter(Boolean) as { originalPath: string; normalizedPath: string }[];

    const seenPaths = new Set<string>();
    const uniqueAssets: { originalPath: string; normalizedPath: string }[] = [];
    for (const asset of localAssets) {
      if (seenPaths.has(asset.normalizedPath)) {
        continue;
      }
      seenPaths.add(asset.normalizedPath);
      uniqueAssets.push(asset);
    }

    console.log(`[downloadExampleScenesFolder] Downloading ${uniqueAssets.length} unique assets for ${xmlDirectory}`);

    const requests = uniqueAssets.map(({ normalizedPath }) => {
      const fullPath = `${basePrefix}/${normalizedPath}`.replace(/\/+/g, '/');
      return fetch(fullPath);
    });

    const responses = await Promise.all(requests);

    for (let i = 0; i < responses.length; i++) {
      const response = responses[i];
      const { originalPath, normalizedPath } = uniqueAssets[i];

      if (!response.ok) {
        console.warn(
          `[downloadExampleScenesFolder] Failed to fetch scene asset ${originalPath}: ${response.status}`
        );
        continue;
      }

      const assetPath = normalizedPath;
      const segments = assetPath.split('/');
      ensureWorkingDirectories(mujoco, segments.slice(0, -1));

      const targetPath = `/working/${assetPath}`;
      try {
        if (isBinaryAsset(normalizedPath) || isBinaryAsset(originalPath)) {
          const arrayBuffer = await response.arrayBuffer();
          mujoco.FS.writeFile(targetPath, new Uint8Array(arrayBuffer));
        } else {
          const text = await response.text();
          mujoco.FS.writeFile(targetPath, text);
        }
      } catch (error) {
        console.error(`[downloadExampleScenesFolder] Error writing ${targetPath}:`, error);
      }
    }
  })();

  sceneDownloadPromises.set(normalizedPath, downloadPromise);
  try {
    await downloadPromise;
  } catch (error) {
    sceneDownloadPromises.delete(normalizedPath);
    throw error;
  }
}
