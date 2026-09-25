import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { createLights, lightSpecularRatio } from './lights';
import { createTexture, createSkyboxTexture } from './textures';
import { createTendonMeshes } from './tendons';

/** How a scene's model bytes are encoded: `add_scene(spec=...)` writes `mjz`, `model=` `mjb`. */
export type ModelFormat = 'mjz' | 'mjb';

export function reflectanceParams(
  mjModel: MjModel,
  matId: number,
  specularRatio = 1
): Pick<THREE.MeshPhysicalMaterialParameters, 'specularIntensity' | 'roughness' | 'metalness'> {
  const specular = matId !== -1 ? (mjModel.mat_specular?.[matId] ?? 0.5) : 0.5;
  const shininess = matId !== -1 ? (mjModel.mat_shininess?.[matId] ?? 0.5) : 0.5;
  const metallic = matId !== -1 ? (mjModel.mat_metallic?.[matId] ?? -1) : -1;
  const roughnessAttr = matId !== -1 ? (mjModel.mat_roughness?.[matId] ?? -1) : -1;

  return {
    specularIntensity: specular * specularRatio,
    roughness: roughnessAttr >= 0 ? roughnessAttr : 1.0 - shininess,
    metalness: metallic >= 0 ? metallic : 0,
  };
}

function isInlineXML(input: string): boolean {
  if (!input) {
    return false;
  }
  const trimmed = input.trim();
  return trimmed.startsWith('<mujoco') || trimmed.startsWith('<?xml');
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

function createInfinitePlaneShaderMaterial(params: {
  color: THREE.Color;
  opacity: number;
  texture: THREE.Texture | null;
  // Precomputed UV scale per axis, matching MuJoCo's GL_OBJECT_PLANE formula:
  //   infinite axis: 0.5 * texrepeat
  //   finite axis:   0.5 * texrepeat / halfSize
  uvScaleX: number;
  uvScaleZ: number;
  planeY: number;
  // World-space center of the geom (for finite-extent clipping)
  centerX: number;
  centerZ: number;
  infiniteX: boolean;
  infiniteZ: boolean;
  halfExtentX: number;
  halfExtentZ: number;
}): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: params.color },
      uOpacity: { value: params.opacity },
      uTexture: { value: params.texture },
      uHasTexture: { value: params.texture !== null },
      uUVScale: { value: new THREE.Vector2(params.uvScaleX, params.uvScaleZ) },
      uPlaneY: { value: params.planeY },
      uCenterX: { value: params.centerX },
      uCenterZ: { value: params.centerZ },
      uInfiniteX: { value: params.infiniteX },
      uInfiniteZ: { value: params.infiniteZ },
      uHalfExtentX: { value: params.halfExtentX },
      uHalfExtentZ: { value: params.halfExtentZ },
      uProjInverse: { value: new THREE.Matrix4() },
      uCamWorldMatrix: { value: new THREE.Matrix4() },
    },
    vertexShader: /* glsl */ `
      varying vec2 vNDC;
      void main() {
        vNDC = position.xy;
        gl_Position = vec4(position.xy, 1.0, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      precision highp float;
      varying vec2 vNDC;
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform sampler2D uTexture;
      uniform bool uHasTexture;
      uniform vec2 uUVScale;
      uniform float uPlaneY;
      uniform float uCenterX;
      uniform float uCenterZ;
      uniform bool uInfiniteX;
      uniform bool uInfiniteZ;
      uniform float uHalfExtentX;
      uniform float uHalfExtentZ;
      uniform mat4 uProjInverse;
      uniform mat4 uCamWorldMatrix;

      void main() {
        // Unproject NDC to view-space ray
        vec4 viewRay = uProjInverse * vec4(vNDC, 1.0, 1.0);
        viewRay /= viewRay.w;

        // Transform to world-space direction
        vec3 worldDir = normalize((uCamWorldMatrix * vec4(viewRay.xyz, 0.0)).xyz);

        // Ray-plane intersection at y = uPlaneY
        float denom = worldDir.y;
        if (abs(denom) < 1e-6) discard;

        float t = (uPlaneY - cameraPosition.y) / denom;
        if (t < 0.0) discard;

        vec3 worldPos = cameraPosition + t * worldDir;

        // Clip finite extents relative to geom center
        if (!uInfiniteX && abs(worldPos.x - uCenterX) > uHalfExtentX) discard;
        if (!uInfiniteZ && abs(worldPos.z - uCenterZ) > uHalfExtentZ) discard;

        if (uHasTexture) {
          // MuJoCo GL_OBJECT_PLANE formula:
          //   infinite: S = worldPos * 0.5 * texrepeat
          //   finite:   S = worldPos * 0.5 * texrepeat / halfSize
          // Both cases are baked into uUVScale on the JS side.
          vec2 uv = fract(worldPos.xz * uUVScale);
          gl_FragColor = texture2D(uTexture, uv) * vec4(uColor, uOpacity);
        } else {
          gl_FragColor = vec4(uColor, uOpacity);
        }
      }
    `,
    transparent: params.opacity < 1.0,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

export async function loadSceneFromURL(
  mujoco: MainModule,
  filename: string,
  parent: {
    scene: THREE.Scene;
    bodies?: Record<number, THREE.Group>;
    lights?: THREE.Light[];
    meshes?: Record<number, THREE.BufferGeometry>;
    mujocoRoot?: THREE.Group;
  }
): Promise<[MjModel, MjData, Record<number, THREE.Group>, THREE.Light[]]> {
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
      const vfs = new mujoco.MjVFS();
      try {
        newModel = mujoco.MjModel.mj_loadModel(modelPath, vfs);
      } finally {
        vfs.delete();
      }
    } else {
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

  const mjModel = newModel;
  const mjData = newData;

  const textDecoder = new TextDecoder('utf-8');
  const namesArray = new Uint8Array(mjModel.names);

  const mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';
  parent.scene.add(mujocoRoot);

  const bodies: Record<number, THREE.Group> = {};
  const meshes: Record<number, THREE.BufferGeometry> = {};
  const specularRatio = lightSpecularRatio(mjModel);

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
        if (size[0] === 0 || size[1] === 0) {
          // PlaneGeometry(2,2) in XY plane: vertices at (±1, ±1, 0)
          // Used as a full-screen clip-space quad by the infinite plane shader
          geometry = new THREE.PlaneGeometry(2, 2);
        } else {
          geometry = new THREE.PlaneGeometry(size[0] * 2.0, size[1] * 2.0);
          geometry.rotateX(-Math.PI / 2);
        }
        break;
      }
      case mujoco.mjtGeom.mjGEOM_HFIELD.value:
        geometry = createHFieldGeometry(mjModel, g);
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
        ...reflectanceParams(mjModel, mjModel.geom_matid[g], specularRatio),
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
                ...reflectanceParams(mjModel, mjModel.geom_matid[g], specularRatio),
                map: faceTex,
              });
              materials.push(m);
            }
            currentMaterial = materials;
          } else {
            (currentMaterial as THREE.MeshPhysicalMaterial).envMap = texture;
            (currentMaterial as THREE.MeshPhysicalMaterial).envMapIntensity =
              mjModel.geom_matid[g] !== -1
                ? (mjModel.mat_reflectance?.[mjModel.geom_matid[g]] ?? 0)
                : 0;
          }
        } else {
          (currentMaterial as THREE.MeshPhysicalMaterial).envMap = texture;
          (currentMaterial as THREE.MeshPhysicalMaterial).envMapIntensity =
            mjModel.geom_matid[g] !== -1
              ? (mjModel.mat_reflectance?.[mjModel.geom_matid[g]] ?? 0)
              : 0;
        }
      }
    }

    if (!geometry) {
      console.warn(`Skipping geometry ${g} (type ${type}): no valid geometry created`);
      continue;
    }

    const mesh = new THREE.Mesh(geometry, currentMaterial as THREE.Material);
    const ignoreDragForce = bodies[b].name === 'terrain';

    mesh.castShadow = ignoreDragForce ? false : g !== 0;
    mesh.receiveShadow = type !== mujoco.mjtGeom.mjGEOM_MESH.value;
    mesh.bodyID = b;
    // The geom behind this mesh, so a size written at runtime can be drawn.
    mesh.userData.geomId = g;
    mesh.userData.ignoreDragForce = ignoreDragForce;
    bodies[b].userData.ignoreDragForce = ignoreDragForce;
    bodies[b].add(mesh);
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      // Read by `Passthrough`, which hides these so an AR session shows the room's floor.
      mesh.userData.isGroundPlane = true;
    }
    getPosition(mjModel.geom_pos, g, mesh.position);
    const isInfinitePlane =
      type === mujoco.mjtGeom.mjGEOM_PLANE.value && (size[0] === 0 || size[1] === 0);
    if (!isInfinitePlane) {
      getQuaternion(mjModel.geom_quat, g, mesh.quaternion);
    }
    if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      mesh.scale.set(size[0], size[2], size[1]);
    }
    if (isInfinitePlane) {
      const baseMat = (
        Array.isArray(currentMaterial) ? currentMaterial[0] : currentMaterial
      ) as THREE.MeshPhysicalMaterial;
      const tex = baseMat.map ?? null;
      const repeatX = tex?.repeat.x ?? 1;
      const repeatY = tex?.repeat.y ?? 1;
      // MuJoCo GL_OBJECT_PLANE formula: coeff = 0.5 * texrepeat / halfSize (finite),
      // or 0.5 * texrepeat (infinite, halfSize=0)
      const uvScaleX = size[0] > 0 ? (repeatX * 0.5) / size[0] : repeatX * 0.5;
      const uvScaleZ = size[1] > 0 ? (repeatY * 0.5) / size[1] : repeatY * 0.5;
      const shaderMat = createInfinitePlaneShaderMaterial({
        color: baseMat.color.clone(),
        opacity: baseMat.opacity,
        texture: tex,
        uvScaleX,
        uvScaleZ,
        planeY: mesh.position.y,
        centerX: mesh.position.x,
        centerZ: mesh.position.z,
        infiniteX: size[0] === 0,
        infiniteZ: size[1] === 0,
        halfExtentX: size[0],
        halfExtentZ: size[1],
      });
      mesh.material = shaderMat;
      mesh.frustumCulled = false;
      mesh.renderOrder = 1;
      mesh.onBeforeRender = (_renderer, _scene, camera) => {
        shaderMat.uniforms.uProjInverse.value.copy(camera.projectionMatrixInverse);
        shaderMat.uniforms.uCamWorldMatrix.value.copy(camera.matrixWorld);
      };
    } else if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
      const frontMat = currentMaterial as THREE.MeshPhysicalMaterial;
      const backMat = frontMat.clone();
      backMat.side = THREE.BackSide;
      backMat.transparent = true;
      backMat.opacity = frontMat.opacity * 0.35;
      backMat.depthWrite = false;
      const backMesh = new THREE.Mesh(geometry, backMat);
      backMesh.userData.isGroundPlane = true;
      backMesh.position.copy(mesh.position);
      backMesh.quaternion.copy(mesh.quaternion);
      backMesh.raycast = () => {};
      bodies[b].add(backMesh);
    }
  }

  createTendonMeshes(mujocoRoot, mjModel);

  const lights: THREE.Light[] = createLights({ mujoco, mjModel, mujocoRoot, bodies });

  for (let b = 0; b < mjModel.nbody; b++) {
    if (!bodies[b]) {
      bodies[b] = new THREE.Group();
      const startIdx = mjModel.name_bodyadr[b];
      let endIdx = startIdx;
      while (endIdx < namesArray.length && namesArray[endIdx] !== 0) {
        endIdx++;
      }
      bodies[b].name = textDecoder.decode(namesArray.subarray(startIdx, endIdx));
      bodies[b].bodyID = b;
      bodies[b].has_custom_mesh = false;
    }

    if (b === 0 || !bodies[0]) {
      mujocoRoot.add(bodies[b]);
    } else {
      bodies[0].add(bodies[b]);
    }
  }

  parent.bodies = bodies;
  parent.lights = lights;
  parent.meshes = meshes;
  parent.mujocoRoot = mujocoRoot;

  const skybox = createSkyboxTexture(mujoco, mjModel);
  parent.scene.background = skybox;

  if (!mjModel || 'deleted' in mjModel) {
    throw new Error('loadSceneFromURL: mjModel is invalid or already deleted');
  }

  return [mjModel, mjData, bodies, lights];
}

function createHFieldGeometry(mjModel: MjModel, geomId: number): THREE.BufferGeometry | undefined {
  const hfieldId = mjModel.geom_dataid[geomId];
  if (hfieldId < 0) {
    return undefined;
  }

  const nrow = mjModel.hfield_nrow[hfieldId];
  const ncol = mjModel.hfield_ncol[hfieldId];
  if (nrow < 2 || ncol < 2) {
    return undefined;
  }

  const sx = mjModel.hfield_size[hfieldId * 4 + 0];
  const sy = mjModel.hfield_size[hfieldId * 4 + 1];
  const sz = mjModel.hfield_size[hfieldId * 4 + 2];
  const adr = mjModel.hfield_adr[hfieldId];
  const data = mjModel.hfield_data.subarray(adr, adr + nrow * ncol);

  const positions = new Float32Array(nrow * ncol * 3);
  const uvs = new Float32Array(nrow * ncol * 2);
  const dx = ncol > 1 ? (2 * sx) / (ncol - 1) : 0;
  const dy = nrow > 1 ? (2 * sy) / (nrow - 1) : 0;

  let vertexOffset = 0;
  let uvOffset = 0;
  for (let r = 0; r < nrow; r++) {
    const yMj = dy * r - sy;
    for (let c = 0; c < ncol; c++) {
      const xMj = dx * c - sx;
      const zMj = data[r * ncol + c] * sz;

      positions[vertexOffset++] = xMj;
      positions[vertexOffset++] = zMj;
      positions[vertexOffset++] = -yMj;

      uvs[uvOffset++] = ncol > 1 ? c / (ncol - 1) : 0;
      uvs[uvOffset++] = nrow > 1 ? r / (nrow - 1) : 0;
    }
  }

  const indexCount = (nrow - 1) * (ncol - 1) * 6;
  const indices = indexCount > 65535
    ? new Uint32Array(indexCount)
    : new Uint16Array(indexCount);

  let indexOffset = 0;
  for (let r = 0; r < nrow - 1; r++) {
    for (let c = 0; c < ncol - 1; c++) {
      const i0 = r * ncol + c;
      const i1 = i0 + 1;
      const i2 = i0 + ncol;
      const i3 = i2 + 1;

      indices[indexOffset++] = i0;
      indices[indexOffset++] = i1;
      indices[indexOffset++] = i3;
      indices[indexOffset++] = i0;
      indices[indexOffset++] = i3;
      indices[indexOffset++] = i2;
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
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
