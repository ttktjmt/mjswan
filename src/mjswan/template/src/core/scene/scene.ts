import * as THREE from 'three';
import type { MainModule, MjData, MjModel } from 'mujoco';
import { sampleCubeTexture } from './cubeMap';
import { geomRgba } from './geomRgba';
import { createLights, lightSpecularRatio } from './lights';
import { createTexture, createSkyboxTexture } from './textures';
import { createTendonMeshes } from './tendons';
import { primitiveGeometry, primitiveScale } from './primitive';
import { modelZfar, planeGeometry } from './plane';
import { VIEWER_GEOM_GROUPS, geomGroupLayer } from './layers';

/** How a scene's model bytes are encoded: `add_scene(spec=...)` writes `mjz`, `model=` `mjb`. */
export type ModelFormat = 'mjz' | 'mjb';

export function reflectanceParams(
  mjModel: MjModel | null,
  matId: number,
  specularRatio = 1
): Pick<THREE.MeshPhysicalMaterialParameters, 'specularIntensity' | 'roughness' | 'metalness'> {
  // No material, or no model (a geom drawn from outside it), takes MuJoCo's defaults.
  const material = matId !== -1 ? mjModel : null;
  const specular = material?.mat_specular?.[matId] ?? 0.5;
  const shininess = material?.mat_shininess?.[matId] ?? 0.5;
  const metallic = material?.mat_metallic?.[matId] ?? -1;
  const roughnessAttr = material?.mat_roughness?.[matId] ?? -1;

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

/** What drawing one geom needs from the scene it is drawn into. */
type GeomMeshContext = {
  mujoco: MainModule;
  mjModel: MjModel;
  bodies: Record<number, THREE.Group>;
  meshes: Record<number, THREE.BufferGeometry>;
  specularRatio: number;
  /** `mjModel.names`, read once for every body. */
  names: Uint8Array;
  /** MuJoCo's far plane, `vis.map.zfar` times `stat.extent`: how far an infinite plane reaches. */
  zfar: number;
};


/** The viewer's own bodies, the pointer's anchor and the tracked hands, are named so. */
const INJECTED_PREFIX = 'mjswan_';

/** The `stat.meansize` MuJoCo leaves a model with no body but the world. */
const NO_BODY_MEANSIZE = 0.2;

/**
 * Give the model back its shipped `stat.extent`: MuJoCo's `setStat` counts the viewer's
 * bodies parked 100 m up, stretching all it sizes (infinite planes, haze, shadow maps) a
 * hundredfold. Recomputed as MuJoCo does without them; left as compiled when the sum with
 * them is not MuJoCo's (an author's `<statistic>`, flex edges).
 */
function restoreShippedExtent(mujoco: MainModule, mjModel: MjModel, mjData: MjData, names: Uint8Array): void {
  const injected = new Set<number>();
  for (let b = 1; b < mjModel.nbody; b++) {
    if (bodyName(mjModel, names, b).startsWith(INJECTED_PREFIX)) injected.add(b);
  }
  if (injected.size === 0) return;
  mujoco.mj_kinematics(mjModel, mjData);
  // Read once: each read of a field makes a view.
  const { xpos, xipos, xanchor, site_xpos: sitePos, geom_xpos: geomPos } = mjData;
  const { jnt_bodyid: jointBody, site_bodyid: siteBody, geom_bodyid: geomBody, body_parentid: parent } = mjModel;
  const rbound = Array.from({ length: mjModel.ngeom }, (_, g) => geomRbound(mujoco, mjModel, g));
  const distance = (a: ArrayLike<number>, i: number, b: ArrayLike<number>, j: number) =>
    Math.hypot(a[i * 3] - b[j * 3], a[i * 3 + 1] - b[j * 3 + 1], a[i * 3 + 2] - b[j * 3 + 2]);
  // Each body's size: its farthest joint from its centre of mass, and its geoms' reach.
  const size = new Float64Array(mjModel.nbody);
  for (let j = 0; j < mjModel.njnt; j++) {
    for (const b of [jointBody[j], parent[jointBody[j]]]) size[b] = Math.max(size[b], distance(xipos, b, xanchor, j));
  }
  size[0] = 0;
  for (let g = 0; g < mjModel.ngeom; g++) {
    const b = geomBody[g];
    if (b > 0 && mjModel.geom_rbound[g] > 0) size[b] = Math.max(size[b], mjModel.geom_rbound[g] + distance(xipos, b, geomPos, g));
  }

  const extent = (skip: (body: number) => boolean) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    const add = (xyz: ArrayLike<number>, i: number, r = 0) => {
      for (let k = 0; k < 3; k++) {
        min[k] = Math.min(min[k], xyz[i * 3 + k] - r);
        max[k] = Math.max(max[k], xyz[i * 3 + k] + r);
      }
    };
    let bodies = 0;
    let meansize = 0;
    for (let b = 1; b < mjModel.nbody; b++) {
      if (skip(b)) continue;
      add(xpos, b);
      add(xipos, b);
      bodies += 1;
      meansize += Math.max(size[b], 1e-5);
    }
    meansize = bodies ? meansize / bodies : NO_BODY_MEANSIZE;
    for (let j = 0; j < mjModel.njnt; j++) if (!skip(jointBody[j])) add(xanchor, j);
    for (let i = 0; i < mjModel.nsite; i++) if (!skip(siteBody[i])) add(sitePos, i);
    for (let g = 0; g < mjModel.ngeom; g++) if (!skip(geomBody[g])) add(geomPos, g, rbound[g]);
    const box = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    return Math.max(Number.isFinite(box) ? box : 0, 1e-5, 2 * meansize);
  };
  const stat = mjModel.stat;
  const full = extent(() => false);
  if (Math.abs(stat.extent - full) > 1e-9 * full) return;
  stat.extent = extent((b) => injected.has(b));
}

/** The radius `setStat` gives a geom: its own, or a tenth of a plane's or height field's largest size. */
function geomRbound(mujoco: MainModule, mjModel: MjModel, g: number): number {
  if (mjModel.geom_rbound[g] > 0) return mjModel.geom_rbound[g];
  const type = mjModel.geom_type[g];
  if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) {
    const [x, y] = [mjModel.geom_size[g * 3], mjModel.geom_size[g * 3 + 1]];
    return x || y ? Math.max(x, y) * 0.1 : 0.01;
  }
  if (type === mujoco.mjtGeom.mjGEOM_HFIELD.value) {
    const h = mjModel.geom_dataid[g] * 4;
    return Math.max(...[0, 1, 2, 3].map((k) => mjModel.hfield_size[h + k])) * 0.1;
  }
  return 0;
}

function bodyName(mjModel: MjModel, names: Uint8Array, b: number): string {
  const start = mjModel.name_bodyadr[b];
  let end = start;
  while (end < names.length && names[end] !== 0) end++;
  return new TextDecoder('utf-8').decode(names.subarray(start, end));
}

/** A group the viewer draws stays on layer 0 as well; any other is a camera sensor's alone. */
function placeOnGroupLayer(object: THREE.Object3D, group: number): void {
  if (group < VIEWER_GEOM_GROUPS) object.layers.enable(geomGroupLayer(group));
  else object.layers.set(geomGroupLayer(group));
}

/** Add geom `g`'s mesh to its body, creating the body's group if it has none yet. */
function addGeomMesh(
  { mujoco, mjModel, bodies, meshes, specularRatio, names, zfar }: GeomMeshContext,
  g: number,
): void {
  const b = mjModel.geom_bodyid[g];
  const type = mjModel.geom_type[g];
  const size = [
    mjModel.geom_size[g * 3 + 0],
    mjModel.geom_size[g * 3 + 1],
    mjModel.geom_size[g * 3 + 2],
  ];

  if (!(b in bodies)) {
    bodies[b] = new THREE.Group();

    bodies[b].name = bodyName(mjModel, names, b);

    bodies[b].bodyID = b;
    bodies[b].has_custom_mesh = false;
  }

  const color = geomRgba(mjModel, g);
  // MuJoCo draws nothing for a geom whose colour is fully transparent.
  if (color[3] === 0) return;

  let geometry: THREE.BufferGeometry | undefined;
  const isPlane = type === mujoco.mjtGeom.mjGEOM_PLANE.value;
  let planeStep: [number, number] = [0, 0];
  switch (type) {
    case mujoco.mjtGeom.mjGEOM_PLANE.value: {
      const plane = planeGeometry(mjModel, g, zfar);
      geometry = plane.geometry;
      planeStep = plane.step;
      break;
    }
    case mujoco.mjtGeom.mjGEOM_HFIELD.value:
      geometry = createHFieldGeometry(mjModel, g);
      break;
    case mujoco.mjtGeom.mjGEOM_SPHERE.value:
      geometry = primitiveGeometry('sphere', size);
      break;
    case mujoco.mjtGeom.mjGEOM_CAPSULE.value:
      geometry = primitiveGeometry('capsule', size);
      break;
    case mujoco.mjtGeom.mjGEOM_ELLIPSOID.value:
      geometry = primitiveGeometry('ellipsoid', size);
      break;
    case mujoco.mjtGeom.mjGEOM_CYLINDER.value:
      geometry = primitiveGeometry('cylinder', size);
      break;
    case mujoco.mjtGeom.mjGEOM_BOX.value:
      geometry = primitiveGeometry('box', size);
      break;
    case mujoco.mjtGeom.mjGEOM_MESH.value: {
      const meshID = mjModel.geom_dataid[g];

      if (!(meshID in meshes)) {
        meshes[meshID] = createMeshGeometry(mjModel, meshID);
      }
      geometry = meshes[meshID];
      bodies[b].has_custom_mesh = true;
      break;
    }
  }

  let texture: THREE.Texture | null = null;
  if (mjModel.geom_matid[g] !== -1) {
    const matId = mjModel.geom_matid[g];
    const role = mujoco.mjtTextureRole.mjTEXROLE_RGB.value;
    const texId = mjModel.mat_texid[matId * mujoco.mjtTextureRole.mjNTEXROLE.value + role];
    if (texId !== -1) {
      // A box samples a cube texture as MuJoCo does, so it keeps MuJoCo's face order.
      const cubeFaceOrder =
        type === mujoco.mjtGeom.mjGEOM_BOX.value ? [0, 1, 2, 3, 4, 5] : undefined;
      texture = createTexture({ mujoco, mjModel, texId, cubeFaceOrder });
      if (texture) {
        // A plane's coordinates already carry `texrepeat`, as MuJoCo generates them.
        if (!isPlane) {
          const repeatX = mjModel.mat_texrepeat ? mjModel.mat_texrepeat[matId * 2 + 0] : 1;
          const repeatY = mjModel.mat_texrepeat ? mjModel.mat_texrepeat[matId * 2 + 1] : 1;
          texture.repeat.set(repeatX, repeatY);
        }
        texture.wrapS = THREE.RepeatWrapping;
        texture.wrapT = THREE.RepeatWrapping;
      }
    }
  }

  const currentMaterial = new THREE.MeshPhysicalMaterial({
    color: new THREE.Color(color[0], color[1], color[2]),
    transparent: color[3] < 1.0,
    opacity: color[3],
    ...reflectanceParams(mjModel, mjModel.geom_matid[g], specularRatio),
  });
  // As written: the scene's graphics settings decide how it is read (`setBaseColor`).
  currentMaterial.userData.rgba = [color[0], color[1], color[2]];

  if (texture) {
    if (!(texture instanceof THREE.CubeTexture)) {
      currentMaterial.map = texture;
    } else if (type === mujoco.mjtGeom.mjGEOM_BOX.value) {
      // MuJoCo samples at the point on the unit box, or on the box itself with `texuniform`.
      const uniform = mjModel.mat_texuniform[mjModel.geom_matid[g]];
      sampleCubeTexture(currentMaterial, texture, uniform ? [1, 1, 1] : size);
    } else {
      currentMaterial.envMap = texture;
      currentMaterial.envMapIntensity =
        mjModel.geom_matid[g] !== -1
          ? (mjModel.mat_reflectance?.[mjModel.geom_matid[g]] ?? 0)
          : 0;
    }
  }

  if (!geometry) {
    console.warn(`Skipping geometry ${g} (type ${type}): no valid geometry created`);
    return;
  }

  const mesh = new THREE.Mesh(geometry, currentMaterial as THREE.Material);
  placeOnGroupLayer(mesh, mjModel.geom_group[g]);
  const ignoreDragForce = bodies[b].name === 'terrain';

  // Nothing is under an infinite plane to shadow, and drawn into the map it shadows itself.
  const infinite = planeStep[0] > 0 || planeStep[1] > 0;
  mesh.castShadow = ignoreDragForce || infinite ? false : g !== 0;
  mesh.receiveShadow = type !== mujoco.mjtGeom.mjGEOM_MESH.value;
  mesh.bodyID = b;
  // The geom behind this mesh, so a size written at runtime can be drawn.
  mesh.userData.geomId = g;
  mesh.userData.ignoreDragForce = ignoreDragForce;
  bodies[b].userData.ignoreDragForce = ignoreDragForce;
  bodies[b].add(mesh);
  getPosition(mjModel.geom_pos, g, mesh.position);
  getQuaternion(mjModel.geom_quat, g, mesh.quaternion);
  if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
    primitiveScale('ellipsoid', size, mesh.scale);
  }
  if (isPlane) {
    const frontMat = currentMaterial;
    const backMat = frontMat.clone();
    backMat.side = THREE.BackSide;
    backMat.transparent = true;
    backMat.opacity = frontMat.opacity * 0.35;
    backMat.depthWrite = false;
    const backMesh = new THREE.Mesh(geometry, backMat);
    backMesh.userData.geomId = g;
    backMesh.position.copy(mesh.position);
    backMesh.quaternion.copy(mesh.quaternion);
    backMesh.raycast = () => {};
    placeOnGroupLayer(backMesh, mjModel.geom_group[g]);
    bodies[b].add(backMesh);
    for (const side of [mesh, backMesh]) {
      // Read by `Passthrough`, which hides these so an AR session shows the room's floor.
      side.userData.isGroundPlane = true;
      if (infinite) {
        side.userData.recentre = { origin: side.position.clone(), step: planeStep };
        side.userData.unbounded = true;
      }
    }
  }
}

/**
 * Draw the geoms of `groups` the viewer leaves out (3 and up, collision by convention) on
 * their group's layer alone, for a camera sensor that enables it. Once per model: a group
 * already drawn is skipped.
 */
export function addSensorGeomMeshes(
  mujoco: MainModule,
  mjModel: MjModel,
  parent: { bodies: Record<number, THREE.Group>; meshes?: Record<number, THREE.BufferGeometry> },
  groups: Iterable<number>,
): void {
  const wanted = new Set([...groups].filter((group) => group >= VIEWER_GEOM_GROUPS));
  for (const body of Object.values(parent.bodies)) {
    body.traverse((child) => {
      const geomId = child.userData.geomId as number | undefined;
      if (geomId !== undefined) wanted.delete(mjModel.geom_group[geomId]);
    });
  }
  if (wanted.size === 0) return;
  const context: GeomMeshContext = {
    mujoco,
    mjModel,
    bodies: parent.bodies,
    meshes: (parent.meshes ??= {}),
    specularRatio: lightSpecularRatio(mjModel),
    names: new Uint8Array(mjModel.names),
    zfar: modelZfar(mjModel),
  };
  for (let g = 0; g < mjModel.ngeom; g++) {
    if (wanted.has(mjModel.geom_group[g])) addGeomMesh(context, g);
  }
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

  const names = new Uint8Array(mjModel.names);
  restoreShippedExtent(mujoco, mjModel, mjData, names);

  const mujocoRoot = new THREE.Group();
  mujocoRoot.name = 'MuJoCo Root';
  parent.scene.add(mujocoRoot);

  const bodies: Record<number, THREE.Group> = {};
  const meshes: Record<number, THREE.BufferGeometry> = {};
  const specularRatio = lightSpecularRatio(mjModel);

  const geomContext: GeomMeshContext = {
    mujoco,
    mjModel,
    bodies,
    meshes,
    specularRatio,
    names,
    zfar: modelZfar(mjModel),
  };
  for (let g = 0; g < mjModel.ngeom; g++) {
    if (mjModel.geom_group[g] < VIEWER_GEOM_GROUPS) addGeomMesh(geomContext, g);
  }

  createTendonMeshes(mujocoRoot, mjModel);

  const lights: THREE.Light[] = createLights({ mujoco, mjModel, mujocoRoot, bodies });

  for (let b = 0; b < mjModel.nbody; b++) {
    if (!bodies[b]) {
      bodies[b] = new THREE.Group();
      bodies[b].name = bodyName(mjModel, names, b);
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

/** A mesh's three geometry, Y-up, built off copies of the model's buffers. */
export function createMeshGeometry(mjModel: MjModel, meshID: number): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Copies, turned Y-up below: MuJoCo collides against the model's own vertices.
  const vertexBuffer = mjModel.mesh_vert.slice(
    mjModel.mesh_vertadr[meshID] * 3,
    (mjModel.mesh_vertadr[meshID] + mjModel.mesh_vertnum[meshID]) * 3
  );
  for (let v = 0; v < vertexBuffer.length; v += 3) {
    const temp = vertexBuffer[v + 1];
    vertexBuffer[v + 1] = vertexBuffer[v + 2];
    vertexBuffer[v + 2] = -temp;
  }

  const normalBuffer = mjModel.mesh_normal.slice(
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
  return geometry;
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
  buffer: ArrayLike<number>,
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
