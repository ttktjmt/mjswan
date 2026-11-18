/**
 * How a scene is drawn, its manifest's `graphics`: as MuJoCo's classic renderer draws
 * it, or with its Filament renderer's colour management, tone mapping and exposure. The
 * look itself (textures, lights, `<visual>`) is the model's either way.
 */
import * as THREE from 'three';
import type { MainModule, MjModel } from 'mujoco';
import { sampledCube } from './cubeMap';
import { createHaze, placeHaze } from './haze';
import { VIEWER_GEOM_GROUPS } from './layers';
import { Mirror } from './mirror';
import { modelZfar, recentrePlane } from './plane';
import { addShaderHook } from './shaderHook';
import { linearCube } from './textures';
import {
  DEFAULT_GRAPHICS,
  decodesBaseColor,
  type GraphicsAvailable,
  type GraphicsSettings,
  type ToneMapping,
} from './graphicsSettings';

export * from './graphicsSettings';

const THREE_TONE_MAPPING: Readonly<Record<ToneMapping, THREE.ToneMapping>> = {
  none: THREE.NoToneMapping,
  linear: THREE.LinearToneMapping,
  reinhard: THREE.ReinhardToneMapping,
  cineon: THREE.CineonToneMapping,
  aces_filmic: THREE.ACESFilmicToneMapping,
  agx: THREE.AgXToneMapping,
  neutral: THREE.NeutralToneMapping,
  // Filament's ACES is three's without the 1/0.6 three brightens it by (`applyToRenderer`).
  aces: THREE.ACESFilmicToneMapping,
  filmic: THREE.CustomToneMapping,
};

// Filament's `filmic` is Narkowicz 2015, which three.js lacks: its custom operator, for every material.
THREE.ShaderChunk.tonemapping_pars_fragment = THREE.ShaderChunk.tonemapping_pars_fragment.replace(
  'vec3 CustomToneMapping( vec3 color ) { return color; }',
  `vec3 CustomToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( ( color * ( 2.51 * color + 0.03 ) ) / ( color * ( 2.43 * color + 0.59 ) + 0.14 ) );
}`,
);

/** The renderer's share: tone mapping, exposure and the output's encoding. */
export function applyToRenderer(renderer: THREE.WebGLRenderer, settings: GraphicsSettings): void {
  const filament = settings.colorManagement === 'filament';
  renderer.toneMapping = filament ? THREE_TONE_MAPPING[settings.toneMapping] : THREE.NoToneMapping;
  renderer.toneMappingExposure = 2 ** settings.exposure * (settings.toneMapping === 'aces' ? 0.6 : 1);
  renderer.outputColorSpace =
    filament && settings.outputColorSpace === 'srgb' ? THREE.SRGBColorSpace : THREE.LinearSRGBColorSpace;
  decodesRgba(renderer).value = decodesBaseColor(settings);
}

const rgbaDecoding = new WeakMap<THREE.WebGLRenderer, { value: boolean }>();

function decodesRgba(renderer: THREE.WebGLRenderer): { value: boolean } {
  let uniform = rgbaDecoding.get(renderer);
  if (!uniform) rgbaDecoding.set(renderer, (uniform = { value: false }));
  return uniform;
}

/**
 * Have the renderer read `material`'s colour (its own, its vertices' or its instances') as
 * it reads a model's `rgba`: for the viewer's own drawings, whose colours are fixed.
 */
export function colorAsRgba<T extends THREE.Material>(material: T): T {
  addShaderHook(material, 'mjRgba', (shader, renderer) => {
    shader.uniforms.mjDecodeRgba = decodesRgba(renderer);
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform bool mjDecodeRgba;')
      .replace(
        '#include <color_fragment>',
        '#include <color_fragment>\nif ( mjDecodeRgba ) diffuseColor = sRGBTransferEOTF( diffuseColor );',
      );
  });
  return material;
}

/** Colour a material from the `rgba` it was made with (`userData.rgba`), decoded or as written. */
export function setBaseColor(material: THREE.Material, decode: boolean): void {
  const rgba = material.userData.rgba as number[] | undefined;
  const color = (material as THREE.MeshBasicMaterial).color;
  if (!rgba || !color) return;
  color.setRGB(rgba[0], rgba[1], rgba[2], decode ? THREE.SRGBColorSpace : THREE.LinearSRGBColorSpace);
}

/** The model's own pieces of a scene's graphics, built once per model. */
export class SceneGraphics {
  private settings: GraphicsSettings = { ...DEFAULT_GRAPHICS };
  private readonly planes: THREE.Object3D[] = [];
  private readonly mirror: Mirror | null = null;
  private readonly haze: { mesh: THREE.Mesh; plane: THREE.Object3D; distance: number } | null = null;
  /** Each sRGB texture's bytes uploaded again as written, for a camera sensor. */
  private readonly twins = new Map<THREE.Texture, THREE.Texture>();
  /**
   * Each sRGB cube texture a geom samples, and how to put another in its place: MuJoCo's
   * classic renderer uploads a cube or skybox as stored, and only a 2D texture as sRGB.
   */
  private readonly cubes: Array<{ texture: THREE.Texture; put: (texture: THREE.Texture) => void }> = [];
  /** The model's skybox, and the same decoded into linear light once Filament first draws it. */
  private readonly sky: THREE.CubeTexture | null;
  private linearSky: THREE.CubeTexture | null = null;
  private readonly eye = new THREE.Vector3();
  /** Which switches change anything in this model. */
  readonly available: GraphicsAvailable;

  constructor(
    mujoco: MainModule,
    mjModel: MjModel,
    private readonly root: THREE.Object3D,
    private readonly scene: THREE.Scene,
    private readonly lights: THREE.Light[],
  ) {
    const meshes = new Map<number, THREE.Mesh[]>();
    root.traverse((object) => {
      const g = object.userData.geomId as number | undefined;
      if (g === undefined || !(object as THREE.Mesh).isMesh) return;
      if (object.userData.recentre) this.planes.push(object);
      meshes.set(g, [...(meshes.get(g) ?? []), object as THREE.Mesh]);
    });

    const { mjGEOM_PLANE, mjGEOM_BOX } = mujoco.mjtGeom;
    let hazePlane: THREE.Object3D | null = null;
    for (let g = 0; g < mjModel.ngeom && !(this.mirror && hazePlane); g++) {
      const drawn = meshes.get(g);
      const type = mjModel.geom_type[g];
      if (!drawn || mjModel.geom_group[g] >= VIEWER_GEOM_GROUPS) continue;
      if (!hazePlane && type === mjGEOM_PLANE.value && !mjModel.geom_size[g * 3] && !mjModel.geom_size[g * 3 + 1]) {
        hazePlane = drawn[0];
      }
      if (this.mirror || (type !== mjGEOM_PLANE.value && type !== mjGEOM_BOX.value)) continue;
      // MuJoCo mirrors in the first opaque plane or box with a reflectance, and that one only.
      const matId = mjModel.geom_matid[g];
      const reflectance = matId >= 0 ? mjModel.mat_reflectance[matId] : 0;
      const alpha = matId >= 0 ? mjModel.mat_rgba[matId * 4 + 3] : mjModel.geom_rgba[g * 4 + 3];
      if (reflectance > 0 && alpha >= 0.995) {
        const height = type === mjGEOM_BOX.value ? mjModel.geom_size[g * 3 + 2] : 0;
        this.mirror = new Mirror(drawn[0], drawn, height, reflectance);
      }
    }

    this.forEachGeomMaterial((material) => {
      const cube = sampledCube(material);
      const sampled = cube?.value;
      if (cube && sampled?.colorSpace === THREE.SRGBColorSpace) {
        this.cubes.push({ texture: sampled, put: (texture) => (cube.value = texture as THREE.CubeTexture) });
      }
      const envMap = material.envMap;
      if (envMap?.colorSpace === THREE.SRGBColorSpace) {
        this.cubes.push({ texture: envMap, put: (texture) => (material.envMap = texture) });
      }
    });
    this.sky = scene.background instanceof THREE.CubeTexture ? scene.background : null;

    let skybox = false;
    for (let t = 0; t < mjModel.ntex; t++) {
      skybox ||= Number(mjModel.tex_type[t]) === mujoco.mjtTexture.mjTEXTURE_SKYBOX.value;
    }
    if (hazePlane && skybox) {
      const vis = mjModel.vis;
      const mesh = createHaze(vis.quality.numslices, vis.map.haze, vis.rgba.haze);
      colorAsRgba(mesh.material as THREE.Material);
      root.add(mesh);
      // MuJoCo's skybox and haze stand at 0.7 of its far plane.
      this.haze = { mesh, plane: hazePlane, distance: 0.7 * modelZfar(mjModel) };
    }
    this.available = {
      shadows: lights.some((light) => !!light.userData.castShadow),
      reflections: this.mirror !== null,
      haze: this.haze !== null,
    };
  }

  apply(settings: GraphicsSettings): void {
    this.settings = settings;
    const decode = decodesBaseColor(settings);
    const classic = settings.colorManagement === 'classic';
    this.forEachGeomMaterial((material) => setBaseColor(material, decode));
    for (const cube of this.cubes) cube.put(classic ? this.twin(cube.texture) : cube.texture);
    // An AR session's passthrough holds the sky meanwhile, and puts back the one it took.
    if (this.sky?.colorSpace === THREE.SRGBColorSpace && this.scene.background !== null) {
      // three tone maps no background it decodes in hardware, so Filament's is decoded here.
      this.scene.background = classic ? this.twin(this.sky) : (this.linearSky ??= linearCube(this.sky));
    }
    for (const light of this.lights) light.castShadow = !!light.userData.castShadow && settings.shadows;
  }

  /** Ready the frame `camera` is about to draw: the infinite planes under it, the haze, the reflection. */
  beforeRender(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, presenting: boolean): void {
    camera.updateWorldMatrix(true, false);
    const eye = this.eye.setFromMatrixPosition(camera.matrixWorld);
    for (const plane of this.planes) recentrePlane(plane, eye);
    if (this.haze) {
      const { mesh, plane, distance } = this.haze;
      // No sky, no haze: an AR session's passthrough takes the skybox away.
      mesh.visible =
        this.settings.haze &&
        this.scene.background !== null &&
        placeHaze(mesh, plane, eye, Math.min(distance, 0.7 * camera.far));
    }
    // A headset would draw the mirrored view once per eye, doubling the frame.
    this.mirror?.render(renderer, this.scene, camera, this.settings.reflections && !presenting);
  }

  /**
   * Draw as mjlab's camera sensors are, through mujoco_warp: `rgba` as written and
   * textures as stored, whatever the renderer, no reflection or haze (the haze is the
   * viewer's alone, by layer), and shadows by the sensor's `use_shadows`.
   */
  asSensor<T>(useShadows: boolean, draw: () => T): T {
    const undo: Array<() => void> = [];
    const decoded = decodesBaseColor(this.settings);
    // As simulated: what only the viewer hides (a skin's robot, an AR floor, a previewed geom) too.
    this.root.traverse((object) => {
      if (object.userData.geomId === undefined || object.visible) return;
      object.visible = true;
      undo.push(() => (object.visible = false));
    });
    this.forEachGeomMaterial((material) => {
      if (!material.visible) {
        material.visible = true;
        undo.push(() => (material.visible = false));
      }
      const { map, envMap } = material;
      if (map?.colorSpace === THREE.SRGBColorSpace) {
        material.map = this.twin(map);
        undo.push(() => (material.map = map));
      }
      if (envMap?.colorSpace === THREE.SRGBColorSpace) {
        material.envMap = this.twin(envMap);
        undo.push(() => (material.envMap = envMap));
      }
      const cube = sampledCube(material);
      const viewed = cube?.value;
      if (cube && viewed?.colorSpace === THREE.SRGBColorSpace) {
        cube.value = this.twin(viewed);
        undo.push(() => (cube.value = viewed));
      }
      if (material.userData.rgba) {
        setBaseColor(material, false);
        undo.push(() => setBaseColor(material, decoded));
      }
    });
    const background = this.scene.background;
    if (this.sky?.colorSpace === THREE.SRGBColorSpace && background !== null) {
      this.scene.background = this.twin(this.sky);
      undo.push(() => (this.scene.background = background));
    }
    for (const light of this.lights) {
      const casts = light.castShadow;
      light.castShadow = !!light.userData.castShadow && useShadows;
      undo.push(() => (light.castShadow = casts));
    }
    const resume = this.mirror?.suspend();
    try {
      return draw();
    } finally {
      for (const each of undo) each();
      resume?.();
    }
  }

  dispose(): void {
    this.mirror?.dispose();
    if (this.haze) {
      this.haze.mesh.removeFromParent();
      this.haze.mesh.geometry.dispose();
      (this.haze.mesh.material as THREE.Material).dispose();
    }
    for (const twin of this.twins.values()) twin.dispose();
    this.twins.clear();
    this.linearSky?.dispose();
  }

  private twin<T extends THREE.Texture>(texture: T): T {
    let twin = this.twins.get(texture) as T | undefined;
    if (!twin) {
      // Shares the bytes; a texture of another colour space is uploaded on its own.
      twin = texture.clone();
      twin.colorSpace = THREE.NoColorSpace;
      twin.needsUpdate = true;
      this.twins.set(texture, twin);
    }
    return twin;
  }

  private forEachGeomMaterial(visit: (material: THREE.MeshPhysicalMaterial) => void): void {
    this.root.traverse((object) => {
      if (object.userData.geomId === undefined) return;
      const material = (object as THREE.Mesh).material;
      for (const each of Array.isArray(material) ? material : [material]) {
        if ((each as THREE.MeshPhysicalMaterial | undefined)?.isMeshPhysicalMaterial) {
          visit(each as THREE.MeshPhysicalMaterial);
        }
      }
    });
  }
}
