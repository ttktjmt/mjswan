import * as THREE from 'three';
import type { MainModule, MjModel } from 'mujoco';

interface CreateTextureParams {
  mujoco: MainModule;
  mjModel: MjModel;
  texId: number;
  /** A cube texture's faces, as indices into MuJoCo's order (+X, -X, +Y, -Y, +Z, -Z). */
  cubeFaceOrder?: number[];
}

function expandChannelsToRGBA(src: Uint8Array, dest: Uint8Array, nchannel: number): void {
  const pixelCount = dest.length / 4;

  switch (nchannel) {
    case 1:
      for (let p = 0; p < pixelCount; p++) {
        const l = src[p];
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = 255;
      }
      break;
    case 2:
      for (let p = 0; p < pixelCount; p++) {
        const l = src[p * 2 + 0];
        const a = src[p * 2 + 1];
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = a;
      }
      break;
    case 3:
      for (let p = 0; p < pixelCount; p++) {
        dest[p * 4 + 0] = src[p * 3 + 0];
        dest[p * 4 + 1] = src[p * 3 + 1];
        dest[p * 4 + 2] = src[p * 3 + 2];
        dest[p * 4 + 3] = 255;
      }
      break;
    case 4:
      dest.set(src);
      break;
    default:
      for (let p = 0; p < pixelCount; p++) {
        const l = p < src.length ? src[p] : 0;
        dest[p * 4 + 0] = l;
        dest[p * 4 + 1] = l;
        dest[p * 4 + 2] = l;
        dest[p * 4 + 3] = 255;
      }
  }
}

function create2DTexture(mjModel: MjModel, texId: number): THREE.DataTexture | null {
  const width = mjModel.tex_width ? Number(mjModel.tex_width[texId]) : 0;
  const height = mjModel.tex_height ? Number(mjModel.tex_height[texId]) : 0;
  if (!width || !height) {
    return null;
  }

  const texAdr = mjModel.tex_adr ? Number(mjModel.tex_adr[texId]) : 0;
  const nchannel = mjModel.tex_nchannel ? Number(mjModel.tex_nchannel[texId]) : 0;

  if (nchannel < 1 || nchannel > 4) {
    console.warn(`Invalid channel count ${nchannel} for texture ${texId}`);
    return null;
  }

  const pixelCount = width * height;
  const srcByteCount = pixelCount * nchannel;

  if (!mjModel.tex_data || mjModel.tex_data.length < texAdr + srcByteCount) {
    console.warn(`Insufficient texture data for texture ${texId}`);
    return null;
  }

  const src = mjModel.tex_data.subarray(texAdr, texAdr + srcByteCount);
  const textureData = new Uint8Array(pixelCount * 4);
  expandChannelsToRGBA(src, textureData, nchannel);

  const texture = new THREE.DataTexture(
    textureData,
    width,
    height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  texture.needsUpdate = true;
  texture.flipY = false;
  texture.anisotropy = 4;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.generateMipmaps = true;

  return texture;
}

function createCubeTexture(mjModel: MjModel, texId: number, faceOrder = [3, 2, 0, 1, 4, 5]): THREE.CubeTexture | null {
  const width = mjModel.tex_width ? Number(mjModel.tex_width[texId]) : 0;
  const height = mjModel.tex_height ? Number(mjModel.tex_height[texId]) : 0;

  if (!width || !height) {
    return null;
  }

  let faceSize = width;
  let faceHeight: number;
  let isRepeated: boolean;

  if (height === width) {
    isRepeated = true;
    faceHeight = height;
  } else if (height === 6 * width) {
    isRepeated = false;
    faceHeight = width;
  } else {
    console.warn(`Invalid dimensions for cube texture texId ${texId}: ${width}x${height}`);
    return null;
  }

  if (faceSize !== faceHeight) {
    console.warn(`Non-square faces for cube texture texId ${texId}`);
    return null;
  }

  const texAdr = mjModel.tex_adr ? Number(mjModel.tex_adr[texId]) : 0;
  const nchannel = mjModel.tex_nchannel ? Number(mjModel.tex_nchannel[texId]) : 0;

  const facePixelCount = faceSize * faceHeight;
  const faceSrcByteCount = facePixelCount * nchannel;

  const faces: Uint8Array[] = [];

  for (let faceIdx = 0; faceIdx < 6; faceIdx++) {
    const faceOffset = texAdr + (isRepeated ? 0 : faceIdx * faceSrcByteCount);

    if (!mjModel.tex_data || mjModel.tex_data.length < faceOffset + faceSrcByteCount) {
      console.warn(`Insufficient texture data for cube face ${faceIdx} in texId ${texId}`);
      return null;
    }

    const src = mjModel.tex_data.subarray(faceOffset, faceOffset + faceSrcByteCount);
    const faceData = new Uint8Array(facePixelCount * 4);
    expandChannelsToRGBA(src, faceData, nchannel);
    faces.push(faceData);
  }

  const reorderedFaces = faceOrder.map(i => faces[i]);

  const cubeTexture = new THREE.CubeTexture();

  const probe = document.createElement('canvas');
  if (!probe.getContext('2d')) {
    console.warn(`2D canvas context unavailable; cannot create cube texture texId ${texId}`);
    return null;
  }

  const images: HTMLCanvasElement[] = [];
  for (let i = 0; i < reorderedFaces.length; i++) {
    const faceData = reorderedFaces[i];
    const canvas = document.createElement('canvas');
    canvas.width = faceSize;
    canvas.height = faceHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.warn(`Failed to acquire 2D context for cube face ${i} in texId ${texId}`);
      return null;
    }
    const imageData = ctx.createImageData(faceSize, faceHeight);
    imageData.data.set(faceData);
    ctx.putImageData(imageData, 0, 0);
    images.push(canvas);
  }

  cubeTexture.image = images as unknown as HTMLImageElement[];

  if (cubeTexture.image.some((img) => img === null)) {
    console.warn(`Failed to create canvas for one or more faces in texId ${texId}`);
    return null;
  }

  cubeTexture.needsUpdate = true;
  cubeTexture.format = THREE.RGBAFormat;
  cubeTexture.flipY = false;
  cubeTexture.magFilter = THREE.LinearFilter;
  cubeTexture.minFilter = THREE.LinearFilter;
  cubeTexture.generateMipmaps = false;

  return cubeTexture;
}

/**
 * MuJoCo's Filament renderer decodes an sRGB texture as it is sampled, and reads any other
 * (`linear`, or `auto` on an image that does not say) as written; its classic renderer
 * does so for a 2D texture, and uploads a cube or skybox as stored (`SceneGraphics.apply`).
 */
function colorSpaceOf(mujoco: MainModule, mjModel: MjModel, texId: number): THREE.ColorSpace {
  const srgb = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB.value;
  return Number(mjModel.tex_colorspace?.[texId]) === srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
}

export function createTexture({
  mujoco,
  mjModel,
  texId,
  cubeFaceOrder,
}: CreateTextureParams): THREE.Texture | null {
  if (!mjModel || texId < 0) {
    return null;
  }

  const type = mjModel.tex_type ? Number(mjModel.tex_type[texId]) : mujoco.mjtTexture.mjTEXTURE_2D.value;

  let texture: THREE.Texture | null = null;
  if (type === mujoco.mjtTexture.mjTEXTURE_2D.value) {
    texture = create2DTexture(mjModel, texId);
  } else if (type === mujoco.mjtTexture.mjTEXTURE_CUBE.value) {
    texture = createCubeTexture(mjModel, texId, cubeFaceOrder);
  } else {
    console.warn(`Unsupported texture type ${type} for texId: ${texId}`);
  }
  if (texture) texture.colorSpace = colorSpaceOf(mujoco, mjModel, texId);
  return texture;
}

export function createSkyboxTexture(mujoco: MainModule, mjModel: MjModel): THREE.CubeTexture | null {
  if (!mjModel.tex_type) {
    return null;
  }
  for (let i = 0; i < mjModel.ntex; i++) {
    if (Number(mjModel.tex_type[i]) === mujoco.mjtTexture.mjTEXTURE_SKYBOX.value) {
      // Use identity face order: MuJoCo stores skybox faces with +Z (index 2) as up and -Z (index 3) as down,
      // which maps directly to Three.js +Y (index 2) up and -Y (index 3) down in the CubeTexture layout.
      const cube = createCubeTexture(mjModel, i, [0, 1, 2, 3, 4, 5]);
      if (cube) {
        // Flip faces horizontally: MuJoCo's cubemap is outside-in, but a Three.js skybox is viewed from inside.
        const faces = cube.image as unknown as HTMLCanvasElement[];
        for (let f = 0; f < faces.length; f++) {
          const src = faces[f];
          const flipped = document.createElement('canvas');
          flipped.width = src.width;
          flipped.height = src.height;
          const ctx = flipped.getContext('2d')!;
          ctx.translate(src.width, 0);
          ctx.scale(-1, 1);
          ctx.drawImage(src, 0, 0);
          faces[f] = flipped;
        }
        cube.colorSpace = colorSpaceOf(mujoco, mjModel, i);
        cube.needsUpdate = true;
      }
      return cube;
    }
  }
  return null;
}

/**
 * An sRGB cube texture decoded into linear half floats, read as written: what Filament
 * draws an sRGB sky as. Its faces are the canvases `createCubeTexture` drew.
 */
export function linearCube(cube: THREE.CubeTexture): THREE.CubeTexture {
  const decode = new Uint16Array(256);
  const colour = new THREE.Color();
  for (let i = 0; i < 256; i++) {
    decode[i] = THREE.DataUtils.toHalfFloat(colour.setRGB(i / 255, i / 255, i / 255, THREE.SRGBColorSpace).r);
  }
  const faces = (cube.image as unknown as HTMLCanvasElement[]).map((canvas) => {
    const { data, width, height } = canvas.getContext('2d')!.getImageData(0, 0, canvas.width, canvas.height);
    const half = new Uint16Array(data.length);
    for (let i = 0; i < data.length; i += 4) {
      half[i] = decode[data[i]];
      half[i + 1] = decode[data[i + 1]];
      half[i + 2] = decode[data[i + 2]];
      half[i + 3] = THREE.DataUtils.toHalfFloat(data[i + 3] / 255);
    }
    return new THREE.DataTexture(half, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  });
  const linear = new THREE.CubeTexture(faces as unknown as HTMLImageElement[]);
  linear.format = THREE.RGBAFormat;
  linear.type = THREE.HalfFloatType;
  linear.colorSpace = THREE.LinearSRGBColorSpace;
  linear.flipY = cube.flipY;
  linear.magFilter = THREE.LinearFilter;
  linear.minFilter = THREE.LinearFilter;
  linear.generateMipmaps = false;
  linear.needsUpdate = true;
  return linear;
}
