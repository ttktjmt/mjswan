/**
 * A glTF parsed from its own bytes alone: one naming a buffer or image elsewhere would have
 * the viewer fetch wherever the file says, so it is refused and nothing it names is fetched.
 */
import * as THREE from 'three';
import { GLTFLoader, type GLTF, type GLTFLoaderPlugin, type GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader.js';

/** A glTF's own bytes while it is parsed: `data:` URIs, and `blob:` URLs for embedded images. */
const SEALED = /^(?:data|blob):/i;

/** Parse the binary glTF `data`, refusing one that names a file outside itself; `plugin` extends the loader (VRM). */
export async function parseSealedGltf(
  data: ArrayBuffer,
  plugin?: (parser: GLTFParser) => GLTFLoaderPlugin,
): Promise<GLTF> {
  const named: string[] = [];
  // Every buffer and image request passes here first. Throwing stops it before it is sent;
  // three only logs a failed texture, so refusals are also counted and thrown after.
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => {
    if (SEALED.test(url)) return url;
    named.push(url);
    throw new Error(`mjswan: a skin or collider may not load ${url}.`);
  });
  const loader = new GLTFLoader(manager);
  if (plugin) loader.register(plugin);
  let gltf: GLTF | undefined;
  let failure: unknown;
  try {
    gltf = await loader.parseAsync(data, '');
  } catch (error) {
    failure = error;
  }
  if (named.length) {
    throw new Error(
      `mjswan: this glTF names files outside itself (${named.join(', ')}). A skin or a ` +
        'collider must embed every buffer and image: export it as one .glb.',
    );
  }
  if (gltf === undefined) throw failure;
  return gltf;
}
