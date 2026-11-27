/**
 * A glTF parsed from its own bytes alone. A document's skins and colliders are `.glb` and
 * `.vrm` files that embed every buffer and image; one naming another file, by URL or by
 * path, would have the viewer fetch from wherever the file says. Such a file is refused
 * and nothing it names is fetched.
 */
import * as THREE from 'three';
import { GLTFLoader, type GLTF, type GLTFLoaderPlugin, type GLTFParser } from 'three/examples/jsm/loaders/GLTFLoader.js';

/** What a glTF's own bytes become while it is parsed: a `data:` URI, or a `blob:` URL for an embedded image. */
const SEALED = /^(?:data|blob):/i;

/**
 * Parse `data`, a binary glTF, refusing it when it names a buffer or an image outside
 * itself. `plugin` is registered on the loader (the VRM extensions, for a VRM).
 */
export async function parseSealedGltf(
  data: ArrayBuffer,
  plugin?: (parser: GLTFParser) => GLTFLoaderPlugin,
): Promise<GLTF> {
  const named: string[] = [];
  // Every request the loader makes for a buffer or an image passes here first. Throwing
  // fails it before any is sent; a texture's failure only logs, so it is counted too.
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
