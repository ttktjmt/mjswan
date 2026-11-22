/**
 * Types for the `mjswan/manifest` build (`dist/manifest.js`).
 *
 * `parseManifest(manifest, byteSource) -> Catalog` turns a build's `manifest.json`
 * plus a byte source into the typed catalog of `SceneInput`/`PolicyInput` the
 * engine consumes. Framework-free; shared by the in-repo app and mjswan Cloud.
 * See src/manifest/.
 */
export * from './src/manifest/index';
