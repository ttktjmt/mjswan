/**
 * Types for the `mjswan/manifest` build (`dist/manifest.js`).
 *
 * `parseManifest(manifest, byteSource, { remote }) -> Catalog` turns a build's
 * `manifest.json` plus a byte source into the typed catalog of
 * `SceneInput`/`PolicyInput` the engine consumes, refusing a document that names
 * anything outside itself. `documentUrl` is where a host fetches a document's file;
 * `engineUrl` and `compareVersions` pick an engine by a document's `version`.
 * Framework-free; shared by the in-repo app and mjswan Cloud. See src/manifest/.
 */
export * from './src/manifest/index';
