/**
 * Type declarations for the mjswan library build (`dist/mjswan.js`).
 *
 * The default and named `createEngine` export prepares a headless, instance-
 * scoped simulation engine (MuJoCo WASM + WebGL, no React) that takes bytes and
 * exposes camera/command/playback verbs plus a subscribable state snapshot.
 * See src/engine/.
 */
export * from './src/engine';
export { default } from './src/engine';
