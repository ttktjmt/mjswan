/** `mujoco/mt` in an SPA built without `MJSWAN_MT=1` (see `vite.config.ts`). */
export default function loadMujoco(): never {
  throw new Error('mjswan: this app was built without multithreading (Builder(mt=True)).');
}
