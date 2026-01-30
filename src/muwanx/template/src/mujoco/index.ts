/**
 * MuJoCo WASM bindings for muwanx
 * Built from MuJoCo 3.4.0 with Emscripten 4.0.10
 */

// Re-export types from the generated declaration file
export type {
  MainModule,
  MjModel,
  MjData,
  MjOption,
  MjStatistic,
  MjVisual,
  MjContact,
  MjContactVec,
  MjvCamera,
  MjvOption,
  MjvScene,
  MjvPerturb,
  MjvFigure,
  ClassHandle,
  EmbindModule,
} from './mujoco_wasm.d.ts';

// Import the WASM module loader
import loadMujocoModule from './mujoco_wasm.js';

// Import the WASM file URL using Vite's asset import
import wasmUrl from './mujoco_wasm.wasm?url';

// Import the MainModule type for return type annotation
import type { MainModule as MujocoMainModule } from './mujoco_wasm.d.ts';

/**
 * Load and initialize the MuJoCo WASM module
 * This wrapper ensures the WASM file is properly located in both dev and prod
 */
export default async function loadMujoco(): Promise<MujocoMainModule> {
  return loadMujocoModule({
    locateFile: (path: string) => {
      if (path.endsWith('.wasm')) {
        return wasmUrl;
      }
      return path;
    },
  });
}
