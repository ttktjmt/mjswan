import type { MjModel, MjData, MjVFS } from '@/mujoco';

// Re-export types for convenience
export type { MjModel, MjData, MjVFS };

// Emscripten File System interface (permissive)
interface EmscriptenFS {
  mkdir(path: string, mode?: number): void;
  mount(type: unknown, opts: { root: string }, mountpoint: string): void;
  readFile(path: string, opts?: Record<string, unknown>): Uint8Array | string;
  writeFile(path: string, data: ArrayBufferView | string, opts?: Record<string, unknown>): void;
  unlink(path: string): void;
  rmdir(path: string): void;
  stat(path: string, dontFollow?: unknown): unknown;
  readdir(path: string): string[];
  analyzePath(path: string, dontResolveLastLink?: boolean): { exists: boolean; isRoot: boolean; error: number };
  [key: string]: unknown;
}

// Emscripten MEMFS interface
interface EmscriptenMEMFS {
  [key: string]: unknown;
}

// MjVFS interface for virtual file system
interface MjVFSInterface {
  makeEmptyFile(filename: string, filesize: number): void;
  [key: string]: unknown;
}

// MjModel constructor interface
interface MjModelConstructor {
  new(model: MjModel): MjModel;
  mj_loadXML(path: string): MjModel;
  mj_loadBinary(path: string, vfs: MjVFSInterface): MjModel;
}

// MjData constructor interface
interface MjDataConstructor {
  new(model: MjModel | null): MjData;
  new(model: MjModel, data: MjData): MjData;
}

// MuJoCo enum value type
interface MjEnumValue {
  value: number;
}

// MjVFS constructor interface
interface MjVFSConstructor {
  new(): MjVFSInterface;
}

// Extended Mujoco interface with all necessary properties
export interface Mujoco {
  // Emscripten filesystem
  FS: EmscriptenFS;
  MEMFS: EmscriptenMEMFS;

  // Class constructors
  MjModel: MjModelConstructor;
  MjData: MjDataConstructor;
  MjVFS: MjVFSConstructor;

  // Core simulation functions
  mj_forward(model: MjModel, data: MjData): void;
  mj_step(model: MjModel, data: MjData): void;
  mj_step1(model: MjModel, data: MjData): void;
  mj_step2(model: MjModel, data: MjData): void;
  mj_resetData(model: MjModel, data: MjData): void;
  mj_resetDataKeyframe(model: MjModel, data: MjData, key: number): void;
  mj_name2id(model: MjModel, type: number, name: string): number;
  mj_id2name(model: MjModel, type: number, id: number): string | null;

  // Enums
  mjtObj: {
    mjOBJ_UNKNOWN: MjEnumValue;
    mjOBJ_BODY: MjEnumValue;
    mjOBJ_XBODY: MjEnumValue;
    mjOBJ_JOINT: MjEnumValue;
    mjOBJ_DOF: MjEnumValue;
    mjOBJ_GEOM: MjEnumValue;
    mjOBJ_SITE: MjEnumValue;
    mjOBJ_CAMERA: MjEnumValue;
    mjOBJ_LIGHT: MjEnumValue;
    mjOBJ_MESH: MjEnumValue;
    mjOBJ_SKIN: MjEnumValue;
    mjOBJ_HFIELD: MjEnumValue;
    mjOBJ_TEXTURE: MjEnumValue;
    mjOBJ_MATERIAL: MjEnumValue;
    mjOBJ_PAIR: MjEnumValue;
    mjOBJ_EXCLUDE: MjEnumValue;
    mjOBJ_EQUALITY: MjEnumValue;
    mjOBJ_TENDON: MjEnumValue;
    mjOBJ_ACTUATOR: MjEnumValue;
    mjOBJ_SENSOR: MjEnumValue;
    mjOBJ_NUMERIC: MjEnumValue;
    mjOBJ_TEXT: MjEnumValue;
    mjOBJ_TUPLE: MjEnumValue;
    mjOBJ_KEY: MjEnumValue;
    mjOBJ_PLUGIN: MjEnumValue;
    mjNOBJECT: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtGeom: {
    mjGEOM_PLANE: MjEnumValue;
    mjGEOM_HFIELD: MjEnumValue;
    mjGEOM_SPHERE: MjEnumValue;
    mjGEOM_CAPSULE: MjEnumValue;
    mjGEOM_ELLIPSOID: MjEnumValue;
    mjGEOM_CYLINDER: MjEnumValue;
    mjGEOM_BOX: MjEnumValue;
    mjGEOM_MESH: MjEnumValue;
    mjGEOM_SDF: MjEnumValue;
    mjNGEOMTYPES: MjEnumValue;
    mjGEOM_ARROW: MjEnumValue;
    mjGEOM_ARROW1: MjEnumValue;
    mjGEOM_ARROW2: MjEnumValue;
    mjGEOM_LINE: MjEnumValue;
    mjGEOM_SKIN: MjEnumValue;
    mjGEOM_LABEL: MjEnumValue;
    mjGEOM_TRIANGLE: MjEnumValue;
    mjGEOM_NONE: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtJoint: {
    mjJNT_FREE: MjEnumValue;
    mjJNT_BALL: MjEnumValue;
    mjJNT_SLIDE: MjEnumValue;
    mjJNT_HINGE: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtTrn: {
    mjTRN_JOINT: MjEnumValue;
    mjTRN_JOINTINPARENT: MjEnumValue;
    mjTRN_SLIDERCRANK: MjEnumValue;
    mjTRN_TENDON: MjEnumValue;
    mjTRN_SITE: MjEnumValue;
    mjTRN_BODY: MjEnumValue;
    mjTRN_UNDEFINED: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtLightType: {
    mjLIGHT_DIRECTIONAL: MjEnumValue;
    mjLIGHT_POINT: MjEnumValue;
    mjLIGHT_SPOT: MjEnumValue;
    mjLIGHT_IMAGE: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtTexture: {
    mjTEXTURE_2D: MjEnumValue;
    mjTEXTURE_CUBE: MjEnumValue;
    mjTEXTURE_SKYBOX: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtTextureRole: {
    mjTEXROLE_USER: MjEnumValue;
    mjTEXROLE_RGB: MjEnumValue;
    mjTEXROLE_OCCLUSION: MjEnumValue;
    mjTEXROLE_ROUGHNESS: MjEnumValue;
    mjTEXROLE_METALLIC: MjEnumValue;
    mjTEXROLE_NORMAL: MjEnumValue;
    mjTEXROLE_OPACITY: MjEnumValue;
    mjTEXROLE_EMISSIVE: MjEnumValue;
    mjTEXROLE_RGBA: MjEnumValue;
    mjTEXROLE_ORM: MjEnumValue;
    mjNTEXROLE: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  mjtWrap: {
    mjWRAP_NONE: MjEnumValue;
    mjWRAP_JOINT: MjEnumValue;
    mjWRAP_PULLEY: MjEnumValue;
    mjWRAP_SITE: MjEnumValue;
    mjWRAP_SPHERE: MjEnumValue;
    mjWRAP_CYLINDER: MjEnumValue;
    [key: string]: MjEnumValue;
  };

  // Allow additional properties from the base module
  [key: string]: unknown;
}

// Re-export MainModule as an alias to Mujoco for compatibility
export type MainModule = Mujoco;
