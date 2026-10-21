/**
 * What a skin loads only once a scene has one, as one chunk: three-vrm and the spring
 * model built on it. A scene without a skin fetches neither.
 */
export { VRMHumanBoneParentMap, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
export { SpringPhysics } from './springs';
