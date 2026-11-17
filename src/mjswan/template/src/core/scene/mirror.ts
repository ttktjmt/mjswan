/**
 * MuJoCo's reflection (`mjRND_REFLECTION`, `render_gl3.c`): the first opaque plane or box
 * top with a `reflectance` mirrors the geoms, scaled by it and added to the surface. The
 * mirrored view is drawn into a texture before the frame and added in the surface's
 * material before tone mapping.
 */
import * as THREE from 'three';
import { SENSOR_LIGHT_LAYER, VIEWER_GEOM_GROUPS, geomGroupLayer } from './layers';
import { addShaderHook } from './shaderHook';

/** Clip space onto texture coordinates. */
const BIAS = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1);

export class Mirror {
  private readonly camera = new THREE.PerspectiveCamera();
  private readonly target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  private readonly uniforms = {
    mjMirror: { value: this.target.texture as THREE.Texture },
    mjMirrorMatrix: { value: new THREE.Matrix4() },
    mjReflectance: { value: 0 },
  };
  private readonly size = new THREE.Vector2();
  private readonly normal = new THREE.Vector3();
  private readonly point = new THREE.Vector3();
  private readonly eye = new THREE.Vector3();
  private readonly view = new THREE.Vector3();
  private readonly look = new THREE.Vector3();
  private readonly rotation = new THREE.Matrix4();
  private readonly plane = new THREE.Plane();
  private readonly clip = new THREE.Vector4();
  private readonly q = new THREE.Vector4();

  /**
   * `surface` is the geom's mesh, whose local +y is the geom's z; `height` lifts the
   * mirror to a box's top. `hidden` are the geom's own meshes, which MuJoCo leaves out.
   */
  constructor(
    private readonly surface: THREE.Mesh,
    private readonly hidden: THREE.Object3D[],
    private readonly height: number,
    private readonly reflectance: number,
  ) {
    // MuJoCo's reflection pass draws the geoms and lights alone: no skin, splat or drawing.
    this.camera.layers.set(SENSOR_LIGHT_LAYER);
    for (let group = 0; group < VIEWER_GEOM_GROUPS; group++) this.camera.layers.enable(geomGroupLayer(group));
    const uniforms = this.uniforms;
    addShaderHook(surface.material as THREE.Material, 'mjMirror', (shader) => {
      Object.assign(shader.uniforms, uniforms);
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          '#include <common>\nuniform mat4 mjMirrorMatrix;\nvarying vec4 vMjMirrorUv;\nvarying float vMjOnMirror;',
        )
        .replace(
          '#include <project_vertex>',
          // A box mirrors on its top face alone.
          '#include <project_vertex>\nvMjMirrorUv = mjMirrorMatrix * modelMatrix * vec4( transformed, 1.0 );\nvMjOnMirror = step( 0.99, objectNormal.y );',
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          '#include <common>\nuniform sampler2D mjMirror;\nuniform float mjReflectance;\nvarying vec4 vMjMirrorUv;\nvarying float vMjOnMirror;',
        )
        .replace(
          '#include <opaque_fragment>',
          'outgoingLight += mjReflectance * vMjOnMirror * texture2DProj( mjMirror, vMjMirrorUv ).rgb;\n#include <opaque_fragment>',
        );
    });
  }

  /**
   * Draw the mirrored view for `camera` before the frame, or show none: when `on` is
   * false, and with the camera behind the surface, as MuJoCo shows none.
   */
  render(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, on: boolean): void {
    this.uniforms.mjReflectance.value = 0;
    if (!on) return;
    this.surface.updateWorldMatrix(true, false);
    camera.updateWorldMatrix(true, false);
    const { normal, point, eye, view, look, rotation } = this;
    normal.set(0, 1, 0).transformDirection(this.surface.matrixWorld);
    point.set(0, this.height, 0).applyMatrix4(this.surface.matrixWorld);
    eye.setFromMatrixPosition(camera.matrixWorld);
    view.subVectors(point, eye);
    if (view.dot(normal) > 0) return;

    // The camera mirrored in the surface (three's `Reflector`).
    view.reflect(normal).negate().add(point);
    rotation.extractRotation(camera.matrixWorld);
    look.set(0, 0, -1).applyMatrix4(rotation).add(eye);
    look.subVectors(point, look).reflect(normal).negate().add(point);
    const mirrored = this.camera;
    mirrored.position.copy(view);
    mirrored.up.set(0, 1, 0).applyMatrix4(rotation).reflect(normal);
    mirrored.lookAt(look);
    mirrored.updateMatrixWorld();
    mirrored.far = camera.far;
    mirrored.projectionMatrix.copy(camera.projectionMatrix);
    this.uniforms.mjMirrorMatrix.value
      .copy(BIAS)
      .multiply(mirrored.projectionMatrix)
      .multiply(mirrored.matrixWorldInverse);
    this.obliqueNearPlane();

    renderer.getDrawingBufferSize(this.size);
    const width = Math.max(1, Math.floor(this.size.x / 2));
    const height = Math.max(1, Math.floor(this.size.y / 2));
    if (this.target.width !== width || this.target.height !== height) this.target.setSize(width, height);

    const background = scene.background;
    const autoUpdate = renderer.shadowMap.autoUpdate;
    const previous = renderer.getRenderTarget();
    // As they were: a host's preview may be hiding the geom it replaces.
    const shown = this.hidden.map((object) => object.visible);
    for (const object of this.hidden) object.visible = false;
    // MuJoCo draws the sky after the reflection, so only geoms are mirrored.
    scene.background = null;
    // Last frame's shadow maps: the lights have not moved enough to draw them twice.
    renderer.shadowMap.autoUpdate = false;
    try {
      renderer.setRenderTarget(this.target);
      renderer.render(scene, mirrored);
    } finally {
      renderer.setRenderTarget(previous);
      renderer.shadowMap.autoUpdate = autoUpdate;
      scene.background = background;
      this.hidden.forEach((object, i) => (object.visible = shown[i]));
    }
    this.uniforms.mjReflectance.value = this.reflectance;
  }

  /** Show no reflection until the returned call: a camera sensor's pass draws none. */
  suspend(): () => void {
    const reflectance = this.uniforms.mjReflectance.value;
    this.uniforms.mjReflectance.value = 0;
    return () => {
      this.uniforms.mjReflectance.value = reflectance;
    };
  }

  dispose(): void {
    this.target.dispose();
  }

  /** Clip what lies behind the surface with the near plane (Lengyel's oblique frustum, as `Reflector` does). */
  private obliqueNearPlane(): void {
    const { plane, clip, q } = this;
    plane.setFromNormalAndCoplanarPoint(this.normal, this.point).applyMatrix4(this.camera.matrixWorldInverse);
    clip.set(plane.normal.x, plane.normal.y, plane.normal.z, plane.constant);
    const p = this.camera.projectionMatrix.elements;
    q.set((Math.sign(clip.x) + p[8]) / p[0], (Math.sign(clip.y) + p[9]) / p[5], -1, (1 + p[10]) / p[14]);
    clip.multiplyScalar(2 / clip.dot(q));
    p[2] = clip.x;
    p[6] = clip.y;
    p[10] = clip.z + 1;
    p[14] = clip.w;
  }
}
