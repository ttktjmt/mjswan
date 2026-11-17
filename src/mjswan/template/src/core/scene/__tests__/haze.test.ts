/** MuJoCo's haze: its cone (`render_context.c`'s `haze`) and where it stands (`mjr_render`). */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { hazeGeometry, placeHaze } from '../haze';

describe('hazeGeometry', () => {
  it('is clear at the plane and at eye level, in the haze colour between', () => {
    const ratio = 0.3;
    const geometry = hazeGeometry(4, ratio, [0.2, 0.4, 0.6]);
    const position = geometry.getAttribute('position');
    const color = geometry.getAttribute('color');
    const ring = (r: number) => r * 5;
    expect(color.itemSize).toBe(4);
    expect([color.getW(ring(0)), color.getW(ring(1)), color.getW(ring(2))]).toEqual([0, 1, 0]);
    expect([color.getX(ring(1)), color.getY(ring(1)), color.getZ(ring(1))]).toEqual([0.2, 0.4, 0.6].map(Math.fround));
    // The plane, at y = -1, a share of the skybox distance in; eye level at y = 0 and radius 1.
    expect([position.getX(ring(0)), position.getY(ring(0))]).toEqual([Math.fround(1 - ratio), -1]);
    expect([position.getX(ring(2)), position.getY(ring(2))]).toEqual([1, 0]);
    const peak = position.getY(ring(1)) + 1;
    expect(peak).toBeGreaterThan(0);
    expect(peak).toBeLessThan(1);
  });
});

describe('placeHaze', () => {
  it('stands on the plane under the eye, and is not drawn from below', () => {
    const plane = new THREE.Object3D();
    const haze = new THREE.Object3D();
    expect(placeHaze(haze, plane, new THREE.Vector3(1, 2, 3), 70)).toBe(true);
    expect(haze.position.toArray()).toEqual([1, 2, 3]);
    expect(haze.scale.toArray()).toEqual([70, 2, 70]);
    expect(placeHaze(haze, plane, new THREE.Vector3(1, -0.5, 3), 70)).toBe(false);
  });
});
