/** The pull arrow is the one place a force's size is visible; it has to say it. */
import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { DragArrow } from '../gizmos';

function tipOf(scene: THREE.Scene): { girth: number; tip: THREE.Vector3 } {
  const group = scene.children[0] as THREE.Group;
  group.updateMatrixWorld(true);
  const [shaft, head] = group.children as THREE.Mesh[];
  head.geometry.computeBoundingBox();
  // The cone's apex is at +y of its own geometry.
  const apex = new THREE.Vector3(0, head.geometry.boundingBox!.max.y, 0).applyMatrix4(head.matrixWorld);
  return { girth: shaft.scale.x, tip: apex };
}

describe('DragArrow', () => {
  const from = new THREE.Vector3(0, 0.5, 0);
  const to = new THREE.Vector3(0.4, 0.8, -0.2);

  it('grows thicker with the load and keeps its tip on the pointer', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    const girths: number[] = [];
    for (const load of [0, 0.1, 0.5, 1]) {
      arrow.show(from, to, { load, saturated: false });
      const { girth, tip } = tipOf(scene);
      girths.push(girth);
      expect(tip.distanceTo(to), `tip at load ${load}`).toBeLessThan(1e-6);
    }
    for (let i = 1; i < girths.length; i++) expect(girths[i]).toBeGreaterThan(girths[i - 1]);
    arrow.dispose();
  });

  it('treats a load past the clamp, or no load at all, as the ends of the range', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    arrow.show(from, to, { load: 1, saturated: true });
    const full = tipOf(scene).girth;
    arrow.show(from, to, { load: 7, saturated: true });
    expect(tipOf(scene).girth).toBe(full);
    arrow.show(from, to, { load: 0, saturated: false });
    const rest = tipOf(scene).girth;
    arrow.show(from, to, { load: Number.NaN, saturated: false });
    expect(tipOf(scene).girth).toBe(rest);
    arrow.dispose();
  });

  it('keeps the tip on the pointer on a pull shorter than the head', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    const near = from.clone().add(new THREE.Vector3(0, 0.03, 0));
    arrow.show(from, near, { load: 1, saturated: false });
    expect(tipOf(scene).tip.distanceTo(near)).toBeLessThan(1e-6);
    arrow.dispose();
  });
});
