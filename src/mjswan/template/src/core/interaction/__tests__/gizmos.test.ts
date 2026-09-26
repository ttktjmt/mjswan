/** The pull arrow, the one place a force's size is visible. */
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

  it('grows thicker with the force and keeps its tip on the pointer', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    const girths: number[] = [];
    for (const force of [0, 10, 100, 1000, 100000]) {
      arrow.show(from, to, force);
      const { girth, tip } = tipOf(scene);
      girths.push(girth);
      expect(tip.distanceTo(to), `tip at ${force} N`).toBeLessThan(1e-6);
    }
    for (let i = 1; i < girths.length; i++) expect(girths[i]).toBeGreaterThan(girths[i - 1]);
    // Bounded without a clamp: past a hard pull, far more force barely thickens it.
    arrow.show(from, to, 1e9);
    expect(tipOf(scene).girth - girths[girths.length - 1]).toBeLessThan(0.01 * girths[girths.length - 1]);
    arrow.dispose();
  });

  it('draws no force, or a nonsense one, at its rest girth', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    arrow.show(from, to, 0);
    const rest = tipOf(scene).girth;
    for (const force of [Number.NaN, -50, Number.NEGATIVE_INFINITY]) {
      arrow.show(from, to, force);
      expect(tipOf(scene).girth, String(force)).toBe(rest);
    }
    arrow.dispose();
  });

  it('keeps the tip on the pointer on a pull shorter than the head', () => {
    const scene = new THREE.Scene();
    const arrow = new DragArrow(scene);
    const near = from.clone().add(new THREE.Vector3(0, 0.03, 0));
    arrow.show(from, near, 1000);
    expect(tipOf(scene).tip.distanceTo(near)).toBeLessThan(1e-6);
    arrow.dispose();
  });
});
