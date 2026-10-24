import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { SeeThrough } from '../seeThrough';

function body(): { group: THREE.Group; mesh: THREE.Mesh; shared: THREE.MeshStandardMaterial } {
  const shared = new THREE.MeshStandardMaterial({ color: 0x808080 });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), [shared, shared]);
  mesh.castShadow = true;
  const group = new THREE.Group();
  group.add(mesh, new THREE.Object3D());
  return { group, mesh, shared };
}

describe('SeeThrough', () => {
  it('draws every mesh see-through, over what is in front, unshadowed', () => {
    const { group, mesh, shared } = body();
    new SeeThrough().apply([group, undefined]);
    const materials = mesh.material as THREE.Material[];
    expect(materials).toHaveLength(2);
    for (const material of materials) {
      expect(material).not.toBe(shared);
      expect(material.transparent).toBe(true);
      expect(material.opacity).toBeLessThan(1);
      expect(material.depthTest).toBe(false);
      expect(material.depthWrite).toBe(false);
    }
    expect(mesh.castShadow).toBe(false);
    // Its own material is untouched: the scene cache may hand it to another load.
    expect(shared.transparent).toBe(false);
  });

  it('puts each mesh back as it was, applied twice or not', () => {
    const { group, mesh, shared } = body();
    const seeThrough = new SeeThrough();
    seeThrough.apply([group]);
    const copies = mesh.material as THREE.Material[];
    seeThrough.apply([group]);
    expect(mesh.material).toBe(copies);
    seeThrough.restore();
    expect(mesh.material).toEqual([shared, shared]);
    expect(mesh.castShadow).toBe(true);
    seeThrough.restore();
    expect(mesh.material).toEqual([shared, shared]);
  });
});
