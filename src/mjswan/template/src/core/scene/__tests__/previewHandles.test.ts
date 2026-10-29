/** What a drag on the preview's handles does to the geom, in MuJoCo's axes. */
import { describe, expect, it } from 'vitest';

import { extent, resized, sizeArrows, turned, type PreviewGeom } from '../previewHandles';

const round = (v: number[]) => v.map((x) => Number(x.toFixed(6)) + 0);
const geom = (over: Partial<PreviewGeom>): PreviewGeom => ({
  type: 'box',
  size: [0.1, 0.1, 0.1],
  pos: [0, 0, 0.1],
  quat: [1, 0, 0, 0],
  rgba: [0.5, 0.5, 0.5, 1],
  ...over,
});
const QUARTER_ABOUT_Z = [Math.SQRT1_2, 0, 0, Math.SQRT1_2];

describe('sizeArrows', () => {
  it("gives every spatial size entry one arrow, along the axis it measures", () => {
    expect(sizeArrows('box')).toEqual([{ axis: 0, index: 0 }, { axis: 1, index: 1 }, { axis: 2, index: 2 }]);
    expect(sizeArrows('sphere')).toEqual([{ axis: 0, index: 0 }]);
    expect(sizeArrows('capsule')).toEqual([{ axis: 0, index: 0 }, { axis: 2, index: 1 }]);
    // A plane's third entry is its grid spacing, not an extent.
    expect(sizeArrows('plane')).toEqual([{ axis: 0, index: 0 }, { axis: 1, index: 1 }]);
  });

  it("starts each arrow on the surface: a capsule's cap counts toward its length", () => {
    expect(extent('capsule', [0.05, 0.1], 2)).toBeCloseTo(0.15);
    expect(extent('capsule', [0.05, 0.1], 0)).toBeCloseTo(0.05);
    expect(extent('cylinder', [0.05, 0.1], 2)).toBeCloseTo(0.1);
  });
});

describe('resized', () => {
  it('moves the face under the arrow by the drag and leaves the opposite one', () => {
    const next = resized(geom({}), 0, 0, 0.2);
    expect(round(next.size)).toEqual([0.2, 0.1, 0.1]);
    expect(next.pos[0] - next.size[0]).toBeCloseTo(-0.1);
    expect(next.pos[0] + next.size[0]).toBeCloseTo(0.3);
    expect(round(next.pos.slice(1))).toEqual([0, 0.1]);
  });

  it('grows a box resting on the floor upward', () => {
    const next = resized(geom({}), 2, 2, 0.1);
    expect(round(next.size)).toEqual([0.1, 0.1, 0.15]);
    expect(next.pos[2] - next.size[2]).toBeCloseTo(0);
  });

  it("keeps the bottom where a radius sets the height too, and not where it does not", () => {
    for (const [type, size, half] of [
      ['sphere', [0.1], (s: number[]) => s[0]],
      ['capsule', [0.05, 0.1], (s: number[]) => s[0] + s[1]],
    ] as const) {
      const start = geom({ type, size: [...size], pos: [0, 0, half([...size])] });
      const next = resized(start, 0, 0, 0.04);
      expect(next.pos[2] - half(next.size), type).toBeCloseTo(0);
      expect(next.pos[0] - next.size[0], type).toBeCloseTo(-size[0]);
    }
    const cylinder = resized(geom({ type: 'cylinder', size: [0.05, 0.1] }), 0, 0, 0.04);
    expect(cylinder.pos[2]).toBeCloseTo(0.1);
  });

  it('stops at a few millimetres, the opposite face still in place', () => {
    const next = resized(geom({}), 0, 0, -1);
    expect(next.size[0]).toBeCloseTo(0.005);
    expect(next.pos[0] - next.size[0]).toBeCloseTo(-0.1);
  });

  it('works in the geom\'s own axes: turned a quarter, its x arrow runs along world y', () => {
    const next = resized(geom({ quat: QUARTER_ABOUT_Z }), 0, 0, 0.2);
    expect(round(next.size)).toEqual([0.2, 0.1, 0.1]);
    expect(round(next.pos)).toEqual([0, 0.1, 0.1]);
  });
});

describe('turned', () => {
  it('turns about its own axis, through its centre', () => {
    const once = turned(geom({}), 2, Math.PI / 2);
    expect(round(once.quat)).toEqual(round(QUARTER_ABOUT_Z));
    expect(once.pos).toEqual([0, 0, 0.1]);
    // About its own x, now lying along world y.
    const twice = turned(once, 0, Math.PI / 2);
    expect(round(twice.quat)).toEqual([0.5, 0.5, 0.5, 0.5]);
  });
});
