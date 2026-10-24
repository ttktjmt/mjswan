/**
 * MuJoCo gives every light a `diffuse` and a separate `specular`; a three.js light
 * has one colour that both terms read. Light intensity follows the diffuse term, so
 * the ratio between the two has to reach the material instead of being dropped —
 * without it, highlights are drawn as if every light shone as hard specularly as it
 * does diffusely, which is 4x too strong under MuJoCo's own defaults.
 */
import { describe, expect, it } from 'vitest';

import type { MjModel } from 'mujoco';
import { lightBudget, lightSpecularRatio } from '../lights';

interface LightSpec {
  diffuse: [number, number, number];
  specular: [number, number, number];
  active?: boolean;
}

function modelWithLights(
  lights: LightSpec[],
  headlight?: { active: boolean; diffuse: number[]; specular: number[] }
): MjModel {
  return {
    nlight: lights.length,
    light_active: Uint8Array.from(lights, (l) => (l.active === false ? 0 : 1)),
    light_diffuse: Float32Array.from(lights.flatMap((l) => l.diffuse)),
    light_specular: Float32Array.from(lights.flatMap((l) => l.specular)),
    vis: headlight ? { headlight } : undefined,
  } as unknown as MjModel;
}

describe('lightSpecularRatio', () => {
  it("matches MuJoCo's default scene, where specular is a quarter of diffuse", () => {
    const model = modelWithLights([{ diffuse: [0.7, 0.7, 0.7], specular: [0.3, 0.3, 0.3] }], {
      active: true,
      diffuse: [0.6, 0.6, 0.6],
      specular: [0, 0, 0],
    });

    expect(lightSpecularRatio(model)).toBeCloseTo(0.3 / 1.3);
  });

  it('leaves the material alone when every light is as specular as it is diffuse', () => {
    const model = modelWithLights([{ diffuse: [1, 1, 1], specular: [1, 1, 1] }]);

    expect(lightSpecularRatio(model)).toBeCloseTo(1);
  });

  it('ignores inactive lights', () => {
    const model = modelWithLights([
      { diffuse: [1, 1, 1], specular: [0, 0, 0] },
      { diffuse: [1, 1, 1], specular: [1, 1, 1], active: false },
    ]);

    expect(lightSpecularRatio(model)).toBe(0);
  });

  it('counts the headlight, which carries no entry in the light arrays', () => {
    const model = modelWithLights([], {
      active: true,
      diffuse: [0.6, 0.6, 0.6],
      specular: [0.9, 0.9, 0.9],
    });

    expect(lightSpecularRatio(model)).toBeCloseTo(1.5);
  });

  it('falls back to 1 for a scene with nothing lit, rather than dividing by zero', () => {
    expect(lightSpecularRatio(modelWithLights([]))).toBe(1);
    expect(lightSpecularRatio(modelWithLights([{ diffuse: [0, 0, 0], specular: [1, 1, 1] }]))).toBe(1);
  });
});

describe('lightBudget', () => {
  it("sums every light's diffuse and ambient with the headlight's, as a toon shader takes them", () => {
    // The Spinkick scene: a directional and a spot light, MuJoCo's default headlight.
    const model = {
      ...modelWithLights(
        [
          { diffuse: [0.7, 0.7, 0.7], specular: [0.3, 0.3, 0.3] },
          { diffuse: [0.7, 0.7, 0.7], specular: [0.3, 0.3, 0.3] },
          { diffuse: [5, 5, 5], specular: [0, 0, 0], active: false },
        ],
        { active: true, diffuse: [0.6, 0.6, 0.6], specular: [0, 0, 0] },
      ),
      light_ambient: Float32Array.from([0, 0, 0, 0, 0, 0, 1, 1, 1]),
    } as unknown as MjModel;
    (model as unknown as { vis: { headlight: { ambient: number[] } } }).vis.headlight.ambient = [0.3, 0.3, 0.3];
    expect(lightBudget(model)).toBeCloseTo(2.3);
  });

  it('is zero for a model with no light', () => {
    expect(lightBudget(modelWithLights([]))).toBe(0);
  });
});
