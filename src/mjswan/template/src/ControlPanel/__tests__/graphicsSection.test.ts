/** Copy as Python: the settings drawn now, as the call that builds them into the scene. */
import { describe, expect, it } from 'vitest';

import { DEFAULT_GRAPHICS, type GraphicsSettings } from '../../engine';
import { graphicsAsPython } from '../GraphicsSection';

const call = (args: string) => `from dataclasses import replace\n\nscene.set_viewer(replace(scene.viewer${args}))\n`;

describe('graphicsAsPython', () => {
  it("names only what differs from the scene's own settings", () => {
    const current: GraphicsSettings = { ...DEFAULT_GRAPHICS, colorManagement: 'filament', toneMapping: 'agx', exposure: 0.30000000000000004, haze: false };
    expect(graphicsAsPython(current, DEFAULT_GRAPHICS)).toBe(
      call(', color_management="filament", tone_mapping="agx", exposure=0.3, enable_haze=False'),
    );
    expect(graphicsAsPython(DEFAULT_GRAPHICS, DEFAULT_GRAPHICS)).toBe(call(''));
  });

  it('clears the Filament settings on the way back to classic, which refuses them', () => {
    const scene: GraphicsSettings = { ...DEFAULT_GRAPHICS, colorManagement: 'filament', toneMapping: 'agx' };
    expect(graphicsAsPython(DEFAULT_GRAPHICS, scene)).toBe(
      call(', color_management="classic", tone_mapping=None, exposure=None, output_color_space=None, base_color_type=None'),
    );
  });
});
