/** While paused, every mode but View stands aside, and a scene's own reason comes first. */
import { describe, expect, it } from 'vitest';

import { PAUSED_REASON, unavailableReason } from '../InteractionManager';
import type { ModeContext } from '../modes/mode';

const ctx = {} as ModeContext;
const pull = { id: 'pull' as const };
const view = { id: 'view' as const };
const anchorless = { id: 'weld' as const, unavailable: () => 'This scene was built without a grab anchor.' };

describe('unavailableReason', () => {
  it('lets every mode act while the simulation runs', () => {
    expect(unavailableReason(pull, ctx, false)).toBeNull();
    expect(unavailableReason(view, ctx, false)).toBeNull();
  });

  it('stands a force mode aside while paused, and never View', () => {
    expect(unavailableReason(pull, ctx, true)).toBe(PAUSED_REASON);
    expect(unavailableReason(view, ctx, true)).toBeNull();
  });

  it("gives the scene's own reason before the pause", () => {
    expect(unavailableReason(anchorless, ctx, true)).toBe('This scene was built without a grab anchor.');
  });
});
