/** Leave the body alone: a press that hits one orbits the camera, as a miss does. */
import type { InteractionMode } from './mode';
import type { PointerClaim } from '../pointer';

export class ViewMode implements InteractionMode {
  readonly id = 'view' as const;

  onDown(): PointerClaim {
    return 'none';
  }

  onCancel(): void {}

  preStep(): void {}
}
