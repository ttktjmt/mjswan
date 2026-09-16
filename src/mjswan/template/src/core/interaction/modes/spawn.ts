/**
 * Throw a box into a running scene.
 *
 * The gesture is a slingshot with **one** axis: press a surface to place the box, drag
 * back to load it, release to fire it the other way. The draw length is measured in CSS
 * pixels rather than metres so the feel does not change with the camera distance, and a
 * release under the tap threshold drops the box where it stands instead of throwing it —
 * which is why there is no separate "place" mode.
 *
 * Size is a slider, not a second drag axis. Splitting the drag into speed and size costs
 * the aim, and the aim is the part you want every throw; the size is not.
 *
 * The launch direction lives in the camera plane through the spawn point. For the usual
 * viewpoint — looking down at a floor — dragging down the screen throws up and away, which
 * is the throw people reach for.
 */
import * as THREE from 'three';

import type { InteractionMode, InteractionSim, ModeContext } from './mode';
import { toMjc } from './mode';
import { TAP_SLOP_PX, type PointerClaim, type PointerGesture } from '../pointer';

/** Screen travel for a full-power throw, CSS px. */
const FULL_DRAW_PX = 160;

export class SpawnMode implements InteractionMode {
  readonly id = 'spawn' as const;
  private aiming = false;
  private readonly spawnPoint = new THREE.Vector3();

  /** Anything drawn, the floor included: a throw needs a surface, not a moving part. */
  pickable(): Set<number> | null {
    return null;
  }

  unavailable(ctx: ModeContext): string | null {
    return ctx.pool.size > 0 ? null : 'This scene was built with no boxes to throw.';
  }

  onDown(gesture: PointerGesture, ctx: ModeContext): PointerClaim {
    const { mjData } = ctx.sim();
    if (!mjData) return 'none';
    this.aiming = true;
    this.spawnPoint.copy(gesture.hit.point);
    ctx.ghost.show(this.spawnPoint, ctx.params().size);
    return 'exclusive';
  }

  onUp(gesture: PointerGesture, ctx: ModeContext): void {
    const { mujoco, mjModel, mjData } = ctx.sim();
    this.aiming = false;
    ctx.ghost.hide();
    ctx.arrow.hide();
    if (!mjModel || !mjData) return;
    const params = ctx.params();
    const launch = this.launch(gesture, params.speed);
    ctx.pool.materialize(
      mujoco,
      mjModel,
      mjData,
      { size: params.size, density: params.density },
      { position: toMjc(this.spawnPoint), velocity: toMjc(launch) },
    );
  }

  onCancel(ctx: ModeContext): void {
    this.aiming = false;
    ctx.ghost.hide();
    ctx.arrow.hide();
  }

  preStep(gesture: PointerGesture | null, ctx: ModeContext): void {
    if (!gesture || !this.aiming) return;
    const params = ctx.params();
    ctx.ghost.show(this.spawnPoint, params.size);
    const launch = this.launch(gesture, params.speed);
    if (launch.lengthSq() === 0) {
      ctx.arrow.hide();
      return;
    }
    // Scaled to something visible rather than to metres per second: this is an aim, and a
    // 6 m/s arrow would leave the scene.
    const drawn = launch.clone().normalize().multiplyScalar(0.15 + 0.35 * (launch.length() / params.speed));
    ctx.arrow.show(this.spawnPoint, this.spawnPoint.clone().add(drawn), gesture.travel >= FULL_DRAW_PX);
  }

  /** World launch velocity: away from the pointer, faster the further it was pulled. */
  private launch(gesture: PointerGesture, speed: number): THREE.Vector3 {
    if (gesture.travel < TAP_SLOP_PX) return new THREE.Vector3();
    const direction = this.spawnPoint.clone().sub(gesture.ray);
    if (direction.lengthSq() < 1e-12) return new THREE.Vector3();
    const power = Math.min(1, gesture.travel / FULL_DRAW_PX);
    return direction.normalize().multiplyScalar(power * speed);
  }
}

/** Referenced only through {@link ModeContext}; kept explicit so the sim type stays exported. */
export type { InteractionSim };
