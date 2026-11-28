/**
 * A policy's traced actuator laws, run as mjlab's entity runs its actuators: action terms
 * write per-joint position, velocity and effort targets, and every substep each law turns
 * its joints' targets and state into their `ctrl`.
 *
 * Targets persist across substeps and control steps until an episode's reset zeroes them
 * (mjlab's `EntityData.clear_state`), so a joint no term drives is held toward zero, as
 * mjlab holds it.
 */

import { ActuatorGraph } from './evaluator';

/** The target an action term writes: mjlab's `joint_{pos,vel,effort}_target`. */
export type TargetKind = 'pos_target' | 'vel_target' | 'effort_target';

/** The target a term of *controlType* sets, as mjlab's term of that kind does. */
export function targetKind(controlType: string): TargetKind {
  if (controlType === 'joint_velocity') return 'vel_target';
  if (controlType === 'torque') return 'effort_target';
  return 'pos_target';
}

/** Where one joint's target lives: a law's input array and the joint's column in it. */
export interface TargetSlot {
  values: Float32Array;
  column: number;
}

/** One law and the model addresses of its joints, in its column order. */
export interface ResolvedActuator {
  name: string;
  graph: ActuatorGraph;
  qposAdr: readonly number[];
  qvelAdr: readonly number[];
  ctrlAdr: readonly number[];
}

export class ActuatorSet {
  private readonly byQpos = new Map<number, { actuator: ResolvedActuator; column: number }>();

  constructor(readonly actuators: readonly ResolvedActuator[]) {
    for (const actuator of actuators) {
      actuator.qposAdr.forEach((adr, column) => this.byQpos.set(adr, { actuator, column }));
    }
  }

  /** Where a term writes the *kind* target of the joint at *qposAdr*, or null when no law drives it. */
  targetOf(qposAdr: number, kind: TargetKind): TargetSlot | null {
    const found = this.byQpos.get(qposAdr);
    return found ? { values: found.actuator.graph.inputs[kind], column: found.column } : null;
  }

  /** mjlab's `clear_state`: every target back to zero, at an episode's start. */
  reset(): void {
    for (const { graph } of this.actuators) {
      graph.inputs.pos_target.fill(0);
      graph.inputs.vel_target.fill(0);
      graph.inputs.effort_target.fill(0);
    }
  }

  /** Every law on the state the last `mj_step` left, each writing its joints' `ctrl`. */
  apply(qpos: ArrayLike<number>, qvel: ArrayLike<number>, ctrl: { [index: number]: number }): void {
    for (const { graph, qposAdr, qvelAdr, ctrlAdr } of this.actuators) {
      const { pos, vel } = graph.inputs;
      for (let c = 0; c < qposAdr.length; c++) {
        pos[c] = qpos[qposAdr[c]];
        vel[c] = qvel[qvelAdr[c]];
      }
      const torque = graph.run();
      for (let c = 0; c < ctrlAdr.length; c++) {
        if (ctrlAdr[c] >= 0) ctrl[ctrlAdr[c]] = torque[c];
      }
    }
  }
}
