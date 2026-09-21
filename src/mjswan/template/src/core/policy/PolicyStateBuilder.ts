import type { MainModule, MjData, MjModel } from 'mujoco';
import type { PolicyState } from './types';

export class PolicyStateBuilder {
  private mujoco: MainModule;
  private mjModel: MjModel;
  private mjData: MjData;
  private qposAdr: number[];
  private qvelAdr: number[];
  private ctrlAdr: number[] | null;
  private jointIndices: number[];
  private numActions: number;

  constructor(mujoco: MainModule, mjModel: MjModel, mjData: MjData, jointNames: string[]) {
    this.mujoco = mujoco;
    this.mjModel = mjModel;
    this.mjData = mjData;
    const available = this.getJointNames(mjModel);
    this.qposAdr = [];
    this.qvelAdr = [];
    this.jointIndices = [];
    this.ctrlAdr = null;

    for (const name of jointNames) {
      const jointIdx = available.indexOf(name);
      if (jointIdx < 0) {
        throw new Error(`Joint "${name}" not found in MuJoCo model`);
      }
      this.qposAdr.push(mjModel.jnt_qposadr[jointIdx]);
      this.qvelAdr.push(mjModel.jnt_dofadr[jointIdx]);
      this.jointIndices.push(jointIdx);
    }

    this.numActions = jointNames.length;
    this.ctrlAdr = this.buildCtrlAdr();
  }

  build(): PolicyState {
    const qpos = this.mjData.qpos;
    const qvel = this.mjData.qvel;
    const jointPos = new Float32Array(this.numActions);
    const jointVel = new Float32Array(this.numActions);

    for (let i = 0; i < this.numActions; i++) {
      jointPos[i] = qpos[this.qposAdr[i]];
      jointVel[i] = qvel[this.qvelAdr[i]];
    }

    const rootPos = new Float32Array([qpos[0], qpos[1], qpos[2]]);
    const rootQuat = new Float32Array([qpos[3], qpos[4], qpos[5], qpos[6]]);
    const rootLinVel = new Float32Array([qvel[0], qvel[1], qvel[2]]);
    const rootAngVel = new Float32Array([qvel[3], qvel[4], qvel[5]]);

    return {
      jointPos,
      jointVel,
      rootPos,
      rootQuat,
      rootLinVel,
      rootAngVel,
    };
  }

  getControlMapping():
    | { ctrlAdr: number[]; qposAdr: number[]; qvelAdr: number[] }
    | null {
    if (!this.ctrlAdr) {
      return null;
    }
    return {
      ctrlAdr: this.ctrlAdr.slice(),
      qposAdr: this.qposAdr.slice(),
      qvelAdr: this.qvelAdr.slice(),
    };
  }

  /**
   * Returns the subset of control addresses for joints matching any of the
   * given regex patterns, along with the action vector indices (positions in
   * the flat policy output) for those joints.
   *
   * Patterns follow MuJoCo/mjlab convention: `".*"` matches all joints.
   * Each pattern is anchored with `^(?:...)$` so partial matches are rejected.
   *
   * Returns `null` if no joints match or control addresses are unavailable.
   */
  getControlMappingFor(
    patterns: string[],
    allJointNames: string[]
  ): {
    ctrlAdr: number[];
    qposAdr: number[];
    qvelAdr: number[];
    actionIndices: number[];
  } | null {
    if (!this.ctrlAdr) {
      return null;
    }

    const regexps = patterns.map((p) => new RegExp(`^(?:${p})$`));
    const matchesAny = (name: string): boolean =>
      regexps.some((re) => re.test(name));

    const ctrlAdr: number[] = [];
    const qposAdr: number[] = [];
    const qvelAdr: number[] = [];
    const actionIndices: number[] = [];

    for (let i = 0; i < allJointNames.length; i++) {
      if (matchesAny(allJointNames[i])) {
        ctrlAdr.push(this.ctrlAdr[i]);
        qposAdr.push(this.qposAdr[i]);
        qvelAdr.push(this.qvelAdr[i]);
        actionIndices.push(i);
      }
    }

    if (ctrlAdr.length === 0) {
      return null;
    }

    return { ctrlAdr, qposAdr, qvelAdr, actionIndices };
  }

  /**
   * Returns control mapping for actuators selected by name (not via joint
   * transmission). Used for muscle/tendon actuators whose transmission type
   * isn't `mjTRN_JOINT` and which therefore can't be located via the
   * joint-name path. Patterns follow the same `^(?:...)$` regex convention as
   * `getControlMappingFor`.
   *
   * Returns `null` if no actuators match.
   */
  getCtrlMappingByActuatorNames(patterns: string[]): {
    ctrlAdr: number[];
    actionIndices: number[];
  } | null {
    const actuatorNames = this.getActuatorNames(this.mjModel);
    const regexps = patterns.map((p) => new RegExp(`^(?:${p})$`));
    const matchesAny = (name: string): boolean =>
      regexps.some((re) => re.test(name));

    const ctrlAdr: number[] = [];
    const actionIndices: number[] = [];
    for (let i = 0; i < actuatorNames.length; i++) {
      if (matchesAny(actuatorNames[i])) {
        ctrlAdr.push(i);
        actionIndices.push(ctrlAdr.length - 1);
      }
    }
    return ctrlAdr.length > 0 ? { ctrlAdr, actionIndices } : null;
  }

  private getActuatorNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let i = 0; i < mjModel.nu; i++) {
      let start = mjModel.name_actuatoradr[i];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      names.push(decoder.decode(namesArray.subarray(start, end)));
    }
    return names;
  }

  private buildCtrlAdr(): number[] | null {
    if (this.mjModel.nu <= 0) {
      return null;
    }

    const jointTransmission = this.mujoco.mjtTrn?.mjTRN_JOINT?.value;
    const actuator2joint: number[] = [];
    for (let i = 0; i < this.mjModel.nu; i++) {
      const trnType = this.mjModel.actuator_trntype[i];
      if (jointTransmission === undefined || trnType === jointTransmission) {
        actuator2joint.push(this.mjModel.actuator_trnid[2 * i]);
      } else {
        actuator2joint.push(-1);
      }
    }

    const ctrlAdr: number[] = [];
    for (const jointIdx of this.jointIndices) {
      const actuatorIdx = actuator2joint.findIndex((jointId) => jointId === jointIdx);
      if (actuatorIdx < 0) {
        return null;
      }
      ctrlAdr.push(actuatorIdx);
    }
    return ctrlAdr;
  }

  private getJointNames(mjModel: MjModel): string[] {
    const namesArray = new Uint8Array(mjModel.names);
    const decoder = new TextDecoder();
    const names: string[] = [];
    for (let j = 0; j < mjModel.njnt; j++) {
      let start = mjModel.name_jntadr[j];
      let end = start;
      while (end < namesArray.length && namesArray[end] !== 0) {
        end++;
      }
      let name = decoder.decode(namesArray.subarray(start, end));
      if (!name && j === 0) {
        name = 'floating_base_joint';
      }
      names.push(name);
    }
    return names;
  }
}
