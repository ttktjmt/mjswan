import type { MjData, MjModel } from '@/mujoco';
import type { Mujoco } from '../../types/mujoco';
import type { PolicyState } from './types';

export class PolicyStateBuilder {
  private mujoco: Mujoco;
  private mjModel: MjModel;
  private mjData: MjData;
  private qposAdr: number[];
  private qvelAdr: number[];
  private ctrlAdr: number[] | null;
  private jointIndices: number[];
  private numActions: number;

  constructor(mujoco: Mujoco, mjModel: MjModel, mjData: MjData, jointNames: string[]) {
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
