"""Dump the actuator-parity fixture for ``actuatorParity.test.ts``: mjlab actuators whose
torque law mjlab computes in Python, stepped in mjlab, with what each physics substep
measured and actuated.

The cases: an ideal PD with an effort limit, a DC motor behind a command delay, and a
velocity term over two laws on one arm, one of them a custom ``compute`` with
Unitree-style friction (WBC-Mjlab's ``UnitreeActuator``, trimmed). Each goes through the
adapter and the Builder's own ``serialize_actuators``, so the test loads the entries and
the graphs that really ship.

**States are replayed, not co-simulated**, as in ``dump_action_fixture.py``: the test
feeds each substep's joint positions and velocities from here and compares the ``ctrl``
it writes.

Regenerate with::

    MUJOCO_GL=disable .venv/bin/python tests/dump_actuator_fixture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

import torch
from mjlab.actuator import DcMotorActuatorCfg, IdealPdActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src/mjswan/template/src/core/actuator/__tests__/fixtures/actuator"

STEPS = 14
#: The episode restarts before this step, so the targets and the delay start over.
RESET_AT = 9
SEED = 0
#: Rotor inertia on every joint, so a torque law integrated explicitly stays stable.
ARMATURE = 0.01


@dataclass(kw_only=True)
class FrictionPdActuatorCfg(IdealPdActuatorCfg):
    fs: float = 0.0
    fd: float = 0.0
    va: float = 0.01

    def build(self, entity, target_ids, target_names):
        return FrictionPdActuator(self, entity, target_ids, target_names)


class FrictionPdActuator(IdealPdActuator):
    """The PD less static friction, smoothed by ``tanh``, and viscous friction."""

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        shape = (data.nworld, len(self._target_names))
        self._fs = torch.full(shape, self.cfg.fs, device=device)
        self._fd = torch.full(shape, self.cfg.fd, device=device)
        self._va = torch.full(shape, self.cfg.va, device=device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        effort = super().compute(cmd)
        return effort - (self._fs * torch.tanh(cmd.vel / self._va) + self._fd * cmd.vel)


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _cases() -> dict[str, tuple[tuple[Any, ...], Any]]:
    """Per case: the arm's actuators and its one action term."""
    import action_term_fixture as fx
    from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg

    return {
        "ideal_pd": (
            (
                IdealPdActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness=20.0,
                    damping=1.0,
                    effort_limit=6.0,
                    armature=ARMATURE,
                ),
            ),
            fx.on_arm(JointPositionActionCfg, scale=0.5),
        ),
        "dc_motor": (
            (
                DcMotorActuatorCfg(
                    target_names_expr=(".*",),
                    stiffness=40.0,
                    damping=1.0,
                    effort_limit=8.0,
                    saturation_effort=10.0,
                    velocity_limit=3.0,
                    delay_min_lag=2,
                    delay_max_lag=2,
                    armature=ARMATURE,
                ),
            ),
            fx.on_arm(JointPositionActionCfg, scale=1.0),
        ),
        "friction_velocity": (
            (
                FrictionPdActuatorCfg(
                    target_names_expr=("j1", "j2"),
                    stiffness=5.0,
                    damping=2.0,
                    armature=ARMATURE,
                    fs=0.6,
                    fd=0.06,
                ),
                IdealPdActuatorCfg(
                    target_names_expr=("j3",),
                    stiffness=5.0,
                    damping=2.0,
                    armature=ARMATURE,
                ),
            ),
            fx.on_arm(JointVelocityActionCfg, scale=2.0),
        ),
    }


def _addresses(env: Any, joints: list[str]) -> dict[str, list[int]]:
    """Per joint, its ``qpos`` and ``qvel`` addresses and its actuator's ``ctrl``."""
    import mujoco

    model = env.sim.mj_model
    found: dict[str, list[int]] = {}
    for name in joints:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ctrl = [
            a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_trnid[a, 0] == joint
        ]
        found[name] = [
            int(model.jnt_qposadr[joint]),
            int(model.jnt_dofadr[joint]),
            int(ctrl[0]),
        ]
    return found


def _dump_case(
    actuators: tuple[Any, ...], action: Any, out_dir: Path
) -> dict[str, Any]:
    import action_term_fixture as fx

    from mjswan.build.mdp import serialize_actions, serialize_actuators
    from mjswan.mjlab import (
        adapt_actions,
        resolve_actuator_delays,
        traced_actuator_cfgs,
    )

    env = fx.arm_env({"joint": action}, actuators=actuators)
    joints = [f"robot/{name}" for name in env.scene["robot"].joint_names]
    adapted = adapt_actions({"joint": action}) or {}
    resolve_actuator_delays(adapted, joints, env.cfg)
    entry = serialize_actions(adapted, env=env, out_dir=out_dir)["joint"]
    term = env.action_manager.get_term("joint")

    substeps: list[dict[str, list[float]]] = []
    physics_step = env.sim.step

    def recording_step() -> None:
        substeps.append(
            {
                "qpos": _flat(env.sim.data.qpos[0]),
                "qvel": _flat(env.sim.data.qvel[0]),
                "ctrl": _flat(env.sim.data.ctrl[0]),
            }
        )
        physics_step()

    env.sim.step = recording_step
    generator = torch.Generator().manual_seed(SEED)
    steps: list[dict[str, Any]] = []
    for index in range(STEPS):
        step: dict[str, Any] = {"reset": index == RESET_AT}
        if step["reset"]:
            with contextlib.redirect_stdout(io.StringIO()):
                env.reset()
        raw = torch.rand((1, len(joints)), generator=generator) * 3 - 1.5
        substeps.clear()
        env.step(raw)
        # What the term's apply set as each joint's target, the encoder bias being 0.
        step["processed"] = _flat(term._processed_actions[0])
        step["substeps"] = list(substeps)
        steps.append(step)

    # After the rollout: the build constructs each law afresh on the env.
    traced = traced_actuator_cfgs(env.cfg, ["robot"])
    payload = {
        "entry": entry,
        "joints": joints,
        "addresses": _addresses(env, joints),
        "actuators": serialize_actuators(traced, joints, env=env, out_dir=out_dir),
        "decimation": env.cfg.decimation,
        "steps": steps,
    }
    env.close()
    return payload


def main() -> None:
    from mjswan.compile.actuator import INPUTS, OUTPUT, SUPPORTED_OPS

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    fixture = {
        "supported_ops": sorted(SUPPORTED_OPS),
        "inputs": list(INPUTS),
        "output": OUTPUT,
        "cases": {
            name: _dump_case(actuators, action, OUT_DIR / name)
            for name, (actuators, action) in _cases().items()
        },
    }
    index = OUT_DIR / "actuator.json"
    index.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for name, payload in fixture["cases"].items():
        graphs = sorted(
            p.relative_to(OUT_DIR).as_posix() for p in (OUT_DIR / name).rglob("*.onnx")
        )
        print(f"{name}: {len(payload['steps'])} steps, graphs: {', '.join(graphs)}")
    print(f"wrote {index} ({index.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
