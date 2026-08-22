"""Gentle Humanoid tracking policy demo."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
import torch
import yaml
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.trace_env import build_single_entity_trace_env

from . import terms

HERE = Path(__file__).resolve().parent
GENTLE_HUMANOID_REPO_URL = os.getenv(
    "MJSWAN_GENTLE_HUMANOID_REPO_URL",
    "https://github.com/Axellwppr/motion_tracking.git",
)
GENTLE_HUMANOID_REPO_COMMIT = os.getenv(
    "MJSWAN_GENTLE_HUMANOID_REPO_COMMIT",
    "5684a5e192cf5fe803bc83fc863e75e45e026a40",
)
GENTLE_HUMANOID_DEP_REPO = HERE / ".dep" / "motion_tracking"


def _run_git(args: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    try:
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("git is required to fetch Gentle Humanoid assets") from exc
    except subprocess.CalledProcessError as exc:
        command = " ".join(["git", *args])
        raise RuntimeError(f"Failed to run `{command}` in {cwd}") from exc


def _ensure_gentle_humanoid_repo() -> Path:
    repo = GENTLE_HUMANOID_DEP_REPO
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", GENTLE_HUMANOID_REPO_URL, str(repo)], cwd=HERE)
    else:
        _run_git(["remote", "set-url", "origin", GENTLE_HUMANOID_REPO_URL], cwd=repo)
        _run_git(["fetch", "--tags", "origin"], cwd=repo)
    _run_git(["checkout", "--detach", GENTLE_HUMANOID_REPO_COMMIT], cwd=repo)
    return repo


def _resolve_gentle_humanoid_root() -> Path:
    configured_root = os.getenv("MJSWAN_GENTLE_HUMANOID_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    return _ensure_gentle_humanoid_repo() / "sim2real"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _map_by_name(
    values: list[float],
    source_names: list[str],
    target_names: list[str],
    *,
    default: float = 0.0,
) -> list[float]:
    by_name = {name: float(values[i]) for i, name in enumerate(source_names)}
    return [by_name.get(name, default) for name in target_names]


def _body_world_npz(
    root_pos: np.ndarray,  # (N, 3)
    root_quat_wxyz: np.ndarray,  # (N, 4), wxyz
    dof_pos: np.ndarray,  # (N, n_source), source joint order
    source_joint_names: list[str],
    target_joint_names: list[str],
    *,
    fps: float = 50.0,
) -> bytes:
    """Convert a root+dof clip to the engine's ``body_world`` format (#79):
    reorder joints source→policy order, pelvis as the single body, zero velocities.
    """
    n = root_pos.shape[0]
    src_idx = {name: i for i, name in enumerate(source_joint_names)}
    joint_pos = np.zeros((n, len(target_joint_names)), dtype=np.float32)
    for j, name in enumerate(target_joint_names):
        i = src_idx.get(name)
        if i is not None:
            joint_pos[:, j] = dof_pos[:, i]

    def _c(a: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(a, dtype=np.float32)

    payload = io.BytesIO()
    np.savez(
        payload,
        fps=np.asarray(float(fps), dtype=np.float32),
        joint_pos=_c(joint_pos),
        joint_vel=_c(np.zeros_like(joint_pos)),
        body_pos_w=_c(root_pos.reshape(n, 1, 3)),
        body_quat_w=_c(root_quat_wxyz.reshape(n, 1, 4)),
        body_lin_vel_w=_c(np.zeros((n, 1, 3))),
        body_ang_vel_w=_c(np.zeros((n, 1, 3))),
    )
    return payload.getvalue()


def _default_clip_bytes(tracking_cfg: dict[str, Any]) -> bytes:
    clips = tracking_cfg.get("motion_clips", [])
    if not isinstance(clips, list):
        raise ValueError("tracking.yaml motion_clips must be a list")
    clip = next((c for c in clips if c.get("name") == "default"), None)
    if clip is None:
        raise ValueError("tracking.yaml motion_clips must include a default clip")
    return _body_world_npz(
        root_pos=np.asarray(clip["root_pos"], dtype=np.float32).reshape(1, 3),
        root_quat_wxyz=np.asarray(clip["root_quat"], dtype=np.float32).reshape(1, 4),
        dof_pos=np.asarray(clip["joint_pos"], dtype=np.float32).reshape(1, -1),
        source_joint_names=list(tracking_cfg["dataset_joint_names"]),
        target_joint_names=list(tracking_cfg["action_joint_names"]),
    )


def _clip_file_bytes(
    path: Path, start: int, end: int, target_joint_names: list[str]
) -> bytes:
    """Load a dataset clip (``root_pos``/``root_rot`` xyzw/``dof_pos``) and window
    ``[start:end]``, converting to the engine's ``body_world`` format."""
    with np.load(path) as npz:
        root_pos = np.asarray(npz["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(npz["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(npz["dof_pos"], dtype=np.float32)
        source_joint_names = [
            s.decode() if isinstance(s, bytes) else str(s) for s in npz["joint_names"]
        ]
    hi = end if end >= 0 else root_pos.shape[0]
    root_quat_wxyz = root_rot_xyzw[start:hi][:, [3, 0, 1, 2]]  # xyzw -> wxyz
    return _body_world_npz(
        root_pos[start:hi],
        root_quat_wxyz,
        dof_pos[start:hi],
        source_joint_names,
        target_joint_names,
    )


class _RefWindow:
    """Trace-time stand-in for the browser ``TrackingCommand``'s look-ahead window.

    The clip lookup is data, not math, so the command stays native (ADR 0005) and
    these reads become graph *inputs* the runtime serves from ``getStateField``.
    Only the shapes matter here — one row per ``time_steps`` offset — so the values
    are the neutral ones (identity quats, ready).
    """

    def __init__(self, num_steps: int, num_joints: int):
        self.ref_root_pos_w = torch.zeros(1, num_steps, 3)
        self.ref_root_quat_w = torch.zeros(1, num_steps, 4)
        self.ref_root_quat_w[..., 0] = 1.0
        self.ref_joint_pos = torch.zeros(1, num_steps, num_joints)
        self.is_ready = torch.ones(1, 1)


class _UiValues:
    """Trace-time stand-in for a browser ``UiCommand``: one value per UI input."""

    def __init__(self, width: int):
        self.command = torch.zeros(1, width)


def _write_generated(name: str, payload: bytes) -> Path:
    path = HERE / ".dep" / "generated" / f"gentle_humanoid_{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != payload:
        path.write_bytes(payload)
    return path


def setup_builder() -> mjswan.Builder:
    """Create the builder for the Gentle Humanoid tracking demo."""
    gentle_humanoid_root = _resolve_gentle_humanoid_root()
    if not gentle_humanoid_root.exists():
        raise FileNotFoundError(
            f"Gentle Humanoid asset root not found: {gentle_humanoid_root}. "
            "Set MJSWAN_GENTLE_HUMANOID_ROOT to override it."
        )

    tracking_cfg = _load_yaml(gentle_humanoid_root / "config" / "tracking.yaml")
    controller_cfg = _load_yaml(gentle_humanoid_root / "config" / "controller.yaml")

    action_joint_names = list(tracking_cfg["action_joint_names"])
    real_joint_names = list(controller_cfg["real_joint_names"])
    default_joint_pos = _map_by_name(
        list(controller_cfg["default_qpos_real"]),
        real_joint_names,
        action_joint_names,
    )
    stiffness = {
        name: value
        for name, value in zip(
            real_joint_names,
            list(tracking_cfg.get("kps_real", controller_cfg["kps_real"])),
            strict=True,
        )
    }
    damping = {
        name: value
        for name, value in zip(
            real_joint_names,
            list(tracking_cfg.get("kds_real", controller_cfg["kds_real"])),
            strict=True,
        )
    }
    default_compliance_force = min(
        20.0,
        max(10.0, float(tracking_cfg.get("compliance_flag_threshold", 10.0))),
    )

    future_steps = [int(step) for step in tracking_cfg["future_steps"]]
    # The checkpoint's joint order is `action_joint_names`, not the model's.
    policy_joints = SceneEntityCfg(
        name="robot",
        joint_names=tuple(action_joint_names),
        preserve_order=True,
    )

    builder = mjswan.Builder()
    project = builder.add_project(name="Gentle Humanoid Tracking")
    spec_path = str(gentle_humanoid_root / "assets" / "g1" / "g1.xml")
    scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        name="Unitree G1",
        spec=mujoco.MjSpec.from_file(spec_path),
    )
    # No mjlab task, so tracing needs its own env plus stand-ins for the browser-only commands.
    scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file(spec_path),
            commands={
                "motion": _RefWindow(len(future_steps), len(action_joint_names)),
                # The `compliance` UI command's two value-bearing inputs (a button would carry none).
                "compliance": _UiValues(2),
            },
        )
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0, 0, 0),
            distance=3,
            elevation=-10,
            azimuth=30,
        )
    )

    policy_path = gentle_humanoid_root / tracking_cfg["policy_path"]
    policy_json = policy_path.with_suffix(".json")
    policy = scene.add_policy(
        name="Gentle Humanoid Tracking",
        policy=onnx.load(str(policy_path), load_external_data=True),
        config_path=str(policy_json),
        commands={
            # Built-in motion player; clips convert to its body_world format at build time (#79).
            # `time_steps` is the window its `ref_*` fields are sampled at, sliced by the terms.
            "motion": mjswan.CommandTermConfig(
                term_name="TrackingCommand",
                params={"time_steps": future_steps},
            ),
            "compliance": mjswan.ui_command(
                [
                    mjswan.CheckboxConfig(
                        name="enabled",
                        label=(
                            "Compliance (turn off for motions with hand-ground contact)"
                        ),
                        default=bool(
                            float(tracking_cfg.get("compliance_flag_value", 1.0))
                        ),
                    ),
                    mjswan.SliderConfig(
                        name="force",
                        label="Force",
                        range=(10.0, 20.0),
                        default=default_compliance_force,
                        step=0.5,
                        enabled_when="enabled",
                    ),
                ]
            ),
        },
        # The offsets live on the `motion` command, so motion-coupled terms read their width off
        # the reference tensors. Proprioceptive terms compute one frame; `history_steps` stacks.
        observations=ObservationGroupCfg(
            terms={
                "boot": ObservationTermCfg(func=terms.boot),
                "tracking": ObservationTermCfg(func=terms.tracking),
                "compliance": ObservationTermCfg(
                    func=terms.compliance,
                    params={"command_name": "compliance"},
                ),
                "target_joint_pos": ObservationTermCfg(
                    func=terms.target_joint_pos,
                    params={"asset_cfg": policy_joints},
                ),
                "target_root_z": ObservationTermCfg(func=terms.target_root_z),
                "target_projected_gravity": ObservationTermCfg(
                    func=terms.target_projected_gravity
                ),
                "root_ang_vel": ObservationTermCfg(
                    func=terms.root_ang_vel,
                    history_steps=tuple(tracking_cfg["root_angvel_history_steps"]),
                ),
                "projected_gravity": ObservationTermCfg(
                    func=terms.projected_gravity,
                    history_steps=tuple(
                        tracking_cfg["projected_gravity_history_steps"]
                    ),
                ),
                "joint_pos": ObservationTermCfg(
                    func=terms.joint_pos,
                    params={"asset_cfg": policy_joints},
                    history_steps=tuple(tracking_cfg["joint_pos_history_steps"]),
                ),
                "joint_vel": ObservationTermCfg(
                    func=terms.joint_vel,
                    params={"asset_cfg": policy_joints},
                    history_steps=tuple(tracking_cfg["joint_vel_history_steps"]),
                ),
                # The runtime already holds `last_action`, so only the stacking is ours
                # — newest first, like the offsets the checkpoint names above.
                "prev_actions": ObservationTermCfg(
                    func=obs_fns.last_action,
                    history_steps=tuple(range(int(tracking_cfg["prev_action_steps"]))),
                ),
            }
        ),
        actions={
            "joint_pos": JointPositionActionCfg(
                actuator_names=(".*",),
                scale=list(tracking_cfg["action_scale"]),
                use_default_offset=True,
                stiffness=stiffness,
                damping=damping,
            )
        },
        policy_joint_names=action_joint_names,
        default_joint_pos=default_joint_pos,
        default=True,
    )

    # The conversion to body_world reorders joints, so the npz's order IS action order.
    policy.add_motion(
        name="default",
        source=str(_write_generated("default", _default_clip_bytes(tracking_cfg))),
        fps=50.0,
        anchor_body_name="pelvis",
        body_names=("pelvis",),
        dataset_joint_names=action_joint_names,
        default=True,
        loop=False,
    )
    for motion_cfg in tracking_cfg["motions"]:
        payload = _clip_file_bytes(
            gentle_humanoid_root / motion_cfg["path"],
            int(motion_cfg.get("start", 0)),
            int(motion_cfg.get("end", -1)),
            action_joint_names,
        )
        policy.add_motion(
            name=motion_cfg["name"],
            source=str(_write_generated(motion_cfg["name"], payload)),
            fps=50.0,
            anchor_body_name="pelvis",
            body_names=("pelvis",),
            dataset_joint_names=action_joint_names,
            loop=False,
        )

    return builder


def main() -> None:
    """Build and optionally launch the Gentle Humanoid tracking demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
