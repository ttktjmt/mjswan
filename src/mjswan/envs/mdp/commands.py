"""Command terms mjswan supplies, and its bindings for mjlab's.

:func:`ui_command` and :func:`velocity_command` are operator-driven presets: nothing
resamples, the value is what the control panel says. mjlab's command classes are bound
by cfg-class name: ``UniformVelocityCommandCfg`` is traced as mjlab wrote it, and
``MotionCommandCfg`` stays native (``TrackingCommand``) with only its reset jitter
traced. A command class only one task uses is that task's to register
(``examples/demo/main.py`` does it for ``LiftingCommandCfg``).
"""

from __future__ import annotations

from typing import Any

from ...managers.command_manager import (
    CommandBinding,
    CommandInput,
    CommandTermConfig,
    CommandUiConfig,
    SliderConfig,
    register_command,
)

try:
    # Module-level, not deferred: the RNG spy patches a term body's *module globals*,
    # so `sample_uniform` has to be one of ours to be seen.
    from mjlab.utils.lab_api.math import sample_uniform
except ImportError:
    pass


def ui_command(inputs: list[CommandInput]) -> CommandTermConfig:
    """Create the built-in manual UI command term."""

    return CommandTermConfig(
        term_name="UiCommand",
        ui=CommandUiConfig(inputs=list(inputs)),
    )


def velocity_command(
    *,
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandTermConfig:
    """Three sliders the operator drives, as a ``ui_command`` preset.

    Not mjlab's ``UniformVelocityCommand``: nothing resamples, and the value is
    whatever the slider says. A scene carrying an mjlab task should pass that cfg
    instead (this module binds it) and get mjlab's own joystick.
    """

    return ui_command(
        [
            SliderConfig(
                name="lin_vel_x",
                label="Forward Velocity",
                range=lin_vel_x,
                default=default_lin_vel_x,
                step=0.05,
            ),
            SliderConfig(
                name="lin_vel_y",
                label="Lateral Velocity",
                range=lin_vel_y,
                default=default_lin_vel_y,
                step=0.05,
            ),
            SliderConfig(
                name="ang_vel_z",
                label="Yaw Rate",
                range=ang_vel_z,
                default=default_ang_vel_z,
                step=0.05,
            ),
        ]
    )


# --- UniformVelocityCommand (mjlab's locomotion tasks, and anything built on them) ---

# mjlab's `_debug_vis_impl` runs Python every frame, so its drawing is restated as data
# `core/command/debugViz.ts` reads.
_ARROW_WIDTH = 0.015  # As mjlab passes to every `add_arrow`.


def _velocity_viz(cfg: Any) -> list[dict[str, Any]]:
    """`UniformVelocityCommand`'s arrows: commanded and actual, linear and angular."""
    entity = getattr(cfg, "entity_name", None) or "robot"
    viz = getattr(cfg, "viz", None)
    scale = float(getattr(viz, "scale", 0.5))
    z_offset = float(getattr(viz, "z_offset", 0.2))
    frame = {
        "entity": entity,
        "pos_field": "root_link_pos_w",
        "quat_field": "root_link_quat_w",
    }

    def arrow(
        source: dict[str, Any],
        components: list[int | None],
        color: tuple[float, float, float, float],
    ) -> dict[str, Any]:
        return {
            "shape": "arrow",
            "color": list(color),
            "width": _ARROW_WIDTH,
            "frame": frame,
            # mjlab scales the whole local offset, so the base rises with it too.
            "origin": {"const": [0.0, 0.0, z_offset * scale]},
            "vector": {**source, "components": components, "scale": scale},
        }

    command = {"state": "vel_command_b"}
    return [
        arrow(command, [0, 1, None], (0.2, 0.2, 0.6, 0.6)),
        arrow(command, [None, None, 2], (0.2, 0.6, 0.2, 0.6)),
        arrow(
            {"entity": entity, "field": "root_link_lin_vel_b"},
            [0, 1, None],
            (0.0, 0.6, 1.0, 0.7),
        ),
        arrow(
            {"entity": entity, "field": "root_link_ang_vel_b"},
            [None, None, 2],
            (0.0, 1.0, 0.4, 0.7),
        ),
    ]


# No `ui=`: the joystick descriptor is recorded from the term's own `create_gui` at
# build time (`mjswan.mjlab.gui`).
register_command(
    "UniformVelocityCommandCfg",
    CommandBinding(
        state_fields=[
            "vel_command_b",
            "vel_command_w",
            "heading_target",
            "is_heading_env",
            "is_standing_env",
            "is_world_env",
            "is_forward_env",
        ],
        command_field="vel_command_b",
        viz=_velocity_viz,
    ),
)


# --- MotionCommand (mjlab's tracking tasks) ---

#: The fields ``TrackingCommand.getStateField`` serves itself. Any other field a graph
#: reads off it comes from the selected motion's ``frame_fields``.
TRACKING_STATE_FIELDS = frozenset(
    {
        "command",
        "is_ready",
        "ref_root_pos_w",
        "ref_root_quat_w",
        "ref_joint_pos",
        "anchor_pos_w",
        "anchor_quat_w",
        "anchor_lin_vel_w",
        "anchor_ang_vel_w",
        "ref_base_height",
        "ref_base_lin_vel_b",
        "ref_base_ang_vel_b",
        "ref_gravity_b",
        "joint_pos",
        "tracked_joint_pos",
        "body_pos_w",
        "robot_anchor_pos_w",
        "robot_anchor_quat_w",
        "robot_body_pos_w",
        "body_pos_relative_w",
    }
)


def serialize_motion_command(cfg: Any) -> dict[str, Any]:
    """Convert mjlab's ``MotionCommandCfg`` into browser tracking metadata."""
    data: dict[str, Any] = {
        "anchor_body_name": getattr(cfg, "anchor_body_name", ""),
        "body_names": list(getattr(cfg, "body_names", ()) or ()),
        "sampling_mode": getattr(cfg, "sampling_mode", "start"),
        "pose_range": {
            key: list(value)
            for key, value in (getattr(cfg, "pose_range", None) or {}).items()
        },
        "velocity_range": {
            key: list(value)
            for key, value in (getattr(cfg, "velocity_range", None) or {}).items()
        },
        "joint_position_range": list(getattr(cfg, "joint_position_range", (0.0, 0.0))),
        "debug_vis": bool(getattr(cfg, "debug_vis", False)),
    }
    viz = getattr(cfg, "viz", None)
    data["viz_mode"] = getattr(viz, "mode", "ghost")
    ghost_color = getattr(viz, "ghost_color", None)
    if ghost_color is not None:
        data["ghost_color"] = [float(c) for c in ghost_color]
    entity_name = getattr(cfg, "entity_name", None)
    if entity_name:
        data["entity_name"] = entity_name
    # Not mjlab's: set by a subclass whose command serves the frame a reset writes first.
    if getattr(cfg, "hold_reset_frame", False):
        data["hold_reset_frame"] = True
    return data


_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _range_tensor(ranges: dict[str, tuple[float, float]] | None, device: Any) -> Any:
    """mjlab's `range_list` -> (6, 2) tensor, missing axes meaning no offset."""
    import torch

    ranges = ranges or {}
    return torch.tensor(
        [tuple(ranges.get(key, (0.0, 0.0))) for key in _POSE_KEYS],
        dtype=torch.float,
        device=device,
    )


def motion_rsi_offset(
    env: Any,
    env_ids: Any,
    *,
    asset_cfg: Any,
    pose_range: dict[str, tuple[float, float]] | None = None,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    joint_position_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reference-state-initialization jitter from ``MotionCommand._resample_command``.

    mjlab perturbs the reference frame it is about to write; this perturbs the frame
    *already written*, in place. The numbers are the same, but no motion clip is needed,
    so it traces like any other reset event. Draws are ordered pose, velocity, joint, as
    mjlab's are.
    """
    import torch
    from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

    asset = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device

    # Root pose: xyz offset, plus a roll/pitch/yaw delta applied as a quaternion.
    pose_ranges = _range_tensor(pose_range, device)
    pose_samples = sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (1, 6), device=device
    )
    root_pos = asset.data.root_link_pos_w + pose_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    root_quat = quat_mul(orientations_delta, asset.data.root_link_quat_w)

    # Root velocity: linear and angular offsets.
    velocity_ranges = _range_tensor(velocity_range, device)
    velocity_samples = sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (1, 6), device=device
    )
    root_lin_vel = asset.data.root_link_lin_vel_w + velocity_samples[:, 0:3]
    root_ang_vel = asset.data.root_link_ang_vel_w + velocity_samples[:, 3:6]

    # Clipped to the soft limits as mjlab does, or a large jitter starts out of range.
    joint_pos = asset.data.joint_pos + sample_uniform(
        joint_position_range[0],
        joint_position_range[1],
        asset.data.joint_pos.shape,
        device=device,
    )
    soft_limits = asset.data.soft_joint_pos_limits
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])

    asset.write_joint_state_to_sim(joint_pos, asset.data.joint_vel, env_ids=env_ids)
    asset.write_root_link_pose_to_sim(
        torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
        torch.cat([root_lin_vel, root_ang_vel], dim=-1), env_ids=env_ids
    )


def _motion_rsi_trace(cfg: Any) -> tuple[Any, dict[str, Any]] | None:
    """The reset graph for a `MotionCommandCfg`, or None if it jitters nothing.

    mjlab's own play-mode override clears `pose_range`/`velocity_range` but keeps
    `joint_position_range` at (-0.1, 0.1), so a deployed tracking policy usually gets
    only the joint jitter; any subset an author keeps goes through the same graph.
    """
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    pose_range = dict(getattr(cfg, "pose_range", None) or {})
    velocity_range = dict(getattr(cfg, "velocity_range", None) or {})
    joint_position_range = tuple(getattr(cfg, "joint_position_range", (0.0, 0.0)))
    if not pose_range and not velocity_range and joint_position_range == (0.0, 0.0):
        return None
    return (
        motion_rsi_offset,
        {
            "asset_cfg": SceneEntityCfg(getattr(cfg, "entity_name", None) or "robot"),
            "pose_range": pose_range,
            "velocity_range": velocity_range,
            "joint_position_range": joint_position_range,
        },
    )


_MOTION_COMMAND_BINDING = CommandBinding(
    ts_name="TrackingCommand",
    serializer=serialize_motion_command,
    reset_trace=_motion_rsi_trace,
)
register_command("MotionCommandCfg", _MOTION_COMMAND_BINDING)


__all__ = [
    "TRACKING_STATE_FIELDS",
    "motion_rsi_offset",
    "serialize_motion_command",
    "ui_command",
    "velocity_command",
]
