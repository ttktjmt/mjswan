"""Trace-friendly rewrites of mjlab command bodies, and their registrations.

A command is a class, not a function, and mjlab's use constructs the tracer cannot
follow: ``Tensor.uniform_`` draws its RNG spy cannot see, and per-``env_ids`` assignment
that branches on live data. ``CommandBinding.trace_override`` rebinds the term to a body
equivalent at ``N=1`` and expressible as one graph.

Each rewrite is a second copy of mjlab's math, so it has to be reread whenever mjlab
moves. ``tests/test_velocity_command.py`` pins it against a live mjlab term, which the
parity harness cannot: that only ever checks "graph == override".
"""

from __future__ import annotations

import types
from typing import Any

from ...command import CommandBinding, register_command

try:
    # Module-level, not deferred: the RNG spy patches a term body's *module globals*,
    # so `sample_uniform` has to be one of ours to be seen.
    import torch
    from mjlab.utils.lab_api.math import sample_uniform, wrap_to_pi
except ImportError:
    pass


# --- UniformVelocityCommand (mjlab's locomotion tasks, and anything built on them) ---

#: `UniformVelocityCommand`'s floor on a forward-only env's commanded speed.
_FORWARD_MIN_SPEED = 0.3

#: Cfg fields the rewrite does not carry: a build error, not a silent difference.
_UNMODELLED_FIELDS: tuple[tuple[str, str], ...] = (
    (
        "init_velocity_prob",
        "starts an episode at the commanded velocity by writing the robot's root "
        "state during resampling, which needs a write gated on that same draw",
    ),
)


def _resample_velocity_command(self: Any, env_ids: Any) -> None:
    """``UniformVelocityCommand._resample_command``, as one graph at ``N=1``.

    In mjlab's own order: the world-frame reference is the raw sample, copied *before*
    the forward clamp.
    """
    cfg = self.cfg
    ranges = cfg.ranges
    n, device = self.num_envs, self.device

    vx = sample_uniform(*ranges.lin_vel_x, (n, 1), device=device)
    vy = sample_uniform(*ranges.lin_vel_y, (n, 1), device=device)
    wz = sample_uniform(*ranges.ang_vel_z, (n, 1), device=device)
    if cfg.heading_command:
        self.heading_target = sample_uniform(*ranges.heading, (n,), device=device)
        self.is_heading_env = (
            sample_uniform(0.0, 1.0, (n,), device=device) <= cfg.rel_heading_envs
        )
    self.is_standing_env = (
        sample_uniform(0.0, 1.0, (n,), device=device) <= cfg.rel_standing_envs
    )
    self.is_world_env = (
        sample_uniform(0.0, 1.0, (n,), device=device) <= cfg.rel_world_envs
    )
    self.vel_command_w = torch.cat([vx, vy, wz], dim=-1)

    self.is_forward_env = (
        sample_uniform(0.0, 1.0, (n,), device=device) <= cfg.rel_forward_envs
    )
    forward = self.is_forward_env.reshape(-1, 1)
    zero = torch.zeros_like(vx)
    vx = torch.where(forward, vx.abs().clamp(min=_FORWARD_MIN_SPEED), vx)
    self.vel_command_b = torch.cat(
        [vx, torch.where(forward, zero, vy), torch.where(forward, zero, wz)], dim=-1
    )


def _update_velocity_command(self: Any) -> None:
    """``UniformVelocityCommand._update_command``, as one graph at ``N=1``.

    Heading tracking, then the world-frame rotation, then standing zeroed last — the
    order mjlab applies them in.
    """
    cfg = self.cfg
    heading_w = self.robot.data.heading_w
    vx = self.vel_command_b[:, 0:1]
    vy = self.vel_command_b[:, 1:2]
    wz = self.vel_command_b[:, 2:3]

    if cfg.heading_command:
        tracked = torch.clip(
            cfg.heading_control_stiffness * wrap_to_pi(self.heading_target - heading_w),
            min=cfg.ranges.ang_vel_z[0],
            max=cfg.ranges.ang_vel_z[1],
        ).reshape(-1, 1)
        wz = torch.where(self.is_heading_env.reshape(-1, 1), tracked, wz)

    world = self.is_world_env.reshape(-1, 1)
    cos_h = torch.cos(heading_w).reshape(-1, 1)
    sin_h = torch.sin(heading_w).reshape(-1, 1)
    vx_w = self.vel_command_w[:, 0:1]
    vy_w = self.vel_command_w[:, 1:2]
    vx = torch.where(world, cos_h * vx_w + sin_h * vy_w, vx)
    vy = torch.where(world, -sin_h * vx_w + cos_h * vy_w, vy)

    standing = self.is_standing_env.reshape(-1, 1)
    command = torch.cat([vx, vy, wz], dim=-1)
    self.vel_command_b = torch.where(standing, torch.zeros_like(command), command)
    self.vel_command_w = torch.where(
        standing, torch.zeros_like(self.vel_command_w), self.vel_command_w
    )


def bind_velocity_override(term: Any) -> None:
    """Swap a built ``UniformVelocityCommand`` onto the two bodies above."""
    for field, behaviour in _UNMODELLED_FIELDS:
        if float(getattr(term.cfg, field, 0.0) or 0.0) > 0.0:
            raise ValueError(
                f"UniformVelocityCommandCfg sets {field}="
                f"{getattr(term.cfg, field)!r}, which mjswan's trace-friendly "
                f"rewrite does not carry: it {behaviour}. Tracing it anyway would "
                "give the browser a command that differs from mjlab's, so extend "
                "`mjswan.envs.mdp.commands` or register your own override via "
                "mjswan.register_command('UniformVelocityCommandCfg', ...)."
            )
    term._resample_command = types.MethodType(_resample_velocity_command, term)
    term._update_command = types.MethodType(_update_velocity_command, term)


# No `ui=`: the joystick descriptor is recorded from the term's own `create_gui` at
# build time (`mjswan.mjlab.gui`). No `viz=`: `mjswan.mjlab.command.default_viz` has it.
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
        trace_override=bind_velocity_override,
    ),
)


__all__ = ["bind_velocity_override"]
