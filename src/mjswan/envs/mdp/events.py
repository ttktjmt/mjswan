"""Custom event registry.

mjswan reimplements none of mjlab's event functions: a task's real function object —
``func(env, env_ids, **params)``, writing via ``entity.write_*_to_sim`` — is traced to
ONNX at build time.

It carries the ``EventBinding`` escape hatch, for a term that cannot be traced at all,
plus :func:`reset_root_state_on_flat_patch` — the one event mjswan owns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    # Only an mjlab task reaches the term below, but the tracer's RNG spy patches a
    # term's *module globals*, so `sample_uniform` has to be one.
    import torch
    from mjlab.utils.lab_api.math import (
        quat_from_euler_xyz,
        quat_mul,
        sample_uniform,
    )
except ImportError:
    pass


@dataclass(frozen=True)
class EventBinding:
    """A hand-written TS event class, bound to an mjlab event name.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Events`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected into
            the bundle at build time. Required — mjswan ships no built-in TS classes,
            so without it the build fails.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, EventBinding] = {}
"""Maps an mjlab event function name to its override.

Populated via :func:`register_event`; consulted by the mjlab adapter when the
config's own ``func`` needs replacing."""


def register_event(mjlab_name: str, sentinel: EventBinding) -> None:
    """Bind one mjlab event to a hand-written TS class.

    Call before :meth:`~mjswan.Builder.build`, so the adapter resolves the name
    and the builder injects ``ts_src`` into the browser bundle.

    Args:
        mjlab_name: The mjlab event function name (e.g. ``"push_robot"``).
        sentinel: An :class:`EventBinding` whose ``ts_src`` implements it.
    """
    _custom_registry[mjlab_name] = sentinel


def reset_root_state_on_flat_patch(
    env: Any,
    env_ids: Any,
    *,
    asset_cfg: Any,
    patches: list[list[float]],
    yaw_range: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Place the root on a uniformly-chosen flat terrain patch, at a random yaw.

    mjlab's own ``reset_root_state_from_flat_patches`` cannot be traced: it draws with
    ``torch.randint``, which the tracer does not record, and indexes per-env terrain
    tensors the browser has no counterpart for. ``patches`` is the generator's sampled
    ``(x, y, z)`` list instead, baked into the graph.

    Standing height and orientation come from ``default_root_state``, as mjlab's do —
    never from the live root pose, which the browser's keyframe restore has zeroed by
    the time a reset event runs.
    """
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state
    device = root_states.device

    table = torch.tensor(patches, dtype=torch.float, device=device)
    count = table.shape[0]
    # A uniform draw scaled to an index, not a randint: `sample_uniform` is what the
    # tracer records, so the browser redraws it from the same seeded stream.
    pick = sample_uniform(0.0, 1.0, (1,), device=device)
    index = torch.clamp((pick * count).long(), max=count - 1)
    patch = table.index_select(0, index)

    root_pos = torch.cat(
        [patch[:, 0:2], patch[:, 2:3] + root_states[:, 2:3]],
        dim=-1,
    )
    yaw = sample_uniform(yaw_range[0], yaw_range[1], (1,), device=device)
    zeros = torch.zeros_like(yaw)
    root_quat = quat_mul(quat_from_euler_xyz(zeros, zeros, yaw), root_states[:, 3:7])
    asset.write_root_link_pose_to_sim(
        torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids
    )


__all__ = [
    "EventBinding",
    "register_event",
    "reset_root_state_on_flat_patch",
    "_custom_registry",
]
