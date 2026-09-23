"""mjlab command terms as mjswan's, and what mjlab's command classes draw.

A command is a class, so ``type(cfg).__name__`` is looked up in the command registry
(:func:`mjswan.register_command`), which either traces the built term or maps it to a
permanently-native TS class.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from ..managers.command_manager import CommandTermConfig as MjswanCommandTermConfig
from ..managers.command_manager import PendingCommandTrace, PendingResetTrace
from ..managers.command_manager import _custom_registry as _custom_command_registry
from .detect import is_from_mjlab


def _adapt_command_cfg(term: Any) -> MjswanCommandTermConfig:
    """Convert a single mjlab ``CommandTermCfg`` to mjswan."""

    if isinstance(term, MjswanCommandTermConfig):
        return term

    class_name = type(term).__name__
    spec = _custom_command_registry.get(class_name)
    if spec is None:
        raise ValueError(
            f"No mjswan mapping for mjlab command config '{class_name}'. "
            f"Register one with mjswan.register_command()."
        )

    if spec.is_onnx_traced:
        assert spec.state_fields is not None and spec.command_field is not None
        ui = spec.ui(term) if callable(spec.ui) else spec.ui
        viz = spec.viz(term) if callable(spec.viz) else spec.viz
        if viz is None:
            viz = default_viz(term)
        return MjswanCommandTermConfig(
            term_name="OnnxCommand",
            pending_trace=PendingCommandTrace(
                mjlab_cfg=term,
                state_fields=spec.state_fields,
                command_field=spec.command_field,
                trace_override=spec.trace_override,
                ui=ui,
                viz=viz,
            ),
        )

    assert spec.serializer is not None
    serialized = dict(spec.serializer(term))
    # A native term may still own a reset-time graph for its randomization.
    reset_trace = spec.reset_trace(term) if spec.reset_trace is not None else None
    return MjswanCommandTermConfig(
        term_name=spec.ts_name,
        params=serialized,
        pending_reset_trace=(
            PendingResetTrace(func=reset_trace[0], params=reset_trace[1])
            if reset_trace is not None
            else None
        ),
    )


def adapt_commands(
    commands: Mapping[str, Any] | None,
) -> dict[str, MjswanCommandTermConfig] | None:
    """Adapt command configs, converting mjlab types if detected."""

    if commands is None:
        return None

    adapted: dict[str, MjswanCommandTermConfig] = {}
    for key, term in commands.items():
        if isinstance(term, MjswanCommandTermConfig):
            adapted[key] = term
            continue
        # A registered name adapts wherever its class lives, even with no mjlab base.
        if is_from_mjlab(term) or type(term).__name__ in _custom_command_registry:
            try:
                adapted[key] = _adapt_command_cfg(term)
            except ValueError as exc:
                warnings.warn(
                    f"Skipping command term '{key}': {exc}",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            continue
        adapted[key] = term
    return adapted


# --- Debug visualization ---
# mjlab's `_debug_vis_impl` runs Python every frame, so the browser cannot use it.
# These restate what each command class draws, as data `core/command/debugViz.ts` reads.

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


def _lifting_viz(cfg: Any) -> list[dict[str, Any]]:
    """`LiftingCommand`'s target sphere, colored from the task's own cfg."""
    viz = getattr(cfg, "viz", None)
    color = list(getattr(viz, "target_color", (1.0, 0.0, 0.0, 1.0)))
    return [
        {
            "shape": "sphere",
            "radius": 0.03,
            "color": color,
            "origin": {"state": "target_pos"},
        }
    ]


_default_viz_builders: dict[str, Callable[[Any], list[dict[str, Any]]]] = {
    "UniformVelocityCommandCfg": _velocity_viz,
    "LiftingCommandCfg": _lifting_viz,
}


def default_viz(mjlab_cfg: Any) -> list[dict[str, Any]] | None:
    """The debug drawing mjswan knows for an mjlab command cfg, or ``None``.

    Keyed by cfg class name, so any task using one of mjlab's own command cfgs gets it.
    """
    builder = _default_viz_builders.get(type(mjlab_cfg).__name__)
    return builder(mjlab_cfg) if builder is not None else None


__all__ = ["adapt_commands", "default_viz"]
