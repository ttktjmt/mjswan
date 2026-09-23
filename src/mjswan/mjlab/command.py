"""mjlab command terms as mjswan's.

A command is a class, so ``type(cfg).__name__`` is looked up in the command registry
(:func:`mjswan.register_command`), which either traces the built term or maps it to a
permanently-native TS class.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
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


__all__ = ["adapt_commands"]
