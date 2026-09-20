"""Termination manager configuration for mjswan.

Provides ``TerminationTermCfg`` with an API compatible with
``mjlab.managers.termination_manager``.

Example (identical to mjlab)::

    from mjlab.envs.mdp import terminations as term_fns
    from mjswan.managers.termination_manager import TerminationTermCfg

    terminations = {
        "time_out": TerminationTermCfg(
            func=term_fns.time_out, time_out=True,
        ),
        "fallen": TerminationTermCfg(
            func=term_fns.bad_orientation,
            params={"limit_angle": 1.0},
        ),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.mdp.terminations import TerminationBinding


@dataclass
class TerminationTermCfg:
    """Configuration for a single termination term.

    Mirrors ``mjlab.managers.termination_manager.TerminationTermCfg``.

    ``func`` is either a :class:`TerminationBinding` (resolved to a TS class by name) or
    a plain ``func(env, **params)`` the build traces to ONNX. mjlab's ``time_out`` is
    classified native by function; a term that reads no state at all fails the build.
    """

    func: TerminationBinding | Callable[..., Any]
    """Termination function — TerminationBinding sentinel (legacy) or a
    plain callable traced to ONNX (ADR 0005)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the function or TS constructor."""

    time_out: bool = False
    """Whether this term is a truncation (time-based) rather than a
    terminal failure.  Maps to the ``time_out`` flag in the JSON config."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize a ``TerminationBinding`` term.

        A plain-callable term needs a live env this method has no access to; the Builder
        calls ``mjswan.build.mdp.serialize_termination`` for those.
        """
        if isinstance(self.func, TerminationBinding):
            return self._to_dict_legacy()
        raise TypeError(
            f"TerminationTermCfg.to_dict() cannot serialize a plain callable "
            f"func ({self.func!r}) — it must be traced to ONNX against a live "
            f"env. Use mjswan.build.mdp.serialize_termination(cfg, env, "
            f"out_dir) instead (the Builder does this automatically)."
        )

    def _to_dict_legacy(self) -> dict[str, Any]:
        func: TerminationBinding = self.func  # type: ignore[assignment]
        entry: dict[str, Any] = {"name": func.ts_name}
        merged: dict[str, Any] = {**func.defaults, **self.params}
        if merged:
            entry["params"] = merged
        if self.time_out:
            entry["time_out"] = True
        return entry


__all__ = ["TerminationTermCfg"]
