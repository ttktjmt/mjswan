"""Event manager configuration for mjswan.

Provides ``EventTermCfg`` for scene-level events: ``reset``, ``interval``
(e.g. periodic disturbances), ``startup`` (e.g. domain randomization run
once at load), and ``manual``, fired from the control panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..envs.mdp.events import EventBinding

EventMode = Literal["reset", "interval", "startup", "manual"]


@dataclass
class EventTermCfg:
    """Configuration for a single event term.

    Mirrors ``mjlab.managers.event_manager.EventTermCfg``.

    ``func`` is either an :class:`EventBinding` (resolved to a TS class by name) or a
    plain ``func(env, env_ids, **params)`` writing via ``entity.write_*_to_sim``, which
    the build traces to ONNX — see :func:`to_dict`.
    """

    func: EventBinding | Callable[..., Any]
    """Event function — EventBinding sentinel (legacy) or a traceable mjlab-style body."""

    mode: EventMode = "reset"
    """Event trigger mode: ``reset``, ``interval``, ``startup``, or ``manual`` — no
    schedule, the control panel's button is the whole trigger. ``manual`` is mjswan's
    own: an mjlab config carrying one is inert there."""

    params: dict[str, Any] = field(default_factory=dict)
    """Parameters forwarded to the TS event constructor or traced function."""

    interval_range_s: tuple[float, float] | None = None
    """``mode="interval"`` only: ``[min, max]`` seconds between firings."""

    is_global_time: bool = False
    """``mode="interval"`` only: timer survives episode reset when true."""

    min_step_count_between_reset: int | None = None
    """``mode="reset"`` only: suppress firing on resets that arrive too soon."""

    label: str | None = None
    """Control-panel text — a ``manual`` term's button, an ``interval`` term's arm
    checkbox. Defaults browser-side to the term name."""

    disabled_when: str | None = None
    """``mode="manual"`` only: the ``mode="interval"`` term that owns the same job. The
    button greys out, and refuses to fire, while that term's schedule is armed."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize an ``EventBinding`` term.

        A plain-callable term needs a live env this method has no access to; the Builder
        calls ``mjswan.build.mdp.serialize_event`` for those.
        """
        if isinstance(self.func, EventBinding):
            entry: dict[str, Any] = {"name": self.func.ts_name}
            merged: dict[str, Any] = {**self.func.defaults, **self.params}
            if merged:
                entry["params"] = merged
            return entry
        raise TypeError(
            f"EventTermCfg.to_dict() cannot serialize a plain callable func "
            f"({self.func!r}) — it must be traced to ONNX against a live env. "
            f"Use mjswan.build.mdp.serialize_event(cfg, env, out_dir) instead "
            f"(the Builder does this automatically)."
        )


__all__ = ["EventTermCfg", "EventMode"]
