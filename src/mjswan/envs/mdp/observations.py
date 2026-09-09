"""Custom observation registry.

mjswan reimplements none of mjlab's observation functions: a task's real function
object is traced to ONNX at build time. This module carries the ``ObservationBinding``
escape hatch, for a term that cannot be traced at all, plus :func:`action_history` —
env-level state the runtime already holds, so no graph is traced for it either.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationBinding:
    """A hand-written TS observation class, bound to an mjlab observation name.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Observations`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected into
            the bundle at build time. Required — mjswan ships no built-in TS classes,
            so without it the build fails.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, ObservationBinding | Callable[..., Any]] = {}
"""Maps an mjlab observation function name to its override.

Populated via :func:`register_observation`; consulted by the mjlab adapter when
the config's own ``func`` needs replacing."""


def register_observation(
    mjlab_name: str, sentinel: ObservationBinding | Callable[..., Any]
) -> None:
    """Override how one mjlab observation function is exported.

    Call before :meth:`~mjswan.Builder.build`. ``sentinel`` is either a traceable
    ``func(env, **params) -> Tensor`` to trace in place of the task's own, or an
    :class:`ObservationBinding` naming a hand-written TS class.

    Args:
        mjlab_name: The mjlab observation function name (e.g. ``"height_scan"``).
        sentinel: A traceable callable, or an :class:`ObservationBinding`.

    Example::

        register_observation(
            "my_custom_obs",
            ObservationBinding(ts_name="MyCustomObs", ts_src="/path/to/MyCustomObs.ts"),
        )
    """
    _custom_registry[mjlab_name] = sentinel


def action_history(env: Any, age: int = 1, action_name: str | None = None) -> Any:
    """The raw policy action from *age* control steps back.

    mjlab's ``ActionManager`` keeps a three-deep window — ``action``, ``prev_action``,
    ``prev_prev_action`` — and a task observing action history reads the older two
    directly, where mjlab's own ``last_action`` only ever returns the newest. The
    browser holds the same window, so like ``last_action`` this needs no graph.

    ``age=0`` is ``last_action``. Point a task's own term at this with
    ``term_cfg.func = action_history`` and ``params={"age": 1}``.
    """
    manager = env.action_manager
    fields = ("action", "prev_action", "prev_prev_action")
    if not 0 <= age < len(fields):
        raise ValueError(f"age must be one of {tuple(range(len(fields)))}, got {age}")
    actions = getattr(manager, fields[age])
    if action_name is None:
        return actions
    offset = 0
    for term_name, dim in zip(
        manager.active_terms, manager.action_term_dim, strict=True
    ):
        if term_name == action_name:
            return actions[:, offset : offset + int(dim)]
        offset += int(dim)
    raise ValueError(
        f"action_history(action_name={action_name!r}) names an action term the scene "
        f"does not define. Available: {', '.join(manager.active_terms) or '(none)'}."
    )


__all__ = [
    "ObservationBinding",
    "action_history",
    "register_observation",
    "_custom_registry",
]
