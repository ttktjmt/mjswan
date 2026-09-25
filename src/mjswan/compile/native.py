"""Terms the runtime evaluates itself, so there is nothing to trace.

Named by function: mjlab's ``last_action`` and ``generated_commands`` read env-level
state the browser already holds every frame, and ``time_out`` compares
``episode_length_buf``, a clock the runtime owns, against the horizon. Each becomes a
marker entry carrying the selector it needs.
"""

from __future__ import annotations

from typing import Any, Callable

NATIVE_OBSERVATION_FUNCS: dict[str, str] = {
    "last_action": "prev_action",
    "generated_commands": "command",
}


NATIVE_TERMINATION_FUNCS: frozenset[str] = frozenset({"time_out"})


def _native_observation_kind(func: Callable[..., Any]) -> str | None:
    return NATIVE_OBSERVATION_FUNCS.get(getattr(func, "__name__", ""))


def is_native_termination(func: Any) -> bool:
    """Whether *func* is the runtime's native `time_out` rather than a traced body."""
    return getattr(func, "__name__", "") in NATIVE_TERMINATION_FUNCS


def native_observation_entry(
    name: str, func: Callable[..., Any], params: dict[str, Any], env: Any
) -> dict[str, Any] | None:
    """The ``native`` marker for an observation the runtime holds, else ``None``.

    Carries the kind and whichever selector it needs; the caller adds ``size`` (and,
    when fusing, the graph ``input`` name) since the two paths resolve widths
    differently. ``action_offset`` is resolved here, outside the caller's width probe:
    that probe swallows exceptions, and a bad ``action_name`` must raise.
    """
    kind = _native_observation_kind(func)
    if kind is None:
        return None
    entry: dict[str, Any] = {"name": name, "native": kind}
    if kind == "command":
        entry["command_name"] = params["command_name"]
    elif params.get("action_name") is not None:
        entry["action_name"] = params["action_name"]
        entry["action_offset"] = action_term_offset(env, params["action_name"])
    return entry


def action_term_offset(env: Any, action_name: str) -> int:
    """Where *action_name*'s slice starts inside the policy's action vector.

    ``last_action(action_name=...)`` is one action term's slice, so the browser (which
    holds the policy output whole) needs this offset to reproduce it. Raises rather
    than falling back to the whole vector, which would look right until a scene has two
    action terms.
    """
    manager = env.action_manager
    names = list(manager.active_terms)
    offset = 0
    for term_name, dim in zip(names, manager.action_term_dim, strict=True):
        if term_name == action_name:
            return offset
        offset += int(dim)
    raise ValueError(
        f"last_action(action_name={action_name!r}) names an action term the scene "
        f"does not define. Available: {', '.join(names) if names else '(none)'}."
    )
