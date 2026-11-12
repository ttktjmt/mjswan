"""Event terms: traced to a graph, or marked native when there is legitimately nothing
to write."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ...envs.mdp.events import EventBinding
from . import graph
from .binding import require_ts_src
from .provenance import graph_meta, resolved_params, term_provenance

if TYPE_CHECKING:
    from ...managers.event_manager import EventTermCfg


def serialize_event(
    name: str,
    term_cfg: EventTermCfg,
    env: Any,
    out_dir: Path,
    *,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Serialize one event term, or ``None`` if there is genuinely nothing to emit."""
    # Before the tracer import: a config mistake should not need a tracer to report.
    if term_cfg.mode == "manual" and term_cfg.interval_range_s is not None:
        raise ValueError(
            f'Event term {name!r} is mode="manual" and carries '
            f"interval_range_s={term_cfg.interval_range_s!r}. A manual term has no "
            "schedule: the operator's button is its only trigger. Declare a second "
            'mode="interval" term if it should also fire on its own.'
        )
    from ...compile import trace_event_term
    from ...compile.event import event_writes_nothing_reason
    from ...compile.export import DivergentGraph
    from ...compile.slot import UnsupportedEnvRead, slots_json

    func = term_cfg.func
    if isinstance(func, EventBinding):
        require_ts_src("Event", name, func)
        return term_cfg.to_dict()

    resolved = resolved_params(term_cfg.params, env)
    if isinstance(func, type):
        from mjlab.managers.manager_base import ManagerTermBase

        if issubclass(func, ManagerTermBase):
            constructor = cast(Callable[..., Any], func)
            func = constructor(replace(term_cfg, params=resolved), env).__call__
    provenance = term_provenance(func, resolved)
    try:
        export = trace_event_term(
            func,
            resolved,
            env,
            name=name,
            mode=term_cfg.mode,
            is_global_time=getattr(term_cfg, "is_global_time", False),
        )
    except DivergentGraph:
        raise
    except (ValueError, UnsupportedEnvRead) as exc:
        nothing_to_write = event_writes_nothing_reason(func, env, resolved)
        if nothing_to_write is not None:
            return {
                "name": name,
                "mode": term_cfg.mode,
                "native": True,
                "reason": nothing_to_write,
                **provenance,
            }
        raise ValueError(
            f"Event term {name!r} could not be traced: {exc} Emitting it as a no-op "
            "would drop a randomization the task is configured to apply, with nothing "
            "said about it in the browser. Either supply a trace-friendly replacement "
            "via mjswan.register_event(), or write the term as a TS class and point an "
            "EventBinding's `ts_src` at it."
        ) from exc

    ref = graph.onnx_ref("event", name, scope)
    graph.write_onnx(
        out_dir, ref, export.onnx_bytes, meta=graph_meta("event", name, func)
    )
    entry: dict[str, Any] = {
        "name": name,
        "mode": term_cfg.mode,
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
        **provenance,
    }
    if export.set_const:
        entry["set_const"] = True
    if term_cfg.mode == "interval":
        entry["interval_range_s"] = (
            list(term_cfg.interval_range_s) if term_cfg.interval_range_s else None
        )
        entry["is_global_time"] = term_cfg.is_global_time
    if term_cfg.mode == "reset" and term_cfg.min_step_count_between_reset:
        entry["min_step_count_between_reset"] = term_cfg.min_step_count_between_reset
    if term_cfg.label is not None:
        entry["label"] = term_cfg.label
    if term_cfg.disabled_when is not None:
        entry["disabled_when"] = term_cfg.disabled_when
    return entry


def _check_disabled_when(events: Mapping[str, EventTermCfg]) -> None:
    """Refuse a `disabled_when` unless a manual term names a `mode="interval"` term.

    A gate that resolves to nothing greys its button out forever, or never.
    """
    for name, term_cfg in events.items():
        gate = getattr(term_cfg, "disabled_when", None)
        if gate is None:
            continue
        if term_cfg.mode != "manual":
            raise ValueError(
                f'Event term {name!r} is mode="{term_cfg.mode}" and carries '
                f"disabled_when={gate!r}. Only a manual term has a button to grey out."
            )
        if getattr(events.get(gate), "mode", None) != "interval":
            raise ValueError(
                f"Event term {name!r} declares disabled_when={gate!r}, which is not a "
                f'mode="interval" term of this scene (it has '
                f"{sorted(n for n, t in events.items() if t.mode == 'interval')})."
            )


def serialize_events(
    events: Mapping[str, EventTermCfg] | None,
    env: Any,
    out_dir: Path,
    on_term: Callable[[str], None] | None = None,
    *,
    scope: str | None = None,
) -> list[dict[str, Any]] | None:
    """Serialize an MDP's events dict to the JSON list the manifest carries.

    ``on_term`` names each term before it is traced, for the build's progress line.
    *scope* is the owning MDP's directory; see :func:`.graph.onnx_ref`.
    """
    if not events:
        return None
    _check_disabled_when(events)
    result = []
    for name, term_cfg in events.items():
        if on_term is not None:
            on_term(name)
        entry = serialize_event(name, term_cfg, env, out_dir, scope=scope)
        if entry is not None:
            result.append(entry)
    return result or None
