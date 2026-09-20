"""Termination terms: the traced ones fused into one graph, the rest as markers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...envs.mdp.terminations import TerminationBinding
from . import graph
from .binding import require_ts_src
from .provenance import graph_meta, resolved_params, term_provenance
from .sensor import structured_sensor_descriptors

if TYPE_CHECKING:
    from ...managers.termination_manager import TerminationTermCfg


def serialize_termination(
    name: str,
    term_cfg: TerminationTermCfg,
    env: Any,
    out_dir: Path,
    *,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Serialize one termination term."""
    from ...compile import trace_term
    from ...compile.tracer import ConstantTerm, slots_json

    func = term_cfg.func
    if isinstance(func, TerminationBinding):
        require_ts_src("Termination", name, func)
        return term_cfg.to_dict()
    if _is_native_termination(term_cfg):
        return _native_termination_entry(name, term_cfg, env)

    try:
        export = trace_term(func, resolved_params(term_cfg.params, env), env, name=name)
    except ConstantTerm as exc:
        raise _constant_termination_error(name) from exc

    ref = graph.onnx_ref("term", name, scope)
    graph.write_onnx(
        out_dir, ref, export.onnx_bytes, meta=graph_meta("term", name, func)
    )
    entry: dict[str, Any] = {
        "name": name,
        "onnx": ref,
        "input_slots": slots_json(export),
        **term_provenance(func, resolved_params(term_cfg.params, env)),
    }
    sensors = structured_sensor_descriptors(
        export, env, owner=f"Termination term {name!r}"
    )
    if sensors:
        entry["sensors"] = sensors
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


def _native_termination_entry(
    name: str, term_cfg: TerminationTermCfg, env: Any
) -> dict[str, Any]:
    """The `time_out` marker: it compares env-level step counters rather than entity
    data, so there is nothing to trace. The threshold travels with it."""
    entry: dict[str, Any] = {
        "name": name,
        "native": "elapsed_s >= episode_length_s",
        "episode_length_s": float(getattr(env, "max_episode_length_s", 0.0)),
        **term_provenance(term_cfg.func, term_cfg.params),
    }
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


def _is_native_termination(term_cfg: TerminationTermCfg) -> bool:
    """Whether the term is mjlab's `time_out`, decided by the function name."""
    from ...compile.tracer import is_native_termination

    return is_native_termination(term_cfg.func)


def _constant_termination_error(name: str) -> ValueError:
    return ValueError(
        f"Termination term {name!r} reads no simulation state, so it would fire every "
        "step or never; a constant is not a termination rule and is not written to the "
        "document. Read the state through env.scene[...], env.command_manager or "
        "env.sim.data, or use mjlab's own `time_out` for the episode timeout."
    )


def serialize_terminations(
    terminations: dict[str, TerminationTermCfg] | None,
    env: Any,
    out_dir: Path,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Serialize a policy's terminations, fusing the traced ones into one graph.

    Native markers (`time_out`) and `*Binding` terms stay as their own entries; the
    fused graph joins them under ``__fused__``.
    """
    result: dict[str, Any] = {}
    if not terminations:
        return result

    fusable: dict[str, TerminationTermCfg] = {}
    for name, term_cfg in terminations.items():
        func = term_cfg.func
        if isinstance(func, TerminationBinding):
            require_ts_src("Termination", name, func)
            result[name] = term_cfg.to_dict()
            continue
        if _is_native_termination(term_cfg):
            result[name] = _native_termination_entry(name, term_cfg, env)
            continue
        fusable[name] = term_cfg

    if not fusable:
        return result
    if len(fusable) == 1:
        # Fusing one term buys nothing and costs a wire shape, so don't.
        name, term_cfg = next(iter(fusable.items()))
        entry = serialize_termination(name, term_cfg, env, out_dir, scope=scope)
        if entry is not None:
            result[name] = entry
        return result

    result[FUSED_TERMINATION_KEY] = _fused_termination_entry(
        fusable, env, out_dir, "terminations", scope=scope
    )
    return result


FUSED_TERMINATION_KEY = "__fused__"
"""Config key the fused termination graph lives under. Cannot collide with a term
name, which is always a Python identifier."""


def _fused_termination_entry(
    terms: dict[str, TerminationTermCfg],
    env: Any,
    out_dir: Path,
    group_name: str,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    from ...compile.tracer import (
        ConstantTerm,
        GroupTermSpec,
        slots_json,
        trace_termination_group,
    )

    specs = [
        GroupTermSpec(name=name, func=cfg.func, params=resolved_params(cfg.params, env))
        for name, cfg in terms.items()
    ]
    try:
        export = trace_termination_group(specs, env, name=group_name)
    except ConstantTerm as exc:
        raise _constant_termination_error(exc.term) from exc
    ref = graph.onnx_ref("term", group_name, scope)
    graph.write_onnx(
        out_dir, ref, export.onnx_bytes, meta=graph_meta("term", group_name)
    )
    by_name = {spec.name: spec for spec in specs}
    return {
        "fused": ref,
        "input_slots": slots_json(export),
        # Lane order is the graph's; `time_out` rides along for the manager's split.
        "lanes": [
            {
                "name": name,
                "time_out": bool(terms[name].time_out),
                **term_provenance(by_name[name].func, by_name[name].params),
            }
            for name in export.lanes
        ],
    }
