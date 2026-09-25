"""Command terms: the manifest entry and ``.onnx`` graph of a traced command.

One generic ``OnnxCommand`` runtime handler interprets every command from this data,
so the entry has to declare everything it needs to allocate state, supply ``rand``,
thread dynamic reads, and apply any ``entity_write``. The shape is defined by
:func:`command_config` and consumed by ``core/command/OnnxCommand.ts``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import graph
from .provenance import graph_meta, resolved_params, term_provenance

if TYPE_CHECKING:
    from ...compile.command import CommandExport
    from ...managers.command_manager import CommandTermConfig


def command_config(
    export: CommandExport,
    *,
    onnx_ref: str,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
    viz: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``OnnxCommand`` config entry from a trace.

    The term's own id is its key in the MDP's ``commands``; ``term_id`` here is only
    for diagnostics.

    ``ui`` (control-panel inputs) and ``viz`` (what mjlab's ``_debug_vis_impl`` draws,
    shown while ``debug_vis`` is on) are not derivable from the trace.
    """
    from ...compile.slot import slots_json

    cfg: dict[str, Any] = {
        "name": "OnnxCommand",
        "term_id": export.name,
        "onnx": onnx_ref,
        "command_field": export.command_field,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "state_fields": export.state_fields,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
        "debug_vis": bool(debug_vis),
    }
    if resampling_time_range is not None:
        cfg["resampling_time_range"] = [float(v) for v in resampling_time_range]
    if ui is not None:
        cfg["ui"] = ui
    if viz is not None:
        _check_viz_state_fields(export, viz)
        cfg["viz"] = viz
    return cfg


def _check_viz_state_fields(export: CommandExport, viz: list[dict[str, Any]]) -> None:
    """Warn for a primitive reading a state field the trace does not have.

    The browser hides a primitive whose source is missing, so a stale field name is a
    silently blank drawing. A warning, not an error: the drawing is presentation, and a
    term that can still be flown is worth shipping.
    """
    declared = {sf["name"] for sf in export.state_fields}
    missing = {
        vec["state"]
        for primitive in viz
        for vec in (primitive.get("origin"), primitive.get("vector"))
        if isinstance(vec, dict) and vec.get("state") not in (None, *declared)
    }
    if missing:
        warnings.warn(
            f"Command term '{export.name}' has debug-vis primitives reading "
            f"{sorted(missing)}, which its trace does not declare "
            f"(state fields: {sorted(declared)}); the browser draws nothing for them.",
            category=RuntimeWarning,
            stacklevel=3,
        )


def write_command_artifact(
    export: CommandExport,
    out_dir: str | Path,
    *,
    scope: str | None = None,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
    viz: list[dict[str, Any]] | None = None,
    meta: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``<out_dir>/[<scope>/]command/<name>.onnx`` and return its config entry.

    *scope* is the owning MDP's directory; see :func:`.graph.onnx_ref`. *meta*
    is stamped into the graph (:func:`.graph.stamp_provenance`).
    """
    ref = graph.onnx_ref("command", export.name, scope)
    graph.write_onnx(Path(out_dir), ref, export.onnx_bytes, meta=meta)
    return command_config(
        export,
        onnx_ref=ref,
        resampling_time_range=resampling_time_range,
        debug_vis=debug_vis,
        ui=ui,
        viz=viz,
    )


def _serialize_reset_graph(
    name: str,
    cmd_cfg: CommandTermConfig,
    env: Any,
    out_dir: Path,
    *,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Trace a native command's reset-time graph, or ``None`` if it has none.

    Deliberately shaped as an event entry, so the browser runs it through the same
    ``OnnxEvent`` handler rather than growing a second graph evaluator.
    """
    pending = cmd_cfg.pending_reset_trace
    if pending is None:
        return None
    from ...compile import trace_event_term
    from ...compile.slot import slots_json

    graph_name = f"{name}_reset"
    export = trace_event_term(
        pending.func,
        resolved_params(pending.params, env),
        env,
        name=graph_name,
        mode="reset",
    )
    ref = graph.onnx_ref("command", graph_name, scope)
    graph.write_onnx(
        out_dir,
        ref,
        export.onnx_bytes,
        meta=graph_meta("command", graph_name, pending.func),
    )
    return {
        "name": graph_name,
        "mode": "reset",
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
        **term_provenance(pending.func, resolved_params(pending.params, env)),
    }


def serialize_command(
    name: str,
    cmd_cfg: CommandTermConfig,
    env: Any,
    out_dir: Path,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Serialize one command term, resolving a pending ONNX trace if needed."""
    if cmd_cfg.pending_trace is None:
        reset_graph = _serialize_reset_graph(name, cmd_cfg, env, out_dir, scope=scope)
        if reset_graph is None:
            return cmd_cfg.to_dict()
        # `to_dict` refuses while a trace is pending; the graph is resolved now.
        resolved = replace(cmd_cfg, pending_reset_trace=None)
        return {**resolved.to_dict(), "reset_graph": reset_graph}

    from ...compile import trace_command_term

    pending = cmd_cfg.pending_trace
    term = pending.mjlab_cfg.build(env)
    if pending.trace_override is not None:
        pending.trace_override(term)

    export = trace_command_term(
        term,
        pending.state_fields,
        name=name,
        command_field=pending.command_field,
    )
    debug_vis = bool(getattr(pending.mjlab_cfg, "debug_vis", False))
    if debug_vis and not pending.viz:
        warnings.warn(
            f"Command term '{name}' has debug_vis=True but mjswan knows no debug "
            "drawing for it, so the browser shows nothing where mjlab's viewer draws. "
            f"Supply one via mjswan.register_command('{type(pending.mjlab_cfg).__name__}'"
            ", CommandBinding(..., viz=[...])).",
            category=RuntimeWarning,
            stacklevel=3,
        )
    # A traced command is a class, not a function.
    entry = write_command_artifact(
        export,
        out_dir,
        scope=scope,
        resampling_time_range=getattr(pending.mjlab_cfg, "resampling_time_range", None),
        debug_vis=debug_vis,
        ui=pending.ui or _record_command_gui(term, name),
        viz=pending.viz,
        meta=graph_meta("command", name, type(term)),
    )
    return {**entry, **term_provenance(type(term))}


def _record_command_gui(term: Any, name: str) -> dict[str, Any] | None:
    """The term's own viewer GUI as a UI descriptor, or ``None``.

    Called after ``trace_command_term``: ``create_gui`` leaves handles on the term that
    ``compute()`` reads. A descriptor is presentation, not behaviour, so a term this
    cannot record still builds.
    """
    from ...mjlab.gui import record_gui

    try:
        return record_gui(term, name)
    except Exception as exc:
        warnings.warn(
            f"Could not record command term '{name}' GUI from its mjlab "
            f"`create_gui` ({type(exc).__name__}: {exc}); the browser gets no "
            "control panel for it. Supply one via "
            f"mjswan.register_command('{type(term.cfg).__name__}', "
            "CommandBinding(..., ui=...)).",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None
