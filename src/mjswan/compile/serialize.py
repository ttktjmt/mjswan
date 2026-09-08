"""Serialize a traced command into its ``policy.json`` entry and ``.onnx`` graph.

One generic ``OnnxCommand`` runtime handler interprets every command from this data,
so the entry has to declare everything it needs to allocate state, supply ``rand``,
thread dynamic reads, and apply any ``entity_write``. The shape is defined by
:func:`command_config` below and consumed by ``core/command/OnnxCommand.ts``; both
sides are covered by tests, so there is no third restatement of it here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .._graph_io import onnx_ref as _onnx_ref
from .._graph_io import write_onnx as _write_onnx
from .tracer import CommandExport, slots_json


def command_config(
    export: CommandExport,
    *,
    onnx_ref: str,
    resampling_time_range: tuple[float, float] | None = None,
    debug_vis: bool = False,
    ui: dict[str, Any] | None = None,
    viz: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the ``OnnxCommand`` config entry for ``policy.json`` from a trace.

    The term's own id is the outer key the author gives this entry in
    ``PolicyConfig.commands``; ``term_id`` here is only for diagnostics.

    ``ui`` (control-panel inputs) and ``viz`` (what mjlab's ``_debug_vis_impl`` draws,
    shown while ``debug_vis`` is on) are not derivable from the trace.
    """
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

    *scope* is the owning policy's id; see :func:`mjswan._graph_io.onnx_ref`. *meta*
    is stamped into the graph (:func:`mjswan._graph_io.stamp_provenance`).
    """
    ref = _onnx_ref("command", export.name, scope)
    _write_onnx(Path(out_dir), ref, export.onnx_bytes, meta=meta)
    return command_config(
        export,
        onnx_ref=ref,
        resampling_time_range=resampling_time_range,
        debug_vis=debug_vis,
        ui=ui,
        viz=viz,
    )
