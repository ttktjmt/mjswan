"""Observation terms and groups: one fused graph where possible, else per term."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...command import ButtonConfig, CommandTermConfig
from ...envs.mdp.observations import ObservationBinding
from . import graph
from .binding import require_ts_src
from .provenance import graph_meta, resolved_params, term_provenance
from .sensor import structured_sensor_descriptors

if TYPE_CHECKING:
    from ...managers.observation_manager import ObservationGroupCfg, ObservationTermCfg


def _tensor_width(value: Any) -> int:
    """Per-env element count of a term's output (batch axis folded away)."""
    return int(value.detach().reshape(1, -1).shape[-1])


def _native_observation_entry(
    name: str, func: Any, params: dict[str, Any], env: Any
) -> dict[str, Any] | None:
    """Classify a known non-``entity.data`` observation func into a native marker.

    ``last_action`` and ``generated_commands`` read env-level state the runtime already
    holds every frame, so they need no graph. Returns ``None`` for anything else, which
    the caller then traces.

    Checked before tracing rather than by catching the tracer's error: a scene pairing
    ``generated_commands`` with a browser-only ``UiCommand`` fails mjlab's own assert
    during discovery.
    """
    from ...compile.native import native_observation_entry

    entry = native_observation_entry(name, func, params, env)
    if entry is None:
        return None

    try:
        width = _tensor_width(func(env, **params))
    except Exception:  # noqa: BLE001 — best-effort; runtime resolves it instead
        width = 0
    if width:
        # A zero width means "no action manager", not a term of no width.
        entry["size"] = width
    return entry


def _apply_observation_pipeline(
    entry: dict[str, Any],
    term_cfg: ObservationTermCfg,
    group_history_length: int | None,
) -> dict[str, Any]:
    """Add the scale/clip/history metadata every entry shape carries, in mjlab's
    compute -> scale -> history order. Noise and delay are training-only, so dropped."""
    if term_cfg.scale is not None:
        entry["scale"] = (
            list(term_cfg.scale)
            if isinstance(term_cfg.scale, tuple)
            else term_cfg.scale
        )
    if term_cfg.clip is not None:
        entry["clip"] = list(term_cfg.clip)
    # A group count replaces the term's whenever set, `0` included, as mjlab does.
    history = (
        group_history_length
        if group_history_length is not None
        else term_cfg.history_length
    )
    if term_cfg.history_steps:
        # Sparse offsets are per-term, so a group count would be a second answer.
        entry["history_offsets"] = [int(step) for step in term_cfg.history_steps]
    elif history:
        entry["history_length"] = history
    # Interleaving describes a stack's layout, so it says nothing without a stack.
    if term_cfg.history_interleaved and (
        "history_offsets" in entry or "history_length" in entry
    ):
        entry["history_interleaved"] = True
    return entry


def serialize_observation_term(
    name: str,
    term_cfg: ObservationTermCfg,
    env: Any,
    out_dir: Path,
    group_history_length: int | None,
    *,
    scope: str | None = None,
) -> dict[str, Any] | None:
    """Serialize one observation term.

    Raises rather than degrading: dropping a term shortens the vector the policy was
    trained on, and baking a time-varying one freezes an input.
    """
    from ...compile import trace_term
    from ...compile.slot import slots_json
    from ...compile.term import ConstantTerm, warn_constant_observation

    func = term_cfg.func
    if isinstance(func, ObservationBinding):
        require_ts_src("Observation", name, func)
        return term_cfg.to_dict()

    params = resolved_params(term_cfg.params, env)

    provenance = term_provenance(func, params)

    native_entry = _native_observation_entry(name, func, params, env)
    if native_entry is not None:
        native_entry.update(provenance)
        return _apply_observation_pipeline(native_entry, term_cfg, group_history_length)

    try:
        export = trace_term(func, params, env, name=name)
    except ConstantTerm:
        # Reads nothing off the env, so bake it from a real call.
        import torch

        value = func(env, **params)
        if not isinstance(value, torch.Tensor):
            raise
        values = value.detach().flatten().tolist()
        warn_constant_observation(name, len(values))
        entry = {
            "name": name,
            "native": "constant",
            "value": values,
            "size": len(values),
            **provenance,
        }
        return _apply_observation_pipeline(entry, term_cfg, group_history_length)
    ref = graph.onnx_ref("obs", name, scope)
    graph.write_onnx(
        out_dir, ref, export.onnx_bytes, meta=graph_meta("obs", name, func)
    )

    # The runtime cannot infer `size`: inference is async, the group layout is not.
    entry = {
        "name": name,
        "onnx": ref,
        "size": _tensor_width(export.reference_output),
        "input_slots": slots_json(export),
        **provenance,
    }
    return _apply_observation_pipeline(entry, term_cfg, group_history_length)


def _effective_history(group: ObservationGroupCfg, term_cfg: ObservationTermCfg) -> int:
    """Stack depth applied to one term — group level wins, as in mjlab.

    Sparse offsets (``history_steps``) count as their own depth: they are per-term by
    construction, so a group level cannot override them.
    """
    if term_cfg.history_steps:
        return len(term_cfg.history_steps)
    if group.history_length is not None:
        return int(group.history_length)
    return int(term_cfg.history_length or 0)


def _group_is_fusable(group: ObservationGroupCfg) -> bool:
    """Whether the whole group can become one graph.

    A ``*Binding`` term has no body to trace, and per-term history deeper than one
    frame cannot fuse: mjlab stacks each term *before* concatenating, so a group-level
    ring buffer over one fused output would give step-major order where mjlab gives
    term-major.
    """
    for term_cfg in group.terms.values():
        if isinstance(term_cfg.func, ObservationBinding):
            return False
        # Sparse offsets disqualify at any length: no fused output holds a delayed frame.
        if term_cfg.history_steps or _effective_history(group, term_cfg) > 1:
            return False
    return True


def policy_native_sizes(
    data: dict[str, Any], commands: Mapping[str, CommandTermConfig] | None
) -> dict[str, int]:
    """Widths of the native observation terms, keyed as :func:`_native_size` reads them.

    A trace env built for a plain ``add_scene()`` scene has neither an action term nor
    the command, so a fused graph takes its fixed widths from the policy config
    instead: the action count, and the command's value-bearing UI inputs (a button
    carries none).
    """
    sizes: dict[str, int] = {}
    num_actions = data.get("policy_num_actions") or len(
        data.get("policy_joint_names") or ()
    )
    if num_actions:
        sizes["prev_action"] = int(num_actions)
    for name, cmd in (commands or {}).items():
        if cmd.ui is None:
            continue
        width = sum(1 for inp in cmd.ui.inputs if not isinstance(inp, ButtonConfig))
        if not width:
            continue
        sizes[f"command:{name}"] = width
    return sizes


def _native_size(
    term_cfg: ObservationTermCfg, native_sizes: dict[str, int]
) -> int | None:
    """Declared width for a native term, or ``None`` if it isn't native."""
    func_name = getattr(term_cfg.func, "__name__", None)
    if func_name == "last_action":
        # The whole vector; a term-scoped one needs `action_offset` for its slice.
        if term_cfg.params.get("action_name") is not None:
            return None
        return native_sizes.get("prev_action")
    if func_name == "generated_commands":
        return native_sizes.get(f"command:{term_cfg.params['command_name']}")
    return None


def _fused_group_entry(
    group: ObservationGroupCfg,
    env: Any,
    out_dir: Path,
    group_name: str,
    native_sizes: dict[str, int] | None = None,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Trace the group as one graph and return the fused config entry."""
    from ...compile.group import GroupTermSpec, trace_observation_group
    from ...compile.slot import slots_json

    specs = [
        GroupTermSpec(
            name=name,
            func=term_cfg.func,
            params=resolved_params(term_cfg.params, env),
            clip=tuple(term_cfg.clip) if term_cfg.clip else None,
            scale=term_cfg.scale,
            native_size=_native_size(term_cfg, native_sizes or {}),
        )
        for name, term_cfg in group.terms.items()
    ]
    export = trace_observation_group(specs, env, name=group_name)
    ref = graph.onnx_ref("obs", group_name, scope)
    graph.write_onnx(
        out_dir, ref, export.onnx_bytes, meta=graph_meta("obs", group_name)
    )
    by_name = {spec.name: spec for spec in specs}
    entry: dict[str, Any] = {
        "fused": ref,
        "input_slots": slots_json(export),
        "native_inputs": export.native_inputs,
        # Per-term widths in concat order, for the runtime's group layout.
        "layout": [
            {
                **row,
                **term_provenance(
                    by_name[row["name"]].func, by_name[row["name"]].params
                ),
            }
            for row in export.layout
        ],
        "size": _tensor_width(export.reference_output),
    }
    sensors = structured_sensor_descriptors(
        export, env, owner=f"Observation group {group_name!r}"
    )
    if sensors:
        # Structured sensors only; a builtin one is a `sensordata` window.
        entry["sensors"] = sensors
    return entry


def serialize_observation_group(
    group: ObservationGroupCfg,
    env: Any,
    out_dir: Path,
    group_name: str = "policy",
    native_sizes: dict[str, int] | None = None,
    *,
    scope: str | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Serialize an observation group — one fused graph where possible, else per term."""
    from ...compile.group import ConstantGroup

    if _group_is_fusable(group):
        try:
            return _fused_group_entry(
                group, env, out_dir, group_name, native_sizes, scope=scope
            )
        except ConstantGroup:
            # Only knowable by tracing, so not a `_group_is_fusable` static check.
            pass
    result = []
    for name, term_cfg in group.terms.items():
        entry = serialize_observation_term(
            name, term_cfg, env, out_dir, group.history_length, scope=scope
        )
        if entry is not None:
            result.append(entry)
    return result
