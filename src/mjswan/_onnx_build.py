"""Bridges the term config dataclasses to the ``mjswan.compile`` tracer.

Traces each plain-callable term body against the scene's live mjlab env, writes the
``.onnx`` bytes under the scene's output directory, and returns the manifest-shaped
JSON entry the runtime consumes. ``*Binding``-typed terms (custom TS classes) keep
serializing through their own ``to_dict()``.

Called from :mod:`mjswan.builder` once per scene, after that scene's ``mjlab_env`` and
``scene_dir`` are both known.
"""

from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .command import ButtonConfig, CommandTermConfig
from .envs.mdp.events import EventBinding
from .envs.mdp.observations import ObservationBinding
from .envs.mdp.terminations import TerminationBinding

if TYPE_CHECKING:
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from .managers.termination_manager import TerminationTermCfg


def _onnx_ref(kind: str, name: str) -> str:
    """Bundle-relative path for a traced term's ``.onnx`` file."""
    return f"{kind}/{name}.onnx"


def _require_ts_src(kind: str, name: str, binding: Any) -> None:
    """A ``*Binding`` without ``ts_src`` names a class the browser does not have.

    mjswan ships no built-in TS term classes, so a binding is only ever the custom-TS
    escape hatch — without the file the term goes missing from a bundle that reports
    itself complete.
    """
    if binding.ts_src:
        return
    raise ValueError(
        f"{kind} term {name!r} is bound to TS class {binding.ts_name or '(unnamed)'!r} "
        "but no `ts_src` was given, so the browser has no implementation to run: "
        "mjswan ships no built-in TS term classes. Either let the build trace the "
        f"term's own function, or point `ts_src` at a `.ts` file exporting "
        f"{binding.ts_name or 'the class'!r}."
    )


def _write_onnx(out_dir: Path, ref: str, onnx_bytes: bytes) -> None:
    path = out_dir / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onnx_bytes)


# --- Observations ---


def _tensor_width(value: Any) -> int:
    """Per-env element count of a term's output (batch axis folded away)."""
    return int(value.detach().reshape(1, -1).shape[-1])


def _resolved_params(params: dict[str, Any], env: Any) -> dict[str, Any]:
    """Resolve every ``SceneEntityCfg`` in *params* against the live scene.

    mjlab's managers do this at ``_prepare_terms``, turning name patterns into concrete
    indices. The Builder serializes from the task config, whose cfgs are still
    unresolved (``site_ids=slice(None)`` — every site), so tracing without this bakes a
    different function than mjlab runs.

    A copy is resolved, since resolution mutates the cfg. Duck-typed to keep mjlab a
    soft dependency.
    """
    resolved = dict(params)
    for key, value in params.items():
        if callable(getattr(value, "resolve", None)) and hasattr(value, "name"):
            entity_cfg = copy.deepcopy(value)
            entity_cfg.resolve(env.scene)
            resolved[key] = entity_cfg
    return resolved


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
    from .compile.tracer import native_observation_entry

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
    # A group count replaces the term's whenever it is set, `0` included, as mjlab's
    # own `ObservationManager` does.
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
) -> dict[str, Any] | None:
    """Serialize one observation term.

    Raises rather than degrading: dropping a term shortens the vector the policy was
    trained on, and baking a time-varying one freezes an input.
    """
    from .compile import trace_term
    from .compile.tracer import ConstantTerm, slots_json

    func = term_cfg.func
    if isinstance(func, ObservationBinding):
        _require_ts_src("Observation", name, func)
        return term_cfg.to_dict()

    params = _resolved_params(term_cfg.params, env)

    native_entry = _native_observation_entry(name, func, params, env)
    if native_entry is not None:
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
        entry = {
            "name": name,
            "native": "constant",
            "value": values,
            "size": len(values),
        }
        return _apply_observation_pipeline(entry, term_cfg, group_history_length)
    ref = _onnx_ref("obs", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)

    # The runtime cannot infer `size`: inference is async, the group layout is not.
    entry = {
        "name": name,
        "onnx": ref,
        "size": _tensor_width(export.reference_output),
        "input_slots": slots_json(export),
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
    data: dict[str, Any], commands: dict[str, CommandTermConfig] | None
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
) -> dict[str, Any]:
    """Trace the group as one graph and return the fused config entry."""
    from .compile.tracer import (
        GroupTermSpec,
        slots_json,
        trace_observation_group,
    )

    specs = [
        GroupTermSpec(
            name=name,
            func=term_cfg.func,
            params=_resolved_params(term_cfg.params, env),
            clip=tuple(term_cfg.clip) if term_cfg.clip else None,
            scale=term_cfg.scale,
            native_size=_native_size(term_cfg, native_sizes or {}),
        )
        for name, term_cfg in group.terms.items()
    ]
    export = trace_observation_group(specs, env, name=group_name)
    ref = _onnx_ref("obs", group_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "fused": ref,
        "input_slots": slots_json(export),
        "native_inputs": export.native_inputs,
        # Per-term widths in concat order, for the runtime's group layout.
        "layout": export.layout,
        "size": _tensor_width(export.reference_output),
    }
    sensors = _structured_sensor_descriptors(
        export, env, owner=f"Observation group {group_name!r}"
    )
    if sensors:
        # Structured sensors only; a builtin one is a `sensordata` window.
        entry["sensors"] = sensors
    return entry


def _mj_element_name(env: Any, obj_type: str, obj_id: int) -> str:
    """Model name of a body/site/geom the sensor's rays are attached to.

    Names travel, not ids: the browser's model is compiled separately, so an id
    from the build env means nothing there.
    """
    mj_model = env.sim.mj_model
    return {"body": mj_model.body, "site": mj_model.site, "geom": mj_model.geom}[
        obj_type
    ](obj_id).name


_CONTACT_HISTORY_FIELDS = ("force", "torque", "dist")
"""Fields mjlab buffers (``ContactSensor.initialize``)."""


def contact_sensor_descriptor(env: Any, sensor_name: str) -> dict[str, Any] | None:
    """What the browser needs to reproduce one ``ContactSensor``, or None if not one.

    mjlab adds a real MuJoCo sensor per ``(primary, field)`` pair, so the values are
    already in ``sensordata`` — only the layout to read them back travels, plus the
    ring buffer the runtime owns.
    """
    sensor = env.scene.sensors.get(sensor_name)
    slots = getattr(sensor, "_slots", None)
    if not slots:
        return None
    fields: dict[str, Any] = {}
    for slot in slots:
        entry = fields.setdefault(slot.field_name, {"sensors": []})
        entry["sensors"].append(slot.sensor_name)
    for entry in fields.values():
        window = env.sim.mj_model.sensor(entry["sensors"][0])
        # `num_slots * dim` per window; the runtime reshapes with `dim`.
        entry["dim"] = int(window.dim[0]) // int(sensor.cfg.num_slots)
    history = [f for f in _CONTACT_HISTORY_FIELDS if f in fields]
    return {
        "kind": "contact",
        "num_slots": int(sensor.cfg.num_slots),
        "history_length": int(sensor.cfg.history_length),
        # Only these have a buffer, whatever `history_length` says.
        "history_fields": history if sensor.cfg.history_length > 0 else [],
        "fields": fields,
    }


def raycast_sensor_descriptor(env: Any, sensor_name: str) -> dict[str, Any] | None:
    """Everything the browser needs to reproduce one ``RayCastSensor``'s readings.

    There is no ``sensordata`` window for a raycast sensor, so the browser casts the
    rays itself with ``mj_ray``. The offsets and directions are baked from the live
    sensor rather than re-implementing mjlab's pattern generators, which means an
    unknown pattern works for free.

    Returns ``None`` if *sensor_name* is not a raycast sensor.
    """
    sensor = env.scene.sensors.get(sensor_name)
    offsets = getattr(sensor, "_local_offsets", None)
    if offsets is None:
        return None
    return {
        "kind": "raycast",
        # [N, 3] each, in the frame's local coordinates.
        "local_offsets": offsets.detach().cpu().tolist(),
        "local_directions": sensor._local_directions.detach().cpu().tolist(),
        "frames": [
            {"type": obj_type, "name": _mj_element_name(env, obj_type, obj_id)}
            for obj_type, obj_id, _ in sensor._frame_infos
        ],
        # "base" | "yaw" | "world" — how the frame's rotation reaches the rays.
        "ray_alignment": sensor.cfg.ray_alignment,
        "max_distance": float(sensor.cfg.max_distance),
        # mjlab excludes each frame's own parent body so a ray cannot self-hit.
        "exclude_parent_body": bool(sensor.cfg.exclude_parent_body),
        # A terrain scan is `(0,)`: without it the rays hit the robot's own legs.
        "include_geom_groups": (
            None
            if sensor.cfg.include_geom_groups is None
            else [int(g) for g in sensor.cfg.include_geom_groups]
        ),
    }


def _structured_sensor_descriptors(
    export: Any, env: Any, *, owner: str
) -> dict[str, Any]:
    """Descriptors for the structured sensors a graph's slots name.

    Without one the runtime would hold a stale value rather than fail, so an
    undescribable sensor fails the build instead.
    """
    from .compile.tracer import _SENSOR_NS

    descriptors: dict[str, Any] = {}
    for namespace, name_part in export.input_slots:
        if namespace != _SENSOR_NS or "." not in name_part:
            continue
        sensor_name, field = name_part.split(".", 1)
        if sensor_name in descriptors:
            continue
        descriptor = raycast_sensor_descriptor(
            env, sensor_name
        ) or contact_sensor_descriptor(env, sensor_name)
        if descriptor is None:
            sensor = env.scene.sensors.get(sensor_name)
            raise ValueError(
                f"{owner} reads {sensor_name!r}.{field}, but the browser has no "
                f"implementation for a {type(sensor).__name__} — only raycast and "
                "contact sensors can serve fields. Implement it in the runtime and emit "
                "a descriptor here, hand the term to the browser as a TS class, or drop "
                "it from the exported set."
            )
        descriptors[sensor_name] = descriptor
    return descriptors


def serialize_observation_group(
    group: ObservationGroupCfg,
    env: Any,
    out_dir: Path,
    group_name: str = "policy",
    native_sizes: dict[str, int] | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Serialize an observation group — one fused graph where possible, else per term."""
    from .compile.tracer import ConstantGroup

    if _group_is_fusable(group):
        try:
            return _fused_group_entry(group, env, out_dir, group_name, native_sizes)
        except ConstantGroup:
            # Only knowable by tracing, so not a `_group_is_fusable` static check.
            pass
    result = []
    for name, term_cfg in group.terms.items():
        entry = serialize_observation_term(
            name, term_cfg, env, out_dir, group.history_length
        )
        if entry is not None:
            result.append(entry)
    return result


# --- Terminations ---


def serialize_termination(
    name: str, term_cfg: TerminationTermCfg, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Serialize one termination term."""
    from .compile import trace_term
    from .compile.tracer import ConstantTerm, slots_json

    func = term_cfg.func
    if isinstance(func, TerminationBinding):
        _require_ts_src("Termination", name, func)
        return term_cfg.to_dict()

    try:
        export = trace_term(
            func, _resolved_params(term_cfg.params, env), env, name=name
        )
    except ConstantTerm:
        # Narrow: `UntraceableTerm` is a ValueError too, and must fail the build.
        return _native_termination_entry(name, term_cfg, env)

    ref = _onnx_ref("term", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "name": name,
        "onnx": ref,
        "input_slots": slots_json(export),
    }
    sensors = _structured_sensor_descriptors(
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
    }
    if term_cfg.time_out:
        entry["time_out"] = True
    return entry


def _is_native_termination(name: str, term_cfg: TerminationTermCfg, env: Any) -> bool:
    """Whether a term reads no time-varying state (so it cannot be traced).

    ponytail: discovers by tracing and discarding the export, so a fused term is traced
    twice. `trace_term` raises before `torch.onnx.export`, so the second pass is one term
    call; give the tracer a discovery-only entry point if that ever shows up in a build.
    The real name is passed so an `UntraceableTerm` escaping here names the term.
    """
    from .compile import trace_term
    from .compile.tracer import ConstantTerm

    try:
        trace_term(
            term_cfg.func,
            _resolved_params(term_cfg.params, env),
            env,
            name=name,
        )
    except ConstantTerm:
        return True
    return False


def serialize_terminations(
    terminations: dict[str, TerminationTermCfg] | None, env: Any, out_dir: Path
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
            _require_ts_src("Termination", name, func)
            result[name] = term_cfg.to_dict()
            continue
        if _is_native_termination(name, term_cfg, env):
            result[name] = _native_termination_entry(name, term_cfg, env)
            continue
        fusable[name] = term_cfg

    if not fusable:
        return result
    if len(fusable) == 1:
        # Fusing one term buys nothing and costs a wire shape, so don't.
        name, term_cfg = next(iter(fusable.items()))
        entry = serialize_termination(name, term_cfg, env, out_dir)
        if entry is not None:
            result[name] = entry
        return result

    result[FUSED_TERMINATION_KEY] = _fused_termination_entry(
        fusable, env, out_dir, "terminations"
    )
    return result


FUSED_TERMINATION_KEY = "__fused__"
"""Config key the fused termination graph lives under. Cannot collide with a term
name, which is always a Python identifier."""


def _fused_termination_entry(
    terms: dict[str, TerminationTermCfg], env: Any, out_dir: Path, group_name: str
) -> dict[str, Any]:
    from .compile.tracer import (
        GroupTermSpec,
        slots_json,
        trace_termination_group,
    )

    specs = [
        GroupTermSpec(
            name=name, func=cfg.func, params=_resolved_params(cfg.params, env)
        )
        for name, cfg in terms.items()
    ]
    export = trace_termination_group(specs, env, name=group_name)
    ref = _onnx_ref("term", group_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    return {
        "fused": ref,
        "input_slots": slots_json(export),
        # Lane order is the graph's; `time_out` rides along for the manager's split.
        "lanes": [
            {"name": name, "time_out": bool(terms[name].time_out)}
            for name in export.lanes
        ],
    }


# --- Events ---


_DR_ENTITY_INDEX_ATTR = {
    "geom": "geom_ids",
    "body": "body_ids",
    "site": "site_ids",
}


def _dr_entity_names(env: Any, asset_cfg: Any, entity_type: str) -> list[str] | None:
    """Names of the model elements a startup-DR event perturbs, or ``None`` for an
    entity type this cannot enumerate (the caller then leaves the event native)."""
    attr = _DR_ENTITY_INDEX_ATTR.get(entity_type)
    if attr is None:
        return None
    asset = env.scene[asset_cfg.name]
    scoped = getattr(asset_cfg, f"{entity_type}_ids", None)
    all_ids = getattr(asset.indexing, attr)
    ids = [int(i) for i in all_ids.tolist()]
    if scoped is not None and not isinstance(scoped, slice):
        # As mjlab's `_get_entity_indices`: positions into the entity's own elements.
        positions = list(scoped) if hasattr(scoped, "__iter__") else [scoped]
        ids = [ids[int(p)] for p in positions]
    accessor = {
        "geom": env.sim.mj_model.geom,
        "body": env.sim.mj_model.body,
        "site": env.sim.mj_model.site,
    }[entity_type]
    return [accessor(i).name for i in ids]


def _dr_arg(func: Any, params: dict[str, Any], key: str) -> Any:
    """A DR keyword as mjlab would see it: the term's value, else *func*'s default.

    Read off the signature, since mjlab's wrappers do not share defaults —
    ``geom_friction`` is ``"abs"``, ``body_mass`` is ``"scale"``.
    """
    if key in params:
        return params[key]
    param = inspect.signature(func).parameters.get(key)
    return (
        None
        if param is None or param.default is inspect.Parameter.empty
        else param.default
    )


def _dr_name_of(value: Any, fallback: str) -> str:
    """``Operation``/``Distribution`` accept an instance as well as a string."""
    if value is None:
        return fallback
    return str(getattr(value, "name", value))


def _dr_target_axes(
    func: Any, params: dict[str, Any], ranges: Any, default_axes: list[int]
) -> list[int]:
    """mjlab's ``_determine_target_axes`` precedence: explicit, then int keys, then default."""
    axes = _dr_arg(func, params, "axes")
    if axes is not None:
        return [int(a) for a in axes]
    if isinstance(ranges, dict):
        return [int(k) for k in ranges]
    return list(default_axes)


def model_field_dr_descriptor(
    term_cfg: EventTermCfg, env: Any, params: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Describe a startup model-field randomization for the browser, or None.

    These events perturb ``mjModel`` rather than ``mjData``, so the tracer captures no
    write. They need no graph either — draw a number per element per axis, combine it
    with the base value, write it back — so the browser does it at startup from the
    seeded PRNG and a session still replays.

    Returns ``None`` for anything undescribable: an unknown entity type, or the
    string-keyed ``ranges`` form mjlab resolves by name pattern.
    """
    func = term_cfg.func
    # Resolved: an unresolved `SceneEntityCfg` widens a scoped event to every geom.
    params = params if params is not None else _resolved_params(term_cfg.params, env)
    field = getattr(func, "_mjswan_dr_field", None) or _DR_FIELD_BY_FUNC.get(
        getattr(func, "__name__", "")
    )
    if field is None:
        return None
    field_name, entity_type, default_axes = field
    ranges = _dr_arg(func, params, "ranges")
    if not isinstance(ranges, (tuple, list, dict)):
        return None
    if isinstance(ranges, dict) and any(isinstance(k, str) for k in ranges):
        # mjlab resolves these by name pattern per element; not described here.
        return None
    asset_cfg = _dr_arg(func, params, "asset_cfg")
    if asset_cfg is None:
        return None
    names = _dr_entity_names(env, asset_cfg, entity_type)
    if names is None:
        return None

    axes = _dr_target_axes(func, params, ranges, default_axes)
    if isinstance(ranges, dict):
        # `_prepare_axis_ranges` drops a range for an axis nobody targets.
        if any(a not in ranges for a in axes):
            return None
        axis_ranges = {a: [float(ranges[a][0]), float(ranges[a][1])] for a in axes}
    else:
        axis_ranges = {a: [float(ranges[0]), float(ranges[1])] for a in axes}

    operation = _dr_name_of(_dr_arg(func, params, "operation"), "abs")
    return {
        "kind": "model_field",
        "field": field_name,
        "entity_type": entity_type,
        "entity_names": names,
        # Axis -> [lo, hi]. Only these are written, so events on different axes compose.
        "axis_ranges": axis_ranges,
        "operation": operation,
        "distribution": _dr_name_of(_dr_arg(func, params, "distribution"), "uniform"),
        "shared_random": bool(_dr_arg(func, params, "shared_random")),
        # Which base the browser reads: `add`/`scale` take the compiled default, so
        # they never accumulate across events on one axis.
        "uses_defaults": operation in _DR_OPS_USING_DEFAULTS,
        "set_const": _dr_needs_recompute(func, field_name),
    }


# mjlab's `Operation.uses_defaults`: `abs` reads the live value, `add`/`scale` the default.
_DR_OPS_USING_DEFAULTS = frozenset({"add", "scale"})

# Fallback when a DR func lacks `requires_model_fields`: fields that invalidate constants.
_SET_CONST_FIELDS = frozenset(
    {"body_ipos", "body_mass", "body_inertia", "dof_armature"}
)


def _dr_needs_recompute(func: Any, field_name: str) -> bool:
    """Whether the browser owes an ``mj_setConst`` after writing this field.

    Read off mjlab's `requires_model_fields` decorator. Its partial recompute levels
    all map to the full `mj_setConst`, the only one MuJoCo's C API exposes.
    """
    recompute = getattr(func, "recompute", None)
    if recompute is not None:
        return int(recompute) > 0
    return field_name in _SET_CONST_FIELDS


# mjlab's DR helpers bake an entity type and axes into `_randomize_model_field` where
# nothing can introspect them. An author's own wrapper can set `_mjswan_dr_field`.
_DR_FIELD_BY_FUNC: dict[str, tuple[str, str, list[int]]] = {
    "geom_friction": ("geom_friction", "geom", [0]),
    "geom_rgba": ("geom_rgba", "geom", [0, 1, 2, 3]),
    "body_com_offset": ("body_ipos", "body", [0, 1, 2]),
    "body_ipos": ("body_ipos", "body", [0, 1, 2]),
    "body_mass": ("body_mass", "body", [0]),
}


# Trace failure is expected for these, for the stated reason. Everything else raises: a
# silently dropped reset randomization is invisible in the build output and the browser.
_EVENTS_WITH_NOTHING_TO_WRITE: dict[str, str] = {
    "randomize_terrain": (
        "it re-draws each env's sub-terrain origin, and the browser has one baked terrain "
        "with one origin"
    ),
    "encoder_bias": (
        "it writes `Entity.data.encoder_bias`, which the runtime applies from the policy "
        "config's `encoder_bias` rather than from an event graph"
    ),
    "reset_scene_to_default": (
        "it restores every entity's default root and joint state, which is what the "
        "runtime's own reset already does (`mj_resetData` to `qpos0`, or keyframe 0) "
        'before any `mode="reset"` event runs'
    ),
}


def _event_writes_nothing_reason(
    term_cfg: EventTermCfg, env: Any, params: dict[str, Any]
) -> str | None:
    """Why this term's trace legitimately captured no write, or ``None``."""
    func_name = getattr(term_cfg.func, "__name__", "")
    reason = _EVENTS_WITH_NOTHING_TO_WRITE.get(func_name)
    if reason is not None:
        return reason
    if not func_name.startswith("reset_root_state"):
        return None
    # A root write cannot move a fixed-base entity, in mjlab either — and its
    # manipulation tasks still configure `reset_base` on their arms, leaving
    # `asset_cfg` to the signature default, hence `_dr_arg` and not `params`.
    entity_name = getattr(_dr_arg(term_cfg.func, params, "asset_cfg"), "name", None)
    if entity_name is None:
        return None
    try:
        entity = env.scene[entity_name]
    except (KeyError, TypeError):
        return None
    if getattr(entity, "is_fixed_base", False):
        return f"entity {entity_name!r} is fixed-base, so a root write cannot move it"
    return None


def serialize_event(
    name: str, term_cfg: EventTermCfg, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Serialize one event term, or ``None`` if there is genuinely nothing to emit."""
    from .compile import trace_event_term
    from .compile.tracer import slots_json

    func = term_cfg.func
    if isinstance(func, EventBinding):
        _require_ts_src("Event", name, func)
        return term_cfg.to_dict()

    resolved = _resolved_params(term_cfg.params, env)
    try:
        export = trace_event_term(
            func,
            resolved,
            env,
            name=name,
            mode=term_cfg.mode,
        )
    except ValueError as exc:
        # An `mjModel` write captures nothing, so describe it and let the browser draw
        # from the seeded PRNG at load.
        descriptor = model_field_dr_descriptor(term_cfg, env, resolved)
        if descriptor is not None:
            return {"name": name, "mode": term_cfg.mode, **descriptor}
        nothing_to_write = _event_writes_nothing_reason(term_cfg, env, resolved)
        if nothing_to_write is not None:
            return {
                "name": name,
                "mode": term_cfg.mode,
                "native": True,
                "reason": nothing_to_write,
            }
        raise ValueError(
            f"Event term {name!r} could not be traced: {exc} Emitting it as a no-op "
            "would drop a randomization the task is configured to apply, with nothing "
            "said about it in the browser. Either supply a trace-friendly replacement "
            "via mjswan.register_event(), or write the term as a TS class and point an "
            "EventBinding's `ts_src` at it."
        ) from exc

    ref = _onnx_ref("event", name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    entry: dict[str, Any] = {
        "name": name,
        "mode": term_cfg.mode,
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
    }
    if term_cfg.mode == "interval":
        entry["interval_range_s"] = (
            list(term_cfg.interval_range_s) if term_cfg.interval_range_s else None
        )
        entry["is_global_time"] = term_cfg.is_global_time
    if term_cfg.mode == "reset" and term_cfg.min_step_count_between_reset:
        entry["min_step_count_between_reset"] = term_cfg.min_step_count_between_reset
    return entry


def serialize_events(
    events: dict[str, EventTermCfg] | None,
    env: Any,
    out_dir: Path,
    on_term: Callable[[str], None] | None = None,
) -> list[dict[str, Any]] | None:
    """Serialize a scene's events dict to the JSON list ``config.json`` carries.

    ``on_term`` names each term before it is traced, for the build's progress line.
    """
    if not events:
        return None
    result = []
    for name, term_cfg in events.items():
        if on_term is not None:
            on_term(name)
        entry = serialize_event(name, term_cfg, env, out_dir)
        if entry is not None:
            result.append(entry)
    return result or None


# --- Commands ---


def _serialize_reset_graph(
    name: str, cmd_cfg: CommandTermConfig, env: Any, out_dir: Path
) -> dict[str, Any] | None:
    """Trace a native command's reset-time graph, or ``None`` if it has none.

    Deliberately shaped as an event entry, so the browser runs it through the same
    ``OnnxEvent`` handler rather than growing a second graph evaluator.
    """
    pending = cmd_cfg.pending_reset_trace
    if pending is None:
        return None
    from .compile import trace_event_term
    from .compile.tracer import slots_json

    graph_name = f"{name}_reset"
    export = trace_event_term(
        pending.func,
        _resolved_params(pending.params, env),
        env,
        name=graph_name,
        mode="reset",
    )
    ref = _onnx_ref("command", graph_name)
    _write_onnx(out_dir, ref, export.onnx_bytes)
    return {
        "name": graph_name,
        "mode": "reset",
        "onnx": ref,
        "rand_dim": export.rand_dim,
        "rand_ranges": export.rand_ranges,
        "input_slots": slots_json(export),
        "write_targets": export.write_targets,
    }


def serialize_command(
    name: str, cmd_cfg: CommandTermConfig, env: Any, out_dir: Path
) -> dict[str, Any]:
    """Serialize one command term, resolving a pending ONNX trace if needed."""
    if cmd_cfg.pending_trace is None:
        reset_graph = _serialize_reset_graph(name, cmd_cfg, env, out_dir)
        if reset_graph is None:
            return cmd_cfg.to_dict()
        # `to_dict` refuses while a trace is pending; the graph is resolved now.
        resolved = replace(cmd_cfg, pending_reset_trace=None)
        return {**resolved.to_dict(), "reset_graph": reset_graph}

    from .compile import trace_command_term
    from .compile.serialize import write_command_artifact

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
    return write_command_artifact(
        export,
        out_dir,
        resampling_time_range=getattr(pending.mjlab_cfg, "resampling_time_range", None),
        debug_vis=debug_vis,
        ui=pending.ui or _record_command_gui(term, name),
        viz=pending.viz,
    )


def _record_command_gui(term: Any, name: str) -> dict[str, Any] | None:
    """The term's own viewer GUI as a UI descriptor, or ``None``.

    Called after ``trace_command_term``: ``create_gui`` leaves handles on the term that
    ``compute()`` reads. A descriptor is presentation, not behaviour, so a term this
    cannot record still builds.
    """
    from .adapters.gui_spy import record_gui

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
