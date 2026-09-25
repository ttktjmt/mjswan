"""Event terms: traced to a graph, described as a startup model-field randomization,
or marked native when there is legitimately nothing to write."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco

from ...envs.mdp.events import EventBinding
from . import graph
from .binding import require_ts_src
from .provenance import graph_meta, resolved_params, term_provenance

if TYPE_CHECKING:
    from ...managers.event_manager import EventTermCfg


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

    Read off the signature, since mjlab's wrappers differ in defaults (``operation`` is
    ``"abs"`` for ``geom_friction``, ``"scale"`` for ``body_mass``).
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
    """As mjlab's ``_determine_target_axes``: explicit, int keys, then the default."""
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
    write. They need no graph either (draw a number per element per axis, combine it
    with the base value, write it back), so the browser does it at startup from the
    seeded PRNG and a session still replays.

    Returns ``None`` for anything undescribable: an unknown entity type, or the
    string-keyed ``ranges`` form mjlab resolves by name pattern.
    """
    func = term_cfg.func
    # Resolved: an unresolved `SceneEntityCfg` widens a scoped event to every geom.
    params = params if params is not None else resolved_params(term_cfg.params, env)
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
    if field_name == "geom_size":
        _require_primitive_geoms(env, names)
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
        # mjlab recomputes the broadphase bounds from the size in the same call.
        "recompute_bounds": field_name == "geom_size",
    }


#: The geom types `dr.geom_size._recompute_geom_bounds` supports.
_PRIMITIVE_GEOM_TYPES = frozenset(
    {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
    }
)


def _require_primitive_geoms(env: Any, names: list[str]) -> None:
    """Refuse a size randomization on a geom whose bounds cannot be recomputed."""
    model = env.sim.mj_model
    unsupported = {
        name: mujoco.mjtGeom(int(model.geom(name).type)).name
        for name in names
        if int(model.geom(name).type) not in _PRIMITIVE_GEOM_TYPES
    }
    if unsupported:
        raise ValueError(
            "dr.geom_size only supports primitive geom types (sphere, capsule, "
            f"ellipsoid, cylinder, box); these are not: {unsupported}. mjlab raises "
            "the same, because the broadphase bounds it recomputes from the size are "
            "only defined for those."
        )


# mjlab's `Operation.uses_defaults`: `abs` reads the live value, the rest the default.
_DR_OPS_USING_DEFAULTS = frozenset({"add", "scale"})


# For a DR func without `requires_model_fields`: the fields that invalidate constants.
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
    "geom_size": ("geom_size", "geom", [0, 1, 2]),
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
    # A root write cannot move a fixed-base entity, in mjlab either, and its
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
    from ...compile.slot import UnsupportedEnvRead, slots_json

    func = term_cfg.func
    if isinstance(func, EventBinding):
        require_ts_src("Event", name, func)
        return term_cfg.to_dict()

    resolved = resolved_params(term_cfg.params, env)
    provenance = term_provenance(func, resolved)
    # Described, not traced: the body writes `env.sim.model`, perturbing the live model
    # under every later trace. The browser draws from the seeded PRNG at load instead.
    descriptor = model_field_dr_descriptor(term_cfg, env, resolved)
    if descriptor is not None:
        return {"name": name, "mode": term_cfg.mode, **descriptor, **provenance}
    try:
        export = trace_event_term(
            func,
            resolved,
            env,
            name=name,
            mode=term_cfg.mode,
        )
    except (ValueError, UnsupportedEnvRead) as exc:
        nothing_to_write = _event_writes_nothing_reason(term_cfg, env, resolved)
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
