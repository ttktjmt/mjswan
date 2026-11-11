"""What a term may read off ``env``, and how each read is named.

The browser's slot reader serves the ``EntityData`` fields in :data:`READER_FIELDS`
natively; any other property is traced through, so its raw reads become ``sim`` slots.
A read that is neither a slot nor a forwarded constant raises
:class:`UnsupportedEnvRead`, so nothing bakes a constant by reaching the real env.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import re
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

import torch

# Only constants are listed, so an unknown field defaults to dynamic: baking a field
# that varies is silent corruption, while threading a constant merely costs an input.
_STATIC_DATA_FIELDS: frozenset[str] = frozenset(
    {
        "default_joint_pos",
        "default_joint_vel",
        "default_root_state",
        "default_mass",
        "default_inertia",
        "joint_pos_limits",
        "soft_joint_pos_limits",
        "joint_vel_limits",
        "soft_joint_vel_limits",
        "joint_effort_limits",
        "soft_joint_effort_limits",
    }
)


def _is_dynamic_field(field_name: str) -> bool:
    """Whether an ``Entity.data`` field must be threaded as a graph input."""
    return field_name not in _STATIC_DATA_FIELDS


# A slot key identifies one tensor read off the env, as ``(namespace, name)``:
#   (entity_name, data_field)        -> env.scene[entity].data.<field>
#   (_SENSOR_NS, sensor_name)        -> env.scene[sensor].data (a whole BuiltinSensor)
#   (_SENSOR_NS, "sensor.field")     -> env.scene[sensor].data.<field> (structured)
#   (_COMMAND_NS, "cmd.attr")        -> env.command_manager.get_term(cmd).<attr>
#   (_SIM_NS, field)                 -> env.sim.data.<field> == entity.data.data.<field>
#                                       (raw; maybe narrowed to rows)
#   (_ENV_NS, buffer)                -> env.<buffer> (an env-level counter)
#   (_ACTION_NS, "action")           -> env.action_manager.action (the policy's last
#                                       action; a term's `raw_action` slices it)
#   (_ACTION_TERM_NS, "term.attr")   -> env.action_manager.get_term(term).<attr> (a traced
#                                       term's state, or any term's processed action)
#   (_MODEL_NS, field[".default"])   -> env.sim.model.<field> (or its compiled default),
#                                       the rows of the elements an event read
SlotKey = tuple[str, str]


# A tagged key identifies one value an event/command body reads off ``env``. Wider than
# a SlotKey because those bodies also read scene-level tensors and control-flow scalars:
#   ("data", entity, field)  -> entity.data.<field>  (tensor; dynamic or const)
#   ("scene", attr)          -> env.scene.<attr>     (constant, e.g. env_origins)
#   ("attr", entity, attr)   -> entity.<attr>        (scalar, e.g. is_fixed_base)
TaggedKey = tuple


_SENSOR_NS = "__sensor__"
_COMMAND_NS = "__command__"
_SIM_NS = "__sim__"
_ENV_NS = "__env__"
_ACTION_NS = "__action__"
_ACTION_TERM_NS = "__action_term__"
_MODEL_NS = "__model__"


#: Per traced action term of the MDP being built, its state fields: what an observation
#: may read off it. A native term's only readable internal is its processed action.
_TRACED_ACTION_STATE: contextvars.ContextVar[Mapping[str, frozenset[str]]] = (
    contextvars.ContextVar("traced_action_state", default={})
)


@contextlib.contextmanager
def traced_action_state(state: Mapping[str, Collection[str]]) -> Iterator[None]:
    """Let an observation traced inside read each traced action term's *state*."""
    token = _TRACED_ACTION_STATE.set({k: frozenset(v) for k, v in state.items()})
    try:
        yield
    finally:
        _TRACED_ACTION_STATE.reset(token)


def traced_action_fields(term: str) -> frozenset[str] | None:
    """*term*'s state fields if it is traced, else ``None``."""
    return _TRACED_ACTION_STATE.get().get(term)


#: Env-level buffers served as slots, with the dtype mjlab keeps each in. The browser
#: feeds float32 like every slot, so the graph casts back: exact below 2**24 steps.
_ENV_SLOT_DTYPES: dict[str, torch.dtype] = {"episode_length_buf": torch.long}

#: Structured sensor fields mjlab keeps in an integer dtype (a ``CameraSensor``'s), cast
#: back the same way: a pixel or an object id is exact in float32.
_SENSOR_FIELD_DTYPES: dict[str, torch.dtype] = {
    "rgb": torch.uint8,
    "segmentation": torch.int32,
}


#: Per narrowed sim slot: the rows the input carries, and the field's full row count.
SimRows = dict[SlotKey, tuple[list[int], int]]


#: Env attributes every proxy forwards from the real env: trace-time constants a term
#: may read for shapes or rates. Anything else is served by a proxy or raises, in
#: discovery as in replay: a read reaching the real env would bake a silent constant.
_FORWARDED_ENV_ATTRS = (
    "num_envs",
    "device",
    "physics_dt",
    "step_dt",
    "max_episode_length",
    "max_episode_length_s",
    "cfg",
)


#: What a value-returning term may read off ``env``, for the error that names them.
_TERM_ENV_READS = (
    "env.scene[...]",
    "env.command_manager",
    "env.sim.data.<field>",
    "env.episode_length_buf",
    "env.action_manager.action",
    "env.action_manager.get_term(...).raw_action",
)
#: What an event or command body may read off ``env`` (``env.sim`` an event's only).
_EVENT_ENV_READS = (
    "env.scene[...]",
    "env.scene.<attr>",
    "env.sim.model.<field>",
    "env.sim.get_default_field(...)",
)


class UnsupportedEnvRead(AttributeError):
    """A term read an ``env`` attribute the tracer neither serves nor forwards.

    An ``AttributeError``, so ``hasattr`` and ``getattr(..., default)`` probes keep
    working; its own class, so a caller can tell it from a proxy's missing attribute.
    """

    def __init__(self, attr: str, served: Sequence[str]):
        super().__init__(
            f"Term read `env.{attr}`, which the tracer neither serves nor forwards. "
            f"A term may read {', '.join(served)}, and the constants "
            f"{', '.join(_FORWARDED_ENV_ATTRS)}. Anything else would be read off the "
            "trace-time env once and baked into the graph as a constant.",
            name=attr,
        )


#: Refused reads the real object could have answered, while a pass is watched.
_REFUSED_READS: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "refused_reads", default=None
)


@contextlib.contextmanager
def watch_refused_reads() -> Iterator[list[str]]:
    """Collect each refused read the real object has. A body that went on without one,
    through a ``getattr`` default or ``hasattr``, took a path mjlab does not take."""
    refused: list[str] = []
    token = _REFUSED_READS.set(refused)
    try:
        yield refused
    finally:
        _REFUSED_READS.reset(token)


def refuse_read(
    real: Any, attr: str, served: Sequence[str], prefix: str = ""
) -> UnsupportedEnvRead:
    """The error refusing ``env.<prefix><attr>``, noted for :func:`watch_refused_reads`
    when *real* has *attr*."""
    refused = _REFUSED_READS.get()
    if refused is not None:
        try:
            # Static, so no property of the real object runs.
            inspect.getattr_static(real, attr)
        except AttributeError:
            pass
        else:
            refused.append(prefix + attr)
    return UnsupportedEnvRead(prefix + attr, served)


#: The episode horizon, which bakes into a graph comparing the episode counter.
_HORIZON_ATTRS = frozenset({"max_episode_length", "max_episode_length_s"})


def _forward_env_attr(real_env: Any, name: str, served: Sequence[str]) -> Any:
    """``real_env.<name>`` for a forwarded constant; raise for anything else."""
    if name not in _FORWARDED_ENV_ATTRS:
        raise refuse_read(real_env, name, served)
    value = getattr(real_env, name)
    if name in _HORIZON_ATTRS and not value > 0:
        raise ValueError(
            f"Term read env.{name} = {value!r}: the trace env has no episode length, so "
            "the graph would end every episode at its first step. Give the trace env "
            "the task's episode_length_s, e.g. build_single_entity_trace_env(..., "
            "control_dt=..., episode_length_s=...)."
        )
    return value


#: ``EntityData`` fields the browser's slot reader serves natively
#: (``core/onnx/slotReader/fields/``): every ``EntityData`` property but
#: ``joint_torques``, which raises in mjlab too, plus the ``gravity_vec_w`` constant.
#: Kept in step with ``FIELD_READERS`` by hand; ``tests/dump_slot_fixture.py`` dumps
#: exactly this set and the browser's parity test refuses a dumped field it cannot read.
READER_FIELDS: frozenset[str] = frozenset(
    {
        "root_link_pose_w",
        "root_link_vel_w",
        "root_com_pose_w",
        "root_com_vel_w",
        "root_link_pos_w",
        "root_link_quat_w",
        "root_link_lin_vel_w",
        "root_link_ang_vel_w",
        "root_com_pos_w",
        "root_com_quat_w",
        "root_com_lin_vel_w",
        "root_com_ang_vel_w",
        "root_link_lin_vel_b",
        "root_link_ang_vel_b",
        "root_com_lin_vel_b",
        "root_com_ang_vel_b",
        "body_link_pose_w",
        "body_link_vel_w",
        "body_com_pose_w",
        "body_com_vel_w",
        "body_external_wrench",
        "body_link_pos_w",
        "body_link_quat_w",
        "body_link_lin_vel_w",
        "body_link_ang_vel_w",
        "body_com_pos_w",
        "body_com_quat_w",
        "body_com_lin_vel_w",
        "body_com_ang_vel_w",
        "body_external_force",
        "body_external_torque",
        "geom_pose_w",
        "geom_vel_w",
        "geom_pos_w",
        "geom_quat_w",
        "geom_lin_vel_w",
        "geom_ang_vel_w",
        "site_pose_w",
        "site_vel_w",
        "site_pos_w",
        "site_quat_w",
        "site_lin_vel_w",
        "site_ang_vel_w",
        "joint_pos",
        "joint_pos_biased",
        "joint_vel",
        "joint_acc",
        "actuator_force",
        "qfrc_actuator",
        "qfrc_external",
        "tendon_len",
        "tendon_vel",
        "gravity_vec_w",
        "projected_gravity_b",
        "heading_w",
    }
)


#: Slot namespaces that are not an entity's ``data``.
_NON_ENTITY_NS = frozenset(
    {_SENSOR_NS, _COMMAND_NS, _SIM_NS, _ENV_NS, _ACTION_NS, _ACTION_TERM_NS, _MODEL_NS}
)


def unserved_fields(keys: Iterable[SlotKey]) -> list[str]:
    """``entity.field`` for each entity-data slot the browser's reader cannot serve."""
    return sorted(
        {
            f"{namespace}.{name}"
            for namespace, name in keys
            if namespace not in _NON_ENTITY_NS and name not in READER_FIELDS
        }
    )


def _reader_fields(fields: Collection[str] | None) -> frozenset[str]:
    """``None`` means the browser's readers; an explicit set overrides them (tests
    pass an empty one to force every property through the traced path)."""
    return READER_FIELDS if fields is None else frozenset(fields)


def _sim_tensor(value: Any) -> Any:
    """A raw sim field as the tensor it wraps; anything else passes through."""
    if isinstance(value, torch.Tensor):
        return value
    # `TorchArray.__getattr__` delegates to its tensor, so `detach()` is that tensor.
    detach = getattr(value, "detach", None)
    return detach() if callable(detach) else value


def _slot_input_name(key: SlotKey) -> str:
    """The ONNX graph input name for a slot.

    Build-time only: the runtime takes it from the slot's ``input`` field
    (:func:`slot_to_json`) and never recomputes it.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return "sensor__" + re.sub(r"\W", "_", name_part)
    if namespace == _COMMAND_NS:
        return "command__" + re.sub(r"\W", "_", name_part)
    if namespace == _SIM_NS:
        return f"sim__{name_part}"
    if namespace == _ENV_NS:
        return f"env__{name_part}"
    if namespace == _ACTION_NS:
        return f"action__{name_part}"
    if namespace == _ACTION_TERM_NS:
        return "action_term__" + re.sub(r"\W", "_", name_part)
    if namespace == _MODEL_NS:
        return "model__" + re.sub(r"\W", "_", name_part)
    return f"{namespace}__{name_part}"


def slot_label(key: SlotKey) -> str:
    """Human-readable slot name for diagnostics (parity reports, logs)."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return f"sensor:{name_part}"
    if namespace == _COMMAND_NS:
        return f"command:{name_part}"
    if namespace == _SIM_NS:
        return f"sim:{name_part}"
    if namespace == _ENV_NS:
        return f"env:{name_part}"
    if namespace == _ACTION_NS:
        return f"action:{name_part}"
    if namespace == _ACTION_TERM_NS:
        return f"action_term:{name_part}"
    if namespace == _MODEL_NS:
        return f"model:{name_part}"
    return f"{namespace}.{name_part}"


def slot_to_json(
    key: SlotKey,
    shape: Sequence[int] | None = None,
    rows: Sequence[int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one input slot for the manifest's MDP entry.

    Eight shapes, told apart by which keys are present: ``{"entity", "field"}``,
    ``{"sensor"}``, ``{"command", "field"}``, ``{"sim"}`` (a raw ``mjData`` field,
    whole, or the ``rows`` of its element axis the graph takes, in order), ``{"env"}``
    (an env-level counter the runtime keeps), ``{"action"}`` (the policy's last
    action), ``{"action_term", "field"}`` (an action term's state), or
    ``{"model", "element", "names"}`` (an ``mjModel`` field's rows for the named
    elements, given as *extra*). All carry ``input`` (the graph input name) and
    ``shape``: the runtime feeds a flat array and cannot recover the rank without it.
    """
    namespace, name_part = key
    entry: dict[str, Any]
    if namespace == _SENSOR_NS:
        sensor_name, dot, sensor_field = name_part.partition(".")
        entry = {"sensor": sensor_name, "input": _slot_input_name(key)}
        if dot:
            # A structured sensor contributes one slot per field the term reads,
            # rather than one window of `sensordata`.
            entry["field"] = sensor_field
    elif namespace == _COMMAND_NS:
        command_name, _, attr = name_part.partition(".")
        entry = {
            "command": command_name,
            "field": attr,
            "input": _slot_input_name(key),
        }
    elif namespace == _SIM_NS:
        entry = {"sim": name_part, "input": _slot_input_name(key)}
        if rows is not None:
            entry["rows"] = [int(r) for r in rows]
    elif namespace == _ENV_NS:
        entry = {"env": name_part, "input": _slot_input_name(key)}
    elif namespace == _ACTION_NS:
        entry = {"action": name_part, "input": _slot_input_name(key)}
    elif namespace == _ACTION_TERM_NS:
        term, _, attr = name_part.partition(".")
        entry = {"action_term": term, "field": attr, "input": _slot_input_name(key)}
    elif namespace == _MODEL_NS:
        entry = {**(extra or {}), "input": _slot_input_name(key)}
    else:
        entry = {
            "entity": namespace,
            "field": name_part,
            "input": _slot_input_name(key),
        }
    if shape is not None:
        entry["shape"] = [int(d) for d in shape]
    return entry


def slots_json(export: Any) -> list[dict[str, Any]]:
    """Serialize the input slots the exported graph actually takes, shapes included.

    Shared by every export kind. Slots the exporter folded into a constant (an
    index tensor baked into the Gather it feeds) are dropped: ORT rejects a feed that
    is not a graph input.
    """
    shapes = getattr(export, "input_shapes", None) or []
    rows = getattr(export, "input_rows", None) or []
    extras = getattr(export, "slot_extras", None) or {}
    entries = [
        slot_to_json(
            key,
            shapes[i] if i < len(shapes) else None,
            rows[i] if i < len(rows) else None,
            extras.get(key),
        )
        for i, key in enumerate(export.input_slots)
    ]
    graph_inputs = _graph_input_names(getattr(export, "onnx_bytes", None))
    if graph_inputs is None:
        return entries
    return [entry for entry in entries if entry["input"] in graph_inputs]


def _graph_input_names(onnx_bytes: bytes | None) -> set[str] | None:
    """Input names of an exported graph, or None when there is no graph to ask.

    Unparseable bytes answer None rather than raising, so a hand-built export in a
    test degrades to no filtering instead of failing.
    """
    if not onnx_bytes:
        return None
    import onnx

    try:
        model = onnx.load_from_string(onnx_bytes)
    except Exception:
        return None
    return {i.name for i in model.graph.input}


def read_slot(
    env: Any, key: SlotKey, rows: Sequence[int] | None = None
) -> torch.Tensor:
    """Read a slot's current value off ``env``; a sim slot narrowed to ``rows``."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        sensor_name, dot, sensor_field = name_part.partition(".")
        data = env.scene[sensor_name].data
        return getattr(data, sensor_field) if dot else data
    if namespace == _COMMAND_NS:
        command_name, _, attr = name_part.partition(".")
        return getattr(env.command_manager.get_term(command_name), attr)
    if namespace == _SIM_NS:
        value = _sim_tensor(getattr(env.sim.data, name_part))
        return value if rows is None else value[:, list(rows)]
    if namespace == _ENV_NS:
        return getattr(env, name_part)
    if namespace == _ACTION_NS:
        return getattr(env.action_manager, name_part)
    if namespace == _ACTION_TERM_NS:
        term, _, attr = name_part.partition(".")
        return getattr(env.action_manager.get_term(term), attr)
    if namespace == _MODEL_NS:
        from .model import read_model_slot

        return read_model_slot(env, key, list(rows or ()))
    return getattr(env.scene[namespace].data, name_part)
