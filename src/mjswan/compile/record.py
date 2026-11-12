"""Discovery pass: run a term once against stand-ins that log what it reads.

A value-returning term reads through :class:`_RecordingEnv`, which logs each read
under a slot key and, for a raw sim field, the rows the read touches, so the graph
input can be narrowed to them. An event or command body reads through
:class:`_EventCaptureEnv`, which logs under tagged keys and captures the
``write_*_to_sim`` calls whose tensors become the graph outputs.
"""

from __future__ import annotations

from typing import Any, Callable, Collection, Sequence, cast

import torch
from torch.utils._pytree import tree_leaves, tree_map

from .gate import GatedIds
from .model import ModelRecorder, ModelSim
from .proxy import (
    _SHAPE_QUERIES,
    _action_term_proxy,
    _command_proxy,
    _FieldProxy,
    _forward_action_attr,
    _is_sensor,
    _plain,
    _sensor_proxy,
    _SimStandIn,
    _traces_through,
    _with_sim_data,
    action_term_window,
    entity_static,
)
from .slot import (
    _ACTION_NS,
    _ACTION_TERM_NS,
    _COMMAND_NS,
    _ENV_NS,
    _ENV_SLOT_DTYPES,
    _EVENT_ENV_READS,
    _SENSOR_NS,
    _SIM_NS,
    _TERM_ENV_READS,
    READER_FIELDS,
    SimRows,
    SlotKey,
    _forward_env_attr,
    _sim_tensor,
    traced_action_fields,
)


def _index_rows(index: Any, size: int | None) -> list[int] | None:
    """The rows of axis 1 an ``x[:, sel, ...]`` read touches, or ``None`` when that
    cannot be told statically: a batch-axis index, a boolean mask, a field with no
    element axis, an index computed from data."""
    parts = cast("tuple[Any, ...]", index) if isinstance(index, tuple) else ()
    if size is None or len(parts) < 2:
        return None
    lead, sel = parts[0], parts[1]
    if not (isinstance(lead, slice) and lead == slice(None)):
        return None
    if isinstance(sel, bool):
        return None
    if isinstance(sel, int):
        return [sel % size]
    if isinstance(sel, slice):
        return list(range(*sel.indices(size)))
    if isinstance(sel, torch.Tensor):
        if sel.dtype == torch.bool or sel.dim() > 1:
            return None
        return [int(v) % size for v in sel.reshape(-1).tolist()]
    if isinstance(sel, (list, tuple)) and all(
        isinstance(v, int) and not isinstance(v, bool) for v in sel
    ):
        return [v % size for v in sel]
    return None


class _RowRecord:
    """Which rows of one raw field a term's reads touch.

    ``rows`` turns ``None`` once any read cannot be narrowed (the whole field used as a
    value, or an index that cannot be told statically); the slot then ships whole.
    """

    def __init__(self, size: int | None):
        self.size = size
        self.rows: set[int] | None = set()

    def note(self, index: Any) -> None:
        if self.rows is None:
            return
        rows = _index_rows(index, self.size)
        if rows is None:
            self.rows = None
        else:
            self.rows.update(rows)

    def whole(self) -> None:
        self.rows = None


class _RecordingField(_FieldProxy):
    """A raw sim field during discovery, noting the rows each ``[:, sel]`` read touches.

    The read's result is a plain tensor, so only indexing *into the field* is noted; any
    other use of the field is the whole of it.
    """

    _record: _RowRecord

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__ and isinstance(args[0], cls):
            args[0]._record.note(args[1])
        elif getattr(func, "__name__", None) not in _SHAPE_QUERIES:
            for leaf in tree_leaves((args, kwargs)):
                if isinstance(leaf, cls):
                    leaf._record.whole()
        return func(*tree_map(_plain, args), **tree_map(_plain, kwargs))


class _RecordingSimData:
    """Wraps the raw ``SimData`` (``env.sim.data``, the object every
    ``Entity.data.data`` is), logging each field read and the rows the reads touch.

    One sim object is shared by every entity, hence its own namespace rather than the
    entity's. Fields are warp-backed ``TorchArray`` proxies; the tensor view is what
    gets logged, and the term gets it as a :class:`_RecordingField`.
    """

    def __init__(self, env: Any, log: list[tuple[SlotKey, Any]]):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_records", {})

    def __getattr__(self, name: str) -> Any:
        value = _sim_tensor(getattr(self._env.sim.data, name))
        if not isinstance(value, torch.Tensor):
            return value
        self._log.append(((_SIM_NS, name), value))
        record = self._records.get(name)
        if record is None:
            # Only an element axis can be narrowed; `time` is `(nworld,)`.
            record = _RowRecord(int(value.shape[1]) if value.dim() > 1 else None)
            self._records[name] = record
        field = value.as_subclass(_RecordingField)
        field._record = record
        return field


def _merge_narrowing(sims: Sequence[_RecordingSimData]) -> SimRows:
    """The rows each sim field can be narrowed to, across every term that read it.

    Terms index one field by different sets (``cvel`` by the root body in one, by every
    site's body in another), so the union is taken and the input carries it once. A
    field any term used whole, or indexed in a way that cannot be told statically, ships
    whole.
    """
    merged: dict[str, tuple[set[int] | None, int | None]] = {}
    for sim in sims:
        for name, record in sim._records.items():  # noqa: SLF001 (internal proxy)
            rows, size = merged.get(name, (set(), record.size))
            if rows is None or not record.rows:
                merged[name] = (None, size)
            else:
                merged[name] = (rows | record.rows, size)
    return {
        (_SIM_NS, name): (sorted(rows), size)
        for name, (rows, size) in merged.items()
        if rows and size is not None
    }


class _RecordingData:
    """Wraps a real ``Entity.data`` for discovery.

    A reader-served field or plain attribute is logged as one slot, value and all; any
    other property runs against a copy whose ``data`` is the recording sim proxy, so the
    raw fields it reads are what get logged.
    """

    def __init__(
        self,
        real: Any,
        entity: str,
        log: list[tuple[SlotKey, Any]],
        reader_fields: Collection[str],
        sim: _RecordingSimData,
    ):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_reader_fields", reader_fields)
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_traced", None)

    def _through(self) -> Any:
        """The copy a traced-through property runs against. Built lazily: a stand-in
        ``Entity.data`` with no sim behind it never needs one."""
        if self._traced is None:
            object.__setattr__(self, "_traced", _with_sim_data(self._real, self._sim))
        return self._traced

    def __getattr__(self, name: str) -> Any:
        if name == "data":
            return self._sim
        if _traces_through(self._real, name, self._reader_fields):
            return getattr(self._through(), name)
        value = getattr(self._real, name)
        self._log.append(((self._entity, name), value))
        return value


class _RecordingEntity:
    def __init__(
        self,
        real: Any,
        name: str,
        log: list[tuple[SlotKey, Any]],
        reader_fields: Collection[str],
        sim: _RecordingSimData,
    ):
        self._real = real
        self.data = _RecordingData(real.data, name, log, reader_fields, sim)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingScene:
    def __init__(
        self,
        real: Any,
        log: list[tuple[SlotKey, Any]],
        sensors: dict[str, Any],
        reader_fields: Collection[str],
        sim: _RecordingSimData,
    ):
        self._real = real
        self._log = log
        self._sensors = sensors
        self._reader_fields = reader_fields
        self._sim = sim

    @property
    def env_origins(self) -> torch.Tensor:
        return self._real.env_origins

    def __getitem__(self, name: str) -> Any:
        real = self._real[name]
        if _is_sensor(self._real, name):
            # Keep the real sensor so the replay pass can subclass its class.
            self._sensors[name] = real
            return _sensor_proxy(real, lambda: self._read_sensor(name, real))
        return _RecordingEntity(real, name, self._log, self._reader_fields, self._sim)

    def _read_sensor(self, name: str, real: Any) -> Any:
        value = real.data
        if isinstance(value, torch.Tensor):
            # A builtin sensor is one `sensordata` window, one slot.
            self._log.append(((_SENSOR_NS, name), value))
            return value
        # A structured sensor has no single tensor; each field read becomes a slot.
        return _RecordingSensorData(value, name, self._log)


class _RecordingSensorData:
    """Wraps a structured sensor's ``.data``, logging each tensor field read."""

    def __init__(self, real: Any, sensor: str, log: list[tuple[SlotKey, Any]]):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        if isinstance(value, torch.Tensor):
            self._log.append(((_SENSOR_NS, f"{self._sensor}.{name}"), value))
        return value


def _command_or_raise(fetch: Callable[[str], Any], name: str) -> Any:
    """``fetch(name)``, raising for a command the trace env has no term for, which mjlab
    answers with ``None`` or a bare ``KeyError``."""
    try:
        value = fetch(name)
    except KeyError as exc:
        value, cause = None, exc
    else:
        cause = None
    if value is None:
        raise ValueError(
            f"Term read command {name!r}, which the trace env has no term for. The build "
            "stands one in for a command whose width it knows (UI inputs, or a traced "
            "command); for any other, give the trace env one: "
            "`build_single_entity_trace_env(commands=...)`."
        ) from cause
    return value


class _RecordingCommandManager:
    """Wraps the real ``CommandManager``, logging command-state tensor reads."""

    def __init__(
        self,
        real: Any,
        log: list[tuple[SlotKey, Any]],
        commands: dict[str, Any],
    ):
        self._real = real
        self._log = log
        self._commands = commands

    def get_term(self, name: str) -> Any:
        real = _command_or_raise(self._real.get_term, name)
        # Keep the real term so the replay pass can subclass its class.
        self._commands[name] = real

        def on_tensor(attr: str, value: Any) -> Any:
            self._log.append(((_COMMAND_NS, f"{name}.{attr}"), value))
            return value

        return _command_proxy(real, on_tensor)

    def get_command(self, name: str) -> Any:
        value = _command_or_raise(self._real.get_command, name)
        self._log.append(((_COMMAND_NS, f"{name}.command"), value))
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingActionManager:
    """Wraps the real ``ActionManager``, logging reads of the policy's last action.

    A term's ``raw_action`` is its slice of that vector, so it logs the whole vector:
    one slot the runtime serves, sliced inside the graph.
    """

    def __init__(self, real: Any, log: list[tuple[SlotKey, Any]]):
        self._real = real
        self._log = log

    @property
    def action(self) -> torch.Tensor:
        return self._logged_action()

    def get_term(self, name: str) -> Any:
        action_term_window(self._real, name)
        real = self._real.get_term(name)

        def raw_action() -> torch.Tensor:
            self._logged_action()
            return real.raw_action

        def read_state(attr: str, value: torch.Tensor) -> torch.Tensor:
            # Logged as the float32 the browser feeds; replay casts back to its dtype.
            key = (_ACTION_TERM_NS, f"{name}.{attr}")
            self._log.append((key, value.to(torch.float32)))
            return value

        return _action_term_proxy(
            real, raw_action, read_state, traced_action_fields(name)
        )

    def _logged_action(self) -> torch.Tensor:
        value = self._real.action
        if value.shape[-1] == 0:
            raise ValueError(
                "Term read the policy's last action, but the trace env has no action "
                "terms to give it a width. For a plain scene the build takes it from "
                "the policy's `policy_num_actions` or `policy_joint_names`; set one. "
                "Outside the Builder, trace within "
                "`mjswan.mjlab.env.policy_actions(env, num_actions)`."
            )
        self._log.append(((_ACTION_NS, "action"), value))
        return value

    def __getattr__(self, name: str) -> Any:
        return _forward_action_attr(self._real, name)


class _RecordingEnv:
    """Proxy env recording the reads a term makes (entity data, sim data, sensors,
    commands, the episode counter, the last action). Same contract as
    :class:`_ReplayEnv`: any other read raises."""

    def __init__(self, real: Any, reader_fields: Collection[str] = READER_FIELDS):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "_sensors", {})
        object.__setattr__(self, "_commands", {})
        object.__setattr__(self, "_sim", _RecordingSimData(real, self._log))
        object.__setattr__(
            self, "sim", _SimStandIn(self._sim, getattr(real, "sim", None))
        )
        object.__setattr__(
            self,
            "scene",
            _RecordingScene(
                real.scene, self._log, self._sensors, reader_fields, self._sim
            ),
        )

    def __getattr__(self, name: str) -> Any:
        if name == "unwrapped":  # mjlab's is the env itself
            return self
        if name == "command_manager":
            return _RecordingCommandManager(
                self._real.command_manager, self._log, self._commands
            )
        if name == "action_manager":
            return _RecordingActionManager(self._real.action_manager, self._log)
        if name in _ENV_SLOT_DTYPES:
            value = getattr(self._real, name)
            # Logged as the float32 the browser feeds; the term still sees mjlab's dtype.
            self._log.append(((_ENV_NS, name), value.to(torch.float32)))
            return value
        return _forward_env_attr(self._real, name, _TERM_ENV_READS)


# Each write call and the tensors it writes, in argument order. A write through a
# draw-selected env set (:mod:`.gate`) adds a trailing ``gate`` output.
_WRITE_FIELDS: dict[str, tuple[str, ...]] = {
    "joint_state": ("position", "velocity"),
    "root_pose": ("pose",),
    "root_velocity": ("velocity",),
    "root_velocity_b": ("velocity",),
}


#: Keyed by entity as well as kind: one term may write several entities (mjlab's
#: `reset_scene_to_default` writes them all), each needing its own target and outputs.
WriteKey = tuple[str | None, str]
WriteCaptures = dict[WriteKey, tuple[torch.Tensor, ...]]


def _write_output_name(key: WriteKey, field_name: str) -> str:
    """Graph-output name for one written tensor; unprefixed when no entity was named."""
    entity, kind = key
    return f"{entity}__{kind}__{field_name}" if entity else f"{kind}__{field_name}"


def _write_fields(key: WriteKey, values: tuple[Any, ...]) -> tuple[str, ...]:
    """The fields a captured write outputs: its kind's, then ``gate`` if it has one."""
    fields = _WRITE_FIELDS[key[1]]
    return (*fields, "gate") if len(values) > len(fields) else fields


def entity_write_target(
    key: WriteKey, values: tuple[Any, ...], entity: str | None
) -> dict[str, Any]:
    """The manifest's description of one captured write."""
    target: dict[str, Any] = {
        "kind": key[1],
        "entity": key[0] or entity,
        "fields": list(_WRITE_FIELDS[key[1]]),
        "outputs": [_write_output_name(key, f) for f in _WRITE_FIELDS[key[1]]],
    }
    if len(values) > len(_WRITE_FIELDS[key[1]]):
        target["gate"] = _write_output_name(key, "gate")
    return target


class _WriteCaptureMixin:
    """Records ``write_*_to_sim`` calls into ``self._captures``.

    The entity comes from the write itself, not the params: a term names its target
    however it likes (``asset_cfg``, or a plain ``ball_name``).
    """

    #: Entity this proxy stands for; None when the cfg names none.
    _name: str | None
    _captures: WriteCaptures

    def _capture(self, kind: str, values: tuple[Any, ...], env_ids: Any) -> None:
        if isinstance(env_ids, GatedIds):
            values = (*values, env_ids.gate.to(torch.float32))
        self._captures[(self._name, kind)] = values

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        self._capture("joint_state", (position, velocity), env_ids)

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self._capture("root_pose", (pose,), env_ids)

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self._capture("root_velocity", (velocity,), env_ids)

    def write_root_link_velocity_b_to_sim(self, velocity_b, env_ids=None):
        self._capture("root_velocity_b", (velocity_b,), env_ids)

    def write_root_state_to_sim(self, root_state, env_ids=None):
        # mjlab's own split of a 13-wide root state into the two writes above.
        self._capture("root_pose", (root_state[..., :7],), env_ids)
        self._capture("root_velocity", (root_state[..., 7:],), env_ids)


def _flatten_captures(
    captures: WriteCaptures,
) -> tuple[list[str], list[torch.Tensor]]:
    """Flatten a captures dict into (output_names, tensors).

    Insertion order is the term's own write-call order, so discovery and the traced
    module agree on output ordering.
    """
    names: list[str] = []
    tensors: list[torch.Tensor] = []
    for key, values in captures.items():
        for field_name, tensor in zip(_write_fields(key, values), values):
            names.append(_write_output_name(key, field_name))
            tensors.append(tensor)
    return names, tensors


class _EvRecData:
    """Records ``entity.data.<field>`` reads as ``("data", entity, field)``."""

    def __init__(self, real, entity, log):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        self._log.append((("data", self._entity, name), value))
        return value


class _EvRecEntity(_WriteCaptureMixin):
    """Records data-field and (non-``data``) attribute reads; captures writes."""

    def __init__(self, real, name, log, captures):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "data", _EvRecData(real.data, name, log))
        object.__setattr__(self, "_captures", captures)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("write_") and name.endswith("_to_sim"):
            # Forwarding it would mutate the live env mid-trace and capture nothing.
            raise ValueError(
                f"Term called {name}() on entity {self._name!r}, which the tracer does "
                "not capture. Add it to `_WriteCaptureMixin` and `_WRITE_FIELDS` with a "
                "runtime counterpart, or hand the term to the browser as a TS class."
            )
        value = getattr(self._real, name)
        # Only tensors and control-flow scalars can be reproduced during replay.
        if isinstance(value, (torch.Tensor, bool, int, float)):
            self._log.append((("attr", self._name, name), value))
            return value
        return entity_static(name, value)


class _EvRecScene:
    """Records scene-level attribute reads (e.g. ``env_origins``); indexes entities."""

    def __init__(self, real, log, captures):
        self._real = real
        self._log = log
        self._captures = captures

    def __getitem__(self, name: str) -> _EvRecEntity:
        if _is_sensor(self._real, name):
            # Letting it through surfaces as a bare assert deep inside the term.
            raise ValueError(
                f"Event/command term read sensor {name!r}; sensor slots are only "
                "supported for observation/termination terms so far. Extend the "
                "tagged-key proxies (_EvRecScene/_EvReplayScene) the same way "
                "_RecordingScene does, or handle this term natively."
            )
        return _EvRecEntity(self._real[name], name, self._log, self._captures)

    def __getattr__(self, name: str) -> Any:
        if name == "entities":
            # Stand-ins, not the live entities: a term iterating the scene would
            # otherwise write into the tracing env, moving the sim under later terms.
            return {
                key: _EvRecEntity(real, key, self._log, self._captures)
                for key, real in self._real.entities.items()
            }
        value = getattr(self._real, name)
        if isinstance(value, (torch.Tensor, bool, int, float)):
            self._log.append((("scene", name), value))
        return value


class _EventCaptureEnv:
    """Proxy env for event and command tracing: records reads, captures writes,
    never mutates the sim. Same contract as :class:`_EventReplayEnv`: any other read
    raises.

    *commands* serves ``env.command_manager``, which only a command body may read, and
    *model* ``env.sim.model``, which only an event body may: its reads and writes go to
    :attr:`model`, a :class:`ModelRecorder` made on first use.
    """

    def __init__(
        self, real, log, captures, *, commands: bool = False, model: bool = False
    ):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "scene", _EvRecScene(real.scene, log, captures))
        object.__setattr__(self, "_serves_model", model)
        object.__setattr__(self, "model", None)
        if commands:
            object.__setattr__(
                self,
                "command_manager",
                _RecordingCommandManager(real.command_manager, log, {}),
            )

    def __getattr__(self, name: str) -> Any:
        if name == "unwrapped":  # mjlab's is the env itself
            return self
        if name == "sim" and self._serves_model:
            recorder = ModelRecorder(self._real.sim)
            object.__setattr__(self, "model", recorder)
            object.__setattr__(self, "sim", ModelSim(recorder, self._real.sim))
            return self.sim
        return _forward_env_attr(self._real, name, _EVENT_ENV_READS)
