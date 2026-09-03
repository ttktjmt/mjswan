"""Trace real mjlab MDP term bodies to ONNX.

A term is ``func(env, **params)`` reading a few fields off ``env``. Each is run once
against a recording proxy to discover those reads, classified into time-varying state
(a graph input) or a model-derived constant (baked in), then exported as an
``nn.Module`` whose ``forward`` takes the dynamic tensors.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch import nn

from .rng import DrawRecorder, ReplayRng

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
        # Randomizable in mjlab, but the browser's action layer already holds one
        # build-time vector for it, so baking is what keeps the two agreeing.
        "encoder_bias",
    }
)


def _is_dynamic_field(field_name: str) -> bool:
    """Whether an ``Entity.data`` field must be threaded as a graph input."""
    return field_name not in _STATIC_DATA_FIELDS


# A slot key identifies one tensor read off the env, as ``(namespace, name)``:
#   (entity_name, data_field)        -> env.scene[entity].data.<field>
#   (_SENSOR_NS, sensor_name)        -> env.scene[sensor].data (a whole BuiltinSensor)
#   (_COMMAND_NS, "cmd.attr")        -> env.command_manager.get_term(cmd).<attr>
SlotKey = tuple[str, str]

# A tagged key identifies one value an event/command body reads off ``env``. Wider than
# a SlotKey because those bodies also read scene-level tensors and control-flow scalars:
#   ("data", entity, field)  -> entity.data.<field>   (tensor; dynamic or const)
#   ("scene", attr)          -> env.scene.<attr>      (scene-level constant, e.g. env_origins)
#   ("attr", entity, attr)   -> entity.<attr>         (control-flow scalar, e.g. is_fixed_base)
TaggedKey = tuple

_SENSOR_NS = "__sensor__"
_COMMAND_NS = "__command__"


def _class_proxy(real: Any, overrides: dict[str, Any]) -> Any:
    """A stand-in for a live mjlab object that still satisfies ``isinstance`` checks.

    Terms assert on concrete classes (``builtin_sensor`` on ``BuiltinSensor``, say),
    so subclass the real object's class and share its ``__dict__``, replacing only
    what ``overrides`` names.
    """
    cls = type(real)
    proxy_cls = type(f"_Proxy{cls.__name__}", (cls,), overrides)
    proxy = object.__new__(proxy_cls)
    proxy.__dict__ = real.__dict__
    return proxy


def _sensor_proxy(real: Any, get_data: Callable[[], Any]) -> Any:
    """A sensor stand-in whose ``.data`` comes from ``get_data``."""
    return _class_proxy(real, {"data": property(lambda _self: get_data())})


def _command_proxy(real: Any, on_tensor: Callable[[str, Any], Any]) -> Any:
    """A command-term stand-in routing every tensor attribute through ``on_tensor``.

    A command's state lives in plain instance attributes, so ``__getattr__`` never
    fires and ``__getattribute__`` is the only hook that sees the read.
    """

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            return on_tensor(attr, value)
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def _is_sensor(scene: Any, name: str) -> bool:
    """Whether ``scene[name]`` resolves to a sensor rather than an entity."""
    sensors = getattr(scene, "sensors", None)
    return bool(sensors) and name in sensors


# --- Recording proxy: discovers which env fields a term reads. ---


class _RecordingData:
    """Wraps a real ``Entity.data``, logging every field access."""

    def __init__(self, real: Any, entity: str, log: list[tuple[SlotKey, Any]]):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._real, name)
        self._log.append(((self._entity, name), value))
        return value


class _RecordingEntity:
    def __init__(self, real: Any, name: str, log: list[tuple[SlotKey, Any]]):
        self._real = real
        self.data = _RecordingData(real.data, name, log)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingScene:
    def __init__(
        self,
        real: Any,
        log: list[tuple[SlotKey, Any]],
        sensors: dict[str, Any],
    ):
        self._real = real
        self._log = log
        self._sensors = sensors

    def __getitem__(self, name: str) -> Any:
        real = self._real[name]
        if _is_sensor(self._real, name):
            # Keep the real sensor so the replay pass can subclass its class.
            self._sensors[name] = real
            return _sensor_proxy(real, lambda: self._read_sensor(name, real))
        return _RecordingEntity(real, name, self._log)

    def _read_sensor(self, name: str, real: Any) -> Any:
        value = real.data
        if isinstance(value, torch.Tensor):
            # A builtin sensor is one `sensordata` window — one slot.
            self._log.append(((_SENSOR_NS, name), value))
            return value
        # A structured sensor has no single tensor to be, so log the fields the term
        # touches and let each become its own slot.
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
        real = self._real.get_term(name)
        # Keep the real term so the replay pass can subclass its class.
        self._commands[name] = real

        def on_tensor(attr: str, value: Any) -> Any:
            self._log.append(((_COMMAND_NS, f"{name}.{attr}"), value))
            return value

        return _command_proxy(real, on_tensor)

    def get_command(self, name: str) -> Any:
        value = self._real.get_command(name)
        self._log.append(((_COMMAND_NS, f"{name}.command"), value))
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _RecordingEnv:
    """Proxy env recording the reads a term makes (entity data, sensors, commands)."""

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_log", [])
        object.__setattr__(self, "_sensors", {})
        object.__setattr__(self, "_commands", {})
        object.__setattr__(
            self, "scene", _RecordingScene(real.scene, self._log, self._sensors)
        )

    def __getattr__(self, name: str) -> Any:
        if name == "command_manager":
            return _RecordingCommandManager(
                self._real.command_manager, self._log, self._commands
            )
        return getattr(self._real, name)


# --- Replay proxy: serves recorded slots to the term during tracing. ---


class _ReplayData:
    def __init__(self, entity: str, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (self._entity, name)
        try:
            return self._slots[key]
        except KeyError:
            raise AttributeError(
                f"Term read undeclared slot {key!r} during tracing. This field "
                "was not seen in the discovery pass — the term's control flow is "
                "input-dependent, which is not traceable (ADR 0005 §Consequences)."
            ) from None


class _ReplayEntity:
    def __init__(self, entity: str, slots: dict[SlotKey, torch.Tensor]):
        self.data = _ReplayData(entity, slots)


class _ReplaySensorData:
    """Serves a structured sensor's recorded fields during the replay pass."""

    def __init__(self, sensor: str, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (_SENSOR_NS, f"{self._sensor}.{name}")
        if key not in self._slots:
            raise AttributeError(
                f"sensor field {self._sensor}.{name} was not recorded during discovery"
            )
        return self._slots[key]


class _ReplayScene:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
    ):
        self._slots = slots
        self._sensors = sensors or {}

    def __getitem__(self, name: str) -> Any:
        real = self._sensors.get(name)
        if real is not None:
            whole = (_SENSOR_NS, name)
            if whole in self._slots:
                return _sensor_proxy(real, lambda: self._slots[whole])
            # Structured sensor: the discovery pass recorded its fields separately.
            return _sensor_proxy(real, lambda: _ReplaySensorData(name, self._slots))
        return _ReplayEntity(name, self._slots)


class _ReplayCommandManager:
    """Serves recorded command-state slots back during tracing."""

    def __init__(self, slots: dict[SlotKey, torch.Tensor], commands: dict[str, Any]):
        self._slots = slots
        self._commands = commands

    def get_term(self, name: str) -> Any:
        real = self._commands.get(name)
        if real is None:
            raise AttributeError(
                f"Term read command {name!r} during tracing that the discovery pass "
                "never saw — the term's control flow is input-dependent, which is "
                "not traceable (ADR 0005 §Consequences)."
            )
        return _command_proxy(
            real, lambda attr, _v: self._slots[(_COMMAND_NS, f"{name}.{attr}")]
        )

    def get_command(self, name: str) -> torch.Tensor:
        return self._slots[(_COMMAND_NS, f"{name}.command")]


class _ReplayEnv:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
        commands: dict[str, Any] | None = None,
        *,
        real_env: Any,
    ):
        self.scene = _ReplayScene(slots, sensors)
        self.command_manager = _ReplayCommandManager(slots, commands or {})
        self._real_env = real_env

    def __getattr__(self, name: str) -> Any:
        # Forwarded, not copied, so nothing drifts from the real env. Anything else
        # raises rather than silently reading a stand-in.
        if name in ("num_envs", "device"):
            return getattr(self._real_env, name)
        raise AttributeError(name)


# --- Shared trace mechanics: constants as buffers, and the export call itself. ---


def _register_consts(
    module: nn.Module, constants: dict[Any, torch.Tensor], prefix: str = "_const"
) -> dict[Any, str]:
    """Register each constant as a buffer, returning ``slot key -> buffer name``.

    Buffers rather than plain attributes so ``torch.onnx.export`` folds them into the
    graph instead of tracing them as free-floating tensors.
    """
    names: dict[Any, str] = {}
    for i, (key, value) in enumerate(constants.items()):
        buffer_name = f"{prefix}_{i}"
        module.register_buffer(buffer_name, value.detach().clone())
        names[key] = buffer_name
    return names


def _const_values(module: nn.Module, names: dict[Any, str]) -> dict[Any, torch.Tensor]:
    """The registered constants, keyed by the slot each one serves."""
    return {key: getattr(module, name) for key, name in names.items()}


def _export_onnx(
    module: nn.Module,
    example: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    batch_axis: list[str],
    opset: int,
) -> bytes:
    """Export ``module`` to ONNX bytes with ``batch_axis`` names given a dynamic axis 0.

    ``dynamo=False``: the TorchScript tracer records the concrete tensor ops we want,
    while torch.export traces Python control flow and trips on the proxies.
    """
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={n: {0: "batch"} for n in batch_axis},
            opset_version=opset,
            dynamo=False,
        )
    return buffer.getvalue()


def _classify_slots(
    log: list[tuple[SlotKey, Any]],
    dynamic: dict[SlotKey, torch.Tensor],
    constants: dict[SlotKey, torch.Tensor],
) -> bool:
    """Split a recorded read log into graph inputs and baked constants.

    Sensor and command-state reads are live state by definition; an entity data field is
    dynamic unless it is a model-derived constant. Returns whether *this* log
    contributed a dynamic slot, which a group's caller needs per term.
    """
    saw_dynamic = False
    for key, value in log:
        if not isinstance(value, torch.Tensor):
            continue  # non-tensor attribute access, not a graph slot
        namespace, field_name = key
        if namespace in (_SENSOR_NS, _COMMAND_NS) or _is_dynamic_field(field_name):
            dynamic.setdefault(key, value)
            saw_dynamic = True
        else:
            constants.setdefault(key, value)
    return saw_dynamic


def _classify_tagged(
    log: list[tuple[TaggedKey, Any]],
) -> tuple[
    dict[SlotKey, torch.Tensor], dict[TaggedKey, torch.Tensor], dict[TaggedKey, Any]
]:
    """Split an event/command read log into dynamic inputs, tensor and scalar constants.

    Only a time-varying ``entity.data`` field becomes a graph input; scene tensors and
    control-flow scalars are baked.
    """
    dynamic: dict[SlotKey, torch.Tensor] = {}
    tensor_consts: dict[TaggedKey, torch.Tensor] = {}
    scalar_consts: dict[TaggedKey, Any] = {}
    for key, value in log:
        is_tensor = isinstance(value, torch.Tensor)
        if key[0] == "data" and _is_dynamic_field(key[2]) and is_tensor:
            dynamic.setdefault((key[1], key[2]), value)
        elif is_tensor:
            tensor_consts.setdefault(key, value)
        else:
            scalar_consts.setdefault(key, value)
    return dynamic, tensor_consts, scalar_consts


class _TermModule(nn.Module):
    """Wraps ``func(env, **params)`` so ``forward`` takes only dynamic tensors.

    Constant slots (defaults, static tensors) are registered as buffers and
    served back to the term; dynamic slots arrive as ``forward`` arguments in the
    order given by ``dynamic_keys``.
    """

    def __init__(
        self,
        func: Callable[..., torch.Tensor],
        params: dict[str, Any],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        *,
        sensors: dict[str, Any] | None = None,
        commands: dict[str, Any] | None = None,
        real_env: Any,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors or {}
        self._commands = commands or {}
        self._real_env = real_env
        self._const_buffers = _register_consts(self, constants)

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        slots.update(_const_values(self, self._const_buffers))
        env = _ReplayEnv(slots, self._sensors, self._commands, real_env=self._real_env)
        return self._func(env, **self._params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ConstantTerm(ValueError):
    """A term that read no simulation state at all — its value is a constant.

    Genuinely env-independent (a fixed-size padding term, say), so a caller may
    safely bake the value. Distinct from :class:`UntraceableTerm` because the two
    look identical from "no graph inputs" alone and must not be handled alike.
    """


class ConstantGroup(ValueError):
    """Every term in a group is native or constant, so the group has no graph.

    Not an error: the caller falls back to the per-term path rather than fusing an
    empty graph.
    """


class UntraceableTerm(ValueError):
    """A term read time-varying state the tracer could not follow into the graph.

    Baking its trace-time value would freeze that state silently, so the build fails
    instead.
    """

    def __init__(self, term: str, touched: list[str]):
        self.term = term
        self.touched = touched
        super().__init__(
            f"Observation term {term!r} reads state the tracer cannot turn into a "
            f"graph input: {', '.join(touched) or '(nothing usable)'}. Baking its "
            "current value would freeze a time-varying input and silently feed the "
            "policy stale numbers. Three ways out: supply a trace-friendly "
            "replacement via mjswan.register_observation(); write the term as a TS "
            "class and register an ObservationBinding whose `ts_src` points at it; or "
            "drop the term from the exported group and retrain — a shorter observation "
            "vector is not interchangeable with the one the policy was trained on."
        )


@dataclass
class TermExport:
    """The result of tracing one term body to ONNX."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    """Dynamic input slots, in ONNX graph input order — ``(entity, field)`` each."""
    input_names: list[str]
    output_name: str
    reference_output: torch.Tensor
    """The term's output on the discovery step (for a trace-time sanity check)."""
    constant_slots: list[SlotKey] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


def _slot_input_name(key: SlotKey) -> str:
    """The ONNX graph input name for a slot.

    A build-time detail: the name travels to the runtime in the slot's own ``input``
    field (:func:`slot_to_json`) rather than being recomputed there.
    """
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return "sensor__" + re.sub(r"\W", "_", name_part)
    if namespace == _COMMAND_NS:
        return "command__" + re.sub(r"\W", "_", name_part)
    return f"{namespace}__{name_part}"


def slot_label(key: SlotKey) -> str:
    """Human-readable slot name for diagnostics (parity reports, logs)."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        return f"sensor:{name_part}"
    if namespace == _COMMAND_NS:
        return f"command:{name_part}"
    return f"{namespace}.{name_part}"


def slot_to_json(key: SlotKey, shape: Sequence[int] | None = None) -> dict[str, Any]:
    """Serialize one input slot for ``policy.json`` / ``config.json``.

    Three shapes, told apart by which keys are present: ``{"entity", "field"}``,
    ``{"sensor"}``, or ``{"command", "field"}``. All carry ``input`` (the graph input
    name) and ``shape`` — the runtime feeds a flat array and cannot recover the rank
    without it.
    """
    namespace, name_part = key
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

    Shared by all three export kinds. Slots the exporter folded into a constant (an
    index tensor baked into the Gather it feeds) are dropped: ORT rejects a feed that
    is not a graph input.
    """
    shapes = getattr(export, "input_shapes", None) or []
    entries = [
        slot_to_json(key, shapes[i] if i < len(shapes) else None)
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


def trace_term(
    func: Callable[..., torch.Tensor],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> TermExport:
    """Trace a value-returning mjlab term body to ONNX against a live ``env``.

    ``params`` come from the env's own manager (``asset_cfg`` already resolved to
    static indices) and ``env`` must be post-reset.

    Raises:
        ConstantTerm: the term reads no simulation state (handle it as native).
        UntraceableTerm: the term reads state the tracer cannot follow.
    """
    # 1. Discovery: run once against the recording env.
    recorder = _RecordingEnv(env)
    recorded = func(recorder, **params)
    if not isinstance(recorded, torch.Tensor):
        raise ValueError(
            f"Term {name!r} returned {type(recorded).__name__}, not a Tensor; "
            "only value-returning terms are traced here."
        )

    # 2. Classify accessed slots into dynamic inputs vs baked constants.
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    _classify_slots(recorder._log, dynamic, constants)  # noqa: SLF001 — internal proxy

    if not dynamic:
        if recorder._log:  # noqa: SLF001 — internal proxy
            # State *was* read; the tracer just could not follow it into a tensor.
            raise UntraceableTerm(
                name, sorted({slot_label(k) for k, _ in recorder._log})
            )  # noqa: SLF001
        raise ConstantTerm(
            f"Term {name!r} reads no simulation state at all; handle it as a native "
            "term (e.g. time_out) or bake its value (ADR 0005)."
        )

    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    # 3. Trace to ONNX.
    sensors = dict(recorder._sensors)  # noqa: SLF001 — internal proxy
    commands = dict(recorder._commands)  # noqa: SLF001 — internal proxy
    module = _TermModule(
        func,
        params,
        dynamic_keys,
        constants,
        sensors=sensors,
        commands=commands,
        real_env=env,
    ).eval()
    output_name = "value"
    onnx_bytes = _export_onnx(
        module,
        example_inputs,
        input_names=input_names,
        output_names=[output_name],
        batch_axis=[*input_names, output_name],
        opset=opset,
    )

    return TermExport(
        name=name,
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=input_names,
        output_name=output_name,
        reference_output=recorded.detach(),
        constant_slots=sorted(constants),
        input_shapes=[list(t.shape) for t in example_inputs],
    )


def read_slot(env: Any, key: SlotKey) -> torch.Tensor:
    """Read an input slot's current value from ``env``."""
    namespace, name_part = key
    if namespace == _SENSOR_NS:
        sensor_name, dot, sensor_field = name_part.partition(".")
        data = env.scene[sensor_name].data
        return getattr(data, sensor_field) if dot else data
    if namespace == _COMMAND_NS:
        command_name, _, attr = name_part.partition(".")
        return getattr(env.command_manager.get_term(command_name), attr)
    return getattr(env.scene[namespace].data, name_part)


# --- Event terms: an event returns None and writes via `entity.write_*_to_sim`, so
# the tensors it would write become the graph outputs, and its randomness is threaded
# in as an explicit `rand` input replayed by ReplayRng. ---

# Each write call and the tensors it writes, in argument order.
_WRITE_FIELDS: dict[str, tuple[str, ...]] = {
    "joint_state": ("position", "velocity"),
    "root_pose": ("pose",),
    "root_velocity": ("velocity",),
}


#: Keyed by entity as well as kind: one term may write several entities (mjlab's
#: `reset_scene_to_default` writes them all), each needing its own target and outputs.
WriteKey = tuple[str | None, str]
WriteCaptures = dict[WriteKey, tuple[torch.Tensor, ...]]


def _write_output_name(key: WriteKey, field_name: str) -> str:
    """Graph-output name for one written tensor; unprefixed when no entity was named."""
    entity, kind = key
    return f"{entity}__{kind}__{field_name}" if entity else f"{kind}__{field_name}"


class _WriteCaptureMixin:
    """Records ``write_*_to_sim`` calls into ``self._captures``.

    The entity comes from the write itself, not the params: a term names its target
    however it likes (``asset_cfg``, or a plain ``ball_name``).
    """

    #: Entity this proxy stands for; None when the cfg names none.
    _name: str | None
    _captures: WriteCaptures

    def _capture(self, kind: str, values: tuple[Any, ...]) -> None:
        self._captures[(self._name, kind)] = values

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        self._capture("joint_state", (position, velocity))

    def write_root_link_pose_to_sim(self, pose, env_ids=None):
        self._capture("root_pose", (pose,))

    def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
        self._capture("root_velocity", (velocity,))

    def write_root_state_to_sim(self, root_state, env_ids=None):
        # mjlab's own split of a 13-wide root state into the two writes above.
        self._capture("root_pose", (root_state[..., :7],))
        self._capture("root_velocity", (root_state[..., 7:],))


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
        for field_name, tensor in zip(_WRITE_FIELDS[key[1]], values):
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
    """Proxy env for event tracing: records reads, captures writes, no sim mutation."""

    def __init__(self, real, log, captures):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "scene", _EvRecScene(real.scene, log, captures))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class _EvReplayData:
    def __init__(self, entity: str, served: dict[TaggedKey, Any]):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_served", served)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("data", self._entity, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared data slot ('data', {self._entity!r}, "
                f"{name!r}) during tracing (input-dependent read?)."
            ) from None


class _EvReplayEntity(_WriteCaptureMixin):
    def __init__(self, entity, served, captures):
        object.__setattr__(self, "_name", entity)
        object.__setattr__(self, "_served", served)
        object.__setattr__(self, "data", _EvReplayData(entity, served))
        object.__setattr__(self, "_captures", captures)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("attr", self._name, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared attr ('attr', {self._name!r}, {name!r})."
            ) from None


class _EvReplayScene:
    def __init__(self, served, captures, real_env: Any):
        self._served = served
        self._captures = captures
        self._real_env = real_env

    def __getitem__(self, name: str) -> _EvReplayEntity:
        return _EvReplayEntity(name, self._served, self._captures)

    def __getattr__(self, name: str) -> Any:
        if name == "entities":
            # Static structure, so it comes from the real env, as `num_envs` does —
            # only the entities' tensors are recorded slots.
            return {
                key: _EvReplayEntity(key, self._served, self._captures)
                for key in self._real_env.scene.entities
            }
        try:
            return self._served[("scene", name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared scene attr ('scene', {name!r})."
            ) from None


class _EventReplayEnv:
    def __init__(self, served, captures, *, real_env: Any):
        self.scene = _EvReplayScene(served, captures, real_env)
        self._real_env = real_env

    def __getattr__(self, name: str) -> Any:
        # Forwarded, not defaulted: replay must see the same N discovery ran against.
        if name in ("num_envs", "device"):
            return getattr(self._real_env, name)
        raise AttributeError(name)


class _EventModule(nn.Module):
    """Wraps a side-effecting event ``func`` so ``forward(*dynamic, rand)`` returns
    the tensors the term would write.

    Dynamic reads arrive as ``forward`` args, constants as buffers or plain Python
    values; all are served back through the replay env.
    """

    def __init__(
        self,
        func: Callable[..., None],
        params: dict[str, Any],
        dynamic_keys: list[SlotKey],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
        *,
        real_env: Any,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._scalar_consts = scalar_consts
        self._real_env = real_env
        self._const_buffers = _register_consts(self, tensor_consts)

    def forward(self, *args: torch.Tensor):
        *dynamic, rand = args
        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        served.update(_const_values(self, self._const_buffers))
        for (entity, field_name), tensor in zip(self._dynamic_keys, dynamic):
            served[("data", entity, field_name)] = tensor
        captures: WriteCaptures = {}
        env = _EventReplayEnv(served, captures, real_env=self._real_env)
        with ReplayRng(self._func, rand):
            self._func(env, None, **self._params)
        _, tensors = _flatten_captures(captures)
        return tuple(tensors)


@dataclass
class EventExport:
    """The result of tracing one event term body to ONNX."""

    name: str
    mode: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` for ``rand`` — the runtime draws with these."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    """Per write-kind descriptor: what the outputs target (entity, kind, fields)."""
    reference_outputs: tuple[torch.Tensor, ...]
    reference_rand: torch.Tensor
    constant_slots: list[str] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


_EXPORT_FILTERS_INSTALLED = False


def _prepare_single_env_export(num_envs: int) -> None:
    """Refuse a batched trace, and silence the three warnings a single-env one raises.

    All three are safe here: the `index_put_` mutation is read back as a graph output,
    `len(env_ids)` bakes the row count this guard pins to 1, and the `torch.tensor`
    constants are config that cannot vary.
    """
    if num_envs != 1:
        raise ValueError(
            f"tracing with num_envs={num_envs}: the graph would bake that row count "
            "while the runtime feeds one row (session.ts pins the batch axis to 1). "
            "Trace single-env, as Project.add_mjlab_task does."
        )
    global _EXPORT_FILTERS_INSTALLED
    if _EXPORT_FILTERS_INSTALLED:
        return
    warnings.filterwarnings(
        "ignore",
        message="ONNX Preprocess - Removing mutation from node aten::index_put_",
        category=UserWarning,
    )
    # No category: `TracerWarning` is not on `torch.jit`'s public stub.
    warnings.filterwarnings("ignore", message="Using len to get tensor shape")
    warnings.filterwarnings(
        "ignore", message="torch.tensor results are registered as constants"
    )
    _EXPORT_FILTERS_INSTALLED = True


def trace_event_term(
    func: Callable[..., None],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    mode: str,
    opset: int = 17,
) -> EventExport:
    """Trace a side-effecting (write-to-sim) event term body to ONNX.

    The written tensors become the graph outputs and randomness arrives as ``rand``.
    Time-varying ``entity.data`` fields become graph inputs; everything else the term
    reads (scene tensors, control-flow scalars) is baked in.
    """
    # 1. Discovery on the live env: record draws + reads + written values.
    log: list[tuple[TaggedKey, Any]] = []
    captures: WriteCaptures = {}
    proxy = _EventCaptureEnv(env, log, captures)
    with DrawRecorder(func) as rec:
        func(proxy, None, **params)

    if not captures:
        raise ValueError(
            f"Event term {name!r} wrote nothing traceable (no write_joint_state / "
            "write_root_state / write_root_link_pose / write_root_link_velocity call); "
            "handle it natively or extend _WRITE_FIELDS."
        )
    output_names, ref_tensors = _flatten_captures(captures)
    ref_rand = rec.rand_vector
    rand_dim = rec.rand_dim
    rand_ranges = rec.rand_ranges

    # 2. Classify recorded reads: dynamic data-field inputs vs baked constants.
    dynamic, tensor_consts, scalar_consts = _classify_tagged(log)

    dynamic_keys = sorted(dynamic)
    dyn_input_names = [_slot_input_name(k) for k in dynamic_keys]
    example = tuple(dynamic[k] for k in dynamic_keys) + (ref_rand,)
    input_names = [*dyn_input_names, "rand"]

    # 3. Trace: rand replayed as an explicit input; written values captured.
    module = _EventModule(
        func, params, dynamic_keys, tensor_consts, scalar_consts, real_env=env
    ).eval()
    _prepare_single_env_export(env.num_envs)
    # `rand` keeps its traced length: it is one flat draw vector, not a batch of rows.
    onnx_bytes = _export_onnx(
        module,
        example,
        input_names=input_names,
        output_names=output_names,
        batch_axis=[*dyn_input_names, *output_names],
        opset=opset,
    )

    # The write says which entity it landed on; `asset_cfg` is the fallback.
    asset_cfg = params.get("asset_cfg")
    asset_name = getattr(asset_cfg, "name", None)
    write_targets = []
    for key in captures:
        entity, kind = key
        target: dict[str, Any] = {
            "kind": kind,
            "entity": entity or asset_name,
            "fields": list(_WRITE_FIELDS[kind]),
            "outputs": [_write_output_name(key, f) for f in _WRITE_FIELDS[kind]],
        }
        # `asset_cfg`'s ids scope its own entity; any other one gets all of its joints.
        joint_ids = _static_ids(getattr(asset_cfg, "joint_ids", None))
        scoped = entity is None or entity == asset_name
        if kind == "joint_state" and scoped and joint_ids is not None:
            target["joint_ids"] = joint_ids
        write_targets.append(target)

    return EventExport(
        name=name,
        mode=mode,
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=dyn_input_names,
        rand_dim=rand_dim,
        rand_ranges=rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_outputs=tuple(t.detach() for t in ref_tensors),
        reference_rand=ref_rand.detach(),
        constant_slots=[":".join(str(p) for p in k) for k in sorted(tensor_consts)],
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
    )


def _static_ids(ids: Any) -> Any:
    if isinstance(ids, slice):
        return "all"
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return ids


# --- Command terms: a CommandTerm's hidden state is promoted to explicit graph I/O,
# so `_resample_command` + `_update_command` trace as one pure function
#
#     forward(prev_state..., resample_mask, rand) -> (next_state..., entity_write?)
#
# with the runtime holding `state` across frames and owning the resample timer. ---

_ENTITY_WRITE_METHODS = {
    "write_joint_state_to_sim": "joint_state",
    "write_root_link_pose_to_sim": "root_pose",
    "write_root_link_velocity_to_sim": "root_velocity",
}


def _entity_attrs(term: Any) -> list[str]:
    """Names of ``term`` attributes that are entities (read state from / write to)."""
    return [
        attr
        for attr, value in vars(term).items()
        if hasattr(value, "data")
        and any(hasattr(type(value), m) for m in _ENTITY_WRITE_METHODS)
    ]


class _RecordCommand:
    """Swap a command's entity attrs + ``_env`` to recording proxies, so its reads are
    logged and its writes captured without mutating the sim.

    Single-entity commands only: all entity attrs are keyed by ``cfg.entity_name``.
    """

    def __init__(self, term: Any, entity_attr_names: list[str], entity_name: str):
        self.term = term
        self._attrs = entity_attr_names
        self._entity_name = entity_name
        self.log: list[tuple[TaggedKey, Any]] = []
        self.captures: WriteCaptures = {}

    def __enter__(self) -> _RecordCommand:
        self._orig = {a: getattr(self.term, a) for a in self._attrs}
        self._orig_env = getattr(self.term, "_env", None)
        for a in self._attrs:
            setattr(
                self.term,
                a,
                _EvRecEntity(self._orig[a], self._entity_name, self.log, self.captures),
            )
        if self._orig_env is not None:
            self.term._env = _EventCaptureEnv(self._orig_env, self.log, self.captures)
        return self

    def __exit__(self, *exc: object) -> None:
        for a, v in self._orig.items():
            setattr(self.term, a, v)
        if self._orig_env is not None:
            self.term._env = self._orig_env


def _snapshot_state(term: Any) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().clone()
        for k, v in vars(term).items()
        if isinstance(v, torch.Tensor)
    }


def _restore_state(term: Any, snap: dict[str, torch.Tensor]) -> None:
    for k, v in snap.items():
        setattr(term, k, v.clone())


def _gate(
    mask: torch.Tensor, resampled: torch.Tensor, prev: torch.Tensor
) -> torch.Tensor:
    shape = [mask.shape[0]] + [1] * (resampled.dim() - 1)
    m = mask.reshape(shape)
    # ONNX Runtime's Where kernel has no bool-branch implementation; select in
    # int64 and cast back so bool state fields (is_*_env) round-trip.
    if resampled.dtype == torch.bool:
        return torch.where(m, resampled.long(), prev.long()).bool()
    return torch.where(m, resampled, prev)


class _CommandModule(nn.Module):
    """Traces a CommandTerm's resample+update as a pure function.

    ``forward(*dynamic_slots, *prev_state, resample_mask, rand)``: state is injected
    and read back, the resample is gated by ``resample_mask``, ``_update_command``
    always runs, and any ``entity_write`` is captured.

    The mask gates the state fields only — captured writes are a fresh draw every call
    and are valid only when it is true, which ``OnnxCommand.step`` enforces.
    """

    def __init__(
        self,
        term: Any,
        state_fields: list[str],
        entity_attr_names: list[str],
        entity_name: str,
        *,
        dynamic_keys: list[SlotKey],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
    ):
        super().__init__()
        self._term = term
        self._state_fields = state_fields
        self._entity_attr_names = entity_attr_names
        self._entity_name = entity_name
        self._dynamic_keys = dynamic_keys
        self._scalar_consts = scalar_consts
        self._env_ids = torch.arange(term.num_envs)
        self._const_buffers = _register_consts(self, tensor_consts)

    def forward(self, *args: torch.Tensor):
        n_dyn = len(self._dynamic_keys)
        n_state = len(self._state_fields)
        dynamic = args[:n_dyn]
        state_inputs = args[n_dyn : n_dyn + n_state]
        resample_mask = args[n_dyn + n_state]
        rand = args[n_dyn + n_state + 1]

        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        served.update(_const_values(self, self._const_buffers))
        for (entity, field_name), tensor in zip(self._dynamic_keys, dynamic):
            served[("data", entity, field_name)] = tensor

        captures: WriteCaptures = {}
        orig = {a: getattr(self._term, a) for a in self._entity_attr_names}
        orig_env = getattr(self._term, "_env", None)
        for a in self._entity_attr_names:
            setattr(self._term, a, _EvReplayEntity(self._entity_name, served, captures))
        if orig_env is not None:
            # `real_env` is the env being swapped out, not the term: `num_envs`
            # forwards to `_env`, so the term would forward to itself.
            self._term._env = _EventReplayEnv(served, captures, real_env=orig_env)
        try:
            prev = {}
            for field_name, value in zip(self._state_fields, state_inputs):
                setattr(self._term, field_name, value)
                prev[field_name] = value.clone()
            with ReplayRng(self._term._resample_command, rand):
                self._term._resample_command(self._env_ids)
                for field_name in self._state_fields:
                    setattr(
                        self._term,
                        field_name,
                        _gate(
                            resample_mask,
                            getattr(self._term, field_name),
                            prev[field_name],
                        ),
                    )
                self._term._update_command()
            outputs = [getattr(self._term, f) for f in self._state_fields]
            _, write_tensors = _flatten_captures(captures)
            return tuple(outputs) + tuple(write_tensors)
        finally:
            for a, v in orig.items():
                setattr(self._term, a, v)
            if orig_env is not None:
                self._term._env = orig_env


@dataclass
class CommandExport:
    """The result of tracing one command term body to ONNX."""

    name: str
    onnx_bytes: bytes
    state_fields: list[dict[str, Any]]
    """Per state field: {name, shape, dtype} — declared in policy.json (§3a)."""
    command_field: str
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` for ``rand`` — the runtime draws with these."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    reference_rand: torch.Tensor
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


def trace_command_term(
    term: Any,
    state_fields: list[str],
    *,
    name: str,
    command_field: str,
    opset: int = 17,
) -> CommandExport:
    """Trace a stateful CommandTerm to ONNX.

    Promotes ``state_fields`` to explicit graph I/O and threads randomness through
    ``rand``. Only ``sample_uniform`` draws are supported — a term using tensor-method
    RNG (``Tensor.uniform_``) needs a trace-friendly override.
    """
    entity_attr_names = _entity_attrs(term)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    snap = _snapshot_state(term)
    state_example = tuple(getattr(term, f).detach().clone() for f in state_fields)

    # 1. Discovery: swap to recording proxies; log reads, capture writes, spy draws.
    with _RecordCommand(term, entity_attr_names, entity_name) as rec_env:
        with DrawRecorder(term._resample_command) as rec:
            term._resample_command(torch.arange(term.num_envs))
            term._update_command()
        log = list(rec_env.log)
        captures = dict(rec_env.captures)
    ref_rand = rec.rand_vector
    rand_dim = rec.rand_dim
    rand_ranges = rec.rand_ranges
    _restore_state(term, snap)

    output_write_names, _ = _flatten_captures(captures)
    write_targets = [
        {
            "kind": kind,
            "entity": entity or entity_name,
            "fields": list(_WRITE_FIELDS[kind]),
            "outputs": [
                _write_output_name((entity, kind), f) for f in _WRITE_FIELDS[kind]
            ],
        }
        for entity, kind in captures
    ]

    # 2. Classify reads: dynamic data inputs vs baked tensor/scalar constants.
    dynamic, tensor_consts, scalar_consts = _classify_tagged(log)

    dynamic_keys = sorted(dynamic)
    dyn_names = [_slot_input_name(k) for k in dynamic_keys]
    prev_names = [f"prev_{f}" for f in state_fields]

    # 3. Trace: dynamic + prev_state + resample_mask=True + rand -> next_state + writes.
    mask = torch.ones(term.num_envs, dtype=torch.bool)
    example = (*(dynamic[k] for k in dynamic_keys), *state_example, mask, ref_rand)
    input_names = [*dyn_names, *prev_names, "resample_mask", "rand"]
    output_names = [f"next_{f}" for f in state_fields] + output_write_names

    module = _CommandModule(
        term,
        state_fields,
        entity_attr_names,
        entity_name,
        dynamic_keys=dynamic_keys,
        tensor_consts=tensor_consts,
        scalar_consts=scalar_consts,
    ).eval()
    _prepare_single_env_export(term.num_envs)
    # `rand` keeps its traced length: it is one flat draw vector, not a batch of rows.
    onnx_bytes = _export_onnx(
        module,
        example,
        input_names=input_names,
        output_names=output_names,
        batch_axis=[*dyn_names, *prev_names, "resample_mask", *output_names],
        opset=opset,
    )
    _restore_state(term, snap)

    # Initial values, as `cfg.build(env)` left them. Without them the runtime
    # zero-fills, which starts a counter or a held previous value wrong.
    state_specs = [
        {
            "name": f,
            "shape": list(getattr(term, f).shape),
            "dtype": str(getattr(term, f).dtype).replace("torch.", ""),
            "init": [
                # bool/int state round-trips as a number; the reader rebuilds dtype.
                bool(v) if getattr(term, f).dtype == torch.bool else v
                for v in getattr(term, f).detach().reshape(-1).tolist()
            ],
        }
        for f in state_fields
    ]

    return CommandExport(
        name=name,
        onnx_bytes=onnx_bytes,
        state_fields=state_specs,
        command_field=command_field,
        input_slots=dynamic_keys,
        input_names=dyn_names,
        rand_dim=rand_dim,
        rand_ranges=rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_rand=ref_rand.detach(),
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
    )


# --- Observation-group fusion ---

# mjlab funcs reading env-level state rather than `entity.data`: nothing to trace,
# since the runtime already holds these values every frame.
NATIVE_OBSERVATION_FUNCS: dict[str, str] = {
    "last_action": "prev_action",
    "action_history": "prev_action",
    "generated_commands": "command",
}


def _native_observation_kind(func: Callable[..., Any]) -> str | None:
    return NATIVE_OBSERVATION_FUNCS.get(getattr(func, "__name__", ""))


def native_observation_entry(
    name: str, func: Callable[..., Any], params: dict[str, Any], env: Any
) -> dict[str, Any] | None:
    """The ``native`` marker for an observation the runtime already holds, else ``None``.

    Carries the kind and whichever selector it needs; the caller adds ``size`` (and, when
    fusing, the graph ``input`` name) since the two paths resolve widths differently.
    ``action_offset`` is resolved here rather than beside the caller's width probe, whose
    swallowed failure would lose it.
    """
    kind = _native_observation_kind(func)
    if kind is None:
        return None
    entry: dict[str, Any] = {"name": name, "native": kind}
    if kind == "command":
        entry["command_name"] = params["command_name"]
        return entry
    # How far back in the action window, `last_action`'s 0 being the newest.
    if params.get("age"):
        entry["age"] = int(params["age"])
    if params.get("action_name") is not None:
        entry["action_name"] = params["action_name"]
        entry["action_offset"] = action_term_offset(env, params["action_name"])
    return entry


def action_term_offset(env: Any, action_name: str) -> int:
    """Where *action_name*'s slice starts inside the policy's action vector.

    ``last_action(action_name=...)`` is one action term's slice, so the browser — which
    holds the policy output whole — needs this offset to reproduce it. Raises rather
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


@dataclass
class GroupTermSpec:
    """One term inside a fused group, as :func:`trace_observation_group` needs it."""

    name: str
    func: Callable[..., torch.Tensor]
    params: dict[str, Any]
    clip: tuple[float, float] | None = None
    scale: Any = None
    """Per-term scale — a float, or a sequence broadcast over the term's width."""
    native_size: int | None = None


@dataclass
class GroupExport:
    """The result of fusing one observation group into a single ONNX graph."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    """Deduplicated union of every term's dynamic slots, in graph input order."""
    input_names: list[str]
    input_shapes: list[list[int]]
    native_inputs: list[dict[str, Any]]
    """Per native term: ``{name, native, input, size, ...}`` — fed by the runtime."""
    layout: list[dict[str, Any]]
    """``{name, size}`` per term, in concat order, for the runtime's group layout."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)


class _GroupModule(nn.Module):
    """Runs a whole observation group: every term body, then clip/scale, then cat.

    Reproduces mjlab's ``compute_group``, sharing one replay env across the terms so a
    slot two of them read is marshalled once. Native terms are graph *inputs* rather
    than bodies, which keeps the output the complete observation vector.
    """

    def __init__(
        self,
        terms: list[GroupTermSpec],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        *,
        sensors: dict[str, Any],
        commands: dict[str, Any],
        native_names: list[str],
        baked: dict[str, torch.Tensor],
        real_env: Any,
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._native_names = native_names
        self._real_env = real_env
        self._const_buffers = _register_consts(self, constants)
        # A term reading no dynamic state is a value, not a function — bake it.
        self._baked_buffers = _register_consts(self, baked, prefix="_baked")

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        split = len(self._dynamic_keys)
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, args[:split]))
        slots.update(_const_values(self, self._const_buffers))
        native = dict(zip(self._native_names, args[split:]))
        env = _ReplayEnv(slots, self._sensors, self._commands, real_env=self._real_env)

        pieces: list[torch.Tensor] = []
        for term in self._terms:
            if term.name in native:
                value = native[term.name]
            elif term.name in self._baked_buffers:
                value = getattr(self, self._baked_buffers[term.name])
            else:
                value = term.func(env, **term.params)
            # mjlab's order: clip, then scale (observation_manager.compute_group).
            if term.clip is not None:
                value = torch.clamp(value, min=term.clip[0], max=term.clip[1])
            if term.scale is not None:
                value = value * _scale_tensor(term.scale, value)
            pieces.append(value.reshape(value.shape[0], -1))
        return torch.cat(pieces, dim=-1)


def _native_example(term: GroupTermSpec, env: Any) -> torch.Tensor:
    """Example value fixing a native term's graph-input width.

    The live env is asked first. A bare trace env has no action terms and no command
    manager, so the build hands the width down as ``native_size`` instead.
    """
    try:
        value = term.func(env, **term.params).detach()
    except Exception:  # noqa: BLE001 — a trace env legitimately has neither
        value = None
    if value is not None and value.reshape(1, -1).shape[-1] > 0:
        return value
    if term.native_size:
        return torch.zeros(1, term.native_size)
    raise ValueError(
        f"Observation term {term.name!r} is native, but neither the trace env nor "
        "the policy config gives its width. Set the policy's "
        "`policy_joint_names`/`policy_num_actions` (for `last_action`), or declare "
        "the command's UI inputs (for `generated_commands`). A term-scoped "
        "`last_action` has no config answer at all — it needs a trace env whose "
        "action manager holds the term."
    )


def _scale_tensor(scale: Any, like: torch.Tensor) -> torch.Tensor:
    """A term's ``scale`` as a tensor broadcastable over its output."""
    if isinstance(scale, torch.Tensor):
        return scale.to(like.dtype)
    if isinstance(scale, (list, tuple)):
        return torch.tensor(list(scale), dtype=like.dtype, device=like.device)
    return torch.tensor(float(scale), dtype=like.dtype, device=like.device)


def trace_observation_group(
    terms: list[GroupTermSpec],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> GroupExport:
    """Fuse an observation group's terms into one ONNX graph.

    One graph per group rather than one per term, since a per-term graph can be a
    single node and the fixed per-``ort.run()`` cost then dominates (ADR 0005 §4).

    Inputs are the deduplicated union of the terms' dynamic slots, then one input per
    native term. The output is the concatenated vector with clip/scale folded in —
    what the policy consumes, minus history.
    """
    # 1. Discovery, per term: what does each read, and is it native?
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    sensors: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    native_inputs: list[dict[str, Any]] = []
    native_examples: list[torch.Tensor] = []
    baked: dict[str, torch.Tensor] = {}
    layout: list[dict[str, Any]] = []

    for term in terms:
        entry = native_observation_entry(term.name, term.func, term.params, env)
        if entry is not None:
            entry["input"] = "native__" + re.sub(r"\W", "_", term.name)
            value = _native_example(term, env)
            entry["size"] = int(value.reshape(1, -1).shape[-1])
            native_inputs.append(entry)
            native_examples.append(value)
            layout.append({"name": term.name, "size": entry["size"]})
            continue

        recorder = _RecordingEnv(env)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Observation term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = _classify_slots(recorder._log, dynamic, constants)  # noqa: SLF001
        sensors.update(recorder._sensors)  # noqa: SLF001 — internal proxy
        commands.update(recorder._commands)  # noqa: SLF001 — internal proxy
        if not term_dynamic:
            # Nothing read means a constant; unfollowable reads mean live state.
            if recorder._log:  # noqa: SLF001 — internal proxy
                raise UntraceableTerm(
                    term.name,
                    sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
                )
            baked[term.name] = recorded.detach()
        layout.append(
            {"name": term.name, "size": int(recorded.reshape(1, -1).shape[-1])}
        )

    if not dynamic:
        raise ConstantGroup(
            f"Observation group {name!r} reads no time-varying state; every term is "
            "native or constant, so there is no graph to run."
        )

    # 2. Fuse and export. Slots sorted for determinism, then natives in declaration
    #    order.
    dynamic_keys = sorted(dynamic)
    slot_names = [_slot_input_name(k) for k in dynamic_keys]
    native_names = [entry["name"] for entry in native_inputs]
    input_names = [*slot_names, *(entry["input"] for entry in native_inputs)]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys) + tuple(native_examples)

    module = _GroupModule(
        terms,
        dynamic_keys,
        constants,
        sensors=sensors,
        commands=commands,
        native_names=native_names,
        baked=baked,
        real_env=env,
    ).eval()
    output_name = "obs"
    with torch.no_grad():
        reference = module(*example_inputs).detach()
    onnx_bytes = _export_onnx(
        module,
        example_inputs,
        input_names=input_names,
        output_names=[output_name],
        batch_axis=[*input_names, output_name],
        opset=opset,
    )

    return GroupExport(
        name=name,
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=slot_names,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        native_inputs=native_inputs,
        layout=layout,
        output_name=output_name,
        reference_output=reference,
        constant_slots=sorted(constants),
    )


# --- Termination-group fusion ---


@dataclass
class TerminationGroupExport:
    """The result of fusing a set of termination terms into a single ONNX graph."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    input_shapes: list[list[int]]
    lanes: list[str]
    """Term names, in output-lane order — lane *i* is `lanes[i]`'s verdict."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)


class _TerminationGroupModule(nn.Module):
    """Every termination body in one graph, emitting one bool lane per term.

    A lane rather than a single OR, so the manager keeps reporting *which* term fired.
    """

    def __init__(
        self,
        terms: list[GroupTermSpec],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        *,
        sensors: dict[str, Any],
        commands: dict[str, Any],
        real_env: Any,
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._real_env = real_env
        self._const_buffers = _register_consts(self, constants)

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        slots.update(_const_values(self, self._const_buffers))
        env = _ReplayEnv(slots, self._sensors, self._commands, real_env=self._real_env)
        lanes = [term.func(env, **term.params).reshape(-1, 1) for term in self._terms]
        return torch.cat(lanes, dim=-1)


def trace_termination_group(
    terms: list[GroupTermSpec],
    env: Any,
    *,
    name: str,
    opset: int = 17,
) -> TerminationGroupExport:
    """Fuse termination terms into one graph, one bool lane each.

    Same mechanics as :func:`trace_observation_group`, but the output is a bool vector
    so the manager keeps its per-term reasons. `time_out` never reaches here — it reads
    no entity state and is classified native first.
    """
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    sensors: dict[str, Any] = {}
    commands: dict[str, Any] = {}

    for term in terms:
        recorder = _RecordingEnv(env)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Termination term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = _classify_slots(recorder._log, dynamic, constants)  # noqa: SLF001
        sensors.update(recorder._sensors)  # noqa: SLF001 — internal proxy
        commands.update(recorder._commands)  # noqa: SLF001 — internal proxy
        if not term_dynamic:
            # Never baked: a termination blind to state never fires or always does.
            raise UntraceableTerm(
                term.name,
                sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
            )

    if not dynamic:
        raise ValueError(
            f"Termination group {name!r} reads no time-varying state; every term "
            "should be native (e.g. time_out)."
        )

    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    module = _TerminationGroupModule(
        terms, dynamic_keys, constants, sensors=sensors, commands=commands, real_env=env
    ).eval()
    output_name = "done"
    with torch.no_grad():
        reference = module(*example_inputs).detach()
    onnx_bytes = _export_onnx(
        module,
        example_inputs,
        input_names=input_names,
        output_names=[output_name],
        batch_axis=[*input_names, output_name],
        opset=opset,
    )

    return TerminationGroupExport(
        name=name,
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=input_names,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        lanes=[term.name for term in terms],
        output_name=output_name,
        reference_output=reference,
        constant_slots=sorted(constants),
    )
