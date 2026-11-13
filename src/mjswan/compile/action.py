"""Trace a task-defined ``ActionTerm``'s ``process_actions`` and ``reset`` to ONNX.

ADR 0009. The term's state, every tensor attribute either body writes, is promoted to
explicit graph I/O, as a stateful command's is (ADR 0005 §3)::

    process(raw, slots..., prev_<f>..., [rand]) -> (next_<f>..., processed, [processed_from])
    reset(slots..., prev_<f>..., [rand])        -> (next_<f>...)

The browser holds the state, runs ``process`` once per control step after the policy and
``reset`` in the reset chain, and applies ``processed`` natively each substep: what the
term's ``apply_actions`` does is not traced but named, by its mjlab base class or its
binding. Before they ship, both graphs run against the term's own bodies over random
actions, resets and episode clocks.
"""

from __future__ import annotations

import contextlib
import enum
import math
from dataclasses import dataclass, field
from typing import Any, Collection, Literal

import numpy as np
import torch
from torch import nn

from .command import (
    _mismatch,
    _restore_state,
    _snapshot_state,
    _state_spec,
)
from .export import (
    GraphRunner,
    _classify_slots,
    _const_values,
    _export_onnx,
    _input_rows,
    _narrow_inputs,
    _narrow_slots,
    _prepare_single_env_export,
    _register_consts,
    reference_runner,
)
from .gate import GatedIndexing, MaskedRows
from .proxy import _CONTAINERS, _attrs, _holds_live, _is_live_object, _is_sensor
from .record import _RecordingEnv
from .replay import _ReplayEnv
from .rng import DrawRecorder, ReplayRng
from .slot import (
    _ENV_NS,
    READER_FIELDS,
    SlotKey,
    _sim_tensor,
    _slot_input_name,
    read_slot,
    unserved_fields,
)

Body = Literal["reset", "process"]

#: The bodies a draw may sit in, up the term's class hierarchy.
_BODIES = ("process_actions", "reset")

#: Plain immutable values: a frozen copy of one is the value itself.
_SCALARS = (bool, int, float, str, type(None))

#: State dtypes the runtime holds natively; any other crosses as float32.
_NATIVE_STATE_DTYPES = (torch.float32, torch.bool)

#: Entity calls that change the sim, which only ``apply_actions`` may make.
_SIM_CALL_PREFIXES = ("write_", "set_", "clear_")
_SIM_CALLS = frozenset({"reset", "update"})

#: Raw sim fields a body could write in place, compared around each discovery pass.
_SIM_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
)


class UntraceableAction(ValueError):
    """An action term whose body the tracer cannot turn into a graph that matches it."""


def _bodies(term: Any) -> tuple[Any, ...]:
    """Each class's own ``process_actions`` and ``reset``: a ``super()`` call runs in
    the base's module, whose globals a draw is looked up in too."""
    return tuple(
        body
        for cls in type(term).__mro__
        for name in _BODIES
        if (body := vars(cls).get(name)) is not None
    )


def _tensor_versions(obj: Any) -> dict[str, tuple[torch.Tensor, int]]:
    """Each tensor attribute and its in-place version. The tensor is kept, so a freed
    one's id cannot pass for a new one's when a body rebinds an attribute twice."""
    return {
        k: (v, v._version) for k, v in vars(obj).items() if isinstance(v, torch.Tensor)
    }


def _rewritten(
    before: dict[str, tuple[torch.Tensor, int]],
    after: dict[str, tuple[torch.Tensor, int]],
) -> set[str]:
    """The attributes rebound or written in place between the two."""
    return {
        k
        for k, (v, version) in after.items()
        if k not in before or before[k][0] is not v or before[k][1] != version
    }


def _views(term: Any) -> list[list[str]]:
    """Tensor attributes that share memory, each group in attribute order."""
    groups: dict[int, list[str]] = {}
    for k, v in vars(term).items():
        if isinstance(v, torch.Tensor) and v.numel():
            groups.setdefault(v.untyped_storage().data_ptr(), []).append(k)
    return [names for names in groups.values() if len(names) > 1]


class _Same:
    """A value compared by identity."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Same) and other.value is self.value

    __hash__ = None  # type: ignore[assignment]


def _freeze(value: Any, within: frozenset[int] = frozenset()) -> Any:
    """*value* as it stands, to compare after a body runs: containers and plain objects
    by content at any depth, tensors and arrays by value, and live objects, code and a
    reference back to a value in *within* (ids of the values enclosing it) by
    identity."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (*_SCALARS, complex, bytes, enum.Enum, np.generic, slice)):
        return value
    if id(value) in within or _is_live_object(value):
        return _Same(value)
    within = within | {id(value)}
    if isinstance(value, dict):
        return (dict, {k: _freeze(v, within) for k, v in value.items()})
    if isinstance(value, _CONTAINERS):
        return (type(value), [_freeze(v, within) for v in value])
    if (attrs := _attrs(value)) is not None:
        return (type(value), {k: _freeze(v, within) for k, v in attrs.items()})
    return _Same(value)


def _same(a: Any, b: Any) -> bool:
    """Whether two :func:`_freeze` results agree, NaN equal to NaN."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.equal(a, b):
            return True
        return a.is_floating_point() and bool(
            ((a == b) | (a.isnan() & b.isnan())).all()
        )
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        nan = a.dtype.kind in "fc" and b.dtype.kind in "fc"
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=nan)
    if (
        isinstance(a, (float, np.floating))
        and isinstance(b, (float, np.floating))
        and math.isnan(a)
        and math.isnan(b)
    ):
        return True
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 (an object without a usable `==`)
        return a is b


def _python_state(term: Any, skip: Collection[str]) -> dict[str, Any]:
    """Every attribute but the tensors and *skip*, frozen to compare after a body."""
    return {
        k: _freeze(v)
        for k, v in vars(term).items()
        if k not in skip and not isinstance(v, torch.Tensor)
    }


def _sim_values(env: Any) -> dict[str, torch.Tensor]:
    """The raw sim fields a body could write in place, copied."""
    data = getattr(getattr(env, "sim", None), "data", None)
    values = {}
    for name in _SIM_FIELDS:
        try:
            field = getattr(data, name, None)
        except TypeError:
            # mjlab cannot wrap an empty vector field on the CPU, such as mocap_pos in
            # a floating-base model; an empty field holds nothing to write.
            continue
        value = _sim_tensor(field)
        if isinstance(value, torch.Tensor):
            values[name] = value.detach().clone()
    return values


def _refuse_sim_write(term_name: str, what: str, body: str) -> UntraceableAction:
    return UntraceableAction(
        f"Action term {term_name!r} {what} in {body}. Targets reach the sim through "
        "apply_actions, which the browser runs natively; a body that writes the sim "
        "cannot be traced."
    )


class _GuardedEntity:
    """An entity for a body: data reads go to *data_source*, other attributes to the
    real entity, and a call that would change the sim is refused."""

    def __init__(self, data_source: Any, real: Any, term_name: str):
        object.__setattr__(self, "_source", data_source)
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_term", term_name)

    def __getattr__(self, name: str) -> Any:
        if name == "data":
            return self._source.data
        if name.startswith(_SIM_CALL_PREFIXES) or name in _SIM_CALLS:
            raise _refuse_sim_write(
                self._term, f"calls entity.{name}()", "process_actions or reset"
            )
        return getattr(self._real, name)


class _GuardedScene:
    """``env.scene`` for a body: each entity in it guarded like the term's own."""

    def __init__(self, scene: Any, real: Any, term_name: str):
        self._scene = scene
        self._real = real
        self._term = term_name

    def __getitem__(self, name: str) -> Any:
        item = self._scene[name]
        if _is_sensor(self._real, name):
            return item
        return _GuardedEntity(item, self._real[name], self._term)


class _GuardedEnv:
    """``self._env`` for a body: the pass's env, with its scene guarded."""

    def __init__(self, env: Any, real: Any, term_name: str):
        object.__setattr__(self, "_env", env)
        object.__setattr__(
            self, "scene", _GuardedScene(env.scene, real.scene, term_name)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


class _Swap:
    """Point a term's ``_env`` and its entity attributes at stand-ins for one pass."""

    def __init__(self, term: Any, env: Any, entity: Any, entity_attrs: list[str]):
        self._term = term
        self._env = env
        self._entity = entity
        self._attrs = entity_attrs

    def __enter__(self) -> None:
        self._saved = {a: getattr(self._term, a) for a in ["_env", *self._attrs]}
        self._term._env = self._env
        for a in self._attrs:
            setattr(self._term, a, self._entity)

    def __exit__(self, *exc: object) -> None:
        for a, v in self._saved.items():
            setattr(self._term, a, v)


def joint_targets(term: Any, name: str) -> list[str]:
    """The joints *term* drives, entity-prefixed, in the order ``processed`` lists them."""
    names = getattr(term, "target_names", None)
    if names is None:
        raise UntraceableAction(
            f"Action term {name!r} ({type(term).__name__}) names no joints it drives "
            "(`target_names`). The browser applies joint targets only, so it cannot "
            "run this term."
        )
    return [f"{term.cfg.entity_name}/{n}" for n in names]


def _entity_attrs(term: Any, name: str) -> tuple[str, list[str]]:
    """The entity the term drives, and every attribute holding it.

    Refuses an attribute holding another live object (the entity's data, another
    entity, the scene, the sim, the env, a sensor or a manager's term): the term cached
    it at init, so its reads would bypass the tracer and bake as constants.
    """
    env = term._env
    entity_name = term.cfg.entity_name
    entity = env.scene[entity_name]
    attrs: list[str] = []
    foreign: list[str] = []
    for attr, value in vars(term).items():
        if value is entity:
            attrs.append(attr)
        elif attr not in ("_env", "cfg") and _holds_live(value, frozenset({id(term)})):
            foreign.append(attr)
    if foreign:
        raise UntraceableAction(
            f"Action term {name!r} keeps {foreign}, which is or holds a live mjlab "
            f"object other than its entity {entity_name!r}, from its __init__. Its "
            "reads would bypass the tracer and freeze at their build-time values; read "
            "it through self._env or the entity in the body instead, via "
            "mjswan.register_action(..., ActionBinding(trace_override=...))."
        )
    return entity_name, attrs


@dataclass
class _Pass:
    """What one body read, wrote and drew during discovery."""

    log: list[tuple[SlotKey, Any]]
    recorder: _RecordingEnv
    written: set[str]
    draws: DrawRecorder


def _discover(
    term: Any,
    name: str,
    body: Body,
    raw: torch.Tensor,
    *,
    entity_name: str,
    entity_attrs: list[str],
    readers: Collection[str],
) -> _Pass:
    """Run *body* against recording proxies, every draw at its range's low end."""
    real_env = term._env
    real_data = real_env.scene[entity_name].data
    recorder = _RecordingEnv(real_env, readers)
    entity = _GuardedEntity(
        recorder.scene[entity_name], real_env.scene[entity_name], name
    )
    env = _GuardedEnv(recorder, real_env, name)
    attrs = set(vars(term))
    tensors = _tensor_versions(term)
    python = _python_state(term, {"_env", "cfg", *entity_attrs})
    buffers = _tensor_versions(real_data)
    sim = _sim_values(real_env)
    with (
        _Swap(term, env, entity, entity_attrs),
        DrawRecorder(*_bodies(term), at_lower_bounds=True) as draws,
        GatedIndexing(),
        MaskedRows(term.num_envs),
    ):
        _run_body(term, body, raw)
    where = _body_name(body)
    if created := sorted(set(vars(term)) - attrs):
        raise UntraceableAction(
            f"Action term {name!r} creates {created} in {where}. Create its state in "
            "__init__, as a tensor, so the build knows its starting value."
        )
    current = vars(term)
    if changed := sorted(
        k
        for k, v in python.items()
        if k not in current or not _same(v, _freeze(current[k]))
    ):
        raise UntraceableAction(
            f"Action term {name!r} changes {changed} in {where}: Python state a graph "
            "cannot carry, so the browser would keep its build-time value. Keep it in "
            "a tensor instead, via mjswan.register_action(..., "
            "ActionBinding(trace_override=...))."
        )
    sim_data = real_env.sim.data if sim else None
    if written := [
        *(
            f"entity.data.{k}"
            for k in sorted(_rewritten(buffers, _tensor_versions(real_data)))
        ),
        *(
            f"sim.data.{k}"
            for k, v in sim.items()
            if not _same(v, _sim_tensor(getattr(sim_data, k)))
        ),
    ]:
        raise _refuse_sim_write(name, f"writes {written}", where)
    return _Pass(
        list(recorder._log),  # noqa: SLF001
        recorder,
        _rewritten(tensors, _tensor_versions(term)),
        draws,
    )


def _body_name(body: Body) -> str:
    return "process_actions" if body == "process" else "reset"


def _run_body(term: Any, body: Body, raw: torch.Tensor | None) -> None:
    if body == "reset":
        term.reset(torch.arange(term.num_envs))
    else:
        term.process_actions(raw)


def _processed(term: Any) -> torch.Tensor:
    """mjlab's ``BaseAction._processed_actions``: what ``apply_actions`` applies."""
    value = getattr(term, "_processed_actions", None)
    if not isinstance(value, torch.Tensor):
        raise UntraceableAction(
            f"{type(term).__name__} keeps no `_processed_actions`, which the browser "
            "applies: derive it from mjlab's BaseAction."
        )
    return value


def _state_example(value: torch.Tensor) -> torch.Tensor:
    """A state field as the runtime feeds it: float32, or bool."""
    return value if value.dtype in _NATIVE_STATE_DTYPES else value.float()


class _ActionModule(nn.Module):
    """One body of an action term as a pure function of its slots and state."""

    def __init__(
        self,
        term: Any,
        name: str,
        body: Body,
        *,
        state_fields: list[str],
        state_dtypes: list[torch.dtype],
        entity_name: str,
        entity_attrs: list[str],
        dynamic_keys: list[SlotKey],
        constants: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any],
        commands: dict[str, Any],
        sim_rows: Any,
        readers: Collection[str],
        with_rand: bool,
        ramp_from: bool,
    ):
        super().__init__()
        self._term = term
        self._name = name
        self._body: Body = body
        self._state_fields = state_fields
        self._state_dtypes = state_dtypes
        self._entity_name = entity_name
        self._entity_attrs = entity_attrs
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._sim_rows = sim_rows
        self._readers = readers
        self._with_rand = with_rand
        self._ramp_from = ramp_from
        self._real_env = term._env
        self._const_buffers = _register_consts(self, constants)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        args_list = list(args)
        raw = args_list.pop(0) if self._body == "process" else None
        dynamic = args_list[: len(self._dynamic_keys)]
        state = args_list[len(self._dynamic_keys) :][: len(self._state_fields)]
        rand = args_list[-1] if self._with_rand else None

        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        _narrow_slots(slots, self._sim_rows)
        slots.update(_const_values(self, self._const_buffers))
        env = _ReplayEnv(
            slots,
            self._sensors,
            self._commands,
            real_env=self._real_env,
            reader_fields=self._readers,
        )
        entity = _GuardedEntity(
            env.scene[self._entity_name],
            self._real_env.scene[self._entity_name],
            self._name,
        )
        guarded = _GuardedEnv(env, self._real_env, self._name)
        with _Swap(self._term, guarded, entity, self._entity_attrs):
            for name, dtype, value in zip(
                self._state_fields, self._state_dtypes, state
            ):
                setattr(self._term, name, value.to(dtype))
            draws = (
                ReplayRng(*_bodies(self._term), rand=rand)
                if rand is not None
                else contextlib.nullcontext()
            )
            with draws, GatedIndexing(), MaskedRows(self._term.num_envs):
                start = (
                    _processed(self._term).clone()
                    if self._body == "process" and self._ramp_from
                    else None
                )
                _run_body(self._term, self._body, raw)
            outputs = [
                _state_example(getattr(self._term, f)).clone()
                for f in self._state_fields
            ]
            if self._body == "process":
                outputs.append(_processed(self._term).float().clone())
                if start is not None:
                    outputs.append(start.float())
        return tuple(outputs)


@dataclass
class ActionGraph:
    """One traced body: its graph, the slots it reads and the draws it takes."""

    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    output_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    input_shapes: list[list[int]] = field(default_factory=list)
    input_rows: list[list[int] | None] = field(default_factory=list)


@dataclass
class ActionExport:
    """A traced action term: its graphs, its state and the joints it drives."""

    name: str
    process: ActionGraph
    reset: ActionGraph | None
    state_fields: list[dict[str, Any]]
    """Per state field: ``{name, shape, dtype, init}``, as ``cfg.build(env)`` left it."""
    target_names: list[str]
    """The joints, entity-prefixed, in the order ``processed`` lists them."""
    ramp_from: bool


def _export_body(
    term: Any,
    name: str,
    body: Body,
    found: _Pass,
    *,
    state_fields: list[str],
    entity_name: str,
    entity_attrs: list[str],
    readers: Collection[str],
    raw: torch.Tensor,
    ramp_from: bool,
    opset: int,
) -> ActionGraph:
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    _classify_slots(found.log, dynamic, constants)
    if unserved := unserved_fields(dynamic):
        raise UntraceableAction(
            f"Action term {name!r} reads {unserved} in {_body_name(body)}, which the "
            "browser's slot reader does not serve, so the graph could never run. Read "
            "a field it serves (READER_FIELDS) instead, via mjswan.register_action(..., "
            "ActionBinding(trace_override=...))."
        )
    sim_rows = _narrow_inputs(dynamic, [found.recorder._sim])  # noqa: SLF001
    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    state_values = [getattr(term, f) for f in state_fields]
    with_rand = found.draws.rand_dim > 0

    module = _ActionModule(
        term,
        name,
        body,
        state_fields=state_fields,
        state_dtypes=[v.dtype for v in state_values],
        entity_name=entity_name,
        entity_attrs=entity_attrs,
        dynamic_keys=dynamic_keys,
        constants=constants,
        sensors=dict(found.recorder._sensors),  # noqa: SLF001
        commands=dict(found.recorder._commands),  # noqa: SLF001
        sim_rows=sim_rows,
        readers=readers,
        with_rand=with_rand,
        ramp_from=ramp_from,
    ).eval()

    prev_names = [f"prev_{f}" for f in state_fields]
    output_names = [f"next_{f}" for f in state_fields]
    example: list[torch.Tensor] = []
    names: list[str] = []
    if body == "process":
        example.append(raw)
        names.append("raw")
        output_names += ["processed", *(["processed_from"] if ramp_from else [])]
    example += [dynamic[k] for k in dynamic_keys]
    example += [_state_example(v) for v in state_values]
    names += input_names + prev_names
    if with_rand:
        example.append(found.draws.rand_vector)
        names.append("rand")

    snapshot = _snapshot_state(term)
    try:
        onnx_bytes = _export_onnx(
            module,
            tuple(example),
            input_names=names,
            output_names=output_names,
            # `rand` keeps its traced length: one flat draw vector, not rows.
            batch_axis=[n for n in names if n != "rand"] + output_names,
            opset=opset,
        )
    finally:
        _restore_state(term, snapshot)
    return ActionGraph(
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=input_names,
        output_names=output_names,
        rand_dim=found.draws.rand_dim,
        rand_ranges=found.draws.rand_ranges,
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        input_rows=_input_rows(dynamic_keys, sim_rows),
    )


def trace_action_term(
    term: Any,
    *,
    name: str,
    ramp_from: bool = False,
    opset: int = 17,
    reader_fields: Collection[str] | None = None,
) -> ActionExport:
    """Trace a built mjlab action term's ``process_actions`` and ``reset`` to ONNX.

    *term* comes from ``cfg.build(env)`` on a single-env trace env and is left as it was
    found. *ramp_from* adds the ``processed_from`` output an interpolating apply starts
    from. Raises :class:`UntraceableAction` for a body the browser could not reproduce.
    """
    _prepare_single_env_export(term.num_envs)
    readers = READER_FIELDS if reader_fields is None else frozenset(reader_fields)
    targets = joint_targets(term, name)
    entity_name, entity_attrs = _entity_attrs(term, name)
    if term.action_dim != len(targets):
        raise UntraceableAction(
            f"Action term {name!r} takes {term.action_dim} action value(s) but drives "
            f"{len(targets)} joint(s). The browser feeds a traced term one action per "
            "joint it drives."
        )
    views = _views(term)

    generator = torch.Generator().manual_seed(0)
    raw = torch.rand((term.num_envs, term.action_dim), generator=generator) * 2 - 1
    snapshot = _snapshot_state(term)
    try:
        passes = {
            body: _discover(
                term,
                name,
                body,
                raw,
                entity_name=entity_name,
                entity_attrs=entity_attrs,
                readers=readers,
            )
            for body in ("reset", "process")
        }
    finally:
        _restore_state(term, snapshot)
    state_fields = sorted(passes["reset"].written | passes["process"].written)
    if shared := [names for names in views if set(names) & set(state_fields)]:
        raise UntraceableAction(
            f"Action term {name!r} keeps {shared[0]} as views of one tensor. The "
            "browser carries each state field as a tensor of its own, so a write "
            "through one would not reach the other; give each its own tensor "
            "(`.clone()`)."
        )

    graphs = {
        body: _export_body(
            term,
            name,
            body,
            passes[body],
            state_fields=state_fields,
            entity_name=entity_name,
            entity_attrs=entity_attrs,
            readers=readers,
            raw=raw,
            ramp_from=ramp_from,
            opset=opset,
        )
        for body in ("reset", "process")
        # A reset that writes no state is mjlab's no-op for this term: no graph.
        if body == "process" or passes["reset"].written
    }
    reset = graphs.get("reset")
    if reset is not None and (_ENV_NS, "episode_length_buf") in reset.input_slots:
        raise UntraceableAction(
            f"Action term {name!r} reads env.episode_length_buf in reset. mjlab zeroes "
            "the clock after its reset terms and the browser before them, so the reset "
            "would see a different value; read it in process_actions instead."
        )
    export = ActionExport(
        name=name,
        process=graphs["process"],
        reset=reset,
        state_fields=[
            _state_spec(f, _state_example(getattr(term, f))) for f in state_fields
        ],
        target_names=targets,
        ramp_from=ramp_from,
    )
    _verify(export, term)
    return export


def compare_body(
    term: Any,
    export: ActionExport,
    body: Body,
    run_graph: GraphRunner,
    *,
    raw: torch.Tensor | None = None,
    rand: torch.Tensor | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> str | None:
    """One body through its graph and through mjlab, from the term's current state;
    what disagreed, or ``None``. Leaves the term in mjlab's next state."""
    graph = export.process if body == "process" else export.reset
    assert graph is not None
    env = term._env
    feeds: dict[str, np.ndarray] = {
        f"prev_{spec['name']}": _state_example(getattr(term, spec["name"]))
        .detach()
        .cpu()
        .numpy()
        for spec in export.state_fields
    }
    for key, input_name, rows in zip(
        graph.input_slots, graph.input_names, graph.input_rows
    ):
        feeds[input_name] = read_slot(env, key, rows).detach().cpu().numpy()
    if raw is not None:
        feeds["raw"] = raw.detach().cpu().numpy().astype(np.float32)
    if rand is not None:
        feeds["rand"] = rand.detach().cpu().numpy().astype(np.float32)
    start = _processed(term).clone() if body == "process" else None
    outputs = run_graph(feeds)

    draws = (
        ReplayRng(*_bodies(term), rand=rand)
        if rand is not None
        else contextlib.nullcontext()
    )
    with draws:
        _run_body(term, body, raw)

    def differs(label: str, got: np.ndarray, want: torch.Tensor) -> str | None:
        return _mismatch(label, got, _state_example(want), atol=atol, rtol=rtol)[1]

    for spec in export.state_fields:
        note = differs(
            spec["name"], outputs[f"next_{spec['name']}"], getattr(term, spec["name"])
        )
        if note:
            return note
    if body == "process":
        if note := differs("processed", outputs["processed"], _processed(term)):
            return note
        if export.ramp_from and start is not None:
            return differs("processed_from", outputs["processed_from"], start)
    return None


#: Episode clocks the check walks through: both sides of any hold up to 20 s at 50 Hz.
_EPISODE_STEPS = (0, 1, 2, 3, 10, 49, 50, 51, 100, 1000)


def _probe(graph: ActionGraph, generator: torch.Generator) -> torch.Tensor | None:
    if graph.rand_dim == 0:
        return None
    ranges = torch.tensor(graph.rand_ranges, dtype=torch.float32).reshape(-1, 2)
    low, high = ranges[:, 0], ranges[:, 1]
    return low + (high - low) * torch.rand(len(low), generator=generator)


def _verify(export: ActionExport, term: Any) -> None:
    """Refuse a graph that disagrees with the term's own body.

    Runs reset, then a process step at each of :data:`_EPISODE_STEPS`, then a reset and
    a few more steps, chaining state through mjlab's bodies, with random actions in
    [-1.5, 1.5] and draws inside their ranges. A branch the trace took one way and the
    body takes another (a Python ``if`` on a tensor) shows up here.
    """
    env = term._env
    process = reference_runner(export.process)
    reset = reference_runner(export.reset) if export.reset is not None else None
    generator = torch.Generator().manual_seed(1)
    snapshot = _snapshot_state(term)
    found = getattr(env, "episode_length_buf", None)
    clock = found if isinstance(found, torch.Tensor) else None
    saved_clock = clock.clone() if clock is not None else None

    def step(episode: int) -> str | None:
        if clock is not None:
            clock[:] = episode
        raw = (
            torch.rand((term.num_envs, term.action_dim), generator=generator) * 3 - 1.5
        )
        rand = _probe(export.process, generator)
        return compare_body(term, export, "process", process, raw=raw, rand=rand)

    def restart() -> str | None:
        if reset is None or export.reset is None:
            _run_body(term, "reset", None)
            return None
        rand = _probe(export.reset, generator)
        return compare_body(term, export, "reset", reset, rand=rand)

    try:
        plan: list[tuple[str, Any]] = [("reset", restart)]
        plan += [(f"process at step {s}", lambda s=s: step(s)) for s in _EPISODE_STEPS]
        plan += [("reset", restart)]
        plan += [(f"process at step {s}", lambda s=s: step(s)) for s in (0, 1, 2)]
        for label, run in plan:
            if note := run():
                raise UntraceableAction(
                    f"Action term {export.name!r} traced to a graph that disagrees with "
                    f"its own body ({label}: {note}). A branch the trace cannot follow, "
                    "such as a Python `if` on a tensor, is the likely cause; supply a "
                    "trace-friendly body via mjswan.register_action(..., "
                    "ActionBinding(trace_override=...))."
                )
    finally:
        _restore_state(term, snapshot)
        if clock is not None and saved_clock is not None:
            clock[:] = saved_clock


__all__ = [
    "ActionExport",
    "ActionGraph",
    "GraphRunner",
    "UntraceableAction",
    "compare_body",
    "joint_targets",
    "reference_runner",
    "trace_action_term",
]
