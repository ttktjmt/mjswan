"""Fuse an observation group, or a set of terminations, into one ONNX graph.

One graph per group rather than one per term, since a per-term graph can be a single
node and the fixed per-``ort.run()`` cost then dominates (ADR 0005 §4). The terms share
one replay env, so a slot two of them read is marshalled once; native terms become
graph inputs the runtime feeds, and a constant observation is baked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Collection

import torch
from torch import nn

from .export import (
    _classify_slots,
    _const_values,
    _export_onnx,
    _input_rows,
    _narrow_inputs,
    _narrow_slots,
    _register_consts,
)
from .native import native_observation_entry
from .record import _RecordingEnv, _RecordingSimData
from .replay import _ReplayEnv
from .slot import (
    READER_FIELDS,
    SimRows,
    SlotKey,
    _reader_fields,
    _slot_input_name,
    slot_label,
)
from .term import ConstantTerm, UntraceableTerm, warn_constant_observation


class ConstantGroup(ValueError):
    """Every term in a group is native or constant, so the group has no graph.

    Not an error: the caller falls back to the per-term path rather than fusing an
    empty graph.
    """


@dataclass
class GroupTermSpec:
    """One term inside a fused group, as :func:`trace_observation_group` needs it."""

    name: str
    func: Callable[..., torch.Tensor]
    params: dict[str, Any]
    clip: tuple[float, float] | None = None
    scale: Any = None
    """Per-term scale: a float, or a sequence broadcast over the term's width."""
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
    """Per native term: ``{name, native, input, size, ...}``, fed by the runtime."""
    layout: list[dict[str, Any]]
    """``{name, size}`` per term, in concat order, for the runtime's group layout."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)
    input_rows: list[list[int] | None] = field(default_factory=list)


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
        reader_fields: Collection[str] = READER_FIELDS,
        sim_rows: SimRows | None = None,
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._native_names = native_names
        self._real_env = real_env
        self._reader_fields = reader_fields
        self._sim_rows = sim_rows or {}
        self._const_buffers = _register_consts(self, constants)
        # A term reading no dynamic state is a value, not a function: bake it.
        self._baked_buffers = _register_consts(self, baked, prefix="_baked")

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        split = len(self._dynamic_keys)
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, args[:split]))
        _narrow_slots(slots, self._sim_rows)
        slots.update(_const_values(self, self._const_buffers))
        native = dict(zip(self._native_names, args[split:]))
        env = _ReplayEnv(
            slots,
            self._sensors,
            self._commands,
            real_env=self._real_env,
            reader_fields=self._reader_fields,
        )

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
    except Exception:  # noqa: BLE001 (a trace env legitimately has neither)
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
        "`last_action` has no config answer at all: it needs a trace env whose "
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
    reader_fields: Collection[str] | None = None,
) -> GroupExport:
    """Fuse an observation group's terms into one ONNX graph.

    One graph per group rather than one per term, since a per-term graph can be a
    single node and the fixed per-``ort.run()`` cost then dominates (ADR 0005 §4).

    Inputs are the deduplicated union of the terms' dynamic slots, then one input per
    native term. The output is the concatenated vector with clip/scale folded in,
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
    sims: list[_RecordingSimData] = []
    readers = _reader_fields(reader_fields)

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

        recorder = _RecordingEnv(env, readers)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Observation term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = _classify_slots(recorder._log, dynamic, constants)  # noqa: SLF001
        sensors.update(recorder._sensors)  # noqa: SLF001 (internal proxy)
        commands.update(recorder._commands)  # noqa: SLF001 (internal proxy)
        sims.append(recorder._sim)  # noqa: SLF001 (internal proxy)
        size = int(recorded.reshape(1, -1).shape[-1])
        if not term_dynamic:
            # Nothing read means a constant; unfollowable reads mean live state.
            if recorder._log:  # noqa: SLF001 (internal proxy)
                raise UntraceableTerm(
                    term.name,
                    sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
                )
            warn_constant_observation(term.name, size)
            baked[term.name] = recorded.detach()
        layout.append({"name": term.name, "size": size})

    if not dynamic:
        raise ConstantGroup(
            f"Observation group {name!r} reads no time-varying state; every term is "
            "native or constant, so there is no graph to run."
        )

    # 2. Fuse and export. Slots sorted for determinism, then natives in declaration
    #    order.
    sim_rows = _narrow_inputs(dynamic, sims)
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
        reader_fields=readers,
        sim_rows=sim_rows,
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
        input_rows=_input_rows(dynamic_keys, sim_rows),
    )


@dataclass
class TerminationGroupExport:
    """The result of fusing a set of termination terms into a single ONNX graph."""

    name: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    input_shapes: list[list[int]]
    lanes: list[str]
    """Term names, in output-lane order: lane *i* is `lanes[i]`'s verdict."""
    output_name: str
    reference_output: torch.Tensor
    constant_slots: list[SlotKey] = field(default_factory=list)
    input_rows: list[list[int] | None] = field(default_factory=list)


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
        reader_fields: Collection[str] = READER_FIELDS,
        sim_rows: SimRows | None = None,
    ):
        super().__init__()
        self._terms = terms
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors
        self._commands = commands
        self._real_env = real_env
        self._reader_fields = reader_fields
        self._sim_rows = sim_rows or {}
        self._const_buffers = _register_consts(self, constants)

    def forward(self, *dynamic: torch.Tensor) -> torch.Tensor:
        slots: dict[SlotKey, torch.Tensor] = dict(zip(self._dynamic_keys, dynamic))
        _narrow_slots(slots, self._sim_rows)
        slots.update(_const_values(self, self._const_buffers))
        env = _ReplayEnv(
            slots,
            self._sensors,
            self._commands,
            real_env=self._real_env,
            reader_fields=self._reader_fields,
        )
        lanes = [term.func(env, **term.params).reshape(-1, 1) for term in self._terms]
        return torch.cat(lanes, dim=-1)


def trace_termination_group(
    terms: list[GroupTermSpec],
    env: Any,
    *,
    name: str,
    opset: int = 17,
    reader_fields: Collection[str] | None = None,
) -> TerminationGroupExport:
    """Fuse termination terms into one graph, one bool lane each.

    Same mechanics as :func:`trace_observation_group`, but the output is a bool vector
    so the manager keeps its per-term reasons. `time_out` never reaches here: it is
    classified native by name first (:func:`is_native_termination`).
    """
    dynamic: dict[SlotKey, torch.Tensor] = {}
    constants: dict[SlotKey, torch.Tensor] = {}
    sensors: dict[str, Any] = {}
    commands: dict[str, Any] = {}
    sims: list[_RecordingSimData] = []
    readers = _reader_fields(reader_fields)

    for term in terms:
        recorder = _RecordingEnv(env, readers)
        recorded = term.func(recorder, **term.params)
        if not isinstance(recorded, torch.Tensor):
            raise ValueError(
                f"Termination term {term.name!r} returned "
                f"{type(recorded).__name__}, not a Tensor."
            )
        term_dynamic = _classify_slots(recorder._log, dynamic, constants)  # noqa: SLF001
        sensors.update(recorder._sensors)  # noqa: SLF001 (internal proxy)
        commands.update(recorder._commands)  # noqa: SLF001 (internal proxy)
        sims.append(recorder._sim)  # noqa: SLF001 (internal proxy)
        if not term_dynamic:
            # Never baked: a termination blind to state never fires or always does.
            if not recorder._log:  # noqa: SLF001 (internal proxy)
                raise ConstantTerm(term.name)
            raise UntraceableTerm(
                term.name,
                sorted({slot_label(k) for k, _ in recorder._log}),  # noqa: SLF001
            )

    if not dynamic:
        raise ValueError(
            f"Termination group {name!r} reads no time-varying state; every term "
            "should be native (e.g. time_out)."
        )

    sim_rows = _narrow_inputs(dynamic, sims)
    dynamic_keys = sorted(dynamic)
    input_names = [_slot_input_name(k) for k in dynamic_keys]
    example_inputs = tuple(dynamic[k] for k in dynamic_keys)

    module = _TerminationGroupModule(
        terms,
        dynamic_keys,
        constants,
        sensors=sensors,
        commands=commands,
        real_env=env,
        reader_fields=readers,
        sim_rows=sim_rows,
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
        input_rows=_input_rows(dynamic_keys, sim_rows),
    )
