"""Trace one value-returning term body to ONNX, and the two ways that can fail.

A term is ``func(env, **params)`` returning a tensor. It is run once against the
recording env to discover its reads, those are classified into graph inputs and baked
constants, and the term is exported as an ``nn.Module`` whose ``forward`` takes the
dynamic tensors. A term that read nothing is a :class:`ConstantTerm`; one whose reads
the tracer could not follow into a tensor is an :class:`UntraceableTerm`.
"""

from __future__ import annotations

import warnings
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
from .record import _RecordingEnv
from .replay import _ReplayEnv
from .slot import (
    _TERM_ENV_READS,
    READER_FIELDS,
    SimRows,
    SlotKey,
    _reader_fields,
    _slot_input_name,
    slot_label,
)


class ConstantTerm(ValueError):
    """A term that read no simulation state at all — its value is a constant.

    An observation so shaped is env-independent (a fixed-size padding term, say), so a
    caller may bake the value; a termination so shaped fires every step or never, so a
    caller must refuse it. Distinct from :class:`UntraceableTerm` because the two look
    identical from "no graph inputs" alone and must not be handled alike.
    """

    def __init__(self, term: str):
        self.term = term
        super().__init__(
            f"Term {term!r} reads no simulation state at all; its value is a "
            "constant (ADR 0005)."
        )


def warn_constant_observation(name: str, size: int) -> None:
    """Name a baked term: right for a padding term, wrong for anything else, and the
    tracer cannot tell the two apart."""
    warnings.warn(
        f"Observation term {name!r} reads no simulation state, so its {size} value(s) "
        "are baked into the build and the policy sees the same numbers every step. "
        "Right for a fixed padding term; a term meant to follow the simulation reads "
        f"it by a path the tracer does not serve ({', '.join(_TERM_ENV_READS)}).",
        category=RuntimeWarning,
        stacklevel=3,
    )


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
    input_rows: list[list[int] | None] = field(default_factory=list)
    """Per input slot, the rows of the raw field it carries; None for a whole field."""


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
        reader_fields: Collection[str] = READER_FIELDS,
        sim_rows: SimRows | None = None,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._dynamic_keys = dynamic_keys
        self._sensors = sensors or {}
        self._commands = commands or {}
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
        return self._func(env, **self._params)


def trace_term(
    func: Callable[..., torch.Tensor],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    opset: int = 17,
    reader_fields: Collection[str] | None = None,
) -> TermExport:
    """Trace a value-returning mjlab term body to ONNX against a live ``env``.

    ``params`` come from the env's own manager (``asset_cfg`` already resolved to
    static indices) and ``env`` must be post-reset. ``reader_fields`` names the
    ``EntityData`` fields served as value slots; see :data:`READER_FIELDS`.

    Raises:
        ConstantTerm: the term reads no simulation state; its value is a constant.
        UntraceableTerm: the term reads state the tracer cannot follow.
    """
    readers = _reader_fields(reader_fields)
    # 1. Discovery: run once against the recording env.
    recorder = _RecordingEnv(env, readers)
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
        raise ConstantTerm(name)

    sim_rows = _narrow_inputs(dynamic, [recorder._sim])  # noqa: SLF001
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
        reader_fields=readers,
        sim_rows=sim_rows,
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
        input_rows=_input_rows(dynamic_keys, sim_rows),
    )
