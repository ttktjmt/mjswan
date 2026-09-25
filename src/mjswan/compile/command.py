"""Trace a stateful ``CommandTerm`` to ONNX.

A command's hidden state is promoted to explicit graph I/O, so ``_resample_command`` +
``_update_command`` trace as one pure function::

    forward(prev_state..., resample_mask, rand) -> (next_state..., entity_write?)

with the runtime holding ``state`` across frames and owning the resample timer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from .export import (
    _classify_tagged,
    _const_values,
    _export_onnx,
    _prepare_single_env_export,
    _register_consts,
)
from .record import (
    _WRITE_FIELDS,
    WriteCaptures,
    _EventCaptureEnv,
    _EvRecEntity,
    _flatten_captures,
    _write_output_name,
)
from .replay import _EventReplayEnv, _EvReplayEntity
from .rng import DrawRecorder, ReplayRng
from .slot import SlotKey, TaggedKey, _slot_input_name

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

    The mask gates the state fields only: captured writes are a fresh draw every call
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
                self._term._update_command(None)
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
    """Per state field: ``{name, shape, dtype}``, written to the manifest entry."""
    command_field: str
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` the runtime draws ``rand`` from."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    reference_rand: torch.Tensor
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots``."""


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
    ``rand``. Only ``sample_uniform`` draws are supported: a term using tensor-method
    RNG (``Tensor.uniform_``) needs a trace-friendly override.
    """
    entity_attr_names = _entity_attrs(term)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    snap = _snapshot_state(term)
    state_example = tuple(getattr(term, f).detach().clone() for f in state_fields)

    with _RecordCommand(term, entity_attr_names, entity_name) as rec_env:
        with DrawRecorder(term._resample_command) as rec:
            term._resample_command(torch.arange(term.num_envs))
            term._update_command(None)
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

    dynamic, tensor_consts, scalar_consts = _classify_tagged(log)

    dynamic_keys = sorted(dynamic)
    dyn_names = [_slot_input_name(k) for k in dynamic_keys]
    prev_names = [f"prev_{f}" for f in state_fields]

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
                # Plain JSON values; the reader rebuilds the typed array from `dtype`.
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
