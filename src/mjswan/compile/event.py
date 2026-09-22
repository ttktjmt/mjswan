"""Trace an event term body to ONNX.

An event returns ``None`` and writes via ``entity.write_*_to_sim``, so the tensors it
would write become the graph outputs, and its randomness is threaded in as an explicit
``rand`` input replayed by :class:`.rng.ReplayRng`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
    _flatten_captures,
    _write_output_name,
)
from .replay import _EventReplayEnv
from .rng import DrawRecorder, ReplayRng
from .slot import SlotKey, TaggedKey, _slot_input_name


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
    """Per-element ``[low, high]`` for ``rand``, the runtime draws with these."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    """Per write-kind descriptor: what the outputs target (entity, kind, fields)."""
    reference_outputs: tuple[torch.Tensor, ...]
    reference_rand: torch.Tensor
    constant_slots: list[str] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots`` (see :func:`slots_json`)."""


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
