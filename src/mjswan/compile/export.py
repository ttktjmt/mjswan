"""Shared trace mechanics: classifying reads, narrowing inputs, baking constants, and
the ``torch.onnx.export`` call itself."""

from __future__ import annotations

import io
import warnings
from typing import Any, Sequence

import torch
from torch import nn

from .record import _merge_narrowing, _RecordingSimData
from .replay import _narrowed_field
from .slot import (
    _COMMAND_NS,
    _SENSOR_NS,
    _SIM_NS,
    SimRows,
    SlotKey,
    TaggedKey,
    _is_dynamic_field,
)


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


def _narrow_slots(slots: dict[SlotKey, torch.Tensor], sim_rows: SimRows) -> None:
    """Hand the replay the narrowed sim inputs as fields that remap the term's ids."""
    for key, (rows, size) in sim_rows.items():
        slots[key] = _narrowed_field(slots[key], rows, size)


def _narrow_inputs(
    dynamic: dict[SlotKey, torch.Tensor], sims: Sequence[_RecordingSimData]
) -> SimRows:
    """Narrow each sim example input to the rows the reads touched; see
    :func:`_merge_narrowing`. Returns what was narrowed, for the module and the export."""
    sim_rows = _merge_narrowing(sims)
    for key, (rows, _size) in sim_rows.items():
        dynamic[key] = dynamic[key][:, rows]
    return sim_rows


def _input_rows(
    dynamic_keys: Sequence[SlotKey], sim_rows: SimRows
) -> list[list[int] | None]:
    """Per input slot, the rows it carries, or None for a whole field."""
    return [sim_rows[k][0] if k in sim_rows else None for k in dynamic_keys]


def _classify_slots(
    log: list[tuple[SlotKey, Any]],
    dynamic: dict[SlotKey, torch.Tensor],
    constants: dict[SlotKey, torch.Tensor],
) -> bool:
    """Split a recorded read log into graph inputs and baked constants.

    Sensor, command-state and raw sim-data reads are live state by definition; an entity
    data field is dynamic unless it is a model-derived constant. Returns whether *this* log
    contributed a dynamic slot, which a group's caller needs per term.
    """
    saw_dynamic = False
    for key, value in log:
        if not isinstance(value, torch.Tensor):
            continue  # non-tensor attribute access, not a graph slot
        namespace, field_name = key
        if namespace in (_SENSOR_NS, _COMMAND_NS, _SIM_NS) or _is_dynamic_field(
            field_name
        ):
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
