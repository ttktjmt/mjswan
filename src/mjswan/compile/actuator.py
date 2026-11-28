"""An actuator's torque law: its ``compute`` traced to a graph the browser evaluates
synchronously every physics substep.

The browser's evaluator runs only elementwise operators over per-joint vectors and
scalars (``SUPPORTED_OPS``) of floats and bools, so a graph using any other is refused,
naming it. The graph holds no state: it is checked against the live ``compute`` over a
run of calls, so a law that keeps state between calls (a filter, a history) disagrees
with it and is refused, as is an actuator with its own ``update``, which mjlab calls
every physics step and the check never does.

See ``context/tracing.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .export import _export_onnx, _prepare_single_env_export, check_against_module

#: ``(actuator, cmd) -> torque``: ``managers.action_manager.ActuatorLaw``.
_Compute = Callable[[Any, Any], Any]

#: What the graph reads per joint, in ``ActuatorCmd``'s order, and what it writes.
INPUTS: tuple[str, ...] = ("pos", "vel", "pos_target", "vel_target", "effort_target")
OUTPUT = "torque"

#: The operators ``template/src/core/actuator/evaluator.ts`` runs. Keep the two in step:
#: ``dump_actuator_fixture.py`` hands this set to the evaluator's test.
SUPPORTED_OPS: frozenset[str] = frozenset(
    {
        "Abs",
        "Add",
        "And",
        "Cast",
        "Clip",
        "Constant",
        "Div",
        "Equal",
        "Exp",
        "Greater",
        "GreaterOrEqual",
        "Identity",
        "Less",
        "LessOrEqual",
        "Max",
        "Min",
        "Mul",
        "Neg",
        "Not",
        "Or",
        "Pow",
        "Sign",
        "Sqrt",
        "Sub",
        "Tanh",
        "Where",
    }
)

#: The example's spread per input: joint speeds past a torque-speed envelope's corners.
_SPREAD = {"pos": 3.0, "vel": 60.0, "pos_target": 3.0, "vel_target": 20.0}
_EFFORT_SPREAD = 50.0

_WAYS_OUT = (
    "Register a law the browser can run with mjswan.register_actuator(<cfg class "
    "name>, compute), written with elementwise tensor operations, or drive these joints "
    "with one of mjlab's builtin actuators."
)


class UnsupportedActuator(ValueError):
    """An actuator whose torque law the browser cannot run."""


@dataclass(frozen=True)
class ActuatorExport:
    """One actuator's traced law."""

    onnx_bytes: bytes
    ops: tuple[str, ...]
    output_names: tuple[str, ...] = (OUTPUT,)


class _Law(nn.Module):
    """*compute* of *actuator* over the five per-joint command fields."""

    def __init__(self, actuator: Any, compute: _Compute) -> None:
        super().__init__()
        self._actuator = actuator
        self._compute = compute

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        pos_target: torch.Tensor,
        vel_target: torch.Tensor,
        effort_target: torch.Tensor,
    ) -> torch.Tensor:
        from mjlab.actuator.actuator import ActuatorCmd

        cmd = ActuatorCmd(
            position_target=pos_target,
            velocity_target=vel_target,
            effort_target=effort_target,
            pos=pos,
            vel=vel,
        )
        return self._compute(self._actuator, cmd)


def _own_compute(actuator: Any, cmd: Any) -> torch.Tensor:
    return actuator.compute(cmd)


def _example(width: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(0)

    def draw(spread: float) -> torch.Tensor:
        return (torch.rand(1, width, generator=generator) * 2 - 1) * spread

    return (*(draw(_SPREAD[name]) for name in INPUTS[:-1]), draw(_EFFORT_SPREAD))


def _has_own_update(actuator: Any) -> bool:
    from mjlab.actuator.actuator import Actuator

    return getattr(type(actuator), "update", Actuator.update) is not Actuator.update


def _evaluator_refusal(model: Any, width: int) -> str | None:
    """Why ``evaluator.ts`` would refuse *model* over *width* joints past its operator
    set, or ``None``: it reads float and bool tensors, each a scalar or one value per
    joint, and casts to those two types only."""
    from onnx import TensorProto

    kinds = (TensorProto.FLOAT, TensorProto.BOOL)

    def name(data_type: int) -> str:
        return TensorProto.DataType.Name(data_type).lower()

    def shape_refusal(dims: list[int], what: str) -> str | None:
        count = math.prod(dims)
        per_joint = (
            count == width
            and sum(d != 1 for d in dims) <= 1
            and bool(dims)
            and dims[-1] == width
        )
        if count == 1 or per_joint:
            return None
        return f"{what} has shape {dims}, neither a scalar nor one value per joint"

    def tensor_refusal(tensor: Any, what: str) -> str | None:
        if tensor.data_type not in kinds:
            return (
                f"{what} holds {name(tensor.data_type)} values, where it reads floats "
                "and bools only (write 1.0 rather than 1, and keep tensors float32)"
            )
        return shape_refusal(list(tensor.dims), what)

    for tensor in model.graph.initializer:
        if why := tensor_refusal(tensor, f"initializer {tensor.name!r}"):
            return why
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            return f"a {node.op_type} node is in operator domain {node.domain!r}"
        attrs = {a.name: a for a in node.attribute}
        if node.op_type == "Constant":
            if "value" in attrs:
                why = tensor_refusal(attrs["value"].t, "a Constant")
            elif "value_floats" in attrs:
                why = shape_refusal([len(attrs["value_floats"].floats)], "a Constant")
            elif "value_float" in attrs:
                why = None
            else:
                why = "a Constant holds neither floats nor bools"
            if why:
                return why
        elif node.op_type == "Cast" and attrs["to"].i not in kinds:
            return (
                f"a Cast makes {name(attrs['to'].i)} values, where it casts to float "
                "and bool only (keep tensors float32)"
            )
    return None


def trace_actuator(
    actuator: Any,
    width: int,
    *,
    what: str,
    compute: _Compute | None = None,
    num_envs: int = 1,
    opset: int = 17,
) -> ActuatorExport:
    """Trace *actuator*'s torque law over *width* joints, or *compute* in its place.

    Raises :class:`UnsupportedActuator` for an actuator with its own ``update``, a law
    that does not export, or a graph the browser's evaluator would refuse, and
    ``DivergentGraph`` for one the graph does not reproduce, as a law keeping state
    between calls does not. *num_envs* is the trace env's, which must be one.
    """
    import onnx

    _prepare_single_env_export(num_envs)
    if compute is None and _has_own_update(actuator):
        raise UnsupportedActuator(
            f"{what} has its own update(), which mjlab calls after every physics step "
            "and the browser's stateless law cannot: any state it advances would be "
            f"frozen into the graph. {_WAYS_OUT}"
        )
    law = _Law(actuator, compute or _own_compute)
    example = _example(width)
    try:
        onnx_bytes = _export_onnx(
            law,
            example,
            input_names=list(INPUTS),
            output_names=[OUTPUT],
            batch_axis=[],
            opset=opset,
        )
    except Exception as exc:
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        raise UnsupportedActuator(
            f"{what} does not trace to a graph ({type(exc).__name__}: {reason}). "
            f"{_WAYS_OUT}"
        ) from exc
    model = onnx.load_from_string(onnx_bytes)
    ops = tuple(sorted({n.op_type for n in model.graph.node}))
    if unsupported := [op for op in ops if op not in SUPPORTED_OPS]:
        raise UnsupportedActuator(
            f"{what} traces to {', '.join(unsupported)}, which the browser's actuator "
            "evaluator does not run: it runs elementwise operators only, every physics "
            f"substep. {_WAYS_OUT}"
        )
    if why := _evaluator_refusal(model, width):
        raise UnsupportedActuator(
            f"{what} traces to a graph the browser's actuator evaluator does not run: "
            f"{why}. {_WAYS_OUT}"
        )
    check_against_module(
        law,
        onnx_bytes,
        example,
        input_names=INPUTS,
        output_names=[OUTPUT],
        what=what,
    )
    return ActuatorExport(onnx_bytes=onnx_bytes, ops=ops)


__all__ = [
    "INPUTS",
    "OUTPUT",
    "SUPPORTED_OPS",
    "ActuatorExport",
    "UnsupportedActuator",
    "trace_actuator",
]
