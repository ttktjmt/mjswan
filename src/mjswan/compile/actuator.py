"""An actuator's torque law: its ``compute`` traced to a graph the browser evaluates
synchronously every physics substep.

The browser's evaluator runs only elementwise operators over per-joint vectors and
scalars (``SUPPORTED_OPS``), so a graph using any other is refused, naming it. The graph
holds no state: it is checked against the live ``compute`` over a run of calls, so a law
that keeps state between calls (a filter, a history) disagrees with it and is refused.

See ``context/tracing.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .export import _export_onnx, check_against_module

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


def trace_actuator(
    actuator: Any,
    width: int,
    *,
    what: str,
    compute: _Compute | None = None,
    opset: int = 17,
) -> ActuatorExport:
    """Trace *actuator*'s torque law over *width* joints, or *compute* in its place.

    Raises :class:`UnsupportedActuator` for a law that does not export or uses an
    operator outside :data:`SUPPORTED_OPS`, and ``DivergentGraph`` for one the graph
    does not reproduce, as a law keeping state between calls does not.
    """
    import onnx

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
    ops = tuple(
        sorted({n.op_type for n in onnx.load_from_string(onnx_bytes).graph.node})
    )
    if unsupported := [op for op in ops if op not in SUPPORTED_OPS]:
        raise UnsupportedActuator(
            f"{what} traces to {', '.join(unsupported)}, which the browser's actuator "
            "evaluator does not run: it runs elementwise operators only, every physics "
            f"substep. {_WAYS_OUT}"
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
