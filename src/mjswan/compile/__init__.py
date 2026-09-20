"""Build-time tracing of mjlab MDP term bodies to ONNX graphs, plus a numeric parity
harness validating the exported graphs against the live mjlab env (ADR 0005)."""

from __future__ import annotations

from .parity import ParityReport, TermReport, run_command_parity, run_parity
from .rng import DrawRecorder, ReplayRng
from .tracer import (
    CommandExport,
    EventExport,
    TermExport,
    trace_command_term,
    trace_event_term,
    trace_term,
)

__all__ = [
    "TermExport",
    "EventExport",
    "CommandExport",
    "trace_term",
    "trace_event_term",
    "trace_command_term",
    "run_parity",
    "run_command_parity",
    "ParityReport",
    "TermReport",
    "DrawRecorder",
    "ReplayRng",
]
