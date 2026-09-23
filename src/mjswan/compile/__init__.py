"""Build-time tracing of mjlab MDP term bodies to ONNX graphs, plus a numeric parity
harness validating the exported graphs against the live mjlab env (ADR 0005).

One pass records what a term reads (:mod:`.record`), a second replays those reads
while torch traces the body (:mod:`.replay`); :mod:`.slot` names the reads and
:mod:`.export` holds the mechanics every tracer shares. :mod:`.term`, :mod:`.event`,
:mod:`.command` and :mod:`.group` trace one kind of term each, and :mod:`.native` names
the terms that need no graph. Files are written by :mod:`mjswan.build.mdp`.
"""

from __future__ import annotations

from .command import CommandExport, trace_command_term
from .event import EventExport, trace_event_term
from .parity import ParityReport, TermReport, run_command_parity, run_parity
from .rng import DrawRecorder, ReplayRng
from .term import TermExport, trace_term

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
