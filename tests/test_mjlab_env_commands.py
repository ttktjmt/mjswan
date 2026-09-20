"""Tracing a term that reads a command the browser owns.

Layer: L2 (real tracer, hand-built env — no mjlab env needed).

Some commands have no Python side to trace against: a `UiCommand` is a slider, and
a native `TrackingCommand` is a clip lookup (data, not math). A term that does
arithmetic on one is still an ordinary traced body, and its reads have to become
graph *inputs* the runtime serves from `getStateField`. `TraceCommandManager` is
what lets the discovery pass see them at all; without it the read raises and the
term looks untraceable.

What must hold: each read lands as a `{command, field}` slot with the shape the
stand-in had (a look-ahead window is rank 3, and the runtime needs the rank to feed
its flat value array), and nothing gets baked — a frozen command value is a policy
that ignores its own controls.
"""

from __future__ import annotations

import numpy as np
import pytest

# `mjswan.compile` imports torch, so the package import below waits for the skip.
torch = pytest.importorskip("torch")

from mjswan.compile import trace_term  # noqa: E402
from mjswan.compile.slot import slots_json  # noqa: E402
from mjswan.mjlab.env import TraceCommandManager  # noqa: E402

STEPS = 3
JOINTS = 2


class _RefWindow:
    """A tracking command's look-ahead window, as the browser would serve it."""

    def __init__(self):
        self.ref_root_pos_w = torch.arange(STEPS * 3, dtype=torch.float32).reshape(
            1, STEPS, 3
        )
        self.ref_joint_pos = torch.zeros(1, STEPS, JOINTS)
        self.is_ready = torch.ones(1, 1)


class _UiValues:
    def __init__(self):
        self.command = torch.tensor([[1.0, 12.0]])


def _env():
    class _Scene:
        sensors: dict = {}

        def __getitem__(self, name):
            raise KeyError(name)

    class _Env:
        def __init__(self):
            self.scene = _Scene()
            self.command_manager = TraceCommandManager(
                {"motion": _RefWindow(), "compliance": _UiValues()}
            )

    return _Env()


def _window_term(env, **_):
    """Two offsets of the reference window, gated by readiness."""
    command = env.command_manager.get_term("motion")
    window = command.ref_root_pos_w
    return torch.cat([window[:, 0], window[:, 2]], dim=-1) * command.is_ready


def _ui_term(env, **_):
    command = env.command_manager.get_command("compliance")
    enabled = (command[:, 0:1] >= 0.5).to(command.dtype)
    return torch.cat([enabled, enabled * command[:, 1:2]], dim=-1)


def test_a_window_read_becomes_a_shaped_command_slot():
    export = trace_term(_window_term, {}, _env(), name="tracking")
    slots = {slot["field"]: slot for slot in slots_json(export)}
    assert set(slots) == {"ref_root_pos_w", "is_ready"}
    assert all(slot["command"] == "motion" for slot in slots.values())
    # Rank matters: the runtime feeds a flat array and reshapes to this.
    assert slots["ref_root_pos_w"]["shape"] == [1, STEPS, 3]
    assert slots["is_ready"]["shape"] == [1, 1]
    # Offsets 0 and 2 of `arange(9)`, times a ready flag of 1.
    assert export.reference_output.flatten().tolist() == [0, 1, 2, 6, 7, 8]


def test_a_ui_command_read_becomes_a_command_slot():
    export = trace_term(_ui_term, {}, _env(), name="compliance")
    (slot,) = slots_json(export)
    assert slot["command"] == "compliance"
    assert slot["field"] == "command"
    assert export.reference_output.flatten().tolist() == [1.0, 12.0]


def test_the_command_value_is_an_input_rather_than_baked():
    """A baked command is a policy that ignores its own slider, silently."""
    onnxruntime = pytest.importorskip("onnxruntime")
    export = trace_term(_ui_term, {}, _env(), name="compliance")
    (slot,) = slots_json(export)
    session = onnxruntime.InferenceSession(export.onnx_bytes)
    fed = session.run(None, {slot["input"]: np.array([[0.0, 20.0]], dtype=np.float32)})[
        0
    ]
    # Disabled at trace time it was not: the graph has to follow the new value.
    assert fed.flatten().tolist() == [0.0, 0.0]


def test_an_unknown_command_names_what_the_trace_env_has():
    manager = TraceCommandManager({"motion": _RefWindow()})
    with pytest.raises(KeyError, match="motion"):
        manager.get_term("compliance")


def test_get_command_reads_the_terms_command_attribute():
    """mjlab's own `CommandManager.get_command` contract."""
    manager = TraceCommandManager({"compliance": _UiValues()})
    assert manager.get_command("compliance").flatten().tolist() == [1.0, 12.0]
