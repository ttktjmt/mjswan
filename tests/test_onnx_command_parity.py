"""Command-term parity: traced graph vs the live mjlab term (ADR 0005 §3).

Layer: L3 (needs the ``examples`` extras and a warp CPU-kernel compile, so
``slow``/``mjlab``).

Commands are the most randomness-heavy layer in the system — `lift_height` draws 7
values per resample, the tracking RSI graph 41 — so every one runs through the RNG
spy/replay harness.

The resolution follows the Builder's own path: the registry's `CommandBinding`
supplies `state_fields`, `command_field` and any `trace_override`, and
`cfg.build(env)` constructs the term. So a registration that drifts from its term
fails here rather than shipping.

Run::

    MUJOCO_GL=disable pytest tests/test_onnx_command_parity.py
"""

from __future__ import annotations

import contextlib
import io
import os
import types
from typing import Any

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

pytestmark = [pytest.mark.slow, pytest.mark.mjlab]

# Every reference task with a traced command, and that command's name in its config.
#
# Tracking-Flat-G1's `motion` is absent: it stays a native `TrackingCommand` with only its
# RSI jitter traced, as a reset graph (see `test_onnx_command_config.py`), and its clip is
# a W&B artifact the env cannot be built offline without.
COMMAND_TASKS = [
    # Stateful, with an `entity_write` side effect on the cube (§3b).
    pytest.param("Mjlab-Lift-Cube-Yam", "lift_height", id="lift-cube-yam"),
    # Heading tracking as a dynamic slot, through the override mjswan itself binds.
    pytest.param("Mjlab-Velocity-Flat-Unitree-G1", "twist", id="velocity-flat-g1"),
    pytest.param("Mjlab-Velocity-Flat-Unitree-Go1", "twist", id="velocity-flat-go1"),
]


@pytest.fixture(scope="module", autouse=True)
def _registrations() -> None:
    """`LiftingCommandCfg`'s traced body lives author-side; load it before resolving.

    Without it the adapter raises for an unregistered class: the footgun that cost a
    spinkick policy its RSI jitter. `UniformVelocityCommandCfg` needs no import: mjswan
    binds that one itself.
    """
    pytest.importorskip("mjswan.mjlab.bindings")


def _lift_update_command(self: Any, env_ids: Any = None) -> None:
    """A port's rewrite of mjlab 1.6's ``LiftingCommand._update_command``.

    1.6 refreshes the live sim there after a timer-expiry teleport, which the tracer
    refuses to bake. The command is ``_resample_command``'s alone, so dropping the
    refresh changes no number, as this file's own comparison shows.
    `examples/demo/main.py` registers this rewrite as a `trace_override`; the engine
    binding this file resolves carries none, so the rewrite is supplied here too.
    """
    del env_ids


def _apply_port_rewrite(task_id: str, term: Any) -> None:
    """What a port writes for a term mjlab does not leave traceable."""
    if task_id == "Mjlab-Lift-Cube-Yam":
        term._update_command = types.MethodType(_lift_update_command, term)


def _traced_command(task_id: str, command_name: str) -> tuple[Any, Any]:
    """The live term and its pending trace, resolved as the Builder resolves them."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.mjlab.command import _adapt_command_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.sim.nconmax = 200_000
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset()

    adapted = _adapt_command_cfg(cfg.commands[command_name])
    pending = adapted.pending_trace
    assert pending is not None, (
        f"{task_id}'s {command_name!r} resolved to a native term "
        f"({adapted.term_name!r}); this file only covers traced commands"
    )
    # As `serialize_command` does: build from the cfg, then apply the override.
    term = pending.mjlab_cfg.build(env)
    if pending.trace_override is not None:
        pending.trace_override(term)
    _apply_port_rewrite(task_id, term)
    return env, (term, pending)


@pytest.fixture(scope="module")
def command_report(request):
    from mjswan.compile import run_command_parity

    task_id, command_name = request.param
    env, (term, pending) = _traced_command(task_id, command_name)
    try:
        yield run_command_parity(
            term,
            pending.state_fields,
            name=command_name,
            command_field=pending.command_field,
            n_draws=16,
        )
    finally:
        env.close()


_PARAMS = [pytest.param((p.values[0], p.values[1]), id=p.id) for p in COMMAND_TASKS]


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_traced_command_matches_the_live_term(command_report):
    assert command_report.passed, "\n" + str(command_report.note)


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_every_draw_was_actually_replayed(command_report):
    """A command that traced but was never *run* would pass the assertion above.

    `passed` is an AND over comparisons and an empty AND is True, so the count of
    replayed draws is what separates "agreed" from "never asked" — the same guard
    the task sweep carries.
    """
    assert command_report.steps_checked == 16, (
        f"{command_report.name} compared {command_report.steps_checked} of 16 draws"
    )
    assert command_report.max_abs_diff is not None
    assert command_report.max_abs_diff <= 1e-5


@pytest.mark.parametrize("command_report", _PARAMS, indirect=True)
def test_the_command_actually_draws_randomness(command_report):
    """§2b is about terms *with* randomness; a `rand_dim` of 0 would be vacuous.

    Both of these resample from a range, so a zero here means the tracer stopped
    seeing the draws — which would make the replay above compare nothing.
    """
    assert command_report.rand_dim and command_report.rand_dim > 0, (
        f"{command_report.name} traced with rand_dim={command_report.rand_dim}; "
        "the replay harness then has no randomness to replay"
    )
