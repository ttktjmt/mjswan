"""What must *not* be in a built bundle (ADR 0005 acceptance criteria 4 and 5).

Both hold by construction today, and construction is exactly what changes.

* **No training-only managers.** Reward, curriculum, metrics and recorders have no
  browser-side runtime. mjswan never reads them, and a build that *rejected* them
  could build nothing (every mjlab task carries rewards), so what is checkable is
  that the emitted artifacts mention none of them.
* **No term source as executable text.** ONNX ships term bodies as graph bytes, so no
  Python body may travel alongside — not as a source string, not as an eval-able payload.

Both walk the real emitted output rather than inspecting the code that writes it,
so a future serializer that starts attaching a `rewards` block or a `source` string
fails here regardless of how it got there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

# Manager categories with no browser-side runtime (ADR 0003 §Coverage).
FORBIDDEN_KEYS = (
    "reward",
    "rewards",
    "curriculum",
    "curriculums",
    "metrics",
    "recorders",
)

# Markers of Python source travelling as text; `def ` alone also appears in prose.
PYTHON_SOURCE_MARKERS = (
    "import torch",
    "torch.",
    "def __",
    "env.scene[",
    "lambda ",
    "__import__",
    "eval(",
)


def _walk(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every (path, value) pair in a nested JSON structure."""
    found: list[tuple[str, Any]] = [(path, node)]
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_walk(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(_walk(value, f"{path}[{index}]"))
    return found


class _TraceEnv:
    """The minimal `env.scene[entity].data.<field>` shape the tracer records."""

    def __init__(self, **fields):
        data = type("_Data", (), dict(fields))()
        entity = type("_Entity", (), {"data": data})()
        self.scene = {"robot": entity}


@pytest.fixture
def built_output(tmp_path, minimal_model, minimal_onnx, monkeypatch) -> Path:
    """A real `_save_web` output tree, with the Node build and template copy mocked.

    Mocking those keeps this L1: what is under test is the *content* the
    serializers write, and neither the bundler nor the template copy contributes
    any of it.
    """
    pytest.importorskip("torch")
    import torch

    import mjswan
    from mjswan.envs.mdp.actions import JointPositionActionCfg
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjswan.managers.termination_manager import TerminationTermCfg

    monkeypatch.setattr("mjswan.build.pipeline.ClientBuilder", MagicMock())
    monkeypatch.setattr(
        "mjswan.build.pipeline.install_spa", MagicMock(return_value=True)
    )

    def joint_pos(env):
        return env.scene["robot"].data.joint_pos

    def too_low(env, *, minimum_height=0.2):
        return env.scene["robot"].data.root_link_pos_w[:, 2] < minimum_height

    builder = mjswan.Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=minimal_model
    )
    # A trace env, so the build really traces terms and emits the `.onnx` files —
    # without them the scans below run over a bundle holding nothing at issue.
    scene._config.mjlab_env = _TraceEnv(  # noqa: SLF001 — the builder's own seam
        joint_pos=torch.tensor([[0.1, 0.2]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]),
    )
    scene.add_policy(
        name="Policy",
        policy=minimal_onnx,
        actions={
            "joint_pos": JointPositionActionCfg(actuator_names=(".*",), scale=0.5),
        },
        observations={
            "policy": ObservationGroupCfg(
                terms={"joint_pos": ObservationTermCfg(func=joint_pos)}
            )
        },
        terminations={"too_low": TerminationTermCfg(func=too_low)},
    )
    out = tmp_path / "out"
    builder._save_web(out)  # noqa: SLF001 — the writer under test
    return out


def _json_files(out: Path) -> list[Path]:
    files = sorted(out.rglob("*.json"))
    assert files, "the build wrote no JSON at all; the fixture is not exercising it"
    return files


def test_the_fixture_really_built_traced_terms(built_output):
    """Guard the guard, part two: the scans must have term content to scan.

    Both absence-assertions pass just as happily over a bundle that carries no
    observation or termination at all — which is what this fixture produced before
    it was given a trace env. Pin that the artifacts contain the traced entries and
    their graphs, so the scans stay meaningful.
    """
    manifest = json.loads((built_output / "manifest.json").read_text())
    mdp = manifest["projects"][0]["scenes"][0]["mdps"][0]
    assert "observations" in mdp, mdp.keys()
    assert "terminations" in mdp, mdp.keys()
    graphs = sorted(p.name for p in built_output.rglob("*.onnx"))
    # The policy network plus a graph for each traced term.
    assert len(graphs) >= 3, graphs


def test_no_training_only_manager_reaches_the_artifacts(built_output):
    for path in _json_files(built_output):
        payload = json.loads(path.read_text())
        for node_path, _ in _walk(payload):
            leaf = node_path.rsplit(".", 1)[-1].split("[")[0].lower()
            assert leaf not in FORBIDDEN_KEYS, (
                f"{path.name} carries a training-only manager at {node_path!r}; "
                "reward/curriculum/metrics/recorders have no browser runtime"
            )


def test_no_python_source_travels_in_the_artifacts(built_output):
    for path in _json_files(built_output):
        text = path.read_text()
        for marker in PYTHON_SOURCE_MARKERS:
            assert marker not in text, (
                f"{path.name} contains {marker!r} — a term body must ship as ONNX "
                "graph bytes, never as source text"
            )


def test_graph_bytes_are_onnx_not_source(built_output):
    """Any `.onnx` the build wrote must be a protobuf, not a script in disguise."""
    for path in sorted(built_output.rglob("*.onnx")):
        head = path.read_bytes()[:64]
        # An ONNX file starts with a protobuf field tag, never with text.
        assert not head.lstrip().startswith(b"def "), path
        assert b"import torch" not in path.read_bytes(), path


def test_the_markers_would_actually_fire():
    """Guard the guard: a scan that matches nothing proves nothing.

    Both tests above assert an absence, so they pass just as happily against an
    empty file or a broken marker list. This pins that the markers match the thing
    they are meant to catch.
    """
    sample = "def base_ang_vel(env):\n    return env.scene['robot'].data.root_link_ang_vel_b\n"
    assert any(marker in sample for marker in PYTHON_SOURCE_MARKERS)
    assert "rewards" in FORBIDDEN_KEYS
