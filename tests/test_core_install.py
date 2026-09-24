"""What ``pip install mjswan`` alone promises: the pipeline, and a sentence per source.

Run by the ``core`` job of ``pytest.yml``, which installs no extra; they skip wherever
an extra is installed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import pytest

import mjswan

_EXTRA_MODULES = ("wandb", "huggingface_hub", "mjlab", "torch")
_INSTALLED = [name for name in _EXTRA_MODULES if importlib.util.find_spec(name)]

if os.environ.get("MJSWAN_CORE_ONLY") and _INSTALLED:
    # The job exists to run these; skipping them there would pass it with nothing run.
    raise RuntimeError(f"MJSWAN_CORE_ONLY is set but {_INSTALLED} are installed.")

pytestmark = pytest.mark.skipif(
    bool(_INSTALLED), reason=f"{_INSTALLED} installed; the core CI job runs these"
)

SCENE_XML = """
<mujoco model="core">
  <worldbody><body><joint name="hip" type="hinge"/><geom size="0.1"/></body></worldbody>
  <actuator><position joint="hip"/></actuator>
</mujoco>
"""


@pytest.fixture
def project():
    return mjswan.Builder().add_project(name="Core")


@pytest.fixture
def scene(project):
    return project.add_scene(
        name="Robot", spec=mujoco.MjSpec.from_string(SCENE_XML), control_dt=0.02
    )


def test_import_loads_no_source_backend():
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, mjswan; "
            f"print([m for m in {_EXTRA_MODULES!r} if m in sys.modules])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert loaded == "[]"


def test_a_model_only_scene_builds(scene, build_manifest, tmp_path: Path):
    manifest = build_manifest(scene._project._builder, tmp_path / "dist")

    assert [s["name"] for s in manifest["projects"][0]["scenes"]] == ["Robot"]


@pytest.mark.parametrize(
    ("call", "extra"),
    [
        (lambda p, s: s.add_policy_hf("org/repo", filename="policy.onnx"), "hf"),
        (lambda p, s: p.add_scene_hf("org/repo", "scene.xml"), "hf"),
        (lambda p, s: s.add_splat_hf("org/repo", "street.spz"), "hf"),
        (lambda p, s: s.add_policy_wandb("e/p/run", only_latest=True), "wandb"),
        (lambda p, s: s.add_policy_wandb("e/p/run", task_id="Task"), "wandb,mjlab"),
        (lambda p, s: p.add_scene_mjlab("Mjlab-Velocity-Flat-Unitree-G1"), "mjlab"),
    ],
    ids=[
        "add_policy_hf",
        "add_scene_hf",
        "add_splat_hf",
        "add_policy_wandb latest",
        "add_policy_wandb convert",
        "add_scene_mjlab",
    ],
)
def test_each_source_names_its_extra(project, scene, call, extra):
    with pytest.raises(ImportError, match=rf"pip install 'mjswan\[{extra}\]'"):
        call(project, scene)
