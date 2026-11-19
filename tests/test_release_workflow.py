"""The release's publish jobs: what they build, and from which commit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from mjswan.build.frontend import ClientBuilder

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"
WORKFLOW = WORKFLOWS / "release.yml"
PUBLISH_PYPI = WORKFLOWS / "publish-pypi.yml"
MARK_STEP = "Mark the prebuilt SPA as reusable"


def _pypi_steps() -> list[dict]:
    return yaml.safe_load(PUBLISH_PYPI.read_text())["jobs"]["publish"]["steps"]


def _pypi_step(name: str) -> dict:
    return next(step for step in _pypi_steps() if step.get("name") == name)


def test_publish_jobs_check_out_the_release_commit_by_sha():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]

    assert jobs["release"]["outputs"]["sha"] == "${{ steps.tag.outputs.sha }}"
    for name in ("publish-pypi", "publish-npm"):
        assert jobs[name]["with"]["ref"] == "${{ needs.release.outputs.sha }}"


def test_pypi_release_builds_the_spa_alone():
    """The library build is publish-npm's; the wheel would only carry it unread."""
    run = _pypi_step("Build web template")["run"]

    assert "npm run build:spa" in run
    assert not re.search(r"npm run build\s*$", run, re.MULTILINE)


def test_pypi_release_marks_the_tree_it_ships():
    """After the version is set and node_modules is gone, before the wheel is built."""
    names = [step.get("name") for step in _pypi_steps()]

    assert (
        names.index("Set version")
        < names.index("Remove artifacts from pypi package")
        < names.index(MARK_STEP)
        < names.index("Build and publish")
    )


@pytest.mark.parametrize(
    "options",
    [{}, {"base_path": "/sub/"}, {"gtm_id": "GTM-TEST"}, {"mt": True}, {"debug": True}],
    ids=["default", "base_path", "gtm_id", "mt", "debug"],
)
def test_the_wheel_spa_is_reused_by_the_default_build_alone(
    tmp_path, monkeypatch, options
):
    """The marker the release writes, read back by `build()` as a pip install runs it."""
    template = tmp_path / "template"
    (template / "src").mkdir(parents=True)
    (template / "src" / "App.tsx").write_text("export default 1;\n")
    (template / "package.json").write_text(
        json.dumps({"name": "mjswan", "scripts": {"build:spa": "vite build"}})
    )
    (template / "dist").mkdir()
    (template / "dist" / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr("mjswan.build.frontend.TEMPLATE_DIR", template)
    run = _pypi_step(MARK_STEP)["run"]
    exec(run.split("<<'PY'\n", 1)[1].rsplit("PY", 1)[0], {})

    builds: list[str] = []
    monkeypatch.setattr(ClientBuilder, "create_env", lambda self, clean=False: None)
    monkeypatch.setattr(ClientBuilder, "sync_version_from_python", lambda self: None)
    monkeypatch.setattr(
        ClientBuilder, "install_dependencies", lambda self, clean=False: None
    )
    monkeypatch.setattr(
        ClientBuilder,
        "run_build_script",
        lambda self, script, env=None: builds.append(script),
    )
    ClientBuilder(template).build(**options)

    assert builds == ([] if not options else ["build:spa"])
