"""The release's jobs: what they build, from which commit, and which job can publish."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from mjswan.build.frontend import ClientBuilder

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
WORKFLOW = WORKFLOWS / "release.yml"
PUBLISH_PYPI = WORKFLOWS / "publish-pypi.yml"
PUBLISH_NPM = WORKFLOWS / "publish-npm.yml"
RELEASE_WORKFLOWS = [WORKFLOW, PUBLISH_PYPI, PUBLISH_NPM]
MARK_STEP = "Mark the prebuilt SPA as reusable"


def _jobs(path: Path) -> dict:
    return yaml.safe_load(path.read_text())["jobs"]


def _pypi_steps() -> list[dict]:
    return _jobs(PUBLISH_PYPI)["build"]["steps"]


def _pypi_step(name: str) -> dict:
    return next(step for step in _pypi_steps() if step.get("name") == name)


def test_publish_jobs_check_out_the_release_commit_by_sha():
    jobs = _jobs(WORKFLOW)

    assert jobs["release"]["outputs"]["sha"] == "${{ steps.tag.outputs.sha }}"
    assert jobs["test"]["steps"][0]["with"]["ref"] == "${{ needs.release.outputs.sha }}"
    for name in ("publish-pypi", "publish-npm"):
        assert jobs[name]["with"]["ref"] == "${{ needs.release.outputs.sha }}"
    for path in (PUBLISH_PYPI, PUBLISH_NPM):
        for job in _jobs(path).values():
            checkouts = [
                s for s in job["steps"] if "actions/checkout" in s.get("uses", "")
            ]
            assert all(s["with"]["ref"] == "${{ inputs.ref }}" for s in checkouts)


def test_the_suite_passes_before_anything_publishes():
    jobs = _jobs(WORKFLOW)

    assert jobs["test"]["needs"] == "release"
    assert jobs["test"]["permissions"] == {"contents": "read"}
    assert "pytest" in jobs["test"]["steps"][-1]["run"]
    for name in ("publish-pypi", "publish-npm"):
        assert jobs[name]["needs"] == ["release", "test"]


@pytest.mark.parametrize("path", [PUBLISH_PYPI, PUBLISH_NPM], ids=["pypi", "npm"])
def test_the_job_that_can_publish_runs_no_build(path):
    """The OIDC token PyPI and npm trust never shares a job with the code a build runs."""
    jobs = _jobs(path)

    assert [n for n, j in jobs.items() if "id-token" in j.get("permissions", {})] == [
        "publish"
    ]
    assert jobs["build"]["permissions"] == {"contents": "read"}
    assert jobs["publish"]["needs"] == "build"
    assert "environment" not in jobs["build"]
    runs = "\n".join(step.get("run", "") for step in jobs["publish"]["steps"])
    for runs_code in (
        "pytest",
        "npm ci",
        "npm install",
        "npm run",
        "uv run",
        "uv build",
    ):
        assert runs_code not in runs
    checkouts = [
        step["with"]
        for job in jobs.values()
        for step in job["steps"]
        if "actions/checkout" in step.get("uses", "")
    ]
    assert checkouts and all(c["persist-credentials"] is False for c in checkouts)


def test_npm_publishes_the_tarball_build_packed():
    """By path: npm reads `npm/<name>.tgz`, with no `./`, as a GitHub repository."""
    run = _jobs(PUBLISH_NPM)["publish"]["steps"][-1]["run"]

    assert run.startswith('npm publish "./npm/mjswan-${VERSION}.tgz" ')


@pytest.mark.parametrize("path", [PUBLISH_PYPI, PUBLISH_NPM], ids=["pypi", "npm"])
def test_what_ships_is_built_without_a_stored_cache(path):
    """A cache another run wrote could change the build; the lock is the one input."""
    for job in _jobs(path).values():
        for step in job["steps"]:
            uses = step.get("uses", "")
            if "setup-uv" in uses:
                assert step["with"]["enable-cache"] is False
            if "setup-node" in uses:
                assert step["with"]["package-manager-cache"] is False
            assert "npm install" not in step.get("run", "")


@pytest.mark.parametrize(
    "path", RELEASE_WORKFLOWS, ids=[p.stem for p in RELEASE_WORKFLOWS]
)
def test_no_expression_is_pasted_into_a_script(path):
    """An expression in a `run:` becomes script text; values go through `env`."""
    for name, job in _jobs(path).items():
        for step in job.get("steps", []):
            assert "${{" not in step.get("run", ""), (name, step.get("name"))


def test_every_action_is_pinned_to_a_commit():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for name, job in _jobs(path).items():
            refs = [job.get("uses")] + [s.get("uses") for s in job.get("steps", [])]
            for ref in filter(None, refs):
                if not ref.startswith("./"):
                    assert re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref), (
                        path.name,
                        name,
                        ref,
                    )


def test_every_uv_install_is_checked_against_the_pinned_release():
    """setup-uv verifies only the uv versions it knows and installs any other unchecked,
    so each step names the version `required-version` pins and its `checksum`: the
    sha256 of that release's `uv-x86_64-unknown-linux-gnu.tar.gz` (ubuntu-latest)."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pinned = pyproject["tool"]["uv"]["required-version"]
    assert pinned.startswith("=="), "a checksum needs one exact uv version"
    steps = [
        (path.name, job_name, step.get("with") or {})
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for job_name, job in _jobs(path).items()
        for step in job.get("steps", [])
        if "setup-uv" in step.get("uses", "")
    ]

    assert steps
    for where in steps:
        inputs = where[2]
        assert inputs.get("version") == pinned.removeprefix("=="), where[:2]
        assert re.fullmatch(r"[0-9a-f]{64}", inputs.get("checksum", "")), where[:2]
    assert len({inputs["checksum"] for _, _, inputs in steps}) == 1


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
        < names.index("Build the wheel and sdist")
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
