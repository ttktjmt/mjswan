"""Where ``build()`` writes: only into a missing or empty directory or over a previous
build, and through a directory beside it, so a failed build leaves the last one."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import pytest

from mjswan import Builder
from mjswan.build import pipeline

MODEL = '<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'


@pytest.fixture
def build(monkeypatch):
    """Build a one-scene app into a path, with a fresh builder each time: a build
    releases its scenes' models."""
    # The Node build and the SPA copy only; the writer runs for real.
    monkeypatch.setattr(pipeline, "ClientBuilder", MagicMock())
    monkeypatch.setattr(pipeline, "install_spa", MagicMock(return_value=True))

    def run(out: Path) -> None:
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", spec=mujoco.MjSpec.from_string(MODEL)
        )
        builder._save_web(out)

    return run


def _leftovers(parent: Path) -> list[str]:
    """What a build left beside its output: staging or a replaced build."""
    return sorted(p.name for p in parent.iterdir() if p.name.startswith("."))


def _fail_the_manifest(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("tracing failed")

    monkeypatch.setattr(pipeline, "write_manifest", fail)


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_a_missing_or_empty_directory_takes_the_build(build, tmp_path, kind):
    out = tmp_path / "out"
    if kind == "empty":
        out.mkdir()

    build(out)

    assert json.loads((out / "manifest.json").read_text())["projects"]
    assert _leftovers(tmp_path) == []


def test_a_previous_build_is_replaced_whole(build, tmp_path):
    out = tmp_path / "out"
    build(out)
    project = json.loads((out / "manifest.json").read_text())["projects"][0]["id"]
    stale = out / project / "stale.onnx"
    stale.write_bytes(b"from a scene since removed")

    build(out)

    assert (out / "manifest.json").is_file()
    assert not stale.exists()
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    "manifest",
    [None, "not json", json.dumps({"name": "a web app manifest"})],
    ids=["none", "not-json", "not-mjswan"],
)
def test_a_directory_holding_anything_else_is_refused_and_kept(
    build, tmp_path, manifest
):
    """`build(".")` or `build(Path.home())` would otherwise delete that directory."""
    out = tmp_path / "work"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    if manifest is not None:
        (out / "manifest.json").write_text(manifest)

    with pytest.raises(ValueError, match="holds no mjswan build"):
        build(out)

    assert (out / "notes.txt").read_text() == "keep me"
    assert _leftovers(tmp_path) == []


def test_a_file_or_a_link_is_refused(build, tmp_path):
    file = tmp_path / "out.txt"
    file.write_text("keep me")
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="not a directory"):
        build(file)
    with pytest.raises(ValueError, match="symbolic link"):
        build(link)
    assert file.read_text() == "keep me"


def test_a_failed_build_leaves_the_previous_one(build, tmp_path, monkeypatch):
    out = tmp_path / "out"
    build(out)
    before = (out / "manifest.json").read_text()
    _fail_the_manifest(monkeypatch)

    with pytest.raises(RuntimeError, match="tracing failed"):
        build(out)

    assert (out / "manifest.json").read_text() == before
    assert _leftovers(tmp_path) == []


def test_a_failed_first_build_leaves_nothing_to_refuse_next_time(
    build, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    with monkeypatch.context() as patch:
        _fail_the_manifest(patch)
        with pytest.raises(RuntimeError, match="tracing failed"):
            build(out)
    assert not out.exists()
    assert _leftovers(tmp_path) == []

    build(out)
    assert (out / "manifest.json").is_file()


def test_a_failed_move_puts_the_previous_build_back(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text('{"projects": []}')
    staging = tmp_path / ".out.0000.tmp"
    staging.mkdir()
    rename = Path.rename

    def refuse_the_staging(self, target):
        if self == staging:
            raise PermissionError("a file in it is open")
        return rename(self, target)

    monkeypatch.setattr(Path, "rename", refuse_the_staging)
    with pytest.raises(PermissionError):
        pipeline._swap_in(staging, out)

    assert (out / "manifest.json").read_text() == '{"projects": []}'
