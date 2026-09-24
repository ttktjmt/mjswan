"""`mjswan new`: a scaffolded project builds as written once its placeholders are in."""

import json
import runpy
from pathlib import Path
from unittest.mock import MagicMock

import onnx
import pytest
from typer.testing import CliRunner

import mjswan
from mjswan.cli import app


def test_the_policy_template_builds_once_its_network_is_in_place(
    tmp_path: Path, minimal_onnx: onnx.ModelProto, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["new", "demo", "--template", "policy"])
    assert result.exit_code == 0, result.output
    project = tmp_path / "demo"
    onnx.save(minimal_onnx, project / "policy.onnx")

    # The script's own build() would run the Node build and launch() would serve, so
    # build() writes the manifest alone and hands back an app whose launch() is inert.
    monkeypatch.setattr("mjswan.build.pipeline.ClientBuilder", MagicMock())
    monkeypatch.setattr(
        "mjswan.build.pipeline.install_spa", MagicMock(return_value=True)
    )

    def build(builder: mjswan.Builder, *args, **kwargs) -> MagicMock:
        builder._save_web(project / "dist")
        return MagicMock()

    monkeypatch.setattr(mjswan.Builder, "build", build)
    monkeypatch.chdir(project)
    runpy.run_path(str(project / "main.py"), run_name="__main__")

    manifest = json.loads((project / "dist" / "manifest.json").read_text())
    (scene,) = manifest["projects"][0]["scenes"]
    assert scene["control_dt"] == 0.02
    assert [p["name"] for p in scene["policies"]] == ["Policy"]
