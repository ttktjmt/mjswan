"""`mjswan info` and the build progress show ids and names with their brackets.

rich parses ``[...]`` as markup, which would drop them.
"""

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from mjswan.builder import Builder
from mjswan.cli import app


def _built(tmp_path: Path, minimal_spec, minimal_onnx, build_manifest) -> Path:
    builder = Builder()
    scene = builder.add_project(name="My Robots").add_scene(
        name="G1 [v2]", spec=minimal_spec, control_dt=0.02
    )
    scene.add_policy(name="[walk]", policy=minimal_onnx)
    out = tmp_path / "dist"
    build_manifest(builder, out)
    return out


def test_info_shows_the_project_id_and_bracketed_names(
    tmp_path, minimal_spec, minimal_onnx, build_manifest
):
    out = _built(tmp_path, minimal_spec, minimal_onnx, build_manifest)
    result = CliRunner().invoke(app, ["info", str(out)])
    assert result.exit_code == 0, result.output
    assert "My Robots  [my_robots]" in result.output
    assert "G1 [v2]  g1_v2/scene.mjz" in result.output
    assert "Policy: [walk]" in result.output


def test_the_build_progress_shows_bracketed_names(
    tmp_path, minimal_spec, minimal_onnx, build_manifest, capsys
):
    _built(tmp_path, minimal_spec, minimal_onnx, build_manifest)
    assert "G1 [v2]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "header",
    [
        ("[link=https://example.invalid]0.12.9[/link]", 1),
        ("[link=https://example.invalid", "]"),
    ],
    ids=["one-value", "across-values"],
)
def test_info_prints_a_document_s_text_inert(
    tmp_path, minimal_spec, minimal_onnx, build_manifest, monkeypatch, *, header
):
    """A received document's strings become neither links nor terminal commands."""
    out = _built(tmp_path, minimal_spec, minimal_onnx, build_manifest)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["version"], manifest["format"] = header
    manifest["projects"][0]["name"] = "Demo\x1b]0;SPOOFED\x07"
    (out / "manifest.json").write_text(json.dumps(manifest))
    # A real terminal's console: CliRunner's own output is not a tty.
    monkeypatch.setattr("mjswan.cli.console", Console(force_terminal=True, width=500))

    result = CliRunner().invoke(app, ["info", str(out)])

    assert result.exit_code == 0, result.output
    assert "\x1b]" not in result.output
    assert "v{}, format {}".format(*header) in result.output
