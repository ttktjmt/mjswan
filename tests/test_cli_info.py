"""`mjswan info` and the build progress show ids and names with their brackets.

rich parses ``[...]`` as markup, which would drop them.
"""

from pathlib import Path

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
