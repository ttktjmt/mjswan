"""Tests for mjswan.Builder — project ID assignment, config JSON structure, and build output.

Layer breakdown:
  L1 (pure logic / lightweight I/O): TestProjectIdAssignment, TestBuilderValidation,
                                     TestSaveConfigJson
  L3 slow (triggers frontend build): TestFullBuild

Run only L1 tests (pre-commit):  pytest -m "not slow"
Run all tests (CI):               pytest
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mjswan
from mjswan.builder import Builder
from mjswan.utils import name2id


# ===========================================================================
# L1 — project ID assignment rules
# ===========================================================================
class TestProjectIdAssignment:
    def test_first_project_without_explicit_id_gets_none(self):
        builder = Builder()
        project = builder.add_project(name="Main Demo")
        assert project.id is None

    def test_second_project_without_explicit_id_gets_auto_id(self):
        builder = Builder()
        builder.add_project(name="Main Demo")
        second = builder.add_project(name="MuJoCo Menagerie")
        assert second.id == name2id("MuJoCo Menagerie")

    def test_auto_id_uses_name2id_transform(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="My Project Name")
        assert second.id == "my_project_name"

    def test_explicit_id_used_as_is_on_first_project(self):
        project = Builder().add_project(name="Main Demo", id="custom")
        assert project.id == "custom"

    def test_explicit_id_used_as_is_on_subsequent_project(self):
        builder = Builder()
        builder.add_project(name="First")
        second = builder.add_project(name="Second", id="explicit_id")
        assert second.id == "explicit_id"

    def test_mixed_id_sequence(self):
        builder = Builder()
        p1 = builder.add_project(name="Project A")
        p2 = builder.add_project(name="Project B")
        p3 = builder.add_project(name="Project C", id="custom")
        assert p1.id is None
        assert p2.id == name2id("Project B")
        assert p3.id == "custom"

    def test_get_projects_returns_independent_copy(self):
        builder = Builder()
        builder.add_project(name="Test")
        copy = builder.get_projects()
        copy.clear()
        assert len(builder.get_projects()) == 1


# ===========================================================================
# L1 — GTM ID handling
# ===========================================================================
class TestBuilderGtmId:
    def test_defaults_to_none(self):
        assert Builder()._gtm_id is None

    def test_stored_when_provided(self):
        assert Builder(gtm_id="GTM-W79HQ38W")._gtm_id == "GTM-W79HQ38W"


# ===========================================================================
# L1 — validation
# ===========================================================================
class TestBuilderValidation:
    def test_build_with_no_projects_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot build an empty application"):
            Builder().build(tmp_path / "out")

    def test_policy_filename_rejects_empty_string(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("")

    def test_policy_filename_rejects_forward_slash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path/policy")

    def test_policy_filename_rejects_backslash(self):
        with pytest.raises(ValueError):
            Builder()._policy_filename("path\\policy")

    def test_policy_filename_accepts_plain_name(self):
        assert Builder()._policy_filename("my_policy") == "my_policy"


# ===========================================================================
# L1 — _save_config_json output structure (no frontend build)
# ===========================================================================
class TestSaveConfigJson:
    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "assets" / "config.json").read_text())

    def test_config_contains_version(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        assert self._read_config(tmp_path)["version"] == mjswan.__version__

    def test_config_has_projects_list(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        config = self._read_config(tmp_path)
        assert isinstance(config["projects"], list)
        assert len(config["projects"]) == 1

    def test_project_name_and_id_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main Demo").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        project = self._read_config(tmp_path)["projects"][0]
        assert project["name"] == "Main Demo"
        assert project["id"] is None

    def test_scene_path_uses_name2id_with_mjb_for_model(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="My Scene", model=minimal_model)
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["name"] == "My Scene"
        assert scene["path"] == "my_scene/scene.mjb"

    def test_scene_path_uses_mjz_for_spec(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="My Scene", spec=minimal_spec)
        builder._save_config_json(tmp_path)
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["path"] == "my_scene/scene.mjz"

    def test_policy_without_config_path_has_no_config_key(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(minimal_onnx, name="Policy")
        builder._save_config_json(tmp_path)
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert policy["name"] == "Policy"
        assert "config" not in policy

    def test_multiple_projects_all_present_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Project A").add_scene(name="S", model=minimal_model)
        builder.add_project(name="Project B").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert len(projects) == 2
        assert projects[0]["name"] == "Project A"
        assert projects[1]["name"] == "Project B"

    def test_second_project_auto_id_reflected_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Main").add_scene(name="S", model=minimal_model)
        builder.add_project(name="MuJoCo Menagerie").add_scene(
            name="S", model=minimal_model
        )
        builder._save_config_json(tmp_path)
        projects = self._read_config(tmp_path)["projects"]
        assert projects[0]["id"] is None
        assert projects[1]["id"] == name2id("MuJoCo Menagerie")


# ===========================================================================
# L3 slow — full build pipeline (triggers frontend compilation)
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuild:
    def test_build_creates_assets_config_json(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "assets" / "config.json").exists()

    def test_build_with_model_creates_mjb_file(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjb").exists()

    def test_build_with_spec_creates_mjz_file(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", spec=minimal_spec)
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "main" / "assets" / "scene"
        assert (scene_dir / "scene.mjz").exists()

    def test_build_project_without_id_uses_main_directory(
        self, tmp_path, minimal_model
    ):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="S", model=minimal_model)
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "main").is_dir()

    def test_build_project_with_id_uses_id_as_directory(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test", id="demo").add_scene(
            name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "demo").is_dir()

    def test_build_returns_mjswan_app_instance(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="S", model=minimal_model)
        app = builder.build(tmp_path / "out")
        assert isinstance(app, mjswan.mjswanApp)


@pytest.mark.slow
class TestFullBuildGtmId:
    def test_gtm_snippet_injected_into_all_html_files(self, tmp_path, minimal_model):
        builder = Builder(gtm_id="GTM-SAMPLE123")
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        out = tmp_path / "out"
        for html_file in [out / "index.html", out / "main" / "index.html"]:
            html = html_file.read_text()
            assert "GTM-SAMPLE123" in html
            assert "googletagmanager.com/gtm.js" in html  # <head> script
            assert "googletagmanager.com/ns.html" in html  # <body> noscript

    def test_no_gtm_without_gtm_id(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(name="Scene", model=minimal_model)
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "googletagmanager.com" not in html
