"""Tests for mjswan.Builder — project ID assignment, config JSON structure, and build output.

Layer breakdown:
  L1 (pure logic / lightweight I/O): TestProjectIdAssignment, TestBuilderValidation,
                                     TestSaveConfigJson, TestSaveWebPolicyJson
  L3 slow (triggers frontend build): TestFullBuild

Run only L1 tests (pre-commit):  pytest -m "not slow"
Run all tests (CI):               pytest
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import onnx
import pytest
from onnx import TensorProto, helper

import mjswan
from mjswan._build_client import ClientBuilder
from mjswan.builder import Builder
from mjswan.envs.mdp import events as evt_fns
from mjswan.envs.mdp import observations as obs_fns
from mjswan.envs.mdp import terminations as term_fns
from mjswan.envs.mdp.actions import JointEffortActionCfg, JointPositionActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.utils import name2id
from mjswan.viewer import ViewerConfig


def _entry(
    manifest: dict, policy_name: str, *, project: int = 0, scene: int = 0
) -> dict:
    """A policy's manifest entry merged with its MDP's: what the engine is handed."""
    scene_entry = manifest["projects"][project]["scenes"][scene]
    policy = next(p for p in scene_entry["policies"] if p["name"] == policy_name)
    mdp = next(m for m in scene_entry["mdps"] if m["id"] == policy["mdp"])
    return {**policy, **{k: v for k, v in mdp.items() if k != "id"}}


# ===========================================================================
# L1 — project ID assignment rules
# ===========================================================================
def _two_output_onnx() -> onnx.ModelProto:
    """A network with two outputs, for slot-table tests."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    return helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["y"]),
                helper.make_node("Identity", ["x"], ["z"]),
            ],
            "two_outputs",
            [x],
            [y, z],
        )
    )


def _two_input_onnx() -> onnx.ModelProto:
    """A network with two inputs (concatenated into one output), for slot-table tests."""
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Concat", ["a", "b"], ["y"], axis=1)
    return helper.make_model(helper.make_graph([node], "two_inputs", [a, b], [y]))


class TestProjectIdAssignment:
    """A project's id is `name2id(name)`, unique in the document (ADR 0006 §4)."""

    def test_first_project_gets_its_sanitized_name(self):
        project = Builder().add_project(name="Main Demo")
        assert project.id == "main_demo"

    def test_second_project_gets_its_sanitized_name(self):
        builder = Builder()
        builder.add_project(name="Main Demo")
        second = builder.add_project(name="MuJoCo Menagerie")
        assert second.id == name2id("MuJoCo Menagerie")

    def test_an_explicit_id_is_not_accepted(self):
        # One object, one identifier: the directory and the ?project= value can never
        # disagree if neither can be set by hand.
        with pytest.raises(TypeError, match="id"):
            Builder().add_project(name="Main Demo", id="custom")

    def test_colliding_names_are_renamed_with_a_warning(self):
        builder = Builder()
        first = builder.add_project(name="Flat Terrain")
        with pytest.warns(RuntimeWarning, match="'flat_terrain_1'"):
            second = builder.add_project(name="flat-terrain")
        with pytest.warns(RuntimeWarning, match="'flat_terrain_2'"):
            third = builder.add_project(name="FLAT TERRAIN")
        assert (first.id, second.id, third.id) == (
            "flat_terrain",
            "flat_terrain_1",
            "flat_terrain_2",
        )

    def test_a_name_with_no_letter_or_digit_is_refused(self):
        with pytest.raises(ValueError, match="empty id"):
            Builder().add_project(name="日本語")

    def test_one_default_project_is_allowed(self):
        builder = Builder()
        builder.add_project(name="A")
        assert builder.add_project(name="B", default=True)._config.default is True

    def test_a_second_default_project_is_refused(self):
        builder = Builder()
        builder.add_project(name="A", default=True)
        with pytest.raises(ValueError, match="already"):
            builder.add_project(name="B", default=True)

    def test_get_projects_returns_independent_copy(self):
        builder = Builder()
        builder.add_project(name="Test")
        copy = builder.get_projects()
        copy.clear()
        assert len(builder.get_projects()) == 1


class TestSceneAndPolicyIds:
    """Scene ids are unique within a project, policy ids within a scene."""

    def test_two_scenes_with_one_name_are_both_kept(self, minimal_model):
        project = Builder().add_project(name="P")
        a = project.add_scene(name="Flat Terrain", model=minimal_model)
        with pytest.warns(RuntimeWarning, match="scene"):
            b = project.add_scene(name="Flat Terrain", model=minimal_model)
        assert (a._config.id, b._config.id) == ("flat_terrain", "flat_terrain_1")

    def test_the_same_scene_name_in_another_project_is_not_a_collision(
        self, minimal_model
    ):
        builder = Builder()
        a = builder.add_project(name="A").add_scene(name="S", model=minimal_model)
        b = builder.add_project(name="B").add_scene(name="S", model=minimal_model)
        assert a._config.id == b._config.id == "s"

    def test_two_policies_with_one_name_are_both_kept(
        self, minimal_model, minimal_onnx
    ):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", model=minimal_model, control_dt=0.02)
        )
        a = scene.add_policy(name="model_2000", policy=minimal_onnx)
        with pytest.warns(RuntimeWarning, match="policy"):
            b = scene.add_policy(name="model_2000", policy=minimal_onnx)
        assert (a._config.id, b._config.id) == ("model_2000", "model_2000_1")

    def test_two_default_policies_fail_the_build(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", model=minimal_model, control_dt=0.02)
        )
        scene.add_policy(name="A", policy=minimal_onnx, default=True)
        scene.add_policy(name="B", policy=minimal_onnx, default=True)
        with pytest.raises(ValueError, match="default=True"):
            build_manifest(scene._project._builder, tmp_path / "out")


# ===========================================================================
# L1 — GTM ID handling
# ===========================================================================
class TestSceneCamera:
    """Every scene carries a resolved `camera`, so the browser never invents a view."""

    def test_a_scene_with_no_viewer_still_gets_the_defaults(
        self, tmp_path, minimal_model, build_manifest
    ):
        project = Builder().add_project(name="P")
        project.add_scene(name="S", model=minimal_model)
        manifest = build_manifest(project._builder, tmp_path / "out")

        camera = manifest["projects"][0]["scenes"][0]["camera"]
        assert camera == ViewerConfig().to_dict()
        # `fovy=None` means 45 to Python; the document has to say so itself.
        assert camera["fovy"] == 45.0
        # AUTO is what makes the view follow the robot rather than watch it leave.
        assert camera["origin_type"] == "AUTO"

    def test_an_explicit_viewer_wins(self, tmp_path, minimal_model, build_manifest):
        project = Builder().add_project(name="P")
        project.add_scene(name="S", model=minimal_model).set_viewer(
            ViewerConfig(distance=9.0, fovy=60.0)
        )
        manifest = build_manifest(project._builder, tmp_path / "out")

        camera = manifest["projects"][0]["scenes"][0]["camera"]
        assert (camera["distance"], camera["fovy"]) == (9.0, 60.0)


class TestBuilderGtmId:
    def test_defaults_to_none(self):
        assert Builder()._gtm_id is None

    def test_stored_when_provided(self):
        assert Builder(gtm_id="GTM-W79HQ38W")._gtm_id == "GTM-W79HQ38W"


class TestClientBuilderCustomTerms:
    """Custom terms are runtime plugins (ADR 0004 §10): author TS is collected for
    esbuild into a standalone ESM the engine loads at runtime."""

    def test_collect_custom_terms_gathers_ts_src_by_kind(self, tmp_path, monkeypatch):
        src = tmp_path / "FooTerm.ts"
        src.write_text("export class FooTerm {}\n")
        monkeypatch.setattr(
            term_fns,
            "_custom_registry",
            {"foo": SimpleNamespace(ts_name="FooTerm", ts_src=str(src))},
        )
        terms = ClientBuilder._collect_custom_terms()
        assert terms["terminations"] == {"FooTerm": src.resolve()}

    def test_collect_custom_terms_skips_declarative_callable(self, monkeypatch):
        # A traced term is a plain callable with no ts_src: skip it, do not crash.
        def base_lin_vel(env, **params):
            del env, params

        monkeypatch.setattr(obs_fns, "_custom_registry", {"base_lin_vel": base_lin_vel})
        assert "observations" not in ClientBuilder._collect_custom_terms()


class TestFrontendBuildCache:
    """The SPA is project-independent, so a matching dist/ is reused (no Node)."""

    def test_cache_matches_only_on_identical_meta(self, tmp_path):
        cb = ClientBuilder(tmp_path)
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<!doctype html>")
        meta = cb._build_meta(base_path="/", gtm_id=None, mt=False, debug=False)
        (dist / ".mjswan-build-meta.json").write_text(json.dumps(meta))

        assert cb._cached_spa_matches(meta) is True
        # A different base_path must invalidate the cache.
        other = cb._build_meta(base_path="/sub/", gtm_id=None, mt=False, debug=False)
        assert cb._cached_spa_matches(other) is False

    def test_build_frontend_false_without_cache_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no matching prebuilt"):
            ClientBuilder(tmp_path).build(build_frontend=False)

    def test_source_change_invalidates_fingerprint(self, tmp_path):
        (tmp_path / "src").mkdir()
        app = tmp_path / "src" / "App.tsx"
        app.write_text("export default 1;\n")
        cb = ClientBuilder(tmp_path)
        before = cb._source_fingerprint()
        app.write_text("export default 2;\n")
        assert cb._source_fingerprint() != before

    def test_version_bump_alone_does_not_churn_fingerprint(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text(
            json.dumps({"name": "mjswan", "version": "0.7.0", "scripts": {}})
        )
        cb = ClientBuilder(tmp_path)
        before = cb._source_fingerprint()
        # Version is keyed separately and normalized out of the fingerprint.
        pkg.write_text(
            json.dumps({"name": "mjswan", "version": "0.8.0", "scripts": {}})
        )
        assert cb._source_fingerprint() == before


@pytest.mark.slow
class TestBuildPluginsModule:
    """esbuild-bundles author terms into a standalone ESM (needs Node/esbuild)."""

    def test_bundles_author_term_into_standalone_esm(self, tmp_path, monkeypatch):
        template = Path(mjswan.__file__).parent / "template"
        esbuild = template / "node_modules" / ".bin" / "esbuild"
        if not esbuild.exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        src = tmp_path / "MyEvent.ts"
        src.write_text(
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class MyEvent extends EventBase {\n"
            "  onReset(_ctx: EventContext): void {}\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"my_event": SimpleNamespace(ts_name="MyEvent", ts_src=str(src))},
        )
        out = tmp_path / "plugins.js"
        assert ClientBuilder(template).build_plugins_module(out) is True
        code = out.read_text()
        assert (
            "MyEvent" in code and "events" in code and "export {" in code
        )  # grouped export
        assert "EventBase" in code  # base class bundled (self-contained)
        assert "mjswan/event" not in code  # no bare imports remain

    def test_term_importing_three_reuses_engine_instance(self, tmp_path, monkeypatch):
        # `three` must resolve to the engine's instance; a duplicate breaks instanceof.
        template = Path(mjswan.__file__).parent / "template"
        if not (template / "node_modules" / ".bin" / "esbuild").exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        src = tmp_path / "ThreeObs.ts"
        src.write_text(
            "import * as THREE from 'three';\n"
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class ThreeObs extends EventBase {\n"
            "  onReset(_ctx: EventContext): void { void new THREE.Vector3(); }\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"three_obs": SimpleNamespace(ts_name="ThreeObs", ts_src=str(src))},
        )
        out = tmp_path / "plugins.js"
        assert ClientBuilder(template).build_plugins_module(out) is True
        code = out.read_text()
        assert "__mjswanThree" in code  # `three` resolved to the shared-instance shim
        assert "three" not in code.split("\n")[0]  # no bare `three` import survived


# ===========================================================================
# L1 — validation
# ===========================================================================
class TestBuilderValidation:
    def test_build_with_no_projects_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot build an empty application"):
            Builder().build(tmp_path / "out")

    def test_scene_with_a_policy_needs_a_control_dt(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        # Nothing about a wrong control rate raises at playback — it just plays at the
        # wrong speed — so the build refuses to guess.
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy", policy=minimal_onnx)
        with pytest.raises(ValueError, match="has policies but no control_dt"):
            build_manifest(builder, tmp_path / "out")

    def test_scene_without_a_policy_needs_no_control_dt(
        self, tmp_path, minimal_model, build_manifest
    ):
        # A viewer-only scene has no trained rate to match, so this must not raise.
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        build_manifest(builder, tmp_path / "out")

    def test_non_positive_control_dt_is_rejected(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="S", model=minimal_model, control_dt=0.0
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        with pytest.raises(ValueError, match="must be a positive number of seconds"):
            build_manifest(builder, tmp_path / "out")

    def test_control_dt_reaches_the_scene_entry(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="S", model=minimal_model, control_dt=0.05
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        manifest = build_manifest(builder, tmp_path / "out")
        assert manifest["projects"][0]["scenes"][0]["control_dt"] == 0.05

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
# L1 — from_mjlab wiring (no mjlab/wandb required, dependencies monkeypatched)
# ===========================================================================
class TestFromMjlab:
    """Verify Builder.from_mjlab forwards run_path to SceneHandle.add_policy_wandb."""

    @staticmethod
    def _patch(monkeypatch):
        """Patch add_scene_mjlab to skip the real mjlab loader and return a mock SceneHandle."""
        from mjswan.project import ProjectHandle

        scene_handle = MagicMock(name="SceneHandle")
        seen: dict[str, object] = {}

        def _stub(self, task_id, *, play, env_cfg):
            seen["task_id"], seen["play"], seen["env_cfg"] = task_id, play, env_cfg
            return scene_handle

        monkeypatch.setattr(ProjectHandle, "add_scene_mjlab", _stub)
        # Neither keyword has a default here on purpose: the caller always passes both, so a
        # stub default would hide a regression in what it passes.
        scene_handle.seen = seen
        return scene_handle

    def test_no_run_path_does_not_call_add_policy_wandb(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat")
        scene_handle.add_policy_wandb.assert_not_called()

    def test_str_run_path_forwarded_with_task_id(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat", run_path="org/proj/abc")
        scene_handle.add_policy_wandb.assert_called_once_with(
            "org/proj/abc", task_id="go2_flat"
        )

    def test_list_run_path_forwarded_as_is(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        run_paths = ["org/proj/a", "org/proj/b"]
        Builder.from_mjlab("go2_flat", run_path=run_paths)
        scene_handle.add_policy_wandb.assert_called_once_with(
            run_paths, task_id="go2_flat"
        )

    def test_forwards_play_unresolved(self, monkeypatch):
        """`play` must arrive as `None`, not as a materialised `True`.

        `add_scene_mjlab` rejects `play` together with `env_cfg`, so a wrapper that
        resolved the default here would make every `env_cfg=` call trip that guard.
        Which config `None` ends up meaning is pinned in `test_project.py`, against the
        real method.
        """
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat")
        assert scene_handle.seen["play"] is None
        assert scene_handle.seen["env_cfg"] is None

    def test_play_false_still_reaches_the_scene(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        Builder.from_mjlab("go2_flat", play=False)
        assert scene_handle.seen["play"] is False

    def test_env_cfg_reaches_the_scene(self, monkeypatch):
        scene_handle = self._patch(monkeypatch)
        sentinel = object()
        Builder.from_mjlab("go2_flat", env_cfg=sentinel)
        assert scene_handle.seen["env_cfg"] is sentinel
        assert scene_handle.seen["play"] is None


# ===========================================================================
# L1 — manifest.json structure (frontend build mocked)
# ===========================================================================
class TestSaveConfigJson:
    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "out" / "manifest.json").read_text())

    def test_config_contains_version(self, tmp_path, minimal_model, build_manifest):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        assert self._read_config(tmp_path)["version"] == mjswan.__version__

    def test_config_stamps_the_document_format(
        self, tmp_path, minimal_model, build_manifest
    ):
        # `format` is the structure, `version` the release: a reader gates on the first
        # only (ADR 0006 §7), so both travel and neither stands in for the other.
        from mjswan.document import DOCUMENT_FORMAT

        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        config = self._read_config(tmp_path)
        assert config["format"] == DOCUMENT_FORMAT
        assert isinstance(config["format"], int)

    def test_config_has_projects_list(self, tmp_path, minimal_model, build_manifest):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        config = self._read_config(tmp_path)
        assert isinstance(config["projects"], list)
        assert len(config["projects"]) == 1

    def test_project_name_and_id_in_config(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="Main Demo").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        project = self._read_config(tmp_path)["projects"][0]
        assert project["name"] == "Main Demo"
        assert project["id"] == "main_demo"
        # Unset means "first in document order"; the key is written only when set.
        assert "default" not in project

    def test_config_omits_plugins_when_declarative(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        config = self._read_config(tmp_path)
        assert config["uses_custom_js"] is False
        assert "plugins" not in config

    def test_config_references_plugins_when_custom_js(
        self, tmp_path, minimal_model, build_manifest, monkeypatch
    ):
        # A registered ts_src term flips uses_custom_js and adds the plugin ref.
        monkeypatch.setattr(
            evt_fns,
            "_custom_registry",
            {"e": SimpleNamespace(ts_src="x.ts", ts_name="E")},
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        config = self._read_config(tmp_path)
        assert config["uses_custom_js"] is True
        assert config["plugins"] == "assets/plugins.js"

    def test_scene_path_uses_name2id_with_mjb_for_model(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="My Scene", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["name"] == "My Scene"
        assert scene["id"] == "my_scene"
        assert scene["scene"] == "scene.mjb"
        assert (tmp_path / "out" / "p" / "my_scene" / "scene.mjb").is_file()

    def test_scene_path_uses_mjz_for_spec(self, tmp_path, minimal_spec, build_manifest):
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="My Scene", spec=minimal_spec
        )
        build_manifest(builder, tmp_path / "out")
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        assert scene["scene"] == "scene.mjz"
        assert (tmp_path / "out" / "p" / "my_scene" / "scene.mjz").is_file()

    def test_policy_entry_carries_its_id_mdp_and_onnx_path(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        build_manifest(builder, tmp_path / "out")
        scene = self._read_config(tmp_path)["projects"][0]["scenes"][0]
        policy = scene["policies"][0]
        assert policy["name"] == "Policy"
        assert policy["id"] == "policy"
        assert policy["mdp"] == "policy"
        assert policy["onnx"] == "policy/policy.onnx"
        assert (tmp_path / "out" / "p" / "s" / policy["onnx"]).is_file()
        assert [m["id"] for m in scene["mdps"]] == ["policy"]
        # No per-policy JSON anywhere: the manifest is the one descriptor.
        assert not list((tmp_path / "out" / "p").rglob("*.json"))

    def test_policy_motions_are_inlined_on_the_entry(
        self, tmp_path, minimal_model, minimal_onnx, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx).add_motion(
            name="Spin Kick",
            source="motion.npz",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            default=True,
        )

        build_manifest(builder, tmp_path / "out")
        policy = self._read_config(tmp_path)["projects"][0]["scenes"][0]["policies"][0]
        assert [(m["name"], m["default"]) for m in policy["motions"]] == [
            ("Spin Kick", True)
        ]
        assert policy["motions"][0]["path"] == "assets/spin_kick.npz"

    def test_multiple_projects_all_present_in_config(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="Project A").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.add_project(name="Project B").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        projects = self._read_config(tmp_path)["projects"]
        assert len(projects) == 2
        assert projects[0]["name"] == "Project A"
        assert projects[1]["name"] == "Project B"

    def test_second_project_auto_id_reflected_in_config(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="Main").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.add_project(name="MuJoCo Menagerie").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        projects = self._read_config(tmp_path)["projects"]
        assert projects[0]["id"] == "main"
        assert projects[1]["id"] == name2id("MuJoCo Menagerie")

    def test_default_project_flag_reaches_the_config(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        builder.add_project(name="A").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.add_project(name="B", default=True).add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        projects = self._read_config(tmp_path)["projects"]
        assert "default" not in projects[0]
        assert projects[1]["default"] is True


# ===========================================================================
# L1 — uses_custom_js manifest flag (ADR 0003)
# ===========================================================================
class TestUsesCustomJsFlag:
    """Builds with any `ts_src`-bearing sentinel must mark themselves
    custom-JS so mjswan Cloud can refuse them.  Declarative-only builds must
    be marked clean.
    """

    def _read_config(self, tmp_path: Path) -> dict:
        return json.loads((tmp_path / "out" / "manifest.json").read_text())

    def _isolate_registries(self, monkeypatch):
        """Swap each MDP custom-registry for an empty dict so the test does
        not see registrations leaked in from other tests / modules."""
        from mjswan import command as command_mod
        from mjswan.envs.mdp import events as events_mod
        from mjswan.envs.mdp import observations as obs_mod
        from mjswan.envs.mdp import terminations as term_mod

        monkeypatch.setattr(obs_mod, "_custom_registry", {})
        monkeypatch.setattr(term_mod, "_custom_registry", {})
        monkeypatch.setattr(events_mod, "_custom_registry", {})
        monkeypatch.setattr(command_mod, "_custom_registry", {})

    def test_clean_build_is_false(
        self, tmp_path, minimal_model, build_manifest, monkeypatch
    ):
        self._isolate_registries(monkeypatch)
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        assert self._read_config(tmp_path)["uses_custom_js"] is False

    def test_custom_obs_ts_src_flips_flag_true(
        self, tmp_path, minimal_model, build_manifest, monkeypatch
    ):
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import observations as obs_mod

        obs_mod._custom_registry["my_term"] = obs_mod.ObservationBinding(
            ts_name="MyTerm", ts_src="/tmp/whatever.ts"
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        assert self._read_config(tmp_path)["uses_custom_js"] is True

    def test_custom_term_ts_src_flips_flag_true(
        self, tmp_path, minimal_model, build_manifest, monkeypatch
    ):
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import terminations as term_mod

        term_mod._custom_registry["my_term"] = term_mod.TerminationBinding(
            ts_name="MyTerm", ts_src="/tmp/whatever.ts"
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        assert self._read_config(tmp_path)["uses_custom_js"] is True

    def test_declarative_override_does_not_flip_flag(
        self, tmp_path, minimal_model, build_manifest, monkeypatch
    ):
        """Registered sentinel without ts_src is a declarative param override —
        the build is still declarative-only."""
        self._isolate_registries(monkeypatch)
        from mjswan.envs.mdp import terminations as term_mod

        term_mod._custom_registry["out_of_terrain_bounds"] = (
            term_mod.TerminationBinding(
                ts_name="OutOfTerrainBounds",
                defaults={"limit_x": 5.0, "limit_y": 5.0},
            )
        )
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "out")
        assert self._read_config(tmp_path)["uses_custom_js"] is False


# ===========================================================================
# A live env for tracing, standing in for a task's own when the scene is a raw
# `add_scene(model=...)`. Satisfies `env.scene[name].data.<field>` with torch tensors.
def _fake_trace_env():
    import torch

    class _Data:
        def __init__(self, **fields):
            for k, v in fields.items():
                setattr(self, k, v)

    class _Entity:
        def __init__(self, data):
            self.data = data

    class _Scene:
        def __init__(self, entities):
            self._entities = entities

        def __getitem__(self, name):
            return self._entities[name]

    class _ActionTerm:
        def __init__(self, raw_action):
            self.raw_action = raw_action

    class _ActionManager:
        """Two terms, so `last_action(action_name=…)` is a real slice.

        Every buildable mjlab task declares exactly one action term, where a term's
        slice and the whole vector coincide — so a single-term fake could not tell a
        correct `action_offset` from a missing one. `arm` takes [0,3) and `gripper`
        the tail, mirroring `ActionManager.process_action`'s split.
        """

        def __init__(self):
            self.action = torch.tensor([[1.0, 2.0, 3.0, 9.0]])
            self.active_terms = ["arm", "gripper"]
            self.action_term_dim = [3, 1]

        def get_term(self, name):
            offset = 0
            for term_name, dim in zip(self.active_terms, self.action_term_dim):
                if term_name == name:
                    return _ActionTerm(self.action[:, offset : offset + dim])
                offset += dim
            raise KeyError(name)

    class _Env:
        def __init__(self, entities):
            self.scene = _Scene(entities)
            self.action_manager = _ActionManager()

    data = _Data(
        joint_pos=torch.tensor([[0.1, 0.2]]),
        default_joint_pos=torch.tensor([[0.0, 0.0]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.5]]),
    )
    return _Env({"robot": _Entity(data)})


# Named `last_action` because the tracer classifies natives by `func.__name__`.
def last_action(env, *, action_name=None, **_):
    if action_name is None:
        return env.action_manager.action
    return env.action_manager.get_term(action_name).raw_action


def _fake_joint_pos_rel(env, *, entity_name="robot", **_):
    d = env.scene[entity_name].data
    return d.joint_pos - d.default_joint_pos


def _fake_projected_gravity(env, *, entity_name="robot", **_):
    return env.scene[entity_name].data.projected_gravity_b


def _fake_bad_orientation(env, *, limit_angle, entity_name="robot", **_):
    import torch

    pg = env.scene[entity_name].data.projected_gravity_b
    return torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)).abs() > limit_angle


def _fake_root_height_below_minimum(env, *, minimum_height, entity_name="robot", **_):
    return env.scene[entity_name].data.root_link_pos_w[:, 2] < minimum_height


def _fake_time_out(env, *, max_episode_length=1e9, **_):
    import torch

    del env, max_episode_length
    return torch.tensor([False])


# ===========================================================================
# L1 — _save_web: actions/terminations serialization into policy JSON
# ===========================================================================
class TestSaveWebPolicyJson:
    """Tests for _save_web: verify actions/terminations are emitted into the
    generated policy JSON, covering both the no-config_path and config_path
    branches.  The frontend build and template copy are mocked out so these
    tests remain fast (L1).
    """

    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        """Skip the Node.js frontend build and the large template copytree."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.install_spa", MagicMock(return_value=True))

    def _run(self, builder: Builder, tmp_path: Path) -> Path:
        """Call _save_web and return the output directory."""
        out = tmp_path / "out"
        builder._save_web(out)
        return out

    def _policy_json(self, out: Path, policy_name: str) -> dict:
        """The policy's manifest entry merged with its MDP's: what the engine is handed."""
        return _entry(json.loads((out / "manifest.json").read_text()), policy_name)

    def test_spec_scene_config_path_matches_written_file(self, tmp_path, minimal_spec):
        """`_save_web` frees `scene.spec` right after writing `scene.mjz`, and writes
        `config.json` after that — the recorded path must still be the `.mjz` on disk,
        or the app 404s on the scene it just shipped."""
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", spec=minimal_spec)
        out = self._run(builder, tmp_path)

        scene = json.loads((out / "manifest.json").read_text())["projects"][0][
            "scenes"
        ][0]
        assert scene["scene"] == "scene.mjz"
        assert (out / "p" / scene["id"] / scene["scene"]).is_file()

    def test_terms_without_a_trace_env_name_the_missing_call(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A plain `add_scene` scene has no env to trace against until it is given one.

        The tracer used to reach `None.scene` and raise `AttributeError` from four
        frames down, which says nothing about the one line the author is missing.
        """

        def joint_pos(env):
            return env.scene["robot"].data.joint_pos

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={"joint_pos": ObservationTermCfg(func=joint_pos)}
                )
            },
        )
        with pytest.raises(ValueError, match="set_trace_env"):
            self._run(builder, tmp_path)

    # -----------------------------------------------------------------------
    # no-config_path branch
    # -----------------------------------------------------------------------

    def test_no_config_path_actions_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), scale=0.5, use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert "joint_pos" in data["actions"]
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["actions"]["joint_pos"]["scale"] == 0.5

    def test_no_config_path_terminations_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" in data
        assert "time_out" in data["terminations"]
        # A term reading no dynamic entity state is classified native (ADR 0005).
        entry = data["terminations"]["time_out"]
        assert entry["native"] == "elapsed_s >= episode_length_s"
        assert entry["time_out"] is True

    def test_no_config_path_both_blocks_emitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=2.0),
            },
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 1.2},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert "actions" in data
        assert "terminations" in data
        assert data["actions"]["effort"]["type"] == "torque"
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "mdp/policy/term/fallen.onnx"
        assert (out / "p" / "s" / fallen["onnx"]).exists()

    def test_no_config_path_actions_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data

    def test_no_config_path_terminations_absent_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "terminations" not in data

    def test_a_bare_policy_still_gets_an_entry_and_an_empty_mdp(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx)
        out = self._run(builder, tmp_path)
        manifest = json.loads((out / "manifest.json").read_text())
        scene_entry = manifest["projects"][0]["scenes"][0]
        assert scene_entry["mdps"] == [{"id": "policy"}]
        assert scene_entry["policies"] == [
            {
                "id": "policy",
                "name": "Policy",
                "mdp": "policy",
                "onnx": "policy/policy.onnx",
            }
        ]
        assert not (out / "p" / "s" / "mdp").exists()

    def test_no_config_path_onnx_path_in_json(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",))},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"] == "policy/policy.onnx"

    def test_no_config_path_motions_emitted_and_copied(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        motion_file = tmp_path / "spin_kick.npz"
        motion_file.write_bytes(b"motion-bytes")

        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        ).add_motion(
            name="Spin Kick",
            source=str(motion_file),
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            dataset_joint_names=["joint_a"],
            default=True,
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert data["motions"] == [
            {
                "name": "Spin Kick",
                # Scene-scoped and name-derived: no policy prefix, so the checkpoints of
                # one run reference a single bundled copy under the scene's assets/.
                "path": "assets/spin_kick.npz",
                "fps": 50.0,
                "anchor_body_name": "torso_link",
                "body_names": ["pelvis", "torso_link"],
                "dataset_joint_names": ["joint_a"],
                "default": True,
            }
        ]
        motion_out = out / "p" / "s" / "assets" / "spin_kick.npz"
        assert motion_out.read_bytes() == b"motion-bytes"

    def test_one_clip_shared_by_several_policies_is_written_once(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """The checkpoints of one run share a clip; a copy each meant N copies of it."""
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        clip = tmp_path / "spin_kick.npz"
        clip.write_bytes(b"motion-bytes")
        for name in ("model_0", "model_50"):
            scene.add_policy(name=name, policy=minimal_onnx).add_motion(
                name="Spin Kick",
                source=str(clip),
                anchor_body_name="torso_link",
                body_names=("pelvis",),
            )

        out = self._run(builder, tmp_path)
        assets_dir = out / "p" / "s" / "assets"

        assert sorted(p.name for p in assets_dir.glob("*.npz")) == ["spin_kick.npz"]
        for name in ("model_0", "model_50"):
            assert (
                self._policy_json(out, name)["motions"][0]["path"]
                == "assets/spin_kick.npz"
            )

    def test_same_name_different_content_gets_a_numbered_suffix(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        for payload in (b"clip-one", b"clip-two"):
            clip = tmp_path / f"{payload.decode()}.npz"
            clip.write_bytes(payload)
            scene.add_policy(
                name=f"policy_{payload.decode()}", policy=minimal_onnx
            ).add_motion(
                name="Motion",
                source=str(clip),
                anchor_body_name="b",
                body_names=("b",),
            )

        out = self._run(builder, tmp_path)
        assets_dir = out / "p" / "s" / "assets"

        assert sorted(p.name for p in assets_dir.glob("*.npz")) == [
            "motion.npz",
            "motion_1.npz",
        ]
        assert (assets_dir / "motion.npz").read_bytes() == b"clip-one"
        assert (assets_dir / "motion_1.npz").read_bytes() == b"clip-two"
        paths = {
            name: self._policy_json(out, name)["motions"][0]["path"]
            for name in ("policy_clip-one", "policy_clip-two")
        }
        assert paths == {
            "policy_clip-one": "assets/motion.npz",
            "policy_clip-two": "assets/motion_1.npz",
        }

    def test_no_config_path_commands_emitted_as_command_terms(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            commands={"velocity": mjswan.velocity_command()},
        )

        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["commands"]["velocity"]["name"] == "UiCommand"
        assert len(data["commands"]["velocity"]["ui"]["inputs"]) == 3
        assert data["commands"]["velocity"]["ui"]["inputs"][0]["name"] == "lin_vel_x"

    def test_plain_observation_term_traces_against_live_env(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # A plain-callable observation func — mjlab's own or self-authored, treated the
        # same — is traced against the scene's live env, never reimplemented.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(
                            func=_fake_joint_pos_rel, scale=0.5
                        ),
                    }
                ),
            },
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        # The group fuses: one graph named for it, under the owning policy's directory,
        # with `scale` folded in rather than shipped.
        group = data["observations"]["policy"]
        assert group["fused"] == "mdp/policy/obs/policy.onnx"
        assert [(r["name"], r["size"]) for r in group["layout"]] == [("joint_pos", 2)]
        assert group["layout"][0]["func"] == f"{__name__}:_fake_joint_pos_rel"
        assert group["size"] == 2
        assert "scale" not in group
        assert (out / "p" / "s" / group["fused"]).exists()

    def test_two_policies_in_one_scene_keep_their_own_graphs(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A graph's path is scoped to its MDP, so a sibling cannot overwrite it.

        Both policies key their one observation group the same way, as every single-input
        policy does, so unscoped they would share one file. Nothing would raise: the
        loser's config still declares its own `size`, and `conformToSize` pads or
        truncates the winner's vector to it silently.
        """
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        groups = {
            "Walk": {"joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel)},
            "Crawl": {
                "joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel),
                "gravity": ObservationTermCfg(func=_fake_projected_gravity),
            },
        }
        for policy_name, terms in groups.items():
            scene.add_policy(
                name=policy_name,
                policy=minimal_onnx,
                observations={"policy": ObservationGroupCfg(terms=terms)},
            )

        out = self._run(builder, tmp_path)
        walk = self._policy_json(out, "Walk")["observations"]["policy"]
        crawl = self._policy_json(out, "Crawl")["observations"]["policy"]

        assert walk["fused"] == "mdp/walk/obs/policy.onnx"
        assert crawl["fused"] == "mdp/crawl/obs/policy.onnx"
        assert (walk["size"], crawl["size"]) == (2, 5)

        scene_dir = out / "p" / "s"
        walk_bytes = (scene_dir / walk["fused"]).read_bytes()
        crawl_bytes = (scene_dir / crawl["fused"]).read_bytes()
        assert walk_bytes != crawl_bytes

    def test_two_policies_sharing_one_mdp_trace_it_once(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """The checkpoints of one run share an MDP: one directory, one set of graphs."""
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        mdp = mjswan.MdpConfig(
            observations={
                "policy": ObservationGroupCfg(
                    terms={"joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel)}
                )
            },
            commands={},
        )
        scene.add_policy(name="model_1000", policy=minimal_onnx, mdp=mdp)
        scene.add_policy(name="model_2000", policy=minimal_onnx, mdp=mdp)

        out = self._run(builder, tmp_path)
        manifest = json.loads((out / "manifest.json").read_text())
        scene_entry = manifest["projects"][0]["scenes"][0]
        assert [m["id"] for m in scene_entry["mdps"]] == ["mdp_0"]
        assert [p["mdp"] for p in scene_entry["policies"]] == ["mdp_0", "mdp_0"]
        assert sorted(d.name for d in (out / "p" / "s" / "mdp").iterdir()) == ["mdp_0"]
        assert (out / "p" / "s" / "mdp" / "mdp_0" / "obs" / "policy.onnx").is_file()
        assert sorted(f.name for f in (out / "p" / "s" / "policy").iterdir()) == [
            "model_1000.onnx",
            "model_2000.onnx",
        ]

    def test_policies_sharing_an_mdp_must_agree_on_their_joint_names(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        mdp = mjswan.MdpConfig(commands={})
        scene.add_policy(
            name="A", policy=minimal_onnx, mdp=mdp, policy_joint_names=["j1"]
        )
        scene.add_policy(
            name="B", policy=minimal_onnx, mdp=mdp, policy_joint_names=["j2"]
        )
        with pytest.raises(ValueError, match="share one MdpConfig but disagree"):
            self._run(builder, tmp_path)

    def test_last_action_with_a_term_name_emits_its_slice_offset(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # The two-action-term shape, through the real Builder. No mjlab task has it, so
        # otherwise the offset is only ever checked against a stub action manager.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel),
                        "gripper_action": ObservationTermCfg(
                            func=last_action, params={"action_name": "gripper"}
                        ),
                    }
                ),
            },
        )

        group = self._policy_json(self._run(builder, tmp_path), "Policy")[
            "observations"
        ]["policy"]
        native = next(
            entry
            for entry in group["native_inputs"]
            if entry["name"] == "gripper_action"
        )
        assert native["native"] == "prev_action"
        assert native["action_name"] == "gripper"
        # `arm` holds [0,3), so `gripper` starts at 3. Without the offset the runtime feeds
        # `arm`'s first element at `gripper`'s width: right width, wrong term.
        assert native["action_offset"] == 3
        assert native["size"] == 1
        # And the group still fuses around it: the native term is a graph *input*.
        assert [(r["name"], r["size"]) for r in group["layout"]] == [
            ("joint_pos", 2),
            ("gripper_action", 1),
        ]

    def test_last_action_naming_no_action_term_fails_the_build(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # Degrading would hand over the action vector's head — the silent wrong answer.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel),
                        "grip": ObservationTermCfg(
                            func=last_action, params={"action_name": "grippr"}
                        ),
                    }
                ),
            },
        )
        with pytest.raises(ValueError, match=r"does not define.*arm, gripper"):
            self._run(builder, tmp_path)

    def test_per_term_observations_when_the_group_cannot_fuse(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        # A term stacking its own history keeps the per-term path: mjlab stacks before
        # concatenating, so a fused output would order the group's history differently.
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "policy": ObservationGroupCfg(
                    terms={
                        "joint_pos": ObservationTermCfg(
                            func=_fake_joint_pos_rel, scale=0.5, history_length=3
                        ),
                    }
                ),
            },
        )

        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        terms = data["observations"]["policy"]
        assert isinstance(terms, list)
        assert terms[0]["onnx"] == "mdp/policy/obs/joint_pos.onnx"
        assert terms[0]["scale"] == 0.5
        assert terms[0]["history_length"] == 3

    # -----------------------------------------------------------------------
    # config_path branch
    # -----------------------------------------------------------------------

    def test_config_path_actions_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=True
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" in data
        assert data["actions"]["joint_pos"]["type"] == "joint_position"
        assert data["existing_key"] == "kept"

    def test_config_path_terminations_merged_into_existing_config(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps({"onnx": {"path": "old.onnx"}, "existing_key": "kept"})
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 0.8},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert "terminations" in data
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "mdp/policy/term/fallen.onnx"
        assert (out / "p" / "s" / fallen["onnx"]).exists()
        assert data["existing_key"] == "kept"

    def test_config_path_both_blocks_merged(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "effort": JointEffortActionCfg(actuator_names=(".*",), scale=1.5),
            },
            terminations={
                "height": TerminationTermCfg(
                    func=_fake_root_height_below_minimum,
                    params={"minimum_height": 0.3},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        assert data["actions"]["effort"]["type"] == "torque"
        assert data["actions"]["effort"]["scale"] == 1.5
        # A term reading dynamic entity state is traced to ONNX (ADR 0005).
        height_entry = data["terminations"]["height"]
        assert height_entry["onnx"] == "mdp/policy/term/height.onnx"
        assert (out / "p" / "s" / height_entry["onnx"]).exists()

    def test_config_path_overwrites_existing_actions_block(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """actions from policy.actions fully replaces any pre-existing actions block."""
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {
                    "onnx": {"path": "old.onnx"},
                    "actions": {"old_action": {"type": "joint_position"}},
                }
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "new_action": JointPositionActionCfg(actuator_names=(".*",)),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "new_action" in data["actions"]
        assert "old_action" not in data["actions"]

    def test_config_path_action_fields_survive_a_partial_override(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A term names what it changes; the rest of the authored entry stays.

        For a motor-actuator robot the PD gains live in the policy's own config —
        the model has none — so a scene that only wants a different offset must not
        have to restate them.
        """
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {
                    "onnx": {"path": "old.onnx"},
                    "actions": {
                        "joint_pos": {
                            "type": "joint_position",
                            "scale": {"j": 0.5},
                            "stiffness": {"j": 40.0},
                            "damping": {"j": 2.5},
                        }
                    },
                }
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), offset={"j": 0.25}
                )
            },
        )
        entry = self._policy_json(self._run(builder, tmp_path), "Policy")["actions"][
            "joint_pos"
        ]
        assert entry["offset"] == {"j": 0.25}
        assert entry["stiffness"] == {"j": 40.0}
        assert entry["damping"] == {"j": 2.5}
        assert entry["scale"] == {"j": 0.5}

    def test_config_path_slot_tables_are_ignored_with_a_warning(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        """A sidecar's `onnx` block never reaches the output, and its tables are not read.

        The slot tables are declared on `add_policy` now (ADR 0006 §5); a sidecar still
        carrying one is told so rather than silently obeyed.
        """
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(
            json.dumps(
                {"onnx": {"path": "stale.onnx", "meta": {"in_keys": ["obs_history"]}}}
            )
        )
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
        )
        with pytest.warns(RuntimeWarning, match="in_keys"):
            data = self._policy_json(self._run(builder, tmp_path), "Policy")
        # The network's path is the build's, never the sidecar's stale one.
        assert data["onnx"] == "policy/policy.onnx"
        assert "in_keys" not in data
        assert "out_keys" not in data

    def test_declared_slot_tables_land_in_the_entry_and_defaults_are_omitted(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        terms = {"joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel)}
        scene.add_policy(
            name="Two",
            policy=_two_input_onnx(),
            observations={
                "actor": ObservationGroupCfg(terms=terms),
                "command_": ObservationGroupCfg(terms=terms),
            },
            in_keys=["command_", "actor"],
            out_keys=["action"],
        )
        scene.add_policy(
            name="One",
            policy=minimal_onnx,
            observations=ObservationGroupCfg(terms=terms),
        )

        out = self._run(builder, tmp_path)
        two = self._policy_json(out, "Two")
        assert two["in_keys"] == ["command_", "actor"]
        assert "out_keys" not in two  # `["action"]` is the default
        one = self._policy_json(out, "One")
        assert "in_keys" not in one  # a lone group under `actor` needs no table
        assert one["observations"]["actor"]["fused"] == "mdp/one/obs/actor.onnx"

    def test_a_single_input_policy_takes_its_one_group_whatever_it_is_called(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            observations={
                "obs_history": ObservationGroupCfg(
                    terms={"joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel)}
                )
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        # Not the default, so the table is written, and the runtime feeds that group.
        assert data["in_keys"] == ["obs_history"]

    def test_a_multi_input_policy_without_in_keys_is_refused(self, minimal_model):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(control_dt=0.02, name="S", model=minimal_model)
        )
        with pytest.raises(ValueError, match="2 ONNX inputs .* no in_keys"):
            scene.add_policy(name="Two", policy=_two_input_onnx())

    def test_a_multi_output_policy_without_out_keys_is_warned_about(
        self, minimal_model
    ):
        # Which output is the action is unknowable from the network, so the runtime falls
        # back to the first: a wrong guess drives the actuators from the wrong tensor.
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(control_dt=0.02, name="S", model=minimal_model)
        )
        with pytest.warns(RuntimeWarning, match="2 ONNX outputs .* no out_keys"):
            scene.add_policy(name="Two", policy=_two_output_onnx())

    def test_slot_tables_must_match_the_networks_input_and_output_counts(
        self, minimal_model, minimal_onnx
    ):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(control_dt=0.02, name="S", model=minimal_model)
        )
        with pytest.raises(ValueError, match="2 in_keys .* 1 inputs"):
            scene.add_policy(name="P", policy=minimal_onnx, in_keys=["a", "b"])
        with pytest.raises(ValueError, match="2 out_keys .* 1 outputs"):
            scene.add_policy(name="Q", policy=minimal_onnx, out_keys=["a", "b"])

    def test_an_in_key_naming_no_group_fails_the_build(self, tmp_path, minimal_model):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        terms = {"joint_pos": ObservationTermCfg(func=_fake_joint_pos_rel)}
        scene.add_policy(
            name="Two",
            policy=_two_input_onnx(),
            observations={
                "actor": ObservationGroupCfg(terms=terms),
                "command_": ObservationGroupCfg(terms=terms),
            },
            in_keys=["actor", "command"],  # typo: the group is `command_`
        )
        with pytest.raises(ValueError, match=r"\['command'\]"):
            self._run(builder, tmp_path)

    def test_config_path_onnx_path_updated_when_extras_present(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "stale.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
            extras={"model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["onnx"] == "policy/policy.onnx"
        assert data["extras"] == {
            "model_overrides": {"geom_friction": [1.0, 0.5, 0.25]}
        }

    def test_config_path_actions_absent_from_json_when_not_set(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        config_file = tmp_path / "policy_cfg.json"
        config_file.write_text(json.dumps({"onnx": {"path": "old.onnx"}}))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            config_path=str(config_file),
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert "actions" not in data
        assert "terminations" not in data

    # -----------------------------------------------------------------------
    # Serialization correctness
    # -----------------------------------------------------------------------

    def test_joint_effort_action_scale_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"effort": JointEffortActionCfg(actuator_names=(".*",), scale=3.0)},
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["effort"] == {
            "type": "torque",
            "scale": 3.0,
            "actuator_names": [".*"],
        }

    def test_joint_position_default_offset_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(
                    actuator_names=(".*",), use_default_offset=False
                ),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        assert data["actions"]["joint_pos"]["use_default_offset"] is False

    def test_timeout_termination_time_out_flag(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "time_out": TerminationTermCfg(func=_fake_time_out, time_out=True),
            },
        )
        data = self._policy_json(self._run(builder, tmp_path), "Policy")
        term = data["terminations"]["time_out"]
        # A term reading no dynamic entity state is classified native (ADR 0005).
        assert term["native"] == "elapsed_s >= episode_length_s"
        assert term.get("time_out") is True

    def test_bad_orientation_params_serialized(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        pytest.importorskip("torch")
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        scene._config.mjlab_env = _fake_trace_env()
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            terminations={
                "fallen": TerminationTermCfg(
                    func=_fake_bad_orientation,
                    params={"limit_angle": 1.57},
                ),
            },
        )
        out = self._run(builder, tmp_path)
        data = self._policy_json(out, "Policy")
        # Traced to ONNX, with `limit_angle` closed over by the function, not serialized.
        fallen = data["terminations"]["fallen"]
        assert fallen["onnx"] == "mdp/policy/term/fallen.onnx"
        assert (out / "p" / "s" / fallen["onnx"]).exists()
        assert "time_out" not in fallen


# ===========================================================================
# L3 slow — full build pipeline (triggers frontend compilation) Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuild:
    def test_build_creates_the_root_manifest(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "manifest.json").exists()
        assert not (tmp_path / "out" / "assets" / "config.json").exists()

    def test_build_with_model_creates_mjb_file(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "test" / "scene"
        assert (scene_dir / "scene.mjb").exists()

    def test_build_with_spec_creates_mjz_file(self, tmp_path, minimal_spec):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", spec=minimal_spec
        )
        builder.build(tmp_path / "out")
        scene_dir = tmp_path / "out" / "test" / "scene"
        assert (scene_dir / "scene.mjz").exists()

    def test_build_names_the_project_directory_by_its_id(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="My Demo").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "my_demo").is_dir()
        assert not (tmp_path / "out" / "main").exists()

    def test_build_writes_no_per_project_index_or_logo(self, tmp_path, minimal_model):
        # The SPA selects a project from `?project=` against the build-time base URL, so
        # nothing would ever request `<project-id>/index.html`.
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "index.html").is_file()
        assert not (tmp_path / "out" / "test" / "index.html").exists()
        assert not (tmp_path / "out" / "test" / "logo.svg").exists()

    def test_build_returns_mjswan_app_instance(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        app = builder.build(tmp_path / "out")
        assert isinstance(app, mjswan.MjswanApp)

    def test_build_output_excludes_dev_and_test_files(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder.build(out)
        # Dev/test scaffolding must not ship in the published SPA.
        for leaked in (
            "e2e",
            "harness.html",
            "fixtures",
            "vite.config.ts",
            "vite.lib.config.ts",
            "vite.manifest.config.ts",
            "vitest.config.ts",
            "playwright.config.ts",
            "lib.d.ts",
            "manifest.d.ts",
            "src",
            "node_modules",
            "package.json",
        ):
            assert not (out / leaked).exists(), f"dev file leaked into build: {leaked}"
        assert (out / "index.html").exists() and (out / "manifest.json").exists()


# ===========================================================================
# L1 — mt parameter: _save_mt_headers / no-headers when mt=False
# ===========================================================================
class TestMtHeaders:
    def test_mt_defaults_to_false(self):
        assert Builder()._mt is False

    def test_mt_true_stored(self):
        assert Builder(mt=True)._mt is True

    def test_save_mt_headers_creates_headers_file(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        assert (tmp_path / "_headers").exists()

    def test_save_mt_headers_contains_coop(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content

    def test_save_mt_headers_contains_coep(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_mt_headers_applies_wildcard_route(self, tmp_path):
        Builder()._save_mt_headers(tmp_path)
        content = (tmp_path / "_headers").read_text()
        assert content.startswith("/*")

    def test_mt_false_does_not_write_headers(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web with mt=False must not create _headers."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.install_spa", MagicMock(return_value=True))
        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)
        assert not (out / "_headers").exists()

    def test_mt_true_writes_headers(self, tmp_path, minimal_model, monkeypatch):
        """_save_web with mt=True must create _headers with COOP/COEP content."""
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.install_spa", MagicMock(return_value=True))
        builder = Builder(mt=True)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)
        headers_file = out / "_headers"
        assert headers_file.exists()
        content = headers_file.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_save_web_excludes_mt_template_dir(
        self, tmp_path, minimal_model, monkeypatch
    ):
        """_save_web must not copy the template-only _mt directory into output.

        The output is assembled allowlist-style from the built dist/ (+ LICENSE),
        so template-root scaffolding like _mt is excluded by construction.
        """
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.install_spa", MagicMock(return_value=True))

        builder = Builder(mt=False)
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        out = tmp_path / "out"
        builder._save_web(out)

        assert not (out / "_mt").exists()


# ===========================================================================
# L3 slow — Phase 4: cache reuse + custom-JS runtime plugin module Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildPhase4:
    def test_second_build_reuses_cached_frontend(self, tmp_path, minimal_model, capsys):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(tmp_path / "out1")  # warms the cache
        capsys.readouterr()
        builder2 = Builder()
        builder2.add_project(name="Test").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder2.build(tmp_path / "out2")  # identical inputs → no frontend rebuild
        assert "Reusing cached frontend build" in capsys.readouterr().out
        assert (tmp_path / "out2" / "manifest.json").exists()

    def test_custom_js_build_emits_plugins_module(
        self, tmp_path, minimal_model, monkeypatch
    ):
        from mjswan.envs.mdp import events as evt_mod

        template = Path(mjswan.__file__).parent / "template"
        if not (template / "node_modules" / ".bin" / "esbuild").exists():
            pytest.skip("esbuild not installed (run npm install in template)")

        term = tmp_path / "MyEvent.ts"
        term.write_text(
            "import { EventBase, type EventContext } from 'mjswan/event';\n"
            "export class MyEvent extends EventBase {\n"
            "  onReset(_ctx: EventContext): void {}\n"
            "}\n"
        )
        monkeypatch.setattr(
            evt_mod,
            "_custom_registry",
            {"my_event": SimpleNamespace(ts_name="MyEvent", ts_src=str(term))},
        )
        out = tmp_path / "out"
        builder = Builder()
        builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=minimal_model
        )
        builder.build(out)

        config = json.loads((out / "manifest.json").read_text())
        assert config["uses_custom_js"] is True
        assert config["plugins"] == "assets/plugins.js"
        plugins = (out / "assets" / "plugins.js").read_text()
        assert "MyEvent" in plugins and "events" in plugins
        assert "mjswan/event" not in plugins  # standalone (base class bundled)


# ===========================================================================
# L3 slow — full mt=True build (triggers frontend compilation) Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildMt:
    def test_mt_false_no_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "_headers").exists()

    def test_mt_false_no_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=False)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert not (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_writes_headers_file(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        headers = tmp_path / "out" / "_headers"
        assert headers.exists()
        content = headers.read_text()
        assert "Cross-Origin-Opener-Policy: same-origin" in content
        assert "Cross-Origin-Embedder-Policy: require-corp" in content

    def test_mt_true_emits_coi_serviceworker(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        assert (tmp_path / "out" / "coi-serviceworker.js").exists()

    def test_mt_true_injects_sw_script_into_html(self, tmp_path, minimal_model):
        builder = Builder(mt=True)
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "coi-serviceworker.js" in html
        assert "crossOriginIsolated" in html


@pytest.mark.slow
class TestFullBuildGtmId:
    def test_gtm_snippet_injected_into_all_html_files(self, tmp_path, minimal_model):
        builder = Builder(gtm_id="GTM-SAMPLE123")
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "GTM-SAMPLE123" in html
        assert "googletagmanager.com/gtm.js" in html  # <head> script
        assert "googletagmanager.com/ns.html" in html  # <body> noscript

    def test_no_gtm_without_gtm_id(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="Test").add_scene(
            control_dt=0.02, name="Scene", model=minimal_model
        )
        builder.build(tmp_path / "out")
        html = (tmp_path / "out" / "index.html").read_text()
        assert "googletagmanager.com" not in html
