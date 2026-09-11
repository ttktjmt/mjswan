"""Tests for mjswan project/scene/policy data models and fluent-API handles.

Layer: L1 (no I/O beyond in-memory MuJoCo and ONNX objects).
Tests the "contract" of the builder's hierarchical configuration API:
  Builder → ProjectHandle → SceneHandle → PolicyHandle
"""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import pytest

import mjswan
from mjswan.builder import Builder
from mjswan.command import CommandTermConfig, SliderConfig, ui_command
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.project import _collect_mjlab_scene_assets
from mjswan.scene import SceneConfig


class _FakeSceneCfg:
    def __init__(self):
        self.num_envs = 16
        self.terrain = None
        self.entities = {}


class _FakeEnvCfg:
    def __init__(self):
        self.scene = _FakeSceneCfg()
        self.viewer = None
        self.events = None
        # `add_scene_mjlab` reads the control rate off the config now rather than off a
        # constructed env, so a fake without these would look like an unusable task.
        self.sim = type("Sim", (), {"mujoco": type("Mj", (), {"timestep": 0.01})()})()
        self.decimation = 5


def _install_fake_mjlab(monkeypatch, minimal_spec) -> tuple[list[tuple], _FakeEnvCfg]:
    """Stub out mjlab's Scene/env/registry; return the call log and the cfg the
    fake registry hands back from load_env_cfg."""
    calls: list[tuple] = []
    registry_env_cfg = _FakeEnvCfg()

    class FakeScene:
        def __init__(self, scene_cfg, device: str):
            calls.append(("scene", scene_cfg, device))
            self.spec = minimal_spec
            self.terrain = None

    class FakeManagerBasedRlEnv:
        """Stands in for mjlab's real env — ADR 0005 needs a live env to trace term
        bodies, built lazily at build time by `builder._scene_trace_env`."""

        def __init__(self, env_cfg, device: str):
            calls.append(("env", env_cfg, device))

        def reset(self):
            calls.append(("env_reset",))

    def fake_load_env_cfg(task_id: str, play: bool = False):
        calls.append(("load_env_cfg", task_id, play))
        return registry_env_cfg

    mjlab_scene_module = ModuleType("mjlab.scene")
    mjlab_scene_module.Scene = FakeScene
    mjlab_envs_module = ModuleType("mjlab.envs")
    mjlab_envs_module.ManagerBasedRlEnv = FakeManagerBasedRlEnv
    mjlab_registry_module = ModuleType("mjlab.tasks.registry")
    mjlab_registry_module.load_env_cfg = fake_load_env_cfg

    monkeypatch.setitem(sys.modules, "mjlab", ModuleType("mjlab"))
    monkeypatch.setitem(sys.modules, "mjlab.scene", mjlab_scene_module)
    monkeypatch.setitem(sys.modules, "mjlab.envs", mjlab_envs_module)
    monkeypatch.setitem(sys.modules, "mjlab.tasks", ModuleType("mjlab.tasks"))
    monkeypatch.setitem(sys.modules, "mjlab.tasks.registry", mjlab_registry_module)

    return calls, registry_env_cfg


# ===========================================================================
# License files (ADR 0007 §2): detection and the known-assets table
# ===========================================================================
class TestSceneAttributions:
    def test_a_license_beside_the_model_is_detected_at_add_scene(
        self, tmp_path, minimal_spec
    ):
        # `minimal_spec` was loaded from tmp_path/model.xml, so a file there sits
        # beside the model.
        (tmp_path / "LICENSE").write_text(
            "MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\n"
        )
        scene = Builder().add_project(name="P").add_scene(name="S", spec=minimal_spec)
        found = scene._config.attributions
        assert len(found) == 1
        assert found[0].spdx == "MIT"
        assert found[0].origin == str(tmp_path / "LICENSE")

    def test_a_model_without_a_file_or_a_known_name_gets_nothing(self, minimal_spec):
        scene = Builder().add_project(name="P").add_scene(name="S", spec=minimal_spec)
        assert scene._config.attributions == []

    def test_a_known_model_name_gets_a_generated_file(self, tmp_path):
        xml = tmp_path / "go2.xml"
        xml.write_text(
            '<mujoco model="go2 scene"><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
        )
        spec = mujoco.MjSpec.from_file(str(xml))
        scene = Builder().add_project(name="P").add_scene(name="S", spec=spec)
        (found,) = scene._config.attributions
        assert found.component == "unitree_go2"
        assert found.spdx == "BSD-3-Clause"
        assert found.origin.startswith("known asset")

    def test_a_model_given_as_mjmodel_has_no_directory_to_search(self, minimal_model):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        assert scene._config.attributions == []

    def test_add_scene_mjlab_falls_back_to_the_task_id(self, monkeypatch, minimal_spec):
        _install_fake_mjlab(monkeypatch, minimal_spec)
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene_mjlab("Mjlab-Velocity-Flat-Unitree-G1")
        )
        (found,) = scene._config.attributions
        assert found.component == "unitree_g1"
        assert found.license is not None
        assert found.license.split(b"\n")[0] == b"SPDX-License-Identifier: BSD-3-Clause"

    def test_add_scene_mjlab_prefers_a_file_beside_an_entity_spec(
        self, monkeypatch, minimal_spec, tmp_path
    ):
        _, env_cfg = _install_fake_mjlab(monkeypatch, minimal_spec)
        robot_dir = tmp_path / "asset_zoo" / "robots" / "some_robot"
        robot_dir.mkdir(parents=True)
        (robot_dir / "robot.xml").write_text(
            '<mujoco model="r"><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
        )
        (robot_dir / "LICENSE").write_text("Use it however you like.\n")
        robot_spec = mujoco.MjSpec.from_file(str(robot_dir / "robot.xml"))
        env_cfg.scene.entities = {
            "robot": type("E", (), {"spec_fn": staticmethod(lambda: robot_spec)})()
        }

        scene = (
            Builder()
            .add_project(name="P")
            .add_scene_mjlab("Mjlab-Velocity-Flat-Unitree-G1")
        )
        (found,) = scene._config.attributions
        assert found.component == "some_robot"
        assert found.license == b"Use it however you like.\n"

    def test_an_unknown_task_with_no_files_gets_nothing(
        self, monkeypatch, minimal_spec
    ):
        _install_fake_mjlab(monkeypatch, minimal_spec)
        scene = Builder().add_project(name="P").add_scene_mjlab("Mjlab-Cartpole")
        assert scene._config.attributions == []


# ===========================================================================
# SceneConfig — scene_filename property
# ===========================================================================
class TestSceneConfig:
    def test_scene_filename_is_mjz_when_spec_provided(self, minimal_spec):
        cfg = SceneConfig(name="Test", spec=minimal_spec)
        assert cfg.scene_filename == "scene.mjz"

    def test_scene_filename_is_mjb_when_model_provided(self, minimal_model):
        cfg = SceneConfig(name="Test", model=minimal_model)
        assert cfg.scene_filename == "scene.mjb"

    def test_scene_filename_survives_the_build_releasing_the_spec(self, minimal_spec):
        cfg = SceneConfig(name="Test", spec=minimal_spec)
        cfg.spec = None  # what Builder._save_web does after writing scene.mjz
        assert cfg.scene_filename == "scene.mjz"


# ===========================================================================
# ProjectHandle — add_scene validation and return type
# ===========================================================================
class TestProjectHandle:
    def test_add_scene_with_neither_raises(self):
        project = Builder().add_project(name="P")
        with pytest.raises(ValueError):
            project.add_scene(name="S")  # neither model nor spec

    def test_add_scene_with_both_raises(self, minimal_model, minimal_spec):
        project = Builder().add_project(name="P")
        with pytest.raises(ValueError):
            project.add_scene(name="S", model=minimal_model, spec=minimal_spec)

    def test_add_scene_with_model_returns_scene_handle(self, minimal_model):
        project = Builder().add_project(name="P")
        scene = project.add_scene(name="S", model=minimal_model)
        assert isinstance(scene, mjswan.SceneHandle)

    def test_add_scene_with_spec_returns_scene_handle(self, minimal_spec):
        project = Builder().add_project(name="P")
        scene = project.add_scene(name="S", spec=minimal_spec)
        assert isinstance(scene, mjswan.SceneHandle)

    def test_add_scene_appended_to_project_scenes(self, minimal_model):
        builder = Builder()
        project = builder.add_project(name="P")
        project.add_scene(name="Scene A", model=minimal_model)
        project.add_scene(name="Scene B", model=minimal_model)
        scenes = builder.get_projects()[0].scenes
        assert len(scenes) == 2
        assert scenes[0].name == "Scene A"
        assert scenes[1].name == "Scene B"

    def test_project_name_and_id_exposed(self):
        project = Builder().add_project(name="My Project")
        assert project.name == "My Project"
        assert project.id == "my_project"

    def test_collect_mjlab_scene_assets_uses_terrain_and_entities(
        self, monkeypatch, tmp_path: Path
    ):
        def make_spec(label: str) -> mujoco.MjSpec:
            xml_path = tmp_path / f"{label}.xml"
            xml_path.write_text(
                f'<mujoco model="{label}">'
                '<worldbody><geom type="sphere" size="0.1"/></worldbody>'
                "</mujoco>"
            )
            return mujoco.MjSpec.from_file(str(xml_path))

        class FakeCfg:
            def __init__(self, spec: mujoco.MjSpec):
                self._spec = spec

            def spec_fn(self):
                return self._spec

        terrain_spec = make_spec("terrain")
        robot_spec = make_spec("robot")
        prop_spec = make_spec("prop")

        class FakeSceneCfg:
            terrain = FakeCfg(terrain_spec)
            entities = {
                "robot": FakeCfg(robot_spec),
                "prop": FakeCfg(prop_spec),
            }

        def fake_collect_spec_assets(spec):
            return {f"{spec.modelname}.bin": spec.modelname.encode()}

        monkeypatch.setattr(
            "mjswan.project.collect_spec_assets",
            fake_collect_spec_assets,
        )

        assets = _collect_mjlab_scene_assets(FakeSceneCfg())

        assert assets == {
            "terrain.bin": b"terrain",
            "robot.bin": b"robot",
            "prop.bin": b"prop",
        }

    def test_add_scene_mjlab_passes_play_flag_to_load_env_cfg(
        self, monkeypatch, minimal_spec
    ):
        calls, registry_env_cfg = _install_fake_mjlab(monkeypatch, minimal_spec)

        project = Builder().add_project(name="P")
        scene = project.add_scene_mjlab("Mjlab-Velocity-Rough-Unitree-G1", play=True)

        assert isinstance(scene, mjswan.SceneHandle)
        assert [c[0] for c in calls] == ["load_env_cfg", "scene"]
        assert calls[0] == ("load_env_cfg", "Mjlab-Velocity-Rough-Unitree-G1", True)
        assert calls[1] == ("scene", registry_env_cfg.scene, "cpu")
        assert registry_env_cfg.scene.num_envs == 1
        # The tracing env is deferred to build time; the config it will be built from is
        # what the scene keeps.
        assert scene._config.mjlab_env is None
        assert scene._config.mjlab_env_cfg is registry_env_cfg
        # The control rate is `timestep * decimation`, not the timestep.
        assert scene._config.control_dt == 0.05

    def test_add_scene_mjlab_defaults_to_the_play_config(
        self, monkeypatch, minimal_spec
    ):
        """mjswan is a playback tool, so its default is the opposite of mjlab's.

        The training config sets `episode_length_s` to 10-20 s and mjswan serializes that
        into the browser's `time_out` termination, so a viewer built from it resets the
        robot every few seconds. Play also drops `push_robot` and the terrain-bounds
        termination.
        """
        calls, _ = _install_fake_mjlab(monkeypatch, minimal_spec)

        Builder().add_project(name="P").add_scene_mjlab("Mjlab-Cartpole-Balance")

        assert calls[0] == ("load_env_cfg", "Mjlab-Cartpole-Balance", True)

    def test_add_scene_mjlab_play_false_is_honoured(self, monkeypatch, minimal_spec):
        calls, _ = _install_fake_mjlab(monkeypatch, minimal_spec)

        Builder().add_project(name="P").add_scene_mjlab(
            "Mjlab-Cartpole-Balance", play=False
        )

        assert calls[0] == ("load_env_cfg", "Mjlab-Cartpole-Balance", False)

    def test_add_scene_mjlab_rejects_play_together_with_env_cfg(
        self, monkeypatch, minimal_spec
    ):
        """`env_cfg` is already one of the task's two configs, so `play` selects nothing.

        Before the guard the contradiction resolved silently in `env_cfg`'s favour, which
        is how `play=False` next to an `env_cfg=` could read as honoured and not be.
        """
        _install_fake_mjlab(monkeypatch, minimal_spec)
        project = Builder().add_project(name="P")

        with pytest.raises(ValueError, match="not both"):
            project.add_scene_mjlab("t", play=False, env_cfg=_FakeEnvCfg())
        # Redundant-but-agreeing is refused too: there is still nothing for it to select.
        with pytest.raises(ValueError, match="not both"):
            project.add_scene_mjlab("t", play=True, env_cfg=_FakeEnvCfg())

    def test_from_mjlab_reaches_the_play_config(self, monkeypatch, minimal_spec):
        """End-to-end through the wrapper, since its own test only sees `play=None`."""
        calls, _ = _install_fake_mjlab(monkeypatch, minimal_spec)

        Builder.from_mjlab("Mjlab-Cartpole-Balance")

        assert calls[0] == ("load_env_cfg", "Mjlab-Cartpole-Balance", True)

    def test_from_mjlab_env_cfg_skips_the_registry(self, monkeypatch, minimal_spec):
        calls, _ = _install_fake_mjlab(monkeypatch, minimal_spec)
        caller_env_cfg = _FakeEnvCfg()

        Builder.from_mjlab("Mjlab-Cartpole-Balance", env_cfg=caller_env_cfg)

        assert [c[0] for c in calls] == ["scene"]

    def test_from_mjlab_rejects_play_together_with_env_cfg(
        self, monkeypatch, minimal_spec
    ):
        _install_fake_mjlab(monkeypatch, minimal_spec)
        with pytest.raises(ValueError, match="not both"):
            Builder.from_mjlab("t", play=True, env_cfg=_FakeEnvCfg())

    def test_add_scene_mjlab_uses_supplied_env_cfg(self, monkeypatch, minimal_spec):
        """Tracking tasks register with `commands["motion"].motion_file = ""`, so the
        caller has to hand in a cfg with the clip path already filled in — loading the
        task fresh here would build the tracing env against the empty path."""
        calls, _ = _install_fake_mjlab(monkeypatch, minimal_spec)
        caller_env_cfg = _FakeEnvCfg()

        project = Builder().add_project(name="P")
        project.add_scene_mjlab(
            "Mjlab-Tracking-Flat-Unitree-G1", env_cfg=caller_env_cfg
        )

        assert [c[0] for c in calls] == ["scene"]
        assert caller_env_cfg.scene.num_envs == 1


# ===========================================================================
# SceneHandle — add_policy, set_metadata
# ===========================================================================
class TestSceneHandle:
    def test_add_policy_appends_to_scene_policies(self, minimal_model, minimal_onnx):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_policy(name="Policy A", policy=minimal_onnx)
        scene.add_policy(name="Policy B", policy=minimal_onnx)
        policies = builder.get_projects()[0].scenes[0].policies
        assert len(policies) == 2
        assert policies[0].name == "Policy A"
        assert policies[1].name == "Policy B"

    def test_add_policy_returns_policy_handle(self, minimal_model, minimal_onnx):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        handle = scene.add_policy(name="Policy", policy=minimal_onnx)
        assert isinstance(handle, mjswan.PolicyHandle)

    def test_set_metadata_returns_self_for_chaining(self, minimal_model):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        result = scene.set_metadata("key", "value")
        assert result is scene

    def test_set_metadata_stores_value(self, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.set_metadata("author", "tester")
        cfg = builder.get_projects()[0].scenes[0]
        assert cfg.metadata["author"] == "tester"


# ===========================================================================
# PolicyHandle — commands=, set_metadata
# ===========================================================================
class TestPolicyHandle:
    def _make_scene(self, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        return builder, scene

    def _make_policy(self, minimal_model, minimal_onnx):
        builder, scene = self._make_scene(minimal_model)
        policy = scene.add_policy(name="Policy", policy=minimal_onnx)
        return builder, policy

    def test_commands_param_stores_inputs(self, minimal_model, minimal_onnx):
        builder, scene = self._make_scene(minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            commands={
                "custom": ui_command(
                    [SliderConfig(name="x", label="X", range=(-1.0, 1.0))]
                )
            },
        )
        commands = builder.get_projects()[0].scenes[0].policies[0].commands
        assert "custom" in commands
        assert commands["custom"].ui is not None
        assert len(commands["custom"].ui.inputs) == 1

    def test_commands_param_stores_command_term(self, minimal_model, minimal_onnx):
        builder, scene = self._make_scene(minimal_model)
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            commands={
                "goal": CommandTermConfig(term_name="DummyCommand", params={"value": 1})
            },
        )
        commands = builder.get_projects()[0].scenes[0].policies[0].commands
        assert commands["goal"].term_name == "DummyCommand"
        assert commands["goal"].params["value"] == 1

    def test_add_motion_defaults_dataset_joint_names_from_policy(
        self, minimal_model, minimal_onnx
    ):
        _, policy = self._make_policy(minimal_model, minimal_onnx)
        policy._config.policy_joint_names = ["joint_a", "joint_b"]

        motion = policy.add_motion(
            name="Spin Kick",
            source="motion.npz",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
            default=True,
        )

        assert isinstance(motion, mjswan.MotionHandle)
        stored = policy._config.motions[0]
        assert stored.name == "Spin Kick"
        assert stored.dataset_joint_names == ["joint_a", "joint_b"]
        assert stored.default is True

    def test_add_motion_wandb_resolves_run_id_shorthand(
        self, monkeypatch, minimal_model, minimal_onnx
    ):
        _, policy = self._make_policy(minimal_model, minimal_onnx)
        policy._config.policy_joint_names = ["joint_a"]

        called = {}

        def fake_fetch(run_path: str):
            called["run_path"] = run_path
            return "artifact_motion", b"npz-bytes"

        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_motion_npz_from_wandb_run",
            fake_fetch,
        )

        policy.add_motion_wandb(
            run_id="abc123",
            entity="demo-org",
            project="tracking",
            anchor_body_name="torso_link",
            body_names=("pelvis", "torso_link"),
        )

        assert called["run_path"] == "demo-org/tracking/abc123"
        assert policy._config.motions[0].data == b"npz-bytes"

    def test_add_policy_wandb_auto_imports_tracking_motion(
        self, monkeypatch, minimal_model, minimal_onnx
    ):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)

        class MotionCommandCfg:
            __module__ = "mjlab.fake"

            def __init__(self):
                self.anchor_body_name = "torso_link"
                self.body_names = ("pelvis", "torso_link")

        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_onnx_from_wandb_run",
            lambda run_path: ("policy", minimal_onnx),
        )
        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_motion_npz_from_wandb_run",
            lambda run_path: ("motion_asset", b"npz-data"),
        )

        handles = scene.add_policy_wandb(
            "demo-org/tracking/run1",
            only_latest=True,
            commands={"motion": MotionCommandCfg()},
        )

        assert len(handles) == 1
        motion = handles[0]._config.motions[0]
        assert motion.name == "motion_asset"
        assert motion.anchor_body_name == "torso_link"
        assert motion.body_names == ("pelvis", "torso_link")

    def test_tracking_motion_is_imported_when_commands_come_from_the_scene(
        self, monkeypatch, minimal_model, minimal_onnx
    ):
        """The clip has to be found from the *derived* commands, not just explicit ones.

        `add_policy_wandb` scans `commands` for the tracking term to know which clip to
        fetch. It used to scan the parameter, which is `None` once the scene's env config
        supplies the commands — so a tracking scene silently got no motion, and the
        browser's `TrackingCommand` then had nothing to answer `anchor_pos_w` /
        `anchor_quat_w` with. That surfaces only at playback, as a policy that never moves.
        """

        class MotionCommandCfg:
            __module__ = "mjlab.fake"

            def __init__(self):
                self.anchor_body_name = "torso_link"
                self.body_names = ("pelvis", "torso_link")

        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", model=minimal_model, control_dt=0.02)
        )
        env_cfg = _MdpEnvCfg()
        env_cfg.commands = {"motion": MotionCommandCfg()}
        scene._config.mjlab_env_cfg = env_cfg

        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_onnx_from_wandb_run",
            lambda run_path: ("policy", minimal_onnx),
        )
        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_motion_npz_from_wandb_run",
            lambda run_path: ("motion_asset", b"npz-data"),
        )

        # No `commands=`: exactly what the g1_spinkick example does now.
        handles = scene.add_policy_wandb("demo-org/tracking/run1", only_latest=True)

        assert len(handles) == 1
        assert handles[0]._config.motions, "no motion attached"
        motion = handles[0]._config.motions[0]
        assert motion.name == "motion_asset"
        assert motion.anchor_body_name == "torso_link"

    @pytest.mark.slow
    @pytest.mark.mjlab
    def test_tracking_motion_found_on_a_real_mjlab_command(
        self, monkeypatch, minimal_spec, minimal_onnx
    ):
        """Pinned against mjlab's own `MotionCommandCfg`, not a stand-in for it.

        `_extract_tracking_motion_term` recognises the term by class name, else by
        `anchor_body_name` + `body_names`. Both are upstream's spelling, so a rename there
        would silently stop the clip being found — the same playback-only failure as
        scanning the wrong `commands`.
        """
        pytest.importorskip("mjlab")
        import mjlab.tasks  # noqa: F401 — populates the registry
        from mjlab.tasks.registry import load_env_cfg

        from mjswan import wandb_io

        env_cfg = load_env_cfg(
            "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation", play=True
        )
        monkeypatch.setattr(
            wandb_io, "fetch_onnx_from_wandb_run", lambda p: ("model_100", minimal_onnx)
        )
        monkeypatch.setattr(
            wandb_io, "fetch_motion_npz_from_wandb_run", lambda p: ("spinkick", b"npz")
        )

        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", spec=minimal_spec, control_dt=0.02)
        )
        # As `add_scene_mjlab` would leave it, without paying for the live tracing env
        # (which needs a genuine clip on disk).
        scene._config.mjlab_env_cfg = env_cfg

        handles = scene.add_policy_wandb("org/proj/run", only_latest=True)

        motion_cfg: Any = env_cfg.commands["motion"]
        motion = handles[0]._config.motions[0]
        assert motion.name == "spinkick"
        assert motion.anchor_body_name == motion_cfg.anchor_body_name
        assert motion.body_names == tuple(motion_cfg.body_names)

    def test_set_metadata_stores_value(self, minimal_model, minimal_onnx):
        builder, policy = self._make_policy(minimal_model, minimal_onnx)
        policy.set_metadata("version", "1.0")
        cfg = builder.get_projects()[0].scenes[0].policies[0]
        assert cfg.metadata["version"] == "1.0"

    def test_add_policy_wandb_only_latest_preserves_extras(
        self, minimal_model, minimal_onnx, monkeypatch
    ):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)

        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_onnx_from_wandb_run",
            lambda _path: ("latest", minimal_onnx),
        )

        extras = {
            "model_overrides": {"geom_friction": [1.0, 0.5, 0.25]},
            "reset_samples": {"qpos": [[0.0]], "qvel": [[0.0]]},
        }
        handles = scene.add_policy_wandb(
            "entity/project/run",
            only_latest=True,
            extras=extras,
        )

        assert len(handles) == 1
        assert handles[0]._config.extras == extras


# ===========================================================================
# SceneHandle — deriving a policy's term sets from the scene's mjlab env config
# ===========================================================================
class _MdpEnvCfg:
    """An mjlab env config carrying the four term sets, plus its control rate.

    mjlab keeps all four on the env config; mjswan keeps them on the policy, so one
    scene can host several. These tests pin the bridge between those two shapes.
    """

    def __init__(self, *, timestep: float = 0.005, decimation: int = 4):
        self.observations = {
            "actor": mjswan.ObservationGroupCfg(terms={}),
            "critic": mjswan.ObservationGroupCfg(terms={}),
        }
        self.commands: dict[str, Any] = {"velocity": mjswan.velocity_command()}
        self.actions = {"joint_pos": JointPositionActionCfg(actuator_names=(".*",))}
        self.terminations = {"time_out": TerminationTermCfg(func=_never, time_out=True)}
        self.sim = type(
            "Sim", (), {"mujoco": type("Mj", (), {"timestep": timestep})()}
        )()
        self.decimation = decimation


def _never(env):  # a stand-in term body; nothing traces it in these tests
    raise AssertionError("not called")


def _mjlab_scene(minimal_model, env_cfg, control_dt: float = 0.02):
    """A scene as `add_scene_mjlab` leaves one, without needing mjlab installed."""
    scene = (
        Builder()
        .add_project(name="P")
        .add_scene(name="S", model=minimal_model, control_dt=control_dt)
    )
    scene._config.mjlab_env_cfg = env_cfg
    return scene


class TestPolicyTermsDerivedFromEnvCfg:
    def test_all_four_default_from_the_scenes_env_cfg(
        self, minimal_model, minimal_onnx
    ):
        env_cfg = _MdpEnvCfg()
        scene = _mjlab_scene(minimal_model, env_cfg)

        cfg = scene.add_policy(name="Policy", policy=minimal_onnx)._config

        # The actor group under the default slot; critic dropped (ADR 0006 §5).
        assert cfg.observations is not None
        assert list(cfg.observations) == ["actor"]
        assert cfg.observations["actor"] is env_cfg.observations["actor"]
        assert list(cfg.commands) == ["velocity"]
        assert cfg.actions is not None and list(cfg.actions) == ["joint_pos"]
        assert cfg.terminations is not None and list(cfg.terminations) == ["time_out"]

    def test_an_explicit_field_overrides_only_itself(self, minimal_model, minimal_onnx):
        # "The task's observations but my own terminations" should not cost a restatement
        # of the other three.
        env_cfg = _MdpEnvCfg()
        scene = _mjlab_scene(minimal_model, env_cfg)
        mine = {"fallen": TerminationTermCfg(func=_never)}

        cfg = scene.add_policy(
            name="Policy", policy=minimal_onnx, terminations=mine
        )._config

        assert list(cfg.terminations or {}) == ["fallen"]
        assert list(cfg.observations or {}) == ["actor"]
        assert list(cfg.commands) == ["velocity"]

    def test_an_empty_dict_means_none_not_derive(self, minimal_model, minimal_onnx):
        # `{}` is the only way to say "this policy genuinely has no commands"; if it
        # derived, there would be no way to express that at all.
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())

        cfg = scene.add_policy(
            name="Policy", policy=minimal_onnx, commands={}, terminations={}
        )._config

        assert cfg.commands == {}
        assert not cfg.terminations

    def test_a_plain_scene_derives_nothing(self, minimal_model, minimal_onnx):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", model=minimal_model, control_dt=0.02)
        )
        cfg = scene.add_policy(name="Policy", policy=minimal_onnx)._config
        assert cfg.observations is None
        assert cfg.commands == {}
        assert cfg.actions is None
        assert cfg.terminations is None

    def test_a_per_policy_env_cfg_wins_over_the_scenes(
        self, minimal_model, minimal_onnx
    ):
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())
        other = _MdpEnvCfg()
        other.commands = {"other": mjswan.velocity_command()}

        cfg = scene.add_policy(
            name="Policy", policy=minimal_onnx, env_cfg=other
        )._config

        assert list(cfg.commands) == ["other"]

    def test_a_per_policy_env_cfg_at_a_different_rate_is_refused(
        self, minimal_model, minimal_onnx
    ):
        # control_dt is per scene — the runtime derives its substep count and every timer
        # from one value — so honouring this quietly would run the policy at a rate it was
        # not trained for, which is exactly what raises no error at playback.
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg(), control_dt=0.02)
        slower = _MdpEnvCfg(timestep=0.01, decimation=5)  # 0.05 s

        with pytest.raises(ValueError, match="control rate is per scene"):
            scene.add_policy(name="Policy", policy=minimal_onnx, env_cfg=slower)

    def test_a_matching_rate_is_accepted(self, minimal_model, minimal_onnx):
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg(), control_dt=0.02)
        same = _MdpEnvCfg(timestep=0.002, decimation=10)  # 0.02 s
        assert scene.add_policy(name="Policy", policy=minimal_onnx, env_cfg=same)


class _FakeExportContext:
    """Stands in for `PtOnnxExportContext` (a wrapped env plus the metadata it reads)."""

    def __init__(self):
        self.unwrapped = object()
        self.env = type("Wrapped", (), {"unwrapped": self.unwrapped})()
        self.joint_names: list[str] = []
        self.default_joint_pos: list[float] = []
        self.encoder_bias: list[float] = []
        self.closed = False

    def close(self):
        self.closed = True


class TestLatestCheckpointIsTheDefault:
    """The scene opens on the newest checkpoint, not `model_0`.

    Deferring the conversion moved the handles out from under the code that marked the
    default, and the browser then opened on an untrained policy.
    """

    @pytest.fixture
    def checkpoints(self, monkeypatch, minimal_onnx):
        monkeypatch.setattr(
            "mjswan.wandb_io.create_pt_onnx_export_context",
            lambda task_id, env_cfg=None: _FakeExportContext(),
        )
        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_pt_onnx_from_wandb_run",
            lambda run_path, task_id, export_context: [
                ("model_0", minimal_onnx),
                ("model_1000", minimal_onnx),
                ("model_500", minimal_onnx),
            ],
        )

    def test_the_deferred_conversion_still_marks_it(self, checkpoints, minimal_model):
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())

        scene.add_policy_wandb("org/proj/run", task_id="t")
        for pending in scene._config.pending_conversions:
            pending.run(lambda _: None)

        defaults = [p.name for p in scene._config.policies if p.default]
        assert defaults == ["model_1000"]


class TestExportEnvBecomesTheTraceEnv:
    """One `ManagerBasedRlEnv` per scene, not two.

    Converting `.pt` checkpoints builds an env, and so did term tracing at build time —
    the same config twice, so mjlab printed its MDP tables twice per scene and the build
    paid for a second env. The export env is kept instead, which also means terms trace
    against exactly the env the ONNX was exported against.

    The conversion is deferred to `Builder.build`, so each test drives it as the build
    loop does — `add_policy_wandb` alone leaves nothing to assert on.
    """

    @staticmethod
    def _convert(scene):
        """Run what the build's scene loop would run for this scene."""
        for pending in scene._config.pending_conversions:
            pending.run(lambda _: None)
        scene._config.pending_conversions.clear()

    @pytest.fixture
    def context(self, monkeypatch, minimal_onnx):
        context = _FakeExportContext()
        monkeypatch.setattr(
            "mjswan.wandb_io.create_pt_onnx_export_context",
            lambda task_id, env_cfg=None: context,
        )
        monkeypatch.setattr(
            "mjswan.wandb_io.fetch_pt_onnx_from_wandb_run",
            lambda run_path, task_id, export_context: [("model_0", minimal_onnx)],
        )
        return context

    def test_the_scene_keeps_it(self, context, minimal_model):
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())

        scene.add_policy_wandb("org/proj/run", task_id="t")
        self._convert(scene)

        assert scene._config.mjlab_env is context.unwrapped
        assert not context.closed

    def test_a_per_policy_env_cfg_describes_a_different_env(
        self, context, minimal_model
    ):
        # Then it is not the scene's env, so keeping it would trace every *other* policy's
        # terms against the wrong one.
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())

        scene.add_policy_wandb("org/proj/run", task_id="t", env_cfg=_MdpEnvCfg())
        self._convert(scene)

        assert scene._config.mjlab_env is None
        assert context.closed

    def test_an_explicit_trace_env_wins(self, context, minimal_model):
        scene = _mjlab_scene(minimal_model, _MdpEnvCfg())
        mine = object()
        scene.set_trace_env(mine)

        scene.add_policy_wandb("org/proj/run", task_id="t")
        self._convert(scene)

        assert scene._config.mjlab_env is mine
        assert context.closed
