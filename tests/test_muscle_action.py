"""Tests for ``MuscleActivationActionCfg`` and Python build-time muscle dyntype validation.

Covers:
  - ``MuscleActivationActionCfg.to_dict()`` serialization (default, scale/offset
    variants, ``normalize`` toggle).
  - ``validate_muscle_actuators`` helper: name resolution, dyntype enforcement,
    regex semantics.
  - ``Builder._validate_muscle_action_terms`` end-to-end at build time.
  - Builder round-trip of ``policy_num_actions`` / ``initial_qpos`` /
    ``initial_qvel`` together with a muscle action term.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import pytest

from mjswan.builder import Builder
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.envs.mdp.actions.actions import validate_muscle_actuators
from mjswan.utils import name2id

# ---------------------------------------------------------------------------
# Fixtures: minimal muscle / mixed-actuator MuJoCo models
# ---------------------------------------------------------------------------

MUSCLE_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/>
      <site name="s1a"/>
    </body>
    <site name="s1b" pos="1 0 0"/>
    <body>
      <joint name="j2" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/>
      <site name="s2a"/>
    </body>
    <site name="s2b" pos="1 0 0"/>
  </worldbody>
  <tendon>
    <spatial name="t1"><site site="s1a"/><site site="s1b"/></spatial>
    <spatial name="t2"><site site="s2a"/><site site="s2b"/></spatial>
  </tendon>
  <actuator>
    <muscle name="m1" tendon="t1" lengthrange="0 1"/>
    <muscle name="m2" tendon="t2" lengthrange="0 1"/>
  </actuator>
</mujoco>
"""

MIXED_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".1"/>
      <site name="s1a"/>
    </body>
    <site name="s1b" pos="1 0 0"/>
    <body>
      <joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size=".1"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t1"><site site="s1a"/><site site="s1b"/></spatial>
  </tendon>
  <actuator>
    <muscle name="muscle1" tendon="t1" lengthrange="0 1"/>
    <motor name="motor1" joint="j2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def muscle_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MUSCLE_XML)


@pytest.fixture
def mixed_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(MIXED_XML)


@pytest.fixture
def muscle_spec(tmp_path: Path) -> mujoco.MjSpec:
    xml_path = tmp_path / "muscle.xml"
    xml_path.write_text(MUSCLE_XML)
    return mujoco.MjSpec.from_file(str(xml_path))


# ---------------------------------------------------------------------------
# MuscleActivationActionCfg.to_dict()
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_default_omits_scale_offset_and_normalize(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1", "m2"))
        d = cfg.to_dict()
        assert d == {
            "type": "muscle_activation",
            "actuator_names": ["m1", "m2"],
        }

    def test_scale_float_present_when_non_default(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",), scale=2.0)
        d = cfg.to_dict()
        assert d["scale"] == 2.0

    def test_scale_dict_preserved(self):
        cfg = MuscleActivationActionCfg(
            actuator_names=("m1", "m2"), scale={"m1": 1.5, "m2": 0.5}
        )
        d = cfg.to_dict()
        assert d["scale"] == {"m1": 1.5, "m2": 0.5}

    def test_scale_list_preserved(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1", "m2"), scale=[1.5, 0.5])
        d = cfg.to_dict()
        assert d["scale"] == [1.5, 0.5]

    def test_offset_present_when_non_default(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",), offset=-0.5)
        d = cfg.to_dict()
        assert d["offset"] == -0.5

    def test_normalize_false_emitted_as_a_mode(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",), normalize=False)
        d = cfg.to_dict()
        assert d["mode"] == "excitation"

    def test_normalize_true_omitted(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",), normalize=True)
        d = cfg.to_dict()
        assert "mode" not in d and "normalize" not in d

    def test_actuator_names_tuple_serialized_as_list(self):
        cfg = MuscleActivationActionCfg(actuator_names=("a", "b", "c"))
        d = cfg.to_dict()
        assert d["actuator_names"] == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Muscle dyntype is real in the test model (sanity-checks the fixtures)
# ---------------------------------------------------------------------------


class TestMuscleModelFixture:
    def test_muscle_xml_has_two_muscle_actuators(self, muscle_model):
        assert muscle_model.nu == 2
        for i in range(muscle_model.nu):
            assert muscle_model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE, (
                f"actuator {i} dyntype={muscle_model.actuator_dyntype[i]}"
            )

    def test_mixed_xml_has_muscle_and_motor(self, mixed_model):
        assert mixed_model.nu == 2
        muscle_id = mujoco.mj_name2id(
            mixed_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "muscle1"
        )
        motor_id = mujoco.mj_name2id(
            mixed_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor1"
        )
        assert mixed_model.actuator_dyntype[muscle_id] == mujoco.mjtDyn.mjDYN_MUSCLE
        assert mixed_model.actuator_dyntype[motor_id] != mujoco.mjtDyn.mjDYN_MUSCLE


# ---------------------------------------------------------------------------
# validate_muscle_actuators: name resolution + dyntype enforcement
# ---------------------------------------------------------------------------


class TestValidateMuscleActuators:
    def test_all_muscles_literal_names_ok(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=("m1", "m2"))
        assert validate_muscle_actuators(muscle_model, cfg) == [0, 1]

    def test_match_all_regex_ok_when_only_muscles(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=(".*",))
        assert validate_muscle_actuators(muscle_model, cfg) == [0, 1]

    def test_regex_subset_ok(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",))
        assert validate_muscle_actuators(muscle_model, cfg) == [0]

    def test_unknown_literal_name_raises(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=("doesnotexist",))
        with pytest.raises(ValueError, match="doesnotexist"):
            validate_muscle_actuators(muscle_model, cfg)

    def test_regex_with_no_matches_raises(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=("zz.*",))
        with pytest.raises(ValueError, match="zz"):
            validate_muscle_actuators(muscle_model, cfg)

    def test_empty_actuator_names_raises(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=())
        with pytest.raises(ValueError, match="empty"):
            validate_muscle_actuators(muscle_model, cfg)

    def test_motor_actuator_in_explicit_list_raises(self, mixed_model):
        cfg = MuscleActivationActionCfg(actuator_names=("muscle1", "motor1"))
        with pytest.raises(ValueError, match="motor1"):
            validate_muscle_actuators(mixed_model, cfg)

    def test_match_all_regex_with_mixed_actuators_raises(self, mixed_model):
        cfg = MuscleActivationActionCfg(actuator_names=(".*",))
        with pytest.raises(ValueError, match="motor1"):
            validate_muscle_actuators(mixed_model, cfg)

    def test_muscle_only_subset_of_mixed_model_ok(self, mixed_model):
        cfg = MuscleActivationActionCfg(actuator_names=("muscle1",))
        assert validate_muscle_actuators(mixed_model, cfg) == [0]

    def test_term_name_appears_in_error(self, muscle_model):
        cfg = MuscleActivationActionCfg(actuator_names=("doesnotexist",))
        with pytest.raises(ValueError, match="muscles"):
            validate_muscle_actuators(muscle_model, cfg, term_name="muscles")


# ---------------------------------------------------------------------------
# Builder._validate_muscle_action_terms — end-to-end at build time
# ---------------------------------------------------------------------------


class TestBuilderMuscleValidation:
    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

    def _make_builder(self, model, actions, minimal_onnx):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=model
        )
        scene.add_policy(name="Policy", policy=minimal_onnx, actions=actions)
        return builder

    def test_build_succeeds_when_all_muscles(
        self, tmp_path, muscle_model, minimal_onnx
    ):
        builder = self._make_builder(
            muscle_model,
            {"muscles": MuscleActivationActionCfg(actuator_names=("m1", "m2"))},
            minimal_onnx,
        )
        builder._save_web(tmp_path / "out")

        policy_json = json.loads(
            (
                tmp_path / "out" / "main" / "assets" / name2id("S") / "policy.json"
            ).read_text()
        )
        assert "actions" in policy_json
        assert policy_json["actions"]["muscles"]["type"] == "muscle_activation"

    def test_build_raises_on_mixed_actuators(self, tmp_path, mixed_model, minimal_onnx):
        builder = self._make_builder(
            mixed_model,
            {"muscles": MuscleActivationActionCfg(actuator_names=(".*",))},
            minimal_onnx,
        )
        with pytest.raises(ValueError, match="motor1"):
            builder._save_web(tmp_path / "out")

    def test_build_raises_on_unknown_actuator_name(
        self, tmp_path, muscle_model, minimal_onnx
    ):
        builder = self._make_builder(
            muscle_model,
            {"muscles": MuscleActivationActionCfg(actuator_names=("nonexistent",))},
            minimal_onnx,
        )
        with pytest.raises(ValueError, match="nonexistent"):
            builder._save_web(tmp_path / "out")

    def test_build_uses_spec_when_model_absent(
        self, tmp_path, muscle_spec, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", spec=muscle_spec
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={"muscles": MuscleActivationActionCfg(actuator_names=("m1", "m2"))},
        )
        builder._save_web(tmp_path / "out")  # should not raise

    def test_no_muscle_term_skips_validation(
        self, tmp_path, minimal_model, minimal_onnx
    ):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        builder = self._make_builder(
            minimal_model,
            {"jp": JointPositionActionCfg(actuator_names=(".*",))},
            minimal_onnx,
        )
        builder._save_web(tmp_path / "out")  # should not raise


# ---------------------------------------------------------------------------
# Builder round-trip: muscle + policy_num_actions + initial_qpos/qvel
# ---------------------------------------------------------------------------


class TestBuilderMuscleRoundTrip:
    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

    def test_policy_num_actions_initial_qpos_qvel_emitted(
        self, tmp_path, muscle_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=muscle_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            policy_joint_names=[],
            policy_num_actions=2,
            initial_qpos=[0.1, 0.2],
            initial_qvel=[0.0, 0.0],
            actions={"muscles": MuscleActivationActionCfg(actuator_names=("m1", "m2"))},
        )

        builder._save_web(tmp_path / "out")
        data = json.loads(
            (
                tmp_path / "out" / "main" / "assets" / name2id("S") / "policy.json"
            ).read_text()
        )
        assert data["policy_num_actions"] == 2
        assert data["initial_qpos"] == [0.1, 0.2]
        assert data["initial_qvel"] == [0.0, 0.0]
        assert data["actions"]["muscles"]["type"] == "muscle_activation"
        assert data["actions"]["muscles"]["actuator_names"] == ["m1", "m2"]

    def test_muscle_mode_serialized_in_policy_json(
        self, tmp_path, muscle_model, minimal_onnx
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.02, name="S", model=muscle_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "muscles": MuscleActivationActionCfg(
                    actuator_names=("m1", "m2"), normalize=False
                )
            },
        )
        builder._save_web(tmp_path / "out")
        data = json.loads(
            (
                tmp_path / "out" / "main" / "assets" / name2id("S") / "policy.json"
            ).read_text()
        )
        assert data["actions"]["muscles"]["mode"] == "excitation"


class TestMuscleActionMode:
    """``action_mode``, which mjlab/myosuite spells the same way so the adapter copies it."""

    def test_legacy_normalize_maps_onto_the_two_named_modes(self):
        assert MuscleActivationActionCfg(actuator_names=("m1",)).mode == "sigmoid"
        assert (
            MuscleActivationActionCfg(actuator_names=("m1",), normalize=False).mode
            == "excitation"
        )

    def test_action_mode_wins_over_normalize(self):
        cfg = MuscleActivationActionCfg(
            actuator_names=("m1",), normalize=True, action_mode="direct"
        )
        assert cfg.mode == "direct"

    def test_only_the_default_mode_is_left_out_of_the_json(self):
        assert "mode" not in MuscleActivationActionCfg(actuator_names=("m1",)).to_dict()
        for mode in ("direct", "excitation"):
            entry = MuscleActivationActionCfg(
                actuator_names=("m1",), action_mode=mode
            ).to_dict()
            assert entry["mode"] == mode

    def test_an_unknown_mode_is_rejected(self):
        cfg = MuscleActivationActionCfg(actuator_names=("m1",), action_mode="clamp")
        with pytest.raises(ValueError, match="action_mode"):
            _ = cfg.mode
