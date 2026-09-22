"""Joint names for a policy whose export carries none.

mjlab attaches export metadata from its velocity, manipulation and tracking runners
only, so a cartpole checkpoint arrives with no ``joint_names`` at all. The Hub path used
to have nothing else to read them from, which left the browser matching the task's joint
patterns against an empty list, skipping the action term, and writing no control: the
scene rendered and the cart never moved. The task's own action terms name the joints, so
they are read from there, and the call says so when even that comes up empty.
"""

from __future__ import annotations

import warnings

import mujoco
import onnx
import pytest
from onnx import TensorProto, helper

import mjswan
from mjswan.envs.mdp.actions import (
    JointEffortActionCfg,
    JointPositionActionCfg,
    MuscleActivationActionCfg,
)
from mjswan.policy import action_term_joint_names, actuated_joints_in_joint_order

#: mjlab's cartpole shape: entity-namespaced joints, one actuator, one passive hinge.
CARTPOLE_XML = """
<mujoco model="cartpole">
  <worldbody>
    <body name="cartpole/cart">
      <joint name="cartpole/slider" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.05 0.05"/>
      <body name="cartpole/pole_1" pos="0 0 0.05">
        <joint name="cartpole/hinge_1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cartpole/slide_act" joint="cartpole/slider"/>
  </actuator>
</mujoco>
"""

#: Two joints whose actuator block is in the opposite order, which is the whole reason
#: joint order and actuator order have to be told apart.
REORDERED_XML = """
<mujoco model="reordered">
  <worldbody>
    <body>
      <joint name="hip" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3"/>
      <body pos="0 0 -0.3">
        <joint name="knee" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="knee_act" joint="knee"/>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""

CARTPOLE_ACTIONS = {
    "effort": JointEffortActionCfg(actuator_names=("cartpole/slider",), scale=1.0)
}


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def _bare_policy_onnx(path, *, action_width: int):
    """An ONNX with no metadata_props at all, as a cartpole export has none."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, action_width])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, action_width])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "p", [x], [y])
    )
    onnx.save(model, str(path))
    return path


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """The Hub stub of ``test_add_policy_hf``, serving metadata-free files."""
    repo: dict[str, str] = {}

    class _Api:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None, repo_type="model"):
            return list(repo)

    class _Hub:
        HfApi = _Api

        @staticmethod
        def hf_hub_download(
            repo_id, filename, revision=None, repo_type=None, token=None
        ):
            return repo[filename]

    monkeypatch.setattr("mjswan.source.hf._hub", lambda: _Hub)

    def register(filename: str, *, action_width: int = 1) -> str:
        repo[filename] = str(
            _bare_policy_onnx(
                tmp_path / filename.replace("/", "_"), action_width=action_width
            )
        )
        return filename

    return register


@pytest.fixture
def cartpole():
    builder = mjswan.Builder()
    project = builder.add_project(name="Test")
    return project.add_scene(
        name="Cartpole",
        spec=mujoco.MjSpec.from_string(CARTPOLE_XML),
        control_dt=0.05,
    )


class TestActuatedJointsInJointOrder:
    def test_it_is_joint_order_not_actuator_order(self):
        model = _model(REORDERED_XML)

        assert actuated_joints_in_joint_order(model) == ["hip", "knee"]

    def test_a_joint_no_actuator_drives_is_left_out(self):
        model = _model(CARTPOLE_XML)

        # `cartpole/hinge_1` is passive: the policy never drives it.
        assert actuated_joints_in_joint_order(model) == ["cartpole/slider"]

    def test_no_model_has_no_answer(self):
        assert actuated_joints_in_joint_order(None) is None


class TestActionTermJointNames:
    def test_the_cartpole_effort_term_resolves_to_its_slider(self):
        names = action_term_joint_names(CARTPOLE_ACTIONS, _model(CARTPOLE_XML))

        assert names == ["cartpole/slider"]

    def test_terms_concatenate_in_the_order_they_are_declared(self):
        actions = {
            "knee": JointPositionActionCfg(actuator_names=("knee",)),
            "hip": JointPositionActionCfg(actuator_names=("hip",)),
        }

        assert action_term_joint_names(actions, _model(REORDERED_XML)) == [
            "knee",
            "hip",
        ]

    def test_match_all_takes_the_actuated_joints_in_joint_order(self):
        actions = {"joint_pos": JointPositionActionCfg(actuator_names=(".*",))}

        assert action_term_joint_names(actions, _model(REORDERED_XML)) == [
            "hip",
            "knee",
        ]

    def test_a_muscle_term_has_no_answer(self):
        """It names actuators, not joints, and needs no joint mapping at all."""
        actions = {"muscle": MuscleActivationActionCfg(actuator_names=(".*",))}

        assert action_term_joint_names(actions, _model(CARTPOLE_XML)) is None

    def test_a_pattern_matching_nothing_has_no_answer(self):
        actions = {"effort": JointEffortActionCfg(actuator_names=("elbow",))}

        assert action_term_joint_names(actions, _model(CARTPOLE_XML)) is None

    def test_two_terms_claiming_one_joint_have_no_answer(self):
        actions = {
            "a": JointPositionActionCfg(actuator_names=("hip",)),
            "b": JointPositionActionCfg(actuator_names=("hip|knee",)),
        }

        assert action_term_joint_names(actions, _model(REORDERED_XML)) is None

    def test_no_actions_have_no_answer(self):
        assert action_term_joint_names(None, _model(CARTPOLE_XML)) is None


class TestMetadatalessExport:
    def test_the_action_terms_supply_the_joint_names(self, cartpole, fake_hub):
        fake_hub("policy.onnx", action_width=1)

        (policy,) = cartpole.add_policy_hf("my-org/cartpole", actions=CARTPOLE_ACTIONS)

        assert policy._config.policy_joint_names == ["cartpole/slider"]

    def test_a_count_the_actions_cannot_drive_is_refused_and_reported(
        self, cartpole, fake_hub
    ):
        """One name against four actions would misdrive them, silently."""
        fake_hub("policy.onnx", action_width=4)

        with pytest.warns(RuntimeWarning, match="policy_joint_names was left unset"):
            (policy,) = cartpole.add_policy_hf(
                "my-org/cartpole", actions=CARTPOLE_ACTIONS
            )

        assert policy._config.policy_joint_names is None

    def test_the_caller_still_wins(self, cartpole, fake_hub):
        fake_hub("policy.onnx", action_width=1)

        (policy,) = cartpole.add_policy_hf(
            "my-org/cartpole",
            actions=CARTPOLE_ACTIONS,
            policy_joint_names=["cartpole/hinge_1"],
        )

        assert policy._config.policy_joint_names == ["cartpole/hinge_1"]


class TestNothingLeftToTry:
    """`add_policy_hf` promises to fill these, so it says when it cannot."""

    def test_action_terms_it_cannot_resolve_are_reported(self, cartpole, fake_hub):
        """Patterns naming a joint this model does not have leave the term undrivable."""
        fake_hub("policy.onnx", action_width=1)
        actions = {"effort": JointEffortActionCfg(actuator_names=("elbow",))}

        with pytest.warns(RuntimeWarning, match="will write no control at all"):
            (policy,) = cartpole.add_policy_hf("my-org/cartpole", actions=actions)

        assert policy._config.policy_joint_names is None

    def test_a_policy_driving_no_joints_says_nothing(self, cartpole, fake_hub):
        """No action term at all is inert by design, not a mapping that went missing."""
        fake_hub("policy.onnx", action_width=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartpole.add_policy_hf("my-org/cartpole")

    def test_a_muscle_policy_says_nothing(self, cartpole, fake_hub):
        """It names actuators directly, so it never wanted joint names."""
        fake_hub("policy.onnx", action_width=1)
        actions = {"muscle": MuscleActivationActionCfg(actuator_names=(".*",))}

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartpole.add_policy_hf("my-org/cartpole", actions=actions)

    def test_a_sidecar_may_still_carry_them(self, cartpole, fake_hub, tmp_path):
        """It is read at build time, so this call cannot know yet and must not cry."""
        fake_hub("policy.onnx", action_width=1)
        sidecar = tmp_path / "policy.json"
        sidecar.write_text('{"policy_joint_names": ["cartpole/slider"]}')
        actions = {"effort": JointEffortActionCfg(actuator_names=("elbow",))}

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartpole.add_policy_hf(
                "my-org/cartpole", actions=actions, config_path=str(sidecar)
            )

    def test_resolving_the_names_says_nothing(self, cartpole, fake_hub):
        fake_hub("policy.onnx", action_width=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cartpole.add_policy_hf("my-org/cartpole", actions=CARTPOLE_ACTIONS)
