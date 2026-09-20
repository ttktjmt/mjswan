"""``SceneHandle.add_policy_hf`` end to end, with the Hub itself stubbed.

Only ``hf_io._hub`` is replaced, so the filename resolution, the name derivation, the
metadata read and the ``add_policy`` call all run for real — the network is the one
thing that does not.
"""

from __future__ import annotations

import mujoco
import onnx
import pytest
from onnx import TensorProto, helper

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg

ROBOT_XML = """
<mujoco model="two_joint">
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
    <position name="hip_act" joint="hip"/>
    <position name="knee_act" joint="knee"/>
  </actuator>
</mujoco>
"""

MJLAB_METADATA = {
    "run_path": "brisk-cloud-42",
    "joint_names": "hip,knee",
    "default_joint_pos": "-0.312,0.669",
    "joint_stiffness": "40.000,40.000",
    "joint_damping": "1.000,1.000",
    "action_scale": "0.500,0.250",
    "command_names": "velocity",
    "observation_names": "base_lin_vel,joint_pos",
    "observation_terms_scale": "1.000,0.500",
    "observation_terms_clip": "-inf;inf,-inf;inf",
    "observation_terms_history_length": "0.000,0.000",
    "observation_terms_flatten_history_dim": "1.000,1.000",
}


def _policy_onnx(path, *, action_width: int = 2, metadata: dict | None = None):
    """An ONNX whose one output is ``action_width`` wide, written to ``path``."""
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, action_width])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, action_width])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "p", [x], [y])
    )
    for key, value in (metadata or {}).items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Install a Hub stub; the returned callable registers ``{filename: local path}``."""
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

    def register(filename: str, **kwargs) -> str:
        repo[filename] = str(
            _policy_onnx(tmp_path / filename.replace("/", "_"), **kwargs)
        )
        return filename

    return register


@pytest.fixture
def scene():
    builder = mjswan.Builder()
    project = builder.add_project(name="Test")
    return project.add_scene(
        name="Robot",
        spec=mujoco.MjSpec.from_string(ROBOT_XML),
        control_dt=0.02,
    )


class TestMetadataFill:
    def test_joint_names_and_rest_pose_come_from_the_file(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names == ["hip", "knee"]
        assert policy._config.default_joint_pos == [-0.312, 0.669]

    def test_the_action_term_comes_from_the_file(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf("my-org/two-joint")

        actions = policy._config.mdp.actions
        assert list(actions) == ["joint_pos"]
        assert actions["joint_pos"].scale == [0.5, 0.25]
        assert actions["joint_pos"].use_default_offset is True

    def test_the_name_falls_back_to_the_repo(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf("my-org/two-joint")

        assert policy._config.name == "two-joint"

    def test_a_caller_value_wins_over_the_file(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf(
            "my-org/two-joint", default_joint_pos=[0.0, 0.0]
        )

        assert policy._config.default_joint_pos == [0.0, 0.0]
        assert policy._config.policy_joint_names == ["hip", "knee"]

    def test_caller_actions_win_over_the_file(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf(
            "my-org/two-joint",
            actions={"joint_pos": JointPositionActionCfg(scale=2.0)},
        )

        assert policy._config.mdp.actions["joint_pos"].scale == 2.0

    def test_use_metadata_false_ignores_the_file(self, scene, fake_hub):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf("my-org/two-joint", use_metadata=False)

        assert policy._config.policy_joint_names is None
        assert policy._config.default_joint_pos is None

    def test_a_policy_without_metadata_is_added_bare(self, scene, fake_hub):
        fake_hub("policy.onnx")

        (policy,) = scene.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names is None
        assert not policy._config.mdp.actions


class TestJointMappingGuard:
    """The metadata is paired with the scene's model, never taken on faith."""

    def test_a_passive_joint_warns_and_fills_nothing(self, scene, fake_hub):
        """mjlab lists every joint; here one is unactuated, so the lists differ."""
        fake_hub(
            "policy.onnx",
            action_width=2,
            metadata={
                **MJLAB_METADATA,
                "joint_names": "hip,knee,passive",
                "default_joint_pos": "-0.312,0.669,0.000",
                "action_scale": "0.500",
            },
        )

        with pytest.warns(RuntimeWarning, match="3 joints"):
            (policy,) = scene.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names is None
        assert policy._config.default_joint_pos is None
        assert not policy._config.mdp.actions

    def test_actuators_out_of_joint_order_warn_and_fill_nothing(self, fake_hub):
        """The counts agree; only the order does not, which a length check misses."""
        builder = mjswan.Builder()
        reversed_scene = builder.add_project(name="T").add_scene(
            name="Robot",
            spec=mujoco.MjSpec.from_string(
                ROBOT_XML.replace(
                    '<position name="hip_act" joint="hip"/>\n'
                    '    <position name="knee_act" joint="knee"/>',
                    '<position name="knee_act" joint="knee"/>\n'
                    '    <position name="hip_act" joint="hip"/>',
                )
            ),
            control_dt=0.02,
        )
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        with pytest.warns(RuntimeWarning, match="actuator order"):
            (policy,) = reversed_scene.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names is None

    def test_joint_names_are_emitted_as_the_model_spells_them(self, fake_hub):
        """A scene from an mjlab task namespaces its joints; mjlab's export does not."""
        builder = mjswan.Builder()
        namespaced = builder.add_project(name="T").add_scene(
            name="Robot",
            spec=mujoco.MjSpec.from_string(
                ROBOT_XML.replace('"hip"', '"robot/hip"').replace(
                    '"knee"', '"robot/knee"'
                )
            ),
            control_dt=0.02,
        )
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = namespaced.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names == ["robot/hip", "robot/knee"]


class TestSeveralFiles:
    def test_each_file_becomes_a_policy_sharing_one_mdp(self, scene, fake_hub):
        fake_hub("policies/walk.onnx", metadata=MJLAB_METADATA)
        fake_hub("policies/stand.onnx", metadata=MJLAB_METADATA)

        walk, stand = scene.add_policy_hf(
            "my-org/two-joint",
            filename=["policies/walk.onnx", "policies/stand.onnx"],
        )

        assert [walk._config.name, stand._config.name] == ["walk", "stand"]
        assert walk._config.mdp is stand._config.mdp

    def test_the_first_policy_opens_the_scene(self, scene, fake_hub):
        fake_hub("policies/walk.onnx", metadata=MJLAB_METADATA)
        fake_hub("policies/stand.onnx", metadata=MJLAB_METADATA)

        walk, stand = scene.add_policy_hf(
            "my-org/two-joint",
            filename=["policies/walk.onnx", "policies/stand.onnx"],
        )

        assert walk._config.default is True
        assert stand._config.default is False

    def test_a_name_for_several_files_raises(self, scene, fake_hub):
        fake_hub("a.onnx")
        fake_hub("b.onnx")

        with pytest.raises(ValueError, match="A name applies to one policy"):
            scene.add_policy_hf(
                "my-org/two-joint", filename=["a.onnx", "b.onnx"], name="Both"
            )


class TestFilenameResolution:
    def test_policy_onnx_wins_over_other_files(self, scene, fake_hub):
        fake_hub("lineage/step_10.onnx")
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)

        (policy,) = scene.add_policy_hf("my-org/two-joint")

        assert policy._config.policy_joint_names == ["hip", "knee"]

    def test_an_ambiguous_repo_raises(self, scene, fake_hub):
        fake_hub("a.onnx")
        fake_hub("b.onnx")

        with pytest.raises(ValueError, match="Pass filename="):
            scene.add_policy_hf("my-org/two-joint")


class TestBuildRoundTrip:
    def test_the_manifest_carries_what_the_file_described(
        self, scene, fake_hub, build_manifest, tmp_path
    ):
        fake_hub("policy.onnx", metadata=MJLAB_METADATA)
        scene.add_policy_hf("my-org/two-joint")

        manifest = build_manifest(scene._project._builder, tmp_path / "dist")

        entry = manifest["projects"][0]["scenes"][0]["policies"][0]
        assert entry["policy_joint_names"] == ["hip", "knee"]
        assert entry["default_joint_pos"] == [-0.312, 0.669]
