"""Muscle Actuator Demo.

Showcases mjswan's muscle-driven policy support using MyoFinger from MyoHub:
https://github.com/MyoHub/myo_sim

The MyoFinger model has 4 hinge joints (IFadb, IFmcp, IFpip, IFdip) driven
by 5 MuJoCo muscle actuators via ``MuscleActivationActionCfg``.  The policy
is an ONNX graph whose only op is ``RandomUniform`` -- it ignores the
observation and emits fresh uniform-[0, 1] activations on every inference
call, so each muscle is driven by a new random excitation each policy step.

This demo exercises three features used by muscle-driven policies:

1. ``policy_num_actions`` -- declare action count without ``policy_joint_names``.
2. ``initial_qpos`` / ``initial_qvel`` -- override the rest pose on reset.
3. ``MuscleActivationActionCfg`` -- maps outputs through ``sigmoid(5*(a-0.5))``.

The MyoFinger XMLs come from the Hugging Face Hub, so this example adds no Python
dependency on ``myo_sim``. Needs ``pip install 'mjswan[hf]'``.
"""

import os

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from onnx import TensorProto, helper

import mjswan
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.mjlab.env import build_single_entity_trace_env
from mjswan.source import hf

JOINT_NAMES = ("IFadb", "IFmcp", "IFpip", "IFdip")
MUSCLE_NAMES = ("extn", "adabR", "adabL", "mflx", "dflx")
NUM_MUSCLES = len(MUSCLE_NAMES)
NUM_JOINTS = len(JOINT_NAMES)
OBS_DIM = 2 * NUM_JOINTS  # joint_pos + joint_vel

# Slightly flexed finger instead of the model's fully extended default.
INITIAL_QPOS = [0.0, 0.3, 0.3, 0.3]
INITIAL_QVEL = [0.0] * NUM_JOINTS

# Pinned to a commit, not a branch: these files moved from `finger/` to
# `myo_sim/models/legacy/finger/` upstream and took the demo build down with a 404.
# A tag would be better; the repository publishes none.
HF_REPO = "ttktjmt/mjswan"
MYO_SCENE = "scenes/myofinger/myofinger_v0.xml"


def _build_policy() -> onnx.ModelProto:
    """Random-uniform policy: ignores the observation, emits fresh [0, 1] samples."""
    obs_in = helper.make_tensor_value_info("actor", TensorProto.FLOAT, [1, OBS_DIM])
    act_out = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, [1, NUM_MUSCLES]
    )

    random_node = helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["action"],
        shape=[1, NUM_MUSCLES],
        low=0.0,
        high=1.0,
        dtype=TensorProto.FLOAT,
    )

    graph = helper.make_graph(
        nodes=[random_node],
        name="muscle_policy",
        inputs=[obs_in],
        outputs=[act_out],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="mjswan-demo",
    )
    onnx.checker.check_model(model)
    return model


def setup_builder() -> mjswan.Builder:
    builder = mjswan.Builder(debug=True)
    project = builder.add_project(name="Muscle Actuator")

    myofinger_path = str(hf.fetch_dir(HF_REPO, "scenes/myofinger") / "myofinger_v0.xml")
    scene = project.add_scene_hf(
        HF_REPO,
        MYO_SCENE,
        name="MyoFinger",
        control_dt=0.02,  # 50 Hz control step
    )
    trace_env = build_single_entity_trace_env(
        lambda: mujoco.MjSpec.from_file(myofinger_path)
    )
    scene.set_trace_env(trace_env)

    # The build resolves joint_names -> joint_ids against the trace env, as mjlab does.
    finger_joints = SceneEntityCfg(name="robot", joint_names=list(JOINT_NAMES))

    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.2),
            distance=1.5,
            elevation=-20.0,
            azimuth=120.0,
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
        )
    )

    scene.add_policy(
        name="Random Action",
        policy=_build_policy(),
        policy_joint_names=[],
        policy_num_actions=NUM_MUSCLES,
        initial_qpos=INITIAL_QPOS,
        initial_qvel=INITIAL_QVEL,
        observations=ObservationGroupCfg(
            terms={
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel,
                    params={"asset_cfg": finger_joints},
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel,
                    params={"asset_cfg": finger_joints},
                ),
            }
        ),
        actions={
            "muscles": MuscleActivationActionCfg(
                entity_name="",
                actuator_names=MUSCLE_NAMES,
            ),
        },
    )

    return builder


def main():
    """Environment variables:
    MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
