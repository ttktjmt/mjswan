"""Minimum Policy

The smallest possible mjswan example with an ONNX-controlled scene: a box on
a vertical slide is held aloft by a hand-built linear PD policy. The ONNX
graph is two nodes (MatMul + Add) constructed inline with onnx.helper. A
slider in the viewer adjusts the target altitude live.
"""

import mujoco
import numpy as np
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from onnx import TensorProto, helper, numpy_helper

import mjswan
from mjswan.envs.mdp.actions import JointEffortActionCfg
from mjswan.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjswan.mjlab.env import build_single_entity_trace_env

MASS = 0.1
GRAVITY = 9.81
KP, KD = 80.0, 12.0
TARGET_RANGE = (0.3, 1.8)
DEFAULT_TARGET = 1.0


def build_policy() -> onnx.ModelProto:
    """Linear PD with gravity comp: F = -kp*(h - h*) - kd*v + m*g.

    Observation order: [height, velocity, target_height] → action [F].
    """
    weights = np.array([[-KP], [-KD], [KP]], dtype=np.float32)
    bias = np.array([MASS * GRAVITY], dtype=np.float32)

    obs_in = helper.make_tensor_value_info("actor", TensorProto.FLOAT, [1, 3])
    act_out = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 1])

    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["actor", "W"], ["matmul_out"]),
            helper.make_node("Add", ["matmul_out", "b"], ["action"]),
        ],
        name="hover_pd",
        inputs=[obs_in],
        outputs=[act_out],
        initializer=[
            numpy_helper.from_array(weights, name="W"),
            numpy_helper.from_array(bias, name="b"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="mjswan-tutorial",
    )
    onnx.checker.check_model(model)
    return model


def joint_height(env, *, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot")):
    """Absolute joint position (no default-pose subtraction — a bare box on a
    slide joint has no meaningful "default" to subtract; the PD policy wants
    the raw height)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def build_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(f"""
    <mujoco>
      <option gravity="0 0 -{GRAVITY}" timestep="0.002"/>
      <worldbody>
        <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="2 2 0.1" rgba=".9 .9 .92 1"/>
        <body name="hover_box" pos="0 0 0">
          <joint name="lift" type="slide" axis="0 0 1" damping="0"/>
          <geom type="box" size=".1 .1 .1" rgba=".2 .6 .9 1" mass="{MASS}"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="thrust" joint="lift" ctrlrange="-20 20"/>
      </actuator>
      <keyframe>
        <key name="hover" qpos="{DEFAULT_TARGET}"/>
      </keyframe>
    </mujoco>
    """)


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="Minimum Policy")

    target_cmd = mjswan.ui_command(
        [
            mjswan.Slider(
                name="target_height",
                label="Target Height (m)",
                range=TARGET_RANGE,
                default=DEFAULT_TARGET,
                step=0.05,
            ),
        ]
    )

    scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        spec=build_spec(),
        name="Hovering Box",
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 1.0),
            distance=3.5,
            elevation=-30.0,
            azimuth=45.0,
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
        )
    )
    scene.set_trace_env(build_single_entity_trace_env(build_spec))
    scene.add_policy(
        name="PD Hover",
        policy=build_policy(),
        policy_joint_names=["lift"],
        observations=ObservationGroupCfg(
            terms={
                "height": ObservationTermCfg(func=joint_height),
                "velocity": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "target": ObservationTermCfg(
                    func=obs_fns.generated_commands,
                    params={"command_name": "target"},
                ),
            }
        ),
        actions={
            "thrust": JointEffortActionCfg(
                entity_name="",
                actuator_names=("lift",),
            ),
        },
        commands={"target": target_cmd},
    )

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
