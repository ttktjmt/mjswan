"""Simple mjswan Demo

A basic example demonstrating how to use mjswan to create a viewer application
with multiple robot scenes (Go2, Go1, and G1).
"""

import os
from pathlib import Path

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

import mjswan
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.mjlab.env import build_single_entity_trace_env

# This demo's own arm pose offset; the gains it rides on live in `locomotion.json`.
_G1_OFFSET = {
    "left_shoulder_pitch_joint": 0.5,
    "right_shoulder_pitch_joint": -0.5,
    "left_wrist_roll_joint": 2.0,
    "right_wrist_roll_joint": -2.0,
}


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with demo projects configured.

    Creates a builder and adds a project with three robot scenes.
    Does not build or launch the application.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    demo_project = builder.add_project(
        name="mjswan Demo",
    )

    demo_project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_trace_env(
        # The env the observation terms are traced against. An mjlab scene brings its own; this
        # plain MJCF needs the entity built explicitly, from the robot alone.
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_g1/g1.xml")
        )
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=2.5,
            elevation=-10.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    ).add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        actions={
            # Only the offset: scale and PD gains ride with the policy, and a term overrides just
            # the fields it names.
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                offset=_G1_OFFSET,
            ),
        },
        observations=ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
                "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "last_action": ObservationTermCfg(func=obs_fns.last_action),
                "velocity_cmd": ObservationTermCfg(
                    func=obs_fns.generated_commands,
                    params={"command_name": "velocity"},
                ),
            }
        ),
        commands={
            "velocity": mjswan.velocity_command(
                lin_vel_x=(-2.0, 2.0),
                lin_vel_y=(-0.5, 0.5),
                default_lin_vel_x=0.5,
                default_lin_vel_y=0.0,
            )
        },
    )
    demo_project.add_scene(
        # model=mujoco.MjModel.from_xml_path("assets/unitree_go2/scene.xml"),
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
        name="Go2",
    )

    return builder


def main():
    """Main entry point for the simple demo.

    Sets up the builder, builds the application, and launches it in a browser.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    # Build and launch the application
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
