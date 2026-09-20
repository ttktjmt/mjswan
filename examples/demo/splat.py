"""Gaussian Splat Background Demo

Demonstrates how to use add_splat() to attach a real-world Gaussian Splat
background to a scene.

Run with:
    uv run splat
"""

import os
from pathlib import Path

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns

import mjswan
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab.env import build_single_entity_trace_env

SPLAT_URLs = [
    "https://cdn.marble.worldlabs.ai/be100eec-f02e-491d-899e-d702652d424d/cb27e09c-e2ca-46c7-8abf-bcd24d2bf9ed_ceramic_500k.spz",
    "https://cdn.marble.worldlabs.ai/09eaec3b-9114-455a-b7f1-da4d037cc511/660e6ce6-959c-42fb-8a9d-66178cb84f4d_ceramic.spz",
]
SPLAT_NAMEs = ["Traffic Road", "Cafe"]


def setup_builder() -> mjswan.Builder:
    """Set up the builder with a splat-backed scene.

    Returns:
        Configured Builder instance ready to be built.
    """
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path)

    project = builder.add_project(name="Splat Demo")

    scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    )
    scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_g1/g1.xml")
        )
    )

    g1_terminations = {
        "bad_orientation": TerminationTermCfg(
            func=term_fns.bad_orientation, params={"limit_angle": 1.0}
        ),
        "root_height_below_minimum": TerminationTermCfg(
            func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
        ),
    }

    scene.add_policy(
        name="balance",
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        config_path="assets/unitree_g1/balance.json",
        terminations=g1_terminations,
        observations=ObservationGroupCfg(
            terms={
                "base_ang_vel": ObservationTermCfg(
                    func=obs_fns.base_ang_vel, history_length=1
                ),
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity, history_length=1
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel, history_length=1
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel, history_length=1
                ),
                "prev_actions": ObservationTermCfg(func=obs_fns.last_action),
            }
        ),
    )

    scene.add_policy(
        name="locomotion",
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        config_path="assets/unitree_g1/locomotion.json",
        terminations=g1_terminations,
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
                lin_vel_x=(-1.5, 1.5),
                lin_vel_y=(-0.5, 0.5),
                default_lin_vel_x=0.5,
            )
        },
    )

    for name, splat_url in zip(SPLAT_NAMEs, SPLAT_URLs):
        scene.add_splat(
            name=name,
            url=splat_url,
            scale=3.275,
            z_offset=0.708,
            control=True,
        )

    return builder


def main():
    """Build and launch the splat demo."""
    builder = setup_builder()
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
