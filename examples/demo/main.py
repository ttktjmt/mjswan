"""mjswan Demo Application

A tour of what mjswan does, hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/

Two projects:

- **mjlab Tasks**: mjlab's own tasks, taken as they are. Every scene is
  ``add_scene_mjlab`` plus trained checkpoints; nothing is hand-written.
- **Showcase**: what mjswan adds on top of the same engine, namely a Gaussian Splat
  background, a model mjlab has no task for, and muscle actuators.

No asset is stored in this repository. Models come from mjlab or the Hugging Face Hub,
policies from the Hub. The Hub repository is public, so a build needs no credentials.
"""

import os
import re
import types
from pathlib import Path
from typing import Any

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from onnx import TensorProto, helper

import mjswan
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab.env import build_single_entity_trace_env
from mjswan.source import hf

#: Every asset this demo does not get from mjlab. Public, so the build is anonymous.
HF_REPO = "ttktjmt/mjswan"

# Project A: mjlab Tasks

#: mjlab's bundled tasks, in the order they appear in the scene list.
MJLAB_TASKS = (
    "Mjlab-Velocity-Flat-Unitree-G1",
    "Mjlab-Velocity-Rough-Unitree-G1",
    "Mjlab-Velocity-Flat-Unitree-Go1",
    "Mjlab-Velocity-Rough-Unitree-Go1",
    "Mjlab-Lift-Cube-Yam",
    "Mjlab-Cartpole-Balance",
    "Mjlab-Cartpole-Swingup",
)

#: Motion tracking. Two inputs and seven outputs, so it needs `in_keys` and a clip.
TRACKING_TASK = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"
TRACKING_MOTION = "motions/mimickit_spinkick_safe.npz"

TASK_VIEWER_CONFIG_MAP: dict[str, mjswan.ViewerConfig] = {
    "Mjlab-Cartpole-Balance": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 1.0),
        distance=4.0,
        elevation=-15.0,
        azimuth=90.0,
        origin_type=mjswan.ViewerConfig.OriginType.WORLD,
    ),
    "Mjlab-Cartpole-Swingup": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 1.0),
        distance=4.0,
        elevation=-15.0,
        azimuth=90.0,
        origin_type=mjswan.ViewerConfig.OriginType.WORLD,
    ),
    "Mjlab-Lift-Cube-Yam": mjswan.ViewerConfig(
        lookat=(0.2, 0.0, 0.4),
        distance=2.0,
        elevation=-20.0,
        azimuth=45.0,
    ),
    "Mjlab-Velocity-Flat-Unitree-G1": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=3.0,
        elevation=-20.0,
        azimuth=0.0,
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name="torso_link",
    ),
    "Mjlab-Velocity-Flat-Unitree-Go1": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=2.0,
        elevation=-10.0,
        azimuth=0.0,
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name="trunk",
    ),
    "Mjlab-Velocity-Rough-Unitree-G1": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=4.0,
        elevation=-20.0,
        azimuth=30.0,
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name="torso_link",
    ),
    "Mjlab-Velocity-Rough-Unitree-Go1": mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=4.0,
        elevation=-20.0,
        azimuth=30.0,
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name="trunk",
    ),
    TRACKING_TASK: mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 0.0),
        distance=3.5,
        elevation=-15.0,
        azimuth=135.0,
        origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
        body_name="torso_link",
    ),
}


def _lift_update_command(self: Any, env_ids: Any = None) -> None:
    """``LiftingCommand._update_command`` without its ``env.sim.forward()`` call.

    mjlab forwards after a timer-expiry teleport of the cube so the rest of the step
    reads post-teleport kinematics. That writes nothing the command emits
    (``target_pos`` is ``_resample_command``'s alone), and the tracer refuses
    ``env.sim``. The browser forwards on its own next step, which is where the teleport
    (an ``entity_write``) lands.
    """
    del env_ids


def _bind_lift_override(term: Any) -> None:
    term._update_command = types.MethodType(_lift_update_command, term)


def _lift_viz(cfg: Any) -> list[dict[str, Any]]:
    """`LiftingCommand`'s target sphere, colored from the task's own cfg."""
    color = list(getattr(cfg.viz, "target_color", (1.0, 0.0, 0.0, 1.0)))
    return [
        {
            "shape": "sphere",
            "radius": 0.03,
            "color": color,
            "origin": {"state": "target_pos"},
        }
    ]


# mjswan binds no command class that only one task uses, so this task registers its own.
mjswan.register_command(
    "LiftingCommandCfg",
    mjswan.CommandBinding(
        state_fields=["target_pos"],
        command_field="target_pos",
        trace_override=_bind_lift_override,
        viz=_lift_viz,
    ),
)


def _checkpoints_for(task_id: str, repo_onnx: list[str]) -> list[str]:
    """A task's mirrored checkpoints, oldest first.

    ``list_repo_onnx`` sorts as strings (``model_1000`` before ``model_500``), and the
    viewer lists checkpoints in the order they are added, so sort by step to show
    training progress. ``add_policy_hf`` opens the highest step either way.
    """
    prefix = f"checkpoints/{task_id.lower()}/"
    named = [name for name in repo_onnx if name.startswith(prefix)]
    if not named:
        raise ValueError(
            f"No checkpoints under {prefix!r} in {HF_REPO!r}. The mirror is a snapshot "
            "of the W&B runs; re-run scripts/mirror_wandb_to_hf.py if a task was added."
        )
    return sorted(named, key=lambda name: int(re.search(r"_(\d+)\.onnx$", name)[1]))


def _add_mjlab_tasks(builder: mjswan.Builder) -> None:
    """mjlab's tasks, as mjlab gives them, driven by the mirrored checkpoints."""
    project = builder.add_project(name="mjlab Tasks")
    repo_onnx = hf.list_repo_onnx(HF_REPO)

    for task_id in MJLAB_TASKS:
        env_cfg = load_env_cfg(task_id, play=True)
        scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
        if viewer_cfg := TASK_VIEWER_CONFIG_MAP.get(task_id):
            scene.set_viewer(viewer_cfg)
        # Only the files are named here: the MDP is read off `env_cfg`, and the joint
        # mapping and rest pose off the metadata mjlab bakes into each `.onnx`.
        scene.add_policy_hf(HF_REPO, filename=_checkpoints_for(task_id, repo_onnx))

    _add_tracking_scene(project, repo_onnx)


def _add_tracking_scene(project: mjswan.ProjectHandle, repo_onnx: list[str]) -> None:
    """G1 Spinkick: the one task whose policy needs a reference motion beside it."""
    env_cfg = load_env_cfg(TRACKING_TASK, play=True)
    motion_term = env_cfg.commands["motion"]
    scene = project.add_scene_mjlab(TRACKING_TASK, env_cfg=env_cfg)
    if viewer_cfg := TASK_VIEWER_CONFIG_MAP.get(TRACKING_TASK):
        scene.set_viewer(viewer_cfg)

    # Slot tables are positional and the export's own tensor names do not say which
    # slot is which (ADR 0006 §5). Only `action` is read; the six after it are the
    # reference pose the clip already carries, named so the table lines up.
    policies = scene.add_policy_hf(
        HF_REPO,
        filename=_checkpoints_for(TRACKING_TASK, repo_onnx),
        in_keys=["actor", "time_step"],
        out_keys=[
            "action",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ],
    )

    # Unlike a W&B run, the Hub does not record which clip a policy trained against, so
    # the scene names it. The anchor and body list come from the task, as in training.
    for policy in policies:
        policy.add_motion_hf(
            HF_REPO,
            TRACKING_MOTION,
            repo_type="model",
            anchor_body_name=motion_term.anchor_body_name,
            body_names=list(motion_term.body_names),
            default=True,
        )


# Project B: Showcase

G1_TERMINATIONS: dict[str, TerminationTermCfg] = {
    "bad_orientation": TerminationTermCfg(
        func=term_fns.bad_orientation, params={"limit_angle": 1.0}
    ),
    "root_height_below_minimum": TerminationTermCfg(
        func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
    ),
}


def _add_g1_on_street(project: mjswan.ProjectHandle) -> None:
    """A Hub-hosted model and policies, standing in a Gaussian Splat capture.

    mjlab has no task for this G1 (the two policies are third-party, trained against
    this XML), so the model comes from the Hub as a directory: the MJCF, its meshes,
    and the `LICENSE` mjswan copies into the build beside it.
    """
    # Cached by `huggingface_hub`, so `add_scene_hf` below does not fetch it again. The
    # tracer wants the robot-only `g1.xml`; the viewer shows `scene.xml` around it.
    g1_dir = hf.fetch_dir(HF_REPO, "scenes/unitree_g1")

    scene = project.add_scene_hf(
        HF_REPO,
        "scenes/unitree_g1/scene.xml",
        name="G1 on Street",
        control_dt=0.02,  # 50 Hz control step
    )
    scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file(str(g1_dir / "g1.xml"))
        )
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=4.3,
            elevation=-33.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    )

    # Placement aligns this capture with this model; the `.spz` does not carry it.
    scene.add_splat_hf(
        HF_REPO, "splats/street.spz", name="Street", scale=3.275, z_offset=0.708, yaw=40
    )

    # Two calls, not one with a list: the policies read different observation sets, so
    # they are two MDPs. Neither is an mjlab export, so the sidecar JSON beside each on
    # the Hub supplies the joint mapping, rest pose and action term (PD gains included).
    scene.add_policy_hf(
        HF_REPO,
        filename="policies/locomotion.onnx",
        name="locomotion",
        config_path=str(hf.fetch_file(HF_REPO, "policies/locomotion.json")),
        terminations=G1_TERMINATIONS,
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
    scene.add_policy_hf(
        HF_REPO,
        filename="policies/balance.onnx",
        name="balance",
        config_path=str(hf.fetch_file(HF_REPO, "policies/balance.json")),
        terminations=G1_TERMINATIONS,
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


# MyoFinger: 4 hinge joints driven by 5 MuJoCo muscle actuators. mjlab has no task
# for it, so the model comes from the Hub as a directory (one XML includes the other).
MYO_JOINT_NAMES = ("IFadb", "IFmcp", "IFpip", "IFdip")
MYO_MUSCLE_NAMES = ("extn", "adabR", "adabL", "mflx", "dflx")
MYO_OBS_DIM = 2 * len(MYO_JOINT_NAMES)  # joint_pos + joint_vel
MYO_INITIAL_QPOS = [0.0, 0.3, 0.3, 0.3]  # slightly flexed, not fully extended
MYO_INITIAL_QVEL = [0.0] * len(MYO_JOINT_NAMES)
MYO_SCENE = "scenes/myofinger/myofinger_v0.xml"


def _build_muscle_policy() -> onnx.ModelProto:
    """Random-uniform policy: ignores the observation, emits fresh [0, 1] samples.

    The point is the action path, not the network: muscle excitation lives in [0, 1], so
    a lone `RandomUniform` op drives every muscle with a new excitation each step.
    """
    obs_in = helper.make_tensor_value_info("actor", TensorProto.FLOAT, [1, MYO_OBS_DIM])
    act_out = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, [1, len(MYO_MUSCLE_NAMES)]
    )
    random_node = helper.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["action"],
        shape=[1, len(MYO_MUSCLE_NAMES)],
        low=0.0,
        high=1.0,
        dtype=TensorProto.FLOAT,
    )
    model = helper.make_model(
        helper.make_graph(
            nodes=[random_node],
            name="muscle_policy",
            inputs=[obs_in],
            outputs=[act_out],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="mjswan-demo",
    )
    onnx.checker.check_model(model)
    return model


def _add_myofinger(project: mjswan.ProjectHandle) -> None:
    """Muscle actuators: no joint mapping, overridden rest pose, sigmoid activation."""
    # Cached, so `add_scene_hf` below re-uses this rather than fetching twice.
    myofinger_path = str(hf.fetch_dir(HF_REPO, "scenes/myofinger") / "myofinger_v0.xml")
    scene = project.add_scene_hf(
        HF_REPO,
        MYO_SCENE,
        name="MyoFinger",
        control_dt=0.02,  # 50 Hz control step
    )
    scene.set_trace_env(
        build_single_entity_trace_env(lambda: mujoco.MjSpec.from_file(myofinger_path))
    )
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.2),
            distance=1.5,
            elevation=-20.0,
            azimuth=120.0,
            origin_type=mjswan.ViewerConfig.OriginType.WORLD,
        )
    )

    # The build resolves joint_names -> joint_ids against the trace env, as mjlab does.
    finger_joints = SceneEntityCfg(name="robot", joint_names=list(MYO_JOINT_NAMES))

    scene.add_policy(
        name="Random Action",
        policy=_build_muscle_policy(),
        # A muscle is not a joint, so there is no mapping to declare, only a count.
        policy_joint_names=[],
        policy_num_actions=len(MYO_MUSCLE_NAMES),
        initial_qpos=MYO_INITIAL_QPOS,
        initial_qvel=MYO_INITIAL_QVEL,
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
                actuator_names=MYO_MUSCLE_NAMES,
            ),
        },
    )


def _add_showcase(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="Showcase")
    _add_g1_on_street(project)
    _add_myofinger(project)


def setup_builder() -> mjswan.Builder:
    """Assemble the demo. Fetches from mjlab and the public Hub, with no credentials."""
    builder = mjswan.Builder(
        base_path=os.getenv("MJSWAN_BASE_PATH", "/"),
        gtm_id="GTM-W79HQ38W",
        license="Apache-2.0",
        copyright="mjswan Developers",
    )
    _add_mjlab_tasks(builder)
    _add_showcase(builder)
    return builder


def main() -> None:
    """Main entry point for the demo application.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
        MJSWAN_SKIP_BUILD: Set to '1' to skip build and launch the pre-built app
    """
    dist_dir = Path(__file__).resolve().parent / "dist"
    if os.getenv("MJSWAN_SKIP_BUILD") == "1":
        app = mjswan.MjswanApp(dist_dir)
    else:
        app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
