"""mjswan Demo Application

A tour of what mjswan does, hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/

Two projects, and the split is the explanation:

- **mjlab Tasks** — mjlab's own tasks, taken as they are. Every scene here is
  ``add_scene_mjlab`` plus trained checkpoints; nothing is hand-written, so what you see
  is what mjlab gives you through mjswan.
- **Showcase** — the same engine with mjswan-side work on top: a Gaussian Splat
  background, a model mjlab has no task for, muscle actuators.

**No asset is stored in this repository.** Models come from mjlab or from the Hugging
Face Hub, and policies from the Hub. The Hub repository is public, so a build needs no
credentials of any kind — which is why the deploy workflow carries no secret, and one
public host is the only thing it has to reach.
"""

import os
import re
from pathlib import Path

import mujoco
import onnx
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from onnx import TensorProto, helper

import mjswan
import mjswan.mjlab.bindings  # noqa: F401 - registers the mjlab command bindings
from mjswan.envs.mdp.actions import MuscleActivationActionCfg
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
from mjswan.mjlab import register_custom_terminations
from mjswan.mjlab.env import build_single_entity_trace_env
from mjswan.source import hf

#: Every asset this demo does not get from mjlab. Public, so the build is anonymous.
HF_REPO = "ttktjmt/mjswan"

# ─────────────────────────────────────────────────────────────────────────────
# Project A — mjlab Tasks
# ─────────────────────────────────────────────────────────────────────────────

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


def _checkpoints_for(task_id: str, repo_onnx: list[str]) -> list[str]:
    """A task's mirrored checkpoints, oldest first.

    ``list_repo_onnx`` sorts as strings, which puts ``model_1000`` before ``model_500``.
    The viewer lists checkpoints in the order they are added, so training has to be
    re-sorted numerically to read as progress. Which one opens is separate and does not
    depend on this: ``add_policy_hf`` defaults to the highest step either way.
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
        register_custom_terminations(env_cfg)
        scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
        if viewer_cfg := TASK_VIEWER_CONFIG_MAP.get(task_id):
            scene.set_viewer(viewer_cfg)
        # Everything but the files comes from the task: observations, commands, actions,
        # terminations and events are read off `env_cfg`, and the joint mapping and rest
        # pose off the metadata mjlab baked into each `.onnx`.
        scene.add_policy_hf(HF_REPO, filename=_checkpoints_for(task_id, repo_onnx))

    _add_tracking_scene(project, repo_onnx)


def _add_tracking_scene(project: mjswan.ProjectHandle, repo_onnx: list[str]) -> None:
    """G1 Spinkick: the one task whose policy needs a reference motion beside it."""
    env_cfg = load_env_cfg(TRACKING_TASK, play=True)
    motion_term = env_cfg.commands["motion"]
    scene = project.add_scene_mjlab(TRACKING_TASK, env_cfg=env_cfg)
    if viewer_cfg := TASK_VIEWER_CONFIG_MAP.get(TRACKING_TASK):
        scene.set_viewer(viewer_cfg)

    # `in_keys` is positional, so a two-input export cannot be read off the network's
    # own input names (ADR 0006 §5).
    policies = scene.add_policy_hf(
        HF_REPO,
        filename=_checkpoints_for(TRACKING_TASK, repo_onnx),
        in_keys=["actor", "time_step"],
    )

    # `add_policy_wandb` finds the clip in the run's artifacts; the Hub holds published
    # files and knows nothing about which clip a policy was trained against, so the
    # tracking scene names it. The anchor and the body list come from the task itself,
    # which is where the training read them from too.
    for policy in policies:
        policy.add_motion_hf(
            HF_REPO,
            TRACKING_MOTION,
            repo_type="model",
            anchor_body_name=motion_term.anchor_body_name,
            body_names=list(motion_term.body_names),
            default=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Project B — Showcase
# ─────────────────────────────────────────────────────────────────────────────

G1_TERMINATIONS: dict[str, TerminationTermCfg] = {
    "bad_orientation": TerminationTermCfg(
        func=term_fns.bad_orientation, params={"limit_angle": 1.0}
    ),
    "root_height_below_minimum": TerminationTermCfg(
        func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
    ),
}


def _add_g1_on_street(project: mjswan.ProjectHandle) -> None:
    """A Hub-hosted model, Hub-hosted policies, and two captures to stand it in.

    mjlab has no task for this G1 — the two policies are third-party, trained against
    this XML — so the model itself comes from the Hub as a directory: the MJCF, the 35
    meshes it names, and the `LICENSE` that mjswan copies into the build beside it.
    """
    # Cached by `huggingface_hub`, so `add_scene_hf` below re-uses this download rather
    # than fetching the tree twice. The robot-only XML is what the tracer needs; the
    # scene XML that includes it is what the viewer shows.
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

    # The placement values line this capture up with this model. They belong to the
    # capture rather than to the file, so no part of the Hub knows them.
    scene.add_splat_hf(
        HF_REPO, "splats/street.spz", name="Street", scale=3.275, z_offset=0.708, yaw=40
    )

    # Two separate calls, not one with a list: these policies were trained on different
    # observation sets, so they are two MDPs. Neither is an mjlab export, so neither
    # carries metadata — the sidecar JSON beside it on the Hub supplies the PD gains.
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


# MyoFinger: 4 hinge joints (IFadb, IFmcp, IFpip, IFdip) driven by 5 MuJoCo muscle
# actuators. mjlab has no task for it, so the model comes from the Hub — two XMLs, one
# including the other, which is why it is a directory like the G1.
MYO_JOINT_NAMES = ("IFadb", "IFmcp", "IFpip", "IFdip")
MYO_MUSCLE_NAMES = ("extn", "adabR", "adabL", "mflx", "dflx")
MYO_OBS_DIM = 2 * len(MYO_JOINT_NAMES)  # joint_pos + joint_vel
MYO_INITIAL_QPOS = [0.0, 0.3, 0.3, 0.3]  # slightly flexed, not fully extended
MYO_INITIAL_QVEL = [0.0] * len(MYO_JOINT_NAMES)
MYO_SCENE = "scenes/myofinger/myofinger_v0.xml"


def _build_muscle_policy() -> onnx.ModelProto:
    """Random-uniform policy: ignores the observation, emits fresh [0, 1] samples.

    The point is the action path, not the network — a muscle is excited by a value in
    [0, 1], so a graph whose only op is `RandomUniform` drives every muscle with a new
    excitation each policy step and shows the actuator model doing its work.
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
    """Muscle actuators: no joint mapping, an overridden rest pose, sigmoid activation."""
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
        # A muscle is not a joint, so there is no mapping to declare — only a count.
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


# ─────────────────────────────────────────────────────────────────────────────


def setup_builder() -> mjswan.Builder:
    """Assemble the demo. Fetches from the Hub and from MyoHub; needs no credentials."""
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
