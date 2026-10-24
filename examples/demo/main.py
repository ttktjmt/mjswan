"""mjswan Demo Application

A tour of what mjswan does, hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/

Two projects:

- **mjlab Tasks**: mjlab's own tasks, taken as they are. Every scene is
  ``add_scene_mjlab`` plus trained checkpoints; nothing is hand-written.
- **Showcase**: what mjswan adds on top of the same engine, namely a Gaussian Splat
  background, a model mjlab has no task for, muscle actuators, and skins: characters
  drawn over a humanoid and a quadruped.

No asset is stored in this repository. Models come from mjlab or the Hugging Face Hub,
policies and skins from the Hub, and WBC-Mjlab's policy and clips from its deploy
repository on GitHub at a pinned commit. Both are public, so a build needs no
credentials.
"""

import math
import os
import re
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import mujoco
import onnx
import yaml
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
from mjswan.source import hf, url

#: Every asset this demo does not get from mjlab. Public, so the build is anonymous.
HF_REPO = "ttktjmt/mjswan"
# Its checkpoints mirror W&B runs. To add a task's:
#   1. fetch each `model_<step>.pt` of its runs (`mjswan.source.wandb.fetch_checkpoints`);
#   2. export each with `mjswan.mjlab.runner.export_checkpoint`, as `add_policy_wandb`
#      does. That writes no metadata, so attach it as mjlab's task runners do on
#      `save()`: `get_base_metadata(context.env.unwrapped, ...)` and
#      `attach_metadata_to_onnx` from `mjlab.rl.exporter_utils`, plus the motion term's
#      `anchor_body_name` and `body_names` for tracking. A task with no `joint_pos`
#      action term (cartpole) has none;
#   3. upload it as `checkpoints/<task id, lower-cased>/model_<step>.onnx`.

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
    step = re.compile(rf"{re.escape(prefix)}.*_(\d+)\.onnx")
    steps = sorted(
        (int(match[1]), name) for name in repo_onnx if (match := step.fullmatch(name))
    )
    if not steps:
        raise ValueError(
            f"No checkpoints under {prefix!r} in {HF_REPO!r}. The mirror is a snapshot "
            "of the W&B runs; mirror the task's as the comment at HF_REPO lists."
        )
    return [name for _, name in steps]


def _add_mjlab_tasks(builder: mjswan.Builder, repo_onnx: list[str]) -> None:
    """mjlab's tasks, as mjlab gives them, driven by the mirrored checkpoints."""
    project = builder.add_project(name="mjlab Tasks")

    for task_id in MJLAB_TASKS:
        _add_mjlab_task(project, task_id, _checkpoints_for(task_id, repo_onnx))

    _add_tracking_scene(project, _checkpoints_for(TRACKING_TASK, repo_onnx))


def _add_mjlab_task(
    project: mjswan.ProjectHandle,
    task_id: str,
    checkpoints: list[str],
    name: str | None = None,
) -> mjswan.SceneHandle:
    """One of mjlab's tasks, as mjlab gives it, driven by ``checkpoints``."""
    env_cfg = load_env_cfg(task_id, play=True)
    scene = project.add_scene_mjlab(task_id, name=name, env_cfg=env_cfg)
    # Only the files are named here: the MDP and the action order come from
    # `env_cfg`, the rest pose from each `.onnx`'s mjlab metadata, else from the
    # model's keyframe (cartpole's exports carry none).
    scene.add_policy_hf(HF_REPO, filename=checkpoints)
    return scene


def _add_tracking_scene(
    project: mjswan.ProjectHandle, checkpoints: list[str]
) -> mjswan.SceneHandle:
    """G1 Spinkick: the one task whose policy needs a reference motion beside it."""
    env_cfg = load_env_cfg(TRACKING_TASK, play=True)
    motion_term = env_cfg.commands["motion"]
    scene = project.add_scene_mjlab(TRACKING_TASK, env_cfg=env_cfg)

    # Slot tables are positional and the export's own tensor names do not say which
    # slot is which (ADR 0006 §5). Only `action` is read; the six after it are the
    # reference pose the clip already carries, named so the table lines up.
    policies = scene.add_policy_hf(
        HF_REPO,
        filename=checkpoints,
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
    return scene


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
            azimuth=146.0,
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
            azimuth=-60.0,
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


# Skins (ADR 0010): a character drawn over the robot and posed from its bodies, while
# physics, collision and the policy stay the robot's. Each file sits under `skins/` in
# HF_REPO with its license, and the repository's README says where it came from. The
# build bundles each byte for byte, so a VRM keeps the license its meta states.
# AvatarSample_A to C are not CC0: pixiv's terms let anyone use and redistribute them as
# distributed, but never relabelled as CC0, sold, behind a paywall, with an editor for
# their looks, or shown as if pixiv or VRoid endorse mjswan.

#: The G1 body each VRM humanoid bone follows. The waist's yaw and roll go to spine and
#: its pitch to chest; the G1 has no neck, so neck and head follow the torso.
G1_BONES = {
    "hips": "pelvis",
    "spine": "waist_roll_link",
    "chest": "torso_link",
    "neck": "torso_link",
    "head": "torso_link",
    **{
        f"{side}{bone}": f"{side.lower()}_{body}"
        for side in ("left", "right")
        for bone, body in (
            ("UpperLeg", "hip_yaw_link"),
            ("LowerLeg", "knee_link"),
            ("Foot", "ankle_roll_link"),
            ("UpperArm", "shoulder_yaw_link"),
            ("LowerArm", "elbow_link"),
            ("Hand", "wrist_yaw_link"),
        )
    },
}

#: The G1 holding its arms out straight, as every VRM rests: at its zero pose it hangs its
#: upper arms and bends its elbows square.
G1_T_POSE = {
    "left_shoulder_roll_joint": math.pi / 2,
    "right_shoulder_roll_joint": -math.pi / 2,
    "left_elbow_joint": math.pi / 2,
    "right_elbow_joint": math.pi / 2,
}

#: Every VRM rests in a T-pose, so one map and one rest pose fit them all.
G1_VRM = mjswan.SkinConfig(bones=G1_BONES, rest_pose=G1_T_POSE, retarget="rotation")

#: The humanoid's skins, in the Skin menu's order; the first is shown on load.
G1_SKINS = {
    "VRM1 Constraint Twist Sample": "skins/pixiv/VRM1_Constraint_Twist_Sample.vrm",
    "AvatarSample_A": "skins/pixiv/AvatarSample_A.vrm",
    "AvatarSample_B": "skins/pixiv/AvatarSample_B.vrm",
    "AvatarSample_C": "skins/pixiv/AvatarSample_C.vrm",
    "Crimsom": "skins/100avatars/100Avatars_001_Crimsom.vrm",
    "Cool Banana": "skins/100avatars/100Avatars_024_CoolBanana.vrm",
}

# 100Avatars dedicates each VRM to CC0, in its meta and release notes; the CC BY 4.0
# file at its repository's root came with the 2020 release, before the VRMs. The credit
# is a courtesy.
HUNDRED_AVATARS_NOTICE = (
    "Crimsom and Cool Banana: from 100Avatars by Polygonal Mind (CC0 1.0), "
    "https://github.com/PolygonalMind/100Avatars\n"
)

GO1_TASK = "Mjlab-Velocity-Rough-Unitree-Go1"

#: The Go1 body each bone follows: front legs for arms, hind legs for legs. The Go1 has
#: no neck, so the animal's spine, neck and head ride its trunk.
GO1_BONES = {
    **{bone: "trunk" for bone in ("hips", "spine", "chest", "neck", "head")},
    **{
        f"{side}{bone}": f"{leg}_{body}"
        for side, front, rear in (("left", "FL", "RL"), ("right", "FR", "RR"))
        for bone, leg, body in (
            ("UpperArm", front, "thigh"),
            ("LowerArm", front, "calf"),
            ("Hand", front, "calf"),
            ("UpperLeg", rear, "thigh"),
            ("LowerLeg", rear, "calf"),
            ("Foot", rear, "calf"),
            ("Toes", rear, "calf"),
        )
    },
}

#: VRM humanoid bone -> the Pug's bone. Its hips bone is mid-body, as the Go1's trunk
#: origin is; it has no toes.
PUG_HUMANOID = {
    "hips": "Body",
    "spine": "Shoulders",
    "neck": "Neck",
    "head": "Head",
    "leftUpperArm": "FrontUpLeg.L",
    "leftLowerArm": "FrontLowLeg.L",
    "leftHand": "FrontFoot.L",
    "rightUpperArm": "FrontUpLeg.R",
    "rightLowerArm": "FrontLowLeg.R",
    "rightHand": "FrontFoot.R",
    "leftUpperLeg": "BackUpLeg.L",
    "leftLowerLeg": "BackLowLeg.L",
    "leftFoot": "BackFoot.L",
    "rightUpperLeg": "BackUpLeg.R",
    "rightLowerLeg": "BackLowLeg.R",
    "rightFoot": "BackFoot.R",
}

#: Quaternius's flat colours as toon shading: the shaded side 60% of the lit one, the two
#: blending softly, as the sources' do.
QUATERNIUS_TOON = mjswan.ToonShading(shade=0.6, toony=0.5)

#: The Pug of Quaternius's Farm Animals.
PUG = mjswan.SkinConfig(
    bones={bone: GO1_BONES[bone] for bone in PUG_HUMANOID},
    humanoid=PUG_HUMANOID,
    rest_pose="init_state",
    material=QUATERNIUS_TOON,
    license="CC0-1.0",
    notice=(
        "Pug: from Farm Animals Animated by Quaternius (CC0 1.0), "
        "https://quaternius.itch.io/lowpoly-animated-animals\n"
        "Made forward-kinematic and scaled by make_pug.py, beside it in "
        "https://huggingface.co/ttktjmt/mjswan/tree/main/skins/quaternius\n"
    ),
)

#: VRM humanoid bone -> the node of one of Quaternius's animals on Poly Pizza: the front
#: legs as arms, the four bones of a hind leg as upper and lower leg, foot and toes.
POLY_PIZZA_HUMANOID = {
    "hips": "Back",
    "spine": "Torso",
    "chest": "Torso2",
    "neck": "Neck1",
    "head": "Head",
    **{
        f"{side}{bone}": f"{node}.{s}"
        for side, s in (("left", "L"), ("right", "R"))
        for bone, node in (
            ("UpperArm", "FrontUpperLeg"),
            ("LowerArm", "FrontLowerLeg"),
            ("Hand", "IKFrontLeg"),
            ("UpperLeg", "BackLeg"),
            ("LowerLeg", "BackUpperLeg"),
            ("Foot", "BackLowerLeg"),
            ("Toes", "IKBackLeg"),
        )
    },
}

#: Its feet hang from IK controls at the root, so from joint angles alone they would
#: stay behind; hung from the legs they end, the legs carry them.
POLY_PIZZA_FEET = {
    f"IK{end}Leg.{s}": f"{end}LowerLeg.{s}" for end in ("Front", "Back") for s in "LR"
}

POLY_PIZZA = mjswan.SkinConfig(
    bones={bone: GO1_BONES[bone] for bone in POLY_PIZZA_HUMANOID},
    humanoid=POLY_PIZZA_HUMANOID,
    reparent=POLY_PIZZA_FEET,
    rest_pose="init_state",
    material=QUATERNIUS_TOON,
    license="CC0-1.0",
)

#: Each scaled so its shoulders and hips sit 0.376 m apart, as the Go1's front and rear
#: hips do.
SHIBA_INU = replace(
    POLY_PIZZA,
    scale=0.236,
    notice="Shiba Inu by Quaternius (CC0 1.0), https://poly.pizza/m/y4wdQpg767\n",
)
QUATERNIUS_FOX = replace(
    POLY_PIZZA,
    scale=0.2425,
    springs=[
        mjswan.SpringChain(
            root="Tail1",
            stiffness=1.5,
            drag=0.3,
            gravity=0.1,
            hit_radius=0.03,
            # The pelvis and the hind thighs, which the tail must not swing through.
            colliders=(
                mjswan.SpringCollider("Back", radius=0.06),
                *(mjswan.SpringCollider(f"BackLeg.{s}", radius=0.035) for s in "LR"),
            ),
        )
    ],
    notice="Fox by Quaternius (CC0 1.0), https://poly.pizza/m/Bc97C66HKi\n",
)

#: The Go1's skins, in the Skin menu's order: rigged glTF quadrupeds used as they are,
#: their front legs named as arms, resting as the Go1 stands in its keyframe.
GO1_SKINS = {
    "Shiba Inu": ("skins/quaternius/shiba_inu.glb", SHIBA_INU),
    "Fox": ("skins/quaternius/fox.glb", QUATERNIUS_FOX),
    "Pug": ("skins/quaternius/pug.glb", PUG),
}


def _skin_file(path: str) -> str:
    """A skin's file, fetched from HF_REPO and cached like every Hub asset here."""
    return str(hf.fetch_file(HF_REPO, path))


#: WBC-Mjlab's G1 (https://github.com/wbc-mjlab/wbc-mjlab): one whole-body tracking
#: policy over a library of clips, as mjswan_playground's `wbc` task runs it. The
#: `wbc-mjlab` package registers the task; the policy and clips are its deploy repo's.
WBC_TASK = "Wbc-G1"
WBC_DEPLOY = (
    "https://raw.githubusercontent.com/wbc-mjlab/wbc-g1-deploy/"
    "6dabf86fddc2b7b429b09e74999732fcde3441f9"
)


def _wbc_file(path: str) -> Path:
    """A file of wbc-g1-deploy at the pinned commit."""
    return url.fetch_file(f"{WBC_DEPLOY}/{path}")


def _add_humanoid_skins(project: mjswan.ProjectHandle) -> None:
    """WBC-Mjlab's G1 under VRM avatars, tracking any of its clips."""
    contract = yaml.safe_load(
        _wbc_file("config/policy/wbc/params/config.yaml").read_text()
    )
    env_cfg = load_env_cfg(WBC_TASK, play=True)
    # Training-only events the deploy runtime does not apply either.
    for training_only in ("assistive_wrench", "pull_robot"):
        env_cfg.events.pop(training_only, None)
    # The browser runs the PD alone, so UnitreeActuator's friction terms become the
    # joints' own frictionloss and damping, which the exported scene carries.
    for actuator in env_cfg.scene.entities["robot"].articulation.actuators:
        actuator.frictionloss, actuator.viscous_damping = actuator.fs, actuator.fd
        actuator.fs = actuator.fd = 0.0
    scene = project.add_scene_mjlab(WBC_TASK, name="Humanoid skins", env_cfg=env_cfg)

    policy = scene.add_policy(
        name="WBC Tracking",
        policy=onnx.load(str(_wbc_file("config/policy/wbc/params/policy.onnx"))),
        policy_joint_names=[f"robot/{joint}" for joint in contract["joint_names"]],
        default_joint_pos=[float(v) for v in contract["default_joint_pos"]],
    )
    manifest = yaml.safe_load(_wbc_file("config/clips/manifest.yaml").read_text())
    # The manifest's default first: the viewer opens on it.
    clips = sorted(
        manifest["clips"], key=lambda clip: clip["name"] != manifest["default"]
    )
    for clip in clips:
        policy.add_motion(
            name=clip["name"],
            source=str(_wbc_file(f"config/clips/{clip['file']}")),
            fps=50.0,
            anchor_body_name=contract["tracking"]["anchor_body_name"],
            body_names=env_cfg.commands["motion"].body_names,
            dataset_joint_names=list(contract["joint_names"]),
            default=clip is clips[0],
        )
    scene.add_attribution(
        "wbc-g1-deploy", license=_wbc_file("LICENSE"), notice=_wbc_file("NOTICE")
    )

    for name, path in G1_SKINS.items():
        scene.add_skin(name, replace(G1_VRM, source=_skin_file(path)))
    scene.add_attribution(
        "100avatars", license="CC0-1.0", notice=HUNDRED_AVATARS_NOTICE
    )


def _add_quadruped_skins(project: mjswan.ProjectHandle, repo_onnx: list[str]) -> None:
    """The Go1 on rough terrain under three animals, on its final checkpoint alone."""
    checkpoints = _checkpoints_for(GO1_TASK, repo_onnx)[-1:]
    scene = _add_mjlab_task(project, GO1_TASK, checkpoints, name="Quadruped skins")
    for name, (path, config) in GO1_SKINS.items():
        scene.add_skin(name, replace(config, source=_skin_file(path)))


def _add_showcase(builder: mjswan.Builder, repo_onnx: list[str]) -> None:
    project = builder.add_project(name="Showcase")
    _add_g1_on_street(project)
    _add_myofinger(project)
    _add_humanoid_skins(project)
    _add_quadruped_skins(project, repo_onnx)


def setup_builder() -> mjswan.Builder:
    """Assemble the demo. Fetches from mjlab and the public Hub, with no credentials."""
    builder = mjswan.Builder(
        base_path=os.getenv("MJSWAN_BASE_PATH", "/"),
        gtm_id="GTM-W79HQ38W",
        license="Apache-2.0",
        copyright="mjswan Developers",
    )
    repo_onnx = hf.list_repo_onnx(HF_REPO)
    _add_mjlab_tasks(builder, repo_onnx)
    _add_showcase(builder, repo_onnx)
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
