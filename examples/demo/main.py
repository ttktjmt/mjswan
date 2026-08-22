"""mjswan Demo Application

This is a demo application showcasing the usage of mjswan.
The demo app is hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/
Run `uv pip install git+https://github.com/ttktjmt/mjlab_myochallenge.git` before running this script.
"""

import os
import posixpath
from pathlib import Path

import gymnasium.logger as gym_logger
import mujoco
import mujoco.mjx as _mjx
import onnx
from mujoco_playground import registry

# Suppress gymnasium logger output from myosuite
_prev_gym_level = gym_logger.min_level
gym_logger.min_level = gym_logger.ERROR

from myosuite import gym_registry_specs  # noqa: E402
from myosuite.envs.myo import myochallenge  # noqa: E402, F401 - for env registration

gym_logger.min_level = _prev_gym_level

import torch  # noqa: E402
from mjlab.envs.mdp import observations as obs_fns  # noqa: E402
from mjlab.envs.mdp import terminations as term_fns  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from robot_descriptions._descriptions import DESCRIPTIONS  # noqa: E402

import mjswan  # noqa: E402
from mjswan.envs.mdp.actions import (  # noqa: E402
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from mjswan.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjswan.managers.termination_manager import TerminationTermCfg  # noqa: E402
from mjswan.trace_env import build_single_entity_trace_env  # noqa: E402

# --- Demo-specific observations. These scenes have no mjlab task, so each supplies its
# own trace env via SceneHandle.set_trace_env, and the terms below are written against
# the same live-env API mjlab's own use. ---

#: Three frames, newest first. `history_length` stacks chronologically (mjlab's order);
#: these Go2 checkpoints were trained on the reverse, so they name the offsets.
GO2_HISTORY_STEPS = (0, 1, 2)


def joint_pos_abs(env, *, asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot")):
    """Absolute joint positions (no default-pose subtraction)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def impedance_command(env, **_):
    """Impedance command placeholder: 27 zeros (no trained impedance head in this demo)."""
    del env
    return torch.zeros(1, 27)


def velocity_command_padding(env, **_):
    """Zero-padding (13) to fill the oscillator command slots this policy expects."""
    del env
    return torch.zeros(1, 13)


def _fix_unitree_mujoco_macos() -> None:
    """Pre-fix the unitree_mujoco cache on macOS to avoid case-sensitivity errors.

    On macOS (case-insensitive filesystem), robot_descriptions fails to checkout
    the unitree_mujoco repo because git history contains a rename from
    terrain.STL -> terrain.stl, which macOS treats as the same file.

    Fix: clone with --no-checkout so no files exist in the working tree before
    the target commit is checked out, and set core.ignorecase=false so git
    handles the case-rename correctly.
    """
    import platform
    import shutil
    import subprocess

    if platform.system() != "Darwin":
        return

    cache_dir = Path.home() / ".cache/robot_descriptions/unitree_mujoco"

    if cache_dir.exists():
        result = subprocess.run(
            ["git", "config", "core.ignorecase"],
            cwd=cache_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip() == "false":
            return  # Already correctly configured
        shutil.rmtree(cache_dir)

    print("Preparing unitree_mujoco cache for macOS (one-time setup)...")
    subprocess.run(
        [
            "git",
            "clone",
            "--no-checkout",
            "https://github.com/unitreerobotics/unitree_mujoco.git",
            str(cache_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.ignorecase", "false"],
        cwd=cache_dir,
        check=True,
    )


def _add_g1_scene(project) -> None:
    g1_scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=4.3,
            elevation=-33.0,
            azimuth=-34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    )
    g1_scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_g1/g1.xml")
        )
    )
    g1_scene.add_splat(
        name="Street",
        source="assets/unitree_g1/street.spz",
        scale=3.275,
        z_offset=0.708,
        yaw=40,
        control=True,
    )

    g1_terminations = {
        "bad_orientation": TerminationTermCfg(
            func=term_fns.bad_orientation, params={"limit_angle": 1.0}
        ),
        "root_height_below_minimum": TerminationTermCfg(
            func=term_fns.root_height_below_minimum, params={"minimum_height": 0.3}
        ),
    }

    g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
        terminations=g1_terminations,
        commands={
            "velocity": mjswan.velocity_command(
                lin_vel_x=(-1.5, 1.5),
                lin_vel_y=(-0.5, 0.5),
                default_lin_vel_x=0.5,
                default_lin_vel_y=0.0,
            )
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
    )

    g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        name="Balance",
        config_path="assets/unitree_g1/balance.json",
        terminations=g1_terminations,
        # Dict form, not a bare group: `balance.json` declares `in_keys: ["observation"]`,
        # and the key *is* the ONNX input name. A bare group would land under "policy" and
        # the runtime would find no input by that name.
        observations={
            "observation": ObservationGroupCfg(
                terms={
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "prev_actions": ObservationTermCfg(func=obs_fns.last_action),
                }
            )
        },
    )


def _add_go2_scene(project) -> None:
    go2_scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        name="Go2",
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=3.8,
            elevation=-20.0,
            azimuth=34.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="base",
        )
    )
    go2_scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_go2/go2.xml")
        )
    )

    go2_actions = {
        "joint_pos": JointPositionActionCfg(
            scale=0.5,
            stiffness=25.0,
            damping=0.5,
        )
    }
    # Two groups, so this stays a dict: `vanilla.json`/`robust.json` declare
    # `in_keys: ["policy", "is_init", "adapt_hx", "command_"]`. `is_init`/`adapt_hx` are the
    # recurrent carry the runtime supplies itself; the two observation inputs are ours, and
    # each key has to match its input name exactly.
    go2_velocity_obs = {
        "policy": ObservationGroupCfg(
            terms={
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity, history_steps=GO2_HISTORY_STEPS
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel, history_steps=GO2_HISTORY_STEPS
                ),
                "joint_vel": ObservationTermCfg(
                    func=obs_fns.joint_vel_rel, history_steps=GO2_HISTORY_STEPS
                ),
                "prev_actions": ObservationTermCfg(
                    func=obs_fns.last_action,
                    history_steps=GO2_HISTORY_STEPS,
                    history_interleaved=True,
                ),
            }
        ),
        "command_": ObservationGroupCfg(
            terms={
                "velocity_cmd": ObservationTermCfg(
                    func=obs_fns.generated_commands,
                    params={"command_name": "velocity"},
                ),
                "velocity_cmd_pad": ObservationTermCfg(func=velocity_command_padding),
            }
        ),
    }

    go2_scene.add_policy(
        name="Facet",
        policy=onnx.load("assets/unitree_go2/facet.onnx"),
        config_path="assets/unitree_go2/facet.json",
        actions=go2_actions,
        # Genuinely multi-input (`facet.json`: policy + command + the recurrent carry), which
        # is what the dict form is for — one group per ONNX input, keyed by its name.
        observations={
            "policy": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity, history_steps=GO2_HISTORY_STEPS
                    ),
                    "joint_pos": ObservationTermCfg(
                        func=joint_pos_abs, history_steps=GO2_HISTORY_STEPS
                    ),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel, history_steps=GO2_HISTORY_STEPS
                    ),
                    "prev_actions": ObservationTermCfg(
                        func=obs_fns.last_action,
                        history_steps=GO2_HISTORY_STEPS,
                        history_interleaved=True,
                    ),
                }
            ),
            "command": ObservationGroupCfg(
                terms={
                    "impedance_cmd": ObservationTermCfg(func=impedance_command),
                }
            ),
        },
        commands={"velocity": mjswan.velocity_command()},
    )

    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/vanilla.onnx"),
        name="Vanilla",
        config_path="assets/unitree_go2/vanilla.json",
        actions=go2_actions,
        observations=go2_velocity_obs,
        commands={"velocity": mjswan.velocity_command()},
    )

    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/robust.onnx"),
        name="Robust",
        config_path="assets/unitree_go2/robust.json",
        actions=go2_actions,
        observations=go2_velocity_obs,
        commands={"velocity": mjswan.velocity_command()},
    )


def _add_go1_scene(project) -> None:
    go1_scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        spec=mujoco.MjSpec.from_file("assets/unitree_go1/go1.xml"),
        name="Go1",
    ).set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.2),
            distance=3.1,
            elevation=-25.0,
            azimuth=-45.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="trunk",
        )
    )
    go1_scene.set_trace_env(
        build_single_entity_trace_env(
            lambda: mujoco.MjSpec.from_file("assets/unitree_go1/go1.xml")
        )
    )

    # HiMLoco is not shown: it asks for an interleaved group history, and the runtime stores
    # that flag without applying it — the stack it feeds the policy is frame-major regardless.
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/decap.onnx"),
        name="Decap",
        config_path="assets/unitree_go1/decap.json",
        actions={
            "effort": JointEffortActionCfg(
                scale=8.0,
                stiffness=20.0,
                damping=0.5,
            )
        },
        # `decap.json` declares `in_keys: ["obs_history"]`; the key is that input's name.
        observations={
            "obs_history": ObservationGroupCfg(
                terms={
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                        scale=(2.0, 2.0, 0.25),
                    ),
                    "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(
                        func=obs_fns.joint_vel_rel,
                        scale=0.05,
                    ),
                    "prev_actions": ObservationTermCfg(func=obs_fns.last_action),
                }
            )
        },
        commands={"velocity": mjswan.velocity_command()},
    )


def _anymal_c_trace_spec() -> mujoco.MjSpec:
    # scene.mjz already carries an "init_state" keyframe from its own Entity build, and
    # EntityCfg adds one of the same name when wrapping it again — so drop the existing.
    spec = mujoco.MjSpec.from_zip("assets/anymal_c_velocity/scene.mjz")
    for key in list(spec.keys):
        if key.name == "init_state":
            spec.delete(key)
    return spec


def _add_anymal_c_scene(project) -> None:
    anymal_c_scene = project.add_scene(
        control_dt=0.02,  # 50 Hz control step
        name="ANYmal C Velocity",
        spec=mujoco.MjSpec.from_zip("assets/anymal_c_velocity/scene.mjz"),
    )
    anymal_c_scene.set_trace_env(build_single_entity_trace_env(_anymal_c_trace_spec))
    anymal_c_scene.add_policy(
        name="velocity 3000 iters",
        policy=onnx.load(
            "assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.onnx"
        ),
        config_path="assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.json",
        actions={
            "joint_pos": JointPositionActionCfg(
                scale=1.013,
                stiffness=19.739,
                damping=1.257,
            )
        },
        # mjlab's own export names the input `obs`, and the config declares it — so the group
        # is keyed to match rather than handed over bare.
        observations={
            "obs": ObservationGroupCfg(
                terms={
                    "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                    "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                    "projected_gravity": ObservationTermCfg(
                        func=obs_fns.projected_gravity
                    ),
                    "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                    "last_action": ObservationTermCfg(func=obs_fns.last_action),
                    "velocity_cmd": ObservationTermCfg(
                        func=obs_fns.generated_commands,
                        params={"command_name": "velocity"},
                    ),
                }
            )
        },
        commands={
            "velocity": mjswan.velocity_command(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-0.5, 0.5),
                default_lin_vel_x=0.5,
                default_lin_vel_y=0.0,
                default_ang_vel_z=0.0,
            )
        },
    )


def _add_mjswan_demo_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="mjswan Demo")
    _add_g1_scene(project)
    _add_go2_scene(project)
    _add_go1_scene(project)


def _add_robot_descriptions_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="Robot Descriptions", id="robotdesc")

    # ANYmal C Velocity from https://github.com/mujocolab/anymal_c_velocity
    _add_anymal_c_scene(project)

    def _rd_spec(module_name: str) -> mujoco.MjSpec:
        from importlib import import_module

        mjcf_path = Path(import_module(f"robot_descriptions.{module_name}").MJCF_PATH)
        # Prefer scene.xml (floor + lights) over the robot-only MJCF when available.
        scene_path = mjcf_path.parent / "scene.xml"
        return mujoco.MjSpec.from_file(
            str(scene_path if scene_path.exists() else mjcf_path)
        )

    for module, desc in DESCRIPTIONS.items():
        if desc.has_mjcf:
            scene_name = module.replace("_mj_description", "")
            scene_name = " ".join([word.capitalize() for word in scene_name.split("_")])
            project.add_scene(name=scene_name, spec=_rd_spec(module))


def _add_playground_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="MuJoCo Playground", id="playground")

    # TEMPORARY PATCH:
    # Force JAX backend for all environments: mujoco_playground inconsistently
    # migrated to warp as the default — some envs pass impl via config, others
    # call mjx.put_model() with no impl at all. Patching here covers both cases.
    # TODO: Once mujoco_playground fixes all envs to respect config_overrides,
    # replace this patch with simply: registry.load(env_name, config_overrides={"impl": "jax"})
    _orig_put_model = _mjx.put_model
    _mjx.put_model = lambda m, **kw: _orig_put_model(m, **{**kw, "impl": "jax"})

    # TEMPORARY PATCH:
    # reacher.py gates on `mujoco.__version__ >= "3.3.0"`, a string compare that is
    # False for "3.10.0", so it calls the pre-3.3 spec.find_body() that mujoco has
    # since renamed to spec.body(). Restore the old name as an alias.
    if not hasattr(mujoco.MjSpec, "find_body"):
        mujoco.MjSpec.find_body = mujoco.MjSpec.body
    try:
        for env_name in registry.ALL_ENVS:
            if "Sparse" in env_name:
                continue

            env = registry.load(env_name)
            with open(env.xml_path) as f:
                xml_content = f.read()
            spec = mujoco.MjSpec.from_string(xml_content, env.model_assets)

            # model_assets is consumed at parse time but not stored in spec.assets, so remap its
            # basename keys to the `dir/file` paths spec.to_xml() looks up.
            mesh_dir = spec.meshdir or ""
            tex_dir = spec.texturedir or ""

            def _add(directory: str, filename: str) -> None:
                if not filename:
                    return
                key = posixpath.join(directory, filename) if directory else filename
                basename = os.path.basename(key)
                if basename in env.model_assets:
                    spec.assets[key] = env.model_assets[basename]

            for mesh in spec.meshes:
                _add(mesh_dir, mesh.file)
            for texture in spec.textures:
                _add(tex_dir, texture.file)
                for cf in texture.cubefiles:
                    _add(tex_dir, cf)
            for hfield in spec.hfields:
                _add("", hfield.file)

            project.add_scene(name=env_name, spec=spec)
    finally:
        _mjx.put_model = _orig_put_model


def _add_myosuite_project(builder: mjswan.Builder) -> None:
    project = builder.add_project(name="MyoSuite", id="myosuite")

    registry_specs = gym_registry_specs()

    # (display_name, lookat, distance, elevation, azimuth)
    target_envs = {
        "myoChallengeDieReorientP2-v0": (
            "mc22 Die Reorient",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeBaodingP2-v1": (
            "mc22 Baoding",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeRelocateP2-v0": (
            "mc23 Relocate",
            (0.0, -0.1, 1.4),
            1.7,
            -7.0,
            -90.0,
        ),
        "myoChallengeChaseTagP2-v0": (
            "mc23 Chase Tag",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeBimanual-v0": (
            "mc24 Bimanual",
            (-0.1, -0.5, 1.4),
            1.3,
            -9.0,
            -61.0,
        ),
        "myoChallengeOslRunRandom-v0": (
            "mc24 OSL Run",
            (0.0, 0.0, 1.4),
            10.0,
            -15.0,
            -62.0,
        ),
        "myoChallengeTableTennisP2-v0": (
            "mc25 Table Tennis",
            (0.0, -1.0, 1.4),
            3.3,
            -11.0,
            -129.0,
        ),
        "myoChallengeSoccerP2-v0": (
            "mc25 Soccer",
            (0.0, -3.0, 2.0),
            14.7,
            -16.0,
            -172.0,
        ),
    }

    for env_name, (
        display_name,
        lookat,
        distance,
        elevation,
        azimuth,
    ) in target_envs.items():
        model_path = registry_specs[env_name].kwargs["model_path"]
        mjspec = mujoco.MjSpec.from_file(model_path)
        project.add_scene(name=display_name, spec=mjspec).set_viewer(
            mjswan.ViewerConfig(
                lookat=lookat,
                distance=distance,
                elevation=elevation,
                azimuth=azimuth,
                origin_type=mjswan.ViewerConfig.OriginType.WORLD,
            )
        )


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder with all demo projects configured.

    This function creates the builder and adds all projects, scenes, and policies
    but does not build or launch the application. Useful for testing.

    Returns:
        Configured Builder instance ready to be built.
    """
    _fix_unitree_mujoco_macos()
    # Ensure asset-relative paths resolve regardless of current working directory.
    os.chdir(Path(__file__).resolve().parent)
    base_path = os.getenv("MJSWAN_BASE_PATH", "/")
    builder = mjswan.Builder(base_path=base_path, gtm_id="GTM-W79HQ38W")

    _add_mjswan_demo_project(builder)
    _add_robot_descriptions_project(builder)
    _add_playground_project(builder)
    _add_myosuite_project(builder)

    return builder


def _copy_licenses(output_dir: Path) -> None:
    """Copy LICENSE and NOTICE files into the built output.

    - robot_descriptions (robotdesc): copies per scene from each repo's REPOSITORY_PATH.
    - myosuite / mujoco_playground: copies to the project root from the dist-info licenses/.
    """
    import importlib.metadata
    import shutil
    from importlib import import_module

    # Per-scene for robot_descriptions (robotdesc project)
    robotdesc_assets = output_dir / "robotdesc" / "assets"
    if robotdesc_assets.exists():
        for module, desc in DESCRIPTIONS.items():
            if not desc.has_mjcf:
                continue
            scene_id = module.replace("_mj_description", "")
            scene_dir = robotdesc_assets / scene_id
            if not scene_dir.exists():
                continue
            mod = import_module(f"robot_descriptions.{module}")
            if not hasattr(mod, "REPOSITORY_PATH"):
                continue
            for fname in ["LICENSE", "NOTICE"]:
                src = Path(mod.REPOSITORY_PATH) / fname
                if src.exists():
                    shutil.copy2(src, scene_dir / fname)

    # Project-level for myosuite and mujoco_playground
    for project_id, pkg_name in [
        ("myosuite", "myosuite"),
        ("playground", "playground"),
    ]:
        project_dir = output_dir / project_id
        if not project_dir.exists():
            continue
        try:
            dist = importlib.metadata.Distribution.from_name(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for fname in ["LICENSE", "NOTICE"]:
            matches = [
                f
                for f in (dist.files or [])
                if Path(str(f)).name == fname and "dist-info" in str(f)
            ]
            for f in matches:
                src = Path(str(dist.locate_file(f)))
                if src.exists():
                    shutil.copy2(src, project_dir / fname)
                    break


def main():
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
        builder = setup_builder()
        app = builder.build()
        _copy_licenses(dist_dir)
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
