"""mjswan Demo Application

This is a demo application showcasing the usage of mjswan.
The demo app is hosted on GitHub Pages: https://ttktjmt.github.io/mjswan/
"""

import os
import posixpath
from pathlib import Path

import gymnasium.logger as gym_logger
import mujoco
import onnx
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mujoco_playground import registry

# Suppress gymnasium logger output from myosuite
_prev_gym_level = gym_logger.min_level
gym_logger.set_level(gym_logger.DISABLED)

from myosuite import gym_registry_specs  # noqa: E402
from myosuite.envs.myo import myochallenge  # noqa: E402, F401 - for env registration

gym_logger.set_level(_prev_gym_level)

from robot_descriptions._descriptions import DESCRIPTIONS  # noqa: E402

import mjswan  # noqa: E402


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

    # =======================
    # 1. mjswan Demo Project
    # =======================

    demo_project = builder.add_project(name="mjswan Demo")

    # 1.A. Unitree G1
    g1_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_g1/scene.xml"),
        name="G1",
    )
    g1_loco_policy = g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/locomotion.onnx"),
        name="Locomotion",
        config_path="assets/unitree_g1/locomotion.json",
    )
    g1_loco_policy.add_velocity_command(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
    )
    g1_scene.add_policy(
        policy=onnx.load("assets/unitree_g1/balance.onnx"),
        name="Balance",
        config_path="assets/unitree_g1/balance.json",
    )

    # 1.B. Unitree Go2
    go2_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_go2/scene.xml"),
        name="Go2",
    )
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/facet.onnx"),
        name="Facet",
        config_path="assets/unitree_go2/facet.json",
    ).add_velocity_command()
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/vanilla.onnx"),
        name="Vanilla",
        config_path="assets/unitree_go2/vanilla.json",
    ).add_velocity_command()
    go2_scene.add_policy(
        policy=onnx.load("assets/unitree_go2/robust.onnx"),
        name="Robust",
        config_path="assets/unitree_go2/robust.json",
    ).add_velocity_command()

    # 1.C. Unitree Go1
    go1_scene = demo_project.add_scene(
        spec=mujoco.MjSpec.from_file("assets/unitree_go1/go1.xml"),
        name="Go1",
    )
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/himloco.onnx"),
        name="HiMLoco",
        config_path="assets/unitree_go1/himloco.json",
    ).add_velocity_command()
    go1_scene.add_policy(
        policy=onnx.load("assets/unitree_go1/decap.onnx"),
        name="Decap",
        config_path="assets/unitree_go1/decap.json",
    ).add_velocity_command()

    # ============================
    # 2. MuJoCo Menagerie Project
    # ============================

    menagerie_project = builder.add_project(name="MuJoCo Menagerie", id="menagerie")

    # ANYmal C Velocity (uses mjlab, not robot_descriptions)
    anymal_c_velocity_env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Anymal-C")
    anymal_c_velocity_env_cfg.scene.num_envs = 1
    anymal_c_velocity_scene = Scene(anymal_c_velocity_env_cfg.scene, device="cpu")
    anymal_c_scene = menagerie_project.add_scene(
        name="ANYmal C Velocity",
        spec=anymal_c_velocity_scene.spec,
    )
    anymal_c_scene.add_policy(
        name="velocity 3000 iters",
        policy=onnx.load(
            "assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.onnx"
        ),
        config_path="assets/anymal_c_velocity/Mjlab-Velocity-Flat-Anymal-C.3000.json",
    ).add_velocity_command(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        default_lin_vel_x=0.5,
        default_lin_vel_y=0.0,
        default_ang_vel_z=0.0,
    )

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

            menagerie_project.add_scene(name=scene_name, spec=_rd_spec(module))

    # =============================
    # 3. MuJoCo Playground Project
    # =============================

    playground_project = builder.add_project(name="MuJoCo Playground", id="playground")

    for env_name in registry.ALL_ENVS:
        if "Sparse" in env_name:
            continue

        env = registry.load(env_name)
        xml_content = open(env.xml_path).read()
        spec = mujoco.MjSpec.from_string(xml_content, env.model_assets)

        # model_assets is consumed at parse time but not stored in spec.assets.
        # Remap basename keys (as in env.model_assets) to the effective paths
        # that spec.to_xml() looks up: dir/file (or just file when dir is empty).
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

        playground_project.add_scene(name=env_name, spec=spec)

    # ====================
    # 4. MyoSuite Project
    # ====================

    myosuite_project = builder.add_project(name="MyoSuite", id="myosuite")

    registry_specs = gym_registry_specs()

    target_env_name_map = {
        "myoChallengeDieReorientP2-v0": "mc22 Die Reorient",
        "myoChallengeBaodingP2-v1": "mc22 Baoding",
        "myoChallengeRelocateP2-v0": "mc23 Relocate",
        "myoChallengeChaseTagP2-v0": "mc23 Chase Tag",
        "myoChallengeBimanual-v0": "mc24 Bimanual",
        "myoChallengeOslRunRandom-v0": "mc24 OSL Run",
        "myoChallengeTableTennisP2-v0": "mc25 Table Tennis",
        "myoChallengeSoccerP2-v0": "mc25 Soccer",
    }

    for env_name, display_name in target_env_name_map.items():
        model_path = registry_specs[env_name].kwargs["model_path"]
        mjspec = mujoco.MjSpec.from_file(model_path)
        myosuite_project.add_scene(name=display_name, spec=mjspec)

    return builder


def main():
    """Main entry point for the demo application.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    builder = setup_builder()
    # Build and launch the application
    app = builder.build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
