"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.

The cartpole tasks look choppier than the rest: mjlab gives them control_dt=0.05
(timestep 0.01 x decimation 5) against 0.02 elsewhere, and both mjlab's viewer and
mjswan sample render state once per control step, so their 100 Hz physics substeps
never reach a frame. Playback is still 1x real time.
"""

from __future__ import annotations

from mjlab.tasks.registry import load_env_cfg

import mjswan

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.defaults"

import mjswan.mjlab.bindings  # noqa: F401 - registers the command bindings
from mjswan.mjlab import register_custom_terminations

# NOTE: Replace these with your own WandB entity and project.
ENTITY = "ttktjmt-org"
PROJECT = "mjlab"
TASK_RUN_ID_MAP: dict[str, str | list[str]] = {
    "Mjlab-Velocity-Flat-Unitree-G1": "vel-flat-g1",
    "Mjlab-Velocity-Rough-Unitree-G1": ["mowqlkd5", "sif72y3p", "rsb8tc3g", "7veqaznf"],
    "Mjlab-Velocity-Flat-Unitree-Go1": "vel-flat-go1-v3",
    "Mjlab-Velocity-Rough-Unitree-Go1": ["basgo8hx", "ad4peite"],
    "Mjlab-Lift-Cube-Yam": "ajfybu8m",
    "Mjlab-Cartpole-Balance": "cartpole-balance-v2",
    "Mjlab-Cartpole-Swingup": "cartpole-swingup",
}
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
}


def main():
    builder = mjswan.Builder(debug=True)
    project = builder.add_project(name="mjlab Tasks")

    for task_id, wandb_run_id in TASK_RUN_ID_MAP.items():
        env_cfg = load_env_cfg(task_id, play=True)
        register_custom_terminations(env_cfg)
        scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
        if viewer_cfg := TASK_VIEWER_CONFIG_MAP.get(task_id):
            scene.set_viewer(viewer_cfg)
        run_ids = [wandb_run_id] if isinstance(wandb_run_id, str) else wandb_run_id
        wandb_paths = [f"{ENTITY}/{PROJECT}/{rid}" for rid in run_ids]
        scene.add_policy_wandb(wandb_paths)

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
