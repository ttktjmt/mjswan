"""MuscleMimic Fullbody motion-tracking demo.

This example exercises mjswan's muscle-driven mimic playback path for the
myosuite ``myoMimicFullbody-v0`` task:
- registers the task via myosuite's mjlab bootstrap (needs a motion clip on disk)
- pulls a sample clip from the gated ``amathislab/musclemimic-retargeted`` HF dataset
- loads policy checkpoints exported from a W&B training run
- bundles the clip into the browser app so TS observations can read it at runtime

See ``README.md`` for prerequisites (HF auth, musclemimic_models, W&B login).
"""

from __future__ import annotations

import os
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.musclemimic"

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry
import numpy as np
from huggingface_hub import hf_hub_download
from mjlab.tasks.registry import load_env_cfg
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)

import mjswan
from mjswan.command import CommandTermConfig
from mjswan.document.ids import name2id
from mjswan.motion import MotionConfig

from . import (
    observations,  # noqa: F401 - registers all mimic observation terms
    terminations,  # noqa: F401 - registers mimic_deviation termination term
)

CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def _bootstrap_mimic_task() -> Path:
    """Download a registration clip and register ``myoMimicFullbody-v0``.

    Returns the path to the downloaded clip (used to bundle into the app).
    """
    clip_path = Path(
        hf_hub_download(
            repo_id=CLIP_REPO_ID,
            filename=CLIP_FILENAME,
            repo_type="dataset",
        )
    )
    os.environ["MIMIC_CLIP"] = str(clip_path)
    bootstrap_myosuite_mjlab_registry()
    return clip_path


def setup_builder() -> mjswan.Builder:
    """Create the builder for the MuscleMimic Fullbody tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    clip_path = _bootstrap_mimic_task()

    run_paths = [
        "ttktjmt-org/mjlab/v3q5ete6",
        "ttktjmt-org/mjlab/srir3zxo",
        "ttktjmt-org/mjlab/y38ix8cv",
    ]
    task_id = "myoMimicFullbody-v0"

    builder = mjswan.Builder(debug=True)

    project = builder.add_project(name="MuscleMimic Fullbody")

    # TODO: adopt the single-config shape the other mjlab examples use — load `env_cfg`
    # and inject `mimic_deviation` above this line, pass `env_cfg=` here, and drop the
    # term sets from `add_policy_wandb`. Building the scene first means it never sees
    # the injected config, hence the explicit term sets below.
    #
    # Converting makes mjlab's own `TerminationManager` see `mimic_deviation`'s extra
    # params. That should be fine (params are validated at `compute()`, and the trace
    # env is never stepped), but `myoMimicFullbody-v0` is not installed in CI, so run
    # the example once afterwards.
    scene = project.add_scene_mjlab(task_id)

    env_cfg = load_env_cfg(task_id, play=True)

    # mjlab's `mimic_deviation` closure carries no params, so inject them here.
    mimic_dev = env_cfg.terminations.get("mimic_deviation")
    if mimic_dev is not None:
        mimic_dev.params = {
            "clip_url": str(clip_path),
            "site_names": observations.SITE_NAMES,
            "body_names": observations.BODY_NAMES,
            "fps": observations.FPS,
            "site_err_threshold": 1.0,
            "root_err_threshold": 0.3,
        }

    # mimic_lookahead is registered as unsupported and will be skipped by the builder.
    # The four term sets are explicit only because of the load order — see the TODO above.
    policy_handles = scene.add_policy_wandb(
        run_paths,
        task_id=task_id,
        observations=env_cfg.observations["actor"],
        commands=env_cfg.commands,
        actions=env_cfg.actions,
        terminations=env_cfg.terminations,
    )

    # Start from clip frame 0: RSI training makes the T-pose keyframe out of distribution.
    clip_npz = np.load(clip_path, allow_pickle=True)
    initial_qpos = clip_npz["qpos"][0].tolist()
    initial_qvel = clip_npz["qvel"][0].tolist()

    mimic_clip = MotionConfig(
        name="mimic_clip",
        source=str(clip_path),
        clip_format="qpos",
        time_source="sim",
        fps=observations.FPS,
    )
    for handle in policy_handles:
        # Mutate config directly (no fluent setters); set policy_num_actions from ONNX since muscle policies have no joint transmission.
        handle._config.motions.append(mimic_clip)
        handle._config.initial_qpos = initial_qpos
        handle._config.initial_qvel = initial_qvel
        output_shape = handle._config.model.graph.output[0].type.tensor_type.shape
        num_actions = output_shape.dim[1].dim_value
        if num_actions > 0:
            handle._config.policy_num_actions = num_actions

        # Rewrite the filesystem path to the bundled relative URL. Scene-scoped and
        # name-derived, so every checkpoint points at the one copy.
        clip_motion_url = f"{name2id(mimic_clip.name)}.npz"
        if (
            handle._config.terminations
            and "mimic_deviation" in handle._config.terminations
        ):
            handle._config.terminations["mimic_deviation"].params["clip_url"] = (
                clip_motion_url
            )

        # Wire TrackingCommand explicitly: myoMimicFullbody ships no MotionCommandCfg.
        if handle._config.commands is None:
            handle._config.commands = {}
        handle._config.commands["motion"] = CommandTermConfig(
            term_name="TrackingCommand",
        )

    return builder


def main() -> None:
    """Build and optionally launch the MuscleMimic Fullbody demo."""
    app = setup_builder().build()
    app.launch()


if __name__ == "__main__":
    main()
