"""G1 spinkick motion-tracking demo.

This example exercises mjswan's mjlab tracking playback path end-to-end:
- MuJoCo scene from mjlab's play config
- policy checkpoints exported from a W&B training run
- reference motion auto-imported from the run's motion artifact
"""

from __future__ import annotations

import os
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    __package__ = "examples.mjlab.g1_spinkick"

import mjlab.tasks  # noqa: F401 - populates the mjlab task registry

# `examples.mjlab.defaults.commands` registers the traced reset graph for
# `MotionCommandCfg` — the reference-state-initialization jitter (ADR 0005 §3).
# Without it the command still builds, bound to the native `TrackingCommand`, but
# with no jitter graph, so every episode would start from the unjittered reference
# frame where mjlab's play config asks for `joint_position_range=(-0.1, 0.1)`.
import examples.mjlab.defaults.commands  # noqa: F401
import mjswan

# The terminations need no registration: they trace straight from mjlab's own functions.


def setup_builder() -> mjswan.Builder:
    """Create the builder for the G1 spinkick tracking demo."""
    example_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    run_path = "ttktjmt-org/mjlab/mayq0rtd"
    task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

    builder = mjswan.Builder()

    project = builder.add_project(name="mjlab Spinkick")
    scene = project.add_scene_mjlab(task_id)

    # The run supplies everything: the clip (mjlab registers `motion_file=""`, and the
    # tracing env is built from the bundled copy at build time), and every term set via
    # the scene's env config.
    #
    # `in_keys` is the exception: the table is positional, so this run's two-input
    # export cannot be read off the network's own input names (ADR 0006 §5).
    scene.add_policy_wandb(run_path, in_keys=["actor", "time_step"])

    return builder


def main() -> None:
    """Build and optionally launch the G1 spinkick demo."""
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
