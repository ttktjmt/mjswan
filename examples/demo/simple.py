"""Simple mjswan Demo

The shortest thing that still walks: one mjlab task, one trained policy, one browser
window. This is what `mjswan demo` runs, so it is also the first mjswan most people see.

Everything comes from somewhere else — the model from mjlab, the policy from the
Hugging Face Hub — so there is nothing to download by hand and nothing to configure.
`examples/demo/main.py` is the same idea at full size.

Needs `pip install 'mjswan[hf,mjlab]'`.
"""

import os
import re

import mjswan
from mjswan.source import hf

HF_REPO = "ttktjmt/mjswan"
TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


def setup_builder() -> mjswan.Builder:
    """Set up and return the builder, configured but not built.

    Returns:
        Configured Builder instance ready to be built.
    """
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="mjswan Demo")

    # The task carries the model and the whole MDP — observations, commands, actions,
    # terminations — so nothing about them is repeated here.
    scene = project.add_scene_mjlab(TASK_ID)
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=3.0,
            elevation=-20.0,
            azimuth=0.0,
            origin_type=mjswan.ViewerConfig.OriginType.ASSET_BODY,
            body_name="torso_link",
        )
    )

    # The last checkpoint of the mirrored run. `main.py` adds all of them so you can
    # watch training progress; one is enough to see the robot walk.
    prefix = f"checkpoints/{TASK_ID.lower()}/"
    checkpoints = sorted(
        (name for name in hf.list_repo_onnx(HF_REPO) if name.startswith(prefix)),
        key=lambda name: int(re.search(r"_(\d+)\.onnx$", name)[1]),
    )
    scene.add_policy_hf(HF_REPO, filename=checkpoints[-1])

    return builder


def main():
    """Main entry point for the simple demo.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
