"""Gaussian Splat backgrounds.

One concept: standing a MuJoCo model inside a photographed place. A splat is a capture
of somewhere real, so the interesting part is not loading it — it is the six numbers
that line the capture up with the model, which no file can tell you.

`control=True` puts live sliders for those six numbers in the control panel, which is
how you find them in the first place; drop it once they are dialled in. Call
`add_splat_hf` more than once and the viewer shows a selector, fetching only the capture
being displayed.

The model and the capture come from the Hugging Face Hub, so this downloads about 28 MB
the first time and nothing after — `huggingface_hub` caches under `~/.cache/huggingface`.

Needs `pip install 'mjswan[hf]'`. No policy, so no mjlab and no torch.

Run with:
    uv run python examples/tutorial/splat.py
"""

import os

import mjswan

HF_REPO = "ttktjmt/mjswan"


def setup_builder() -> mjswan.Builder:
    builder = mjswan.Builder()
    project = builder.add_project(name="Splat Tutorial")

    # A MuJoCo model is an MJCF plus the meshes it names, so the Hub path is a
    # directory: `add_scene_hf` brings the whole tree down and compiles it where it
    # lands, which is also how the `LICENSE` beside it reaches the build (ADR 0007).
    scene = project.add_scene_hf(HF_REPO, "scenes/unitree_g1/scene.xml", name="G1")
    scene.set_viewer(
        mjswan.ViewerConfig(
            lookat=(0.0, 0.0, 0.7),
            distance=4.3,
            elevation=-33.0,
            azimuth=-34.0,
        )
    )

    # `scale` is the capture's metric scale factor and `z_offset` puts its ground plane
    # at z=0; both usually come from the capture's own metadata. `yaw` is the one that
    # is always eyeballed — it turns the place around the robot.
    scene.add_splat_hf(
        HF_REPO,
        "splats/street.spz",
        name="Street",
        scale=3.275,
        z_offset=0.708,
        yaw=40,
        control=True,
    )

    return builder


def main() -> None:
    """Environment variables:
    MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") == "1":
        return
    app.launch()


if __name__ == "__main__":
    main()
