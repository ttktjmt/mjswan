# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["marimo"]
# ///
"""Train MuJoCo Playground's SO-101 PickCube for mjswan, on molab.

A marimo notebook around `so101_pick_cube.py`: each step runs when its button is pressed,
in a subprocess, so a failed step leaves the notebook running.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Train MuJoCo Playground's SO-101 PickCube for mjswan

    MuJoCo Playground 1.0.1's SO-101 PickCube, trained with Playground's own environment
    and PPO settings, then exported as the ONNX policy an mjswan scene runs. The task is
    `state` (`SO101PickCube`: joints, gripper and cube positions) or `vision`
    (`SO101PickCubeVision`, the task Playground's sim-to-real demo deploys: the wrist
    camera's image instead of the cube's position).

    Turn on the GPU from the notebook specs button first. molab can end a GPU session
    early, so give a Hugging Face repository: every checkpoint is kept there too, and the
    next session resumes from it.
    """)


@app.cell
def _(mo):
    task = mo.ui.dropdown(options=["state", "vision"], value="state", label="Task")
    out = mo.ui.text(value="so101_pick_cube", label="Output folder")
    iterations = mo.ui.number(
        start=0, stop=10000, step=10, value=0, label="Iterations (0: Playground's own)"
    )
    hf_repo = mo.ui.text(
        placeholder="user/so101-pick-cube", label="Hugging Face repository"
    )
    hf_token = mo.ui.text(kind="password", label="Hugging Face token (write)")
    branch = mo.ui.text(value="main", label="mjswan branch to fetch the script from")
    mo.vstack([task, out, iterations, hf_repo, hf_token, branch])
    return branch, hf_repo, hf_token, iterations, out, task


@app.cell
def _(branch, hf_repo, hf_token, task):
    import os
    import subprocess
    import sys
    import urllib.request
    from pathlib import Path

    script = Path("so101_pick_cube.py")

    def run(*args: str) -> None:
        """One step of the script, its output streamed into this notebook."""
        if not script.exists():
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/ttktjmt/mjswan/"
                f"{branch.value}/examples/train/so101_pick_cube.py",
                script,
            )
        env = dict(os.environ)
        if hf_token.value:
            env["HF_TOKEN"] = hf_token.value
        hub = (
            ["--hf-repo", hf_repo.value] if hf_repo.value and args[0] != "setup" else []
        )
        chosen = ["--task", task.value] if args[0] in ("train", "export") else []
        process = subprocess.Popen(
            [sys.executable, str(script), *args, *hub, *chosen],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait() != 0:
            raise RuntimeError(f"`{' '.join(args)}` failed; see the output above.")

    return (run,)


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Install Playground")
    train_button = mo.ui.run_button(label="2. Train")
    export_button = mo.ui.run_button(label="3. Export and upload")
    mo.hstack([setup_button, train_button, export_button], justify="start")
    return export_button, setup_button, train_button


@app.cell
def _(mo, run, setup_button):
    mo.stop(not setup_button.value)
    run("setup")


@app.cell
def _(iterations, mo, out, run, train_button):
    mo.stop(not train_button.value)
    count = ["--iterations", str(iterations.value)] if iterations.value else []
    run("train", "--out", out.value, *count)


@app.cell
def _(export_button, mo, out, run):
    mo.stop(not export_button.value)
    run("export", "--out", out.value)
    run("upload", "--out", out.value)


if __name__ == "__main__":
    app.run()
