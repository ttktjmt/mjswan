"""MuJoCo replicate model gallery

Loads every model from MuJoCo's official `<replicate>` example gallery
(scene.xml is a shared background scene, not a standalone model, so it's
skipped):
https://github.com/google-deepmind/mujoco/tree/main/model/replicate
"""

import os
import tempfile
import urllib.request

import mujoco

import mjswan

REPO_RAW = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco/main/model/replicate"
)

MODEL_NAMES = [
    "bowl",
    "bunnies",
    "container",
    "cylinder",
    "helix",
    # "leaves",  # too heavy
    "newton_cradle",
    "particle",
    "particle_free",
    "particle_free2d",
    "stonehenge",
    "tendon",
]

# Files pulled in via <include>/<mesh>/<texture> by the models above. They
# must sit alongside the model files on disk for MuJoCo to resolve them.
DEPENDENCY_FILES = ["scene.xml", "container.xml", "bunny.obj", "asset/marble.png"]


def _download(path, dest_dir):
    dest = os.path.join(dest_dir, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with (
        urllib.request.urlopen(f"{REPO_RAW}/{path}") as response,
        open(dest, "wb") as f,
    ):
        f.write(response.read())
    return dest


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="MuJoCo Replicate Gallery")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for path in DEPENDENCY_FILES:
            _download(path, tmp_dir)
        for name in MODEL_NAMES:
            model_path = _download(f"{name}.xml", tmp_dir)
            spec = mujoco.MjSpec.from_file(model_path)
            project.add_scene(spec=spec, name=name)

        # build() re-reads mesh files (e.g. bunny.obj) from disk, so it must
        # run before tmp_dir is cleaned up.
        app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
