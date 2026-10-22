"""Spinkick with a VRM avatar

The G1 Spinkick scene from ``main.py`` on its own, with VRM avatars as skins: one is
drawn in the G1's place, posed from it every frame, the G1 itself undrawn; its arms and
legs are bent so its hands and feet land where the G1's are (``retarget="hybrid"``).
The viewer's Skin menu picks which; cleared, it shows the G1 again. Physics, collisions
and the policy are the G1's, exactly as in ``main.py``; only the avatar's hair and
clothes are simulated, by MuJoCo, in a model of their own.

    uv run python examples/demo/spinkick_avatar.py

Each skin's file is named in its config below, and the build copies it into the app,
as a splat's ``.spz`` is. A skin whose file is missing is left out. Run it as a script
(not ``python -m``): it imports this directory's ``main.py``.

Needs `pip install 'mjswan[hf,mjlab]'`.
"""

import os
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path

from main import HF_REPO, add_tracking_scene

import mjswan
from mjswan.source import hf

HERE = Path(__file__).resolve().parent

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

#: At its zero pose the G1 hangs its upper arms and bends its elbows square, forearms
#: forward; a VRM rests in a T-pose. Degrees about the avatar's x, then y, then z (y up,
#: facing +z, its left hand toward +x), palms turned in.
G1_REST_OFFSETS = {
    "leftUpperArm": (0, 0, -90),
    "leftLowerArm": (0, -90, -90),
    "leftHand": (0, -90, -90),
    "rightUpperArm": (0, 0, 90),
    "rightLowerArm": (0, 90, 90),
    "rightHand": (0, 90, 90),
}

# AvatarSample_A, VRoid's official sample model. Download it yourself from VRoid Hub,
# which asks you to agree to the VRoid Hub Terms of Use, and save it beside this script
# as 8590256991748008892.vrm:
#   https://hub.vroid.com/en/characters/2843975675147313744/models/5644550979324015604
# It is not CC0. Its VRoidPreset terms (https://vroid.pixiv.help/hc/en-us/articles/4402394424089)
# allow free use, for profit or not, without credit, provided its embedded license
# metadata stays as distributed (never relabeled as CC0), it is shown only as a fixed
# avatar (no appearance editing), it is never behind a paid feature, and nothing implies
# pixiv or VRoid endorse mjswan. Never commit the file (`*.vrm` is git-ignored).
AVATAR_SAMPLE_A = mjswan.SkinConfig(
    source=str(HERE / "8590256991748008892.vrm"),
    bones=G1_BONES,
    rest_offsets=G1_REST_OFFSETS,
    retarget="hybrid",
)

# CC0 (VRM 0.x): Sendagaya Shino, saved beside this script as SendagayaShino.vrm:
#   https://hub.vroid.com/characters/5860098757548846785/models/6567311261748429976
# VRM normalizes every avatar's rest pose to a T-pose, so the G1 tables above fit it as
# they are: only the file differs.
SENDAGAYA_SHINO = replace(AVATAR_SAMPLE_A, source=str(HERE / "SendagayaShino.vrm"))

#: The viewer's Skin menu, in order; the first one found is shown on load.
SKINS = {
    "AvatarSample_A": AVATAR_SAMPLE_A,
    "Sendagaya Shino": SENDAGAYA_SHINO,
}

NO_SKINS = (
    "Running Spinkick without skins. To add AvatarSample_A, download it from VRoid Hub "
    "(you agree to the VRoid Hub Terms of Use there):\n"
    "  https://hub.vroid.com/en/characters/2843975675147313744/models/5644550979324015604\n"
    f"and save it as {AVATAR_SAMPLE_A.source}"
)


def setup_builder() -> mjswan.Builder:
    """The Spinkick scene of ``main.py``, with every skin whose file is there."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="Spinkick Avatar")
    scene = add_tracking_scene(project, hf.list_repo_onnx(HF_REPO))

    added = 0
    for name, config in SKINS.items():
        try:
            scene.add_skin(name, config)
        except (FileNotFoundError, urllib.error.URLError) as exc:
            print(f"Skipping skin {name!r}: {exc}", file=sys.stderr)
            continue
        added += 1
    if not added:
        print(NO_SKINS, file=sys.stderr)
    return builder


def main() -> None:
    """Build and launch.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    # Its own output directory, so main.py's dist/ is left alone.
    app = setup_builder().build("dist_spinkick_avatar")
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
