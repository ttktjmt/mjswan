"""Go1 with a fox skin

The Go1 velocity task from ``main.py`` on its own, with a fox drawn in the Go1's place,
posed from it every frame, the Go1 itself undrawn; the fox's tail swings, simulated by
MuJoCo in a model of its own. Physics, collisions and the policy are the Go1's, exactly
as in ``main.py``.

    uv run python examples/demo/go1_avatar.py

VRM is a humanoid format, and openly licensed four-legged VRMs are rare, so the script
makes one: it downloads the Khronos glTF sample Fox, a rigged quadruped, and adds what
a skin needs, its humanoid bones (the front legs as arms, the hind legs as legs) and a
spring on its tail. Run it as a script (not ``python -m``): it imports this directory's
``main.py``.

Needs `pip install 'mjswan[hf,mjlab]'`.
"""

import json
import os
import struct
import sys
import urllib.error
from dataclasses import replace
from pathlib import Path

from main import HF_REPO, add_mjlab_task

import mjswan
from mjswan.source import hf, url

TASK = "Mjlab-Velocity-Flat-Unitree-Go1"

# The Fox's mesh is CC0, by PixelMannen; its rig and animation are CC BY 4.0, by
# @tomkranis, which asks for the credit below wherever it is shown.
FOX_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    "15cd02d304e54cb42a4db4eefe719f76190f4b68/Models/Fox/glTF-Binary/Fox.glb"
)
FOX_NOTICE = (
    "Fox: low-poly model by PixelMannen (CC0 1.0), rigged and animated by @tomkranis "
    "(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), glTF conversion by "
    "@AsoboStudio and @scurest. "
    "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox\n"
    "Turned into a VRM, scaled and given a tail spring by mjswan's go1_avatar.py.\n"
)
#: Metres per unit of the glTF: its shoulders and hips then sit 0.37 m apart, as the
#: Go1's front and rear hips do.
FOX_SCALE = 0.0078

#: VRM humanoid bone -> the Fox's node. Its front legs are already named as arms.
FOX_HUMANOID = {
    "hips": "b_Hip_01",
    "spine": "b_Spine01_02",
    "chest": "b_Spine02_03",
    "neck": "b_Neck_04",
    "head": "b_Head_05",
    "leftUpperArm": "b_LeftUpperArm_09",
    "leftLowerArm": "b_LeftForeArm_010",
    "leftHand": "b_LeftHand_011",
    "rightUpperArm": "b_RightUpperArm_06",
    "rightLowerArm": "b_RightForeArm_07",
    "rightHand": "b_RightHand_08",
    "leftUpperLeg": "b_LeftLeg01_015",
    "leftLowerLeg": "b_LeftLeg02_016",
    "leftFoot": "b_LeftFoot01_017",
    "leftToes": "b_LeftFoot02_018",
    "rightUpperLeg": "b_RightLeg01_019",
    "rightLowerLeg": "b_RightLeg02_020",
    "rightFoot": "b_RightFoot01_021",
    "rightToes": "b_RightFoot02_022",
}

#: The Go1 body each VRM bone follows: front legs for arms, hind legs for legs. The Go1
#: has no neck, so the fox's spine, neck and head ride its trunk.
GO1_BONES = {
    **{bone: "trunk" for bone in ("hips", "spine", "chest", "neck", "head")},
    **{
        f"{side}{bone}": f"{leg}_{body}"
        for side, front, rear in (("left", "FL", "RL"), ("right", "FR", "RR"))
        for bone, leg, body in (
            ("UpperArm", front, "thigh"),
            ("LowerArm", front, "calf"),
            ("Hand", front, "calf"),
            ("UpperLeg", rear, "thigh"),
            ("LowerLeg", rear, "calf"),
            ("Foot", rear, "calf"),
            ("Toes", rear, "calf"),
        )
    },
}

#: The inverse of the Go1's standing pose (thighs 0.9 rad, calves -1.8, hips -0.1 on the
#: left and 0.1 on the right), in degrees about the avatar's x, then y, then z: whenever
#: the Go1 stands, the fox stands as it rests, and its legs swing by what the Go1's do
#: from there. A paw keeps its angle to the leg above it.
STANDING = {
    "left": {"thigh": (-51.43, 4.49, 3.57), "calf": (51.43, -4.49, 3.57)},
    "right": {"thigh": (-51.43, -4.49, -3.57), "calf": (51.43, 4.49, -3.57)},
}
GO1_REST_OFFSETS = {
    bone: STANDING["left" if bone.startswith("left") else "right"][body.split("_")[1]]
    for bone, body in GO1_BONES.items()
    if body != "trunk"
}

FOX = mjswan.SkinConfig(
    bones=GO1_BONES,
    rest_offsets=GO1_REST_OFFSETS,
    offset=(0.0, 0.0, 0.0),
    # The Go1's trunk origin is mid-body; the fox's hips bone is its pelvis, behind.
    hips_offset=(-0.16, 0.0, 0.0),
    hide_robot=True,
)


def fox_vrm() -> Path:
    """The Khronos sample Fox as a VRM 1.0 file: scaled, its bones named, its tail a spring."""
    glb = url.fetch_file(FOX_URL)
    data = glb.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20 : 20 + json_length])
    binary_chunk = data[20 + json_length :]
    nodes = {node["name"]: i for i, node in enumerate(gltf["nodes"])}

    gltf["nodes"][nodes["root"]]["scale"] = [FOX_SCALE] * 3
    # The tail's last bone has no child to point along: give it a tip, so it swings too.
    gltf["nodes"].append({"name": "b_Tail04_tip", "translation": [20.0, 0.0, 0.0]})
    gltf["nodes"][nodes["b_Tail03_014"]]["children"] = [len(gltf["nodes"]) - 1]
    tail = [nodes[n] for n in ("b_Tail01_012", "b_Tail02_013", "b_Tail03_014")]

    gltf["extensionsUsed"] = [
        *gltf.get("extensionsUsed", []),
        "VRMC_vrm",
        "VRMC_springBone",
    ]
    gltf.setdefault("extensions", {}).update(
        {
            "VRMC_vrm": {
                "specVersion": "1.0",
                # Every permission open and credit required: the VRM license may not
                # narrow CC BY 4.0, which the rig is under.
                "meta": {
                    "name": "Fox",
                    "authors": ["PixelMannen", "tomkranis", "AsoboStudio", "scurest"],
                    "copyrightInformation": FOX_NOTICE.strip(),
                    "references": [FOX_URL],
                    "licenseUrl": "https://vrm.dev/licenses/1.0/",
                    "otherLicenseUrl": "https://creativecommons.org/licenses/by/4.0/",
                    "avatarPermission": "everyone",
                    "commercialUsage": "corporation",
                    "creditNotation": "required",
                    "allowRedistribution": True,
                    "modification": "allowModificationRedistribution",
                    "allowExcessivelyViolentUsage": True,
                    "allowExcessivelySexualUsage": True,
                    "allowPoliticalOrReligiousUsage": True,
                    "allowAntisocialOrHateUsage": True,
                },
                "humanoid": {
                    "humanBones": {
                        bone: {"node": nodes[node]}
                        for bone, node in FOX_HUMANOID.items()
                    }
                },
            },
            "VRMC_springBone": {
                "specVersion": "1.0",
                # The pelvis and the hind thighs, which the tail must not swing through.
                # Offsets are in the node's own units; radii in metres.
                "colliders": [
                    {"node": nodes["b_Hip_01"], "shape": {"sphere": {"radius": 0.06}}},
                    *(
                        {
                            "node": nodes[thigh],
                            "shape": {
                                "capsule": {"tail": [18.9, 0, 0], "radius": 0.035}
                            },
                        }
                        for thigh in ("b_LeftLeg01_015", "b_RightLeg01_019")
                    ),
                ],
                "colliderGroups": [{"name": "body", "colliders": [0, 1, 2]}],
                "springs": [
                    {
                        "name": "tail",
                        "joints": [
                            *(
                                {
                                    "node": node,
                                    "hitRadius": 0.03,
                                    "stiffness": 1.5,
                                    "gravityPower": 0.1,
                                    "gravityDir": [0, -1, 0],
                                    "dragForce": 0.3,
                                }
                                for node in tail
                            ),
                            {"node": len(gltf["nodes"]) - 1},
                        ],
                        "colliderGroups": [0],
                    }
                ],
            },
        }
    )

    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    out = glb.with_suffix(".vrm")
    out.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 20 + len(text) + len(binary_chunk))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + binary_chunk
    )
    return out


def setup_builder() -> mjswan.Builder:
    """The Go1 velocity scene of ``main.py``, with the fox when it downloads."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="Go1 Avatar")
    scene = add_mjlab_task(project, TASK, hf.list_repo_onnx(HF_REPO))
    try:
        fox = replace(FOX, source=str(fox_vrm()))
    except urllib.error.URLError as exc:
        print(f"Running the Go1 without the fox: {exc}", file=sys.stderr)
        return builder
    scene.add_skin("Fox", fox)
    scene.add_attribution("fox", license="CC-BY-4.0", notice=FOX_NOTICE)
    return builder


def main() -> None:
    """Build and launch.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    # Its own output directory, so main.py's dist/ is left alone.
    app = setup_builder().build("dist_go1_avatar")
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
