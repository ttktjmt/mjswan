"""Go1 with a fox or a pug skin

The Go1 velocity tasks from ``main.py``, flat and rough, on their own, with a fox or a
pug (the Skin menu switches) drawn in the Go1's place, the Go1 itself undrawn. Each
part of the animal turns as the Go1 body it maps to does, and each leg is then bent so
its paw lands where the Go1's foot is (``retarget="hybrid"``); the fox's tail swings,
simulated by MuJoCo in a model of its own. Physics, collisions and the policy are the
Go1's, exactly as in ``main.py``.

    uv run python examples/demo/go1_avatar.py

VRM is a humanoid format, and openly licensed four-legged VRMs are rare, so the script
makes them from rigged glTF quadrupeds, adding what a skin needs: humanoid bones (the
front legs as arms, the hind legs as legs) and, on the fox, a tail spring. The fox is
the Khronos glTF sample Fox, downloaded; the pug is ``assets/pug.glb``, committed, as
its source has no stable URL (``assets/make_pug.py`` made it). Run it as a script (not
``python -m``): it imports this directory's ``main.py``.

Needs `pip install 'mjswan[hf,mjlab]'`.
"""

import json
import os
import struct
import sys
import tempfile
import urllib.error
from dataclasses import replace
from pathlib import Path

from main import HF_REPO, add_mjlab_task
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import INIT_STATE

import mjswan
from mjswan.source import hf, url

#: The scene list, in order: the same Go1 and fox on flat ground and on rough terrain.
TASKS = ("Mjlab-Velocity-Flat-Unitree-Go1", "Mjlab-Velocity-Rough-Unitree-Go1")

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
#: The trunk's height in that standing pose (mjlab's initial state, as are its angles):
#: the skins rest whenever the trunk is there. The model itself puts the trunk at
#: 0.445 m, which would sink them by about 0.1 m.
GO1_STANDING_HEIGHT = INIT_STATE.pos[2]
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
    root_height=GO1_STANDING_HEIGHT,
    hide_robot=True,
    retarget="hybrid",
)

PUG_GLB = Path(__file__).parent / "assets" / "pug.glb"
PUG_NOTICE = (
    "Pug: from Farm Animals Animated by Quaternius (CC0 1.0), "
    "https://quaternius.itch.io/lowpoly-animated-animals\n"
    "Made forward-kinematic and scaled by mjswan's examples/demo/assets/make_pug.py, "
    "turned into a VRM by its go1_avatar.py.\n"
)

#: VRM humanoid bone -> the Pug's bone. Its hips bone is mid-body, as the Go1's trunk
#: origin is; it has no toes.
PUG_HUMANOID = {
    "hips": "Body",
    "spine": "Shoulders",
    "neck": "Neck",
    "head": "Head",
    "leftUpperArm": "FrontUpLeg.L",
    "leftLowerArm": "FrontLowLeg.L",
    "leftHand": "FrontFoot.L",
    "rightUpperArm": "FrontUpLeg.R",
    "rightLowerArm": "FrontLowLeg.R",
    "rightHand": "FrontFoot.R",
    "leftUpperLeg": "BackUpLeg.L",
    "leftLowerLeg": "BackLowLeg.L",
    "leftFoot": "BackFoot.L",
    "rightUpperLeg": "BackUpLeg.R",
    "rightLowerLeg": "BackLowLeg.R",
    "rightFoot": "BackFoot.R",
}

PUG = mjswan.SkinConfig(
    bones={bone: GO1_BONES[bone] for bone in PUG_HUMANOID},
    rest_offsets={
        bone: offset
        for bone, offset in GO1_REST_OFFSETS.items()
        if bone in PUG_HUMANOID
    },
    offset=(0.0, 0.0, 0.0),
    root_height=GO1_STANDING_HEIGHT,
    hide_robot=True,
    # MuJoCo's lights, shared out, leave the beige darker than the source's; this lifts
    # it back, short of clipping.
    brightness=1.25,
    retarget="hybrid",
)


def read_glb(path: Path) -> tuple[dict, bytes]:
    """A .glb's JSON, and its binary chunk with the chunk's own header."""
    data = path.read_bytes()
    json_length = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + json_length]), data[20 + json_length :]


def write_glb(path: Path, gltf: dict, binary_chunk: bytes) -> Path:
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 20 + len(text) + len(binary_chunk))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + binary_chunk
    )
    return path


def vrm_meta(
    name: str, authors: list[str], notice: str, license_url: str, credit: bool
) -> dict:
    """A VRM 1.0 meta with every permission open: the VRM license may not narrow the
    model's own, given as its other license."""
    return {
        "name": name,
        "authors": authors,
        "copyrightInformation": notice.strip(),
        "licenseUrl": "https://vrm.dev/licenses/1.0/",
        "otherLicenseUrl": license_url,
        "avatarPermission": "everyone",
        "commercialUsage": "corporation",
        "creditNotation": "required" if credit else "unnecessary",
        "allowRedistribution": True,
        "modification": "allowModificationRedistribution",
        "allowExcessivelyViolentUsage": True,
        "allowExcessivelySexualUsage": True,
        "allowPoliticalOrReligiousUsage": True,
        "allowAntisocialOrHateUsage": True,
    }


def fox_vrm() -> Path:
    """The Khronos sample Fox as a VRM 1.0 file: scaled, its bones named, its tail a spring."""
    glb = url.fetch_file(FOX_URL)
    gltf, binary_chunk = read_glb(glb)
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
                # Credit required: the rig is CC BY 4.0.
                "meta": {
                    **vrm_meta(
                        "Fox",
                        ["PixelMannen", "tomkranis", "AsoboStudio", "scurest"],
                        FOX_NOTICE,
                        "https://creativecommons.org/licenses/by/4.0/",
                        credit=True,
                    ),
                    "references": [FOX_URL],
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

    return write_glb(glb.with_suffix(".vrm"), gltf, binary_chunk)


def pug_vrm() -> Path:
    """``assets/pug.glb`` as a VRM 1.0 file, its bones named."""
    gltf, binary_chunk = read_glb(PUG_GLB)
    nodes = {node["name"]: i for i, node in enumerate(gltf["nodes"])}
    # Toon materials: the viewer divides their colours by the scene's summed light, which
    # would otherwise wash the pug out to white. The shaded side is 60% of the lit one,
    # the two blending softly, as in the source's diffuse shading.
    for material in gltf["materials"]:
        lit = material["pbrMetallicRoughness"]["baseColorFactor"]
        material["extensions"] = {
            "VRMC_materials_mtoon": {
                "specVersion": "1.0",
                "shadeColorFactor": [c * 0.6 for c in lit[:3]],
                "shadingToonyFactor": 0.5,
            }
        }
    gltf["extensionsUsed"] = [
        *gltf.get("extensionsUsed", []),
        "VRMC_vrm",
        "VRMC_materials_mtoon",
    ]
    gltf.setdefault("extensions", {})["VRMC_vrm"] = {
        "specVersion": "1.0",
        "meta": vrm_meta(
            "Pug",
            ["Quaternius"],
            PUG_NOTICE,
            "https://creativecommons.org/publicdomain/zero/1.0/",
            credit=False,
        ),
        "humanoid": {
            "humanBones": {bone: {"node": nodes[n]} for bone, n in PUG_HUMANOID.items()}
        },
    }
    out = Path(tempfile.mkdtemp(prefix="mjswan-")) / "pug.vrm"
    return write_glb(out, gltf, binary_chunk)


def setup_builder() -> mjswan.Builder:
    """The Go1 velocity scenes of ``main.py``, with the pug, and the fox when it downloads."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="Go1 Avatar")
    # Name -> skin and its attribution: (component, license, notice).
    skins = {}
    try:
        skins["Fox"] = (
            replace(FOX, source=str(fox_vrm())),
            ("fox", "CC-BY-4.0", FOX_NOTICE),
        )
    except urllib.error.URLError as exc:
        print(f"Running the Go1 without the fox: {exc}", file=sys.stderr)
    skins["Pug"] = replace(PUG, source=str(pug_vrm())), ("pug", "CC0-1.0", PUG_NOTICE)
    repo_onnx = hf.list_repo_onnx(HF_REPO)
    for task_id in TASKS:
        scene = add_mjlab_task(project, task_id, repo_onnx)
        for name, (skin, (component, license_id, notice)) in skins.items():
            scene.add_skin(name, skin)
            scene.add_attribution(component, license=license_id, notice=notice)
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
