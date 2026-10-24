"""VRM's humanoid bones, and a glTF's nodes named as them: by a table or a preset."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..document.skin import SkinFile

_FINGERS = ("Index", "Middle", "Ring", "Little")


def _vrm_parents() -> dict[str, str | None]:
    parents: dict[str, str | None] = {
        "hips": None,
        "spine": "hips",
        "chest": "spine",
        "upperChest": "chest",
        "neck": "upperChest",
        "head": "neck",
        "leftEye": "head",
        "rightEye": "head",
        "jaw": "head",
    }
    for side in ("left", "right"):
        hand = f"{side}Hand"
        chains = (
            ("hips", "UpperLeg", "LowerLeg", "Foot", "Toes"),
            ("upperChest", "Shoulder", "UpperArm", "LowerArm", "Hand"),
            (hand, "ThumbMetacarpal", "ThumbProximal", "ThumbDistal"),
            *(
                (hand, f"{f}Proximal", f"{f}Intermediate", f"{f}Distal")
                for f in _FINGERS
            ),
        )
        for parent, *bones in chains:
            for bone in bones:
                parents[f"{side}{bone}"] = parent
                parent = f"{side}{bone}"
    return parents


#: Every humanoid bone three-vrm addresses (VRM 1.0's names) -> the bone it hangs from,
#: as VRM nests them (three-vrm's ``VRMHumanBoneParentMap``).
VRM_PARENTS: dict[str, str | None] = _vrm_parents()

VRM_HUMAN_BONES: frozenset[str] = frozenset(VRM_PARENTS)

#: The arms and legs ``retarget="hybrid"`` bends on a humanoid: (upper, lower, end).
HUMANOID_LIMBS: tuple[tuple[str, str, str], ...] = tuple(
    (f"{side}{upper}", f"{side}{lower}", f"{side}{end}")
    for side in ("left", "right")
    for upper, lower, end in (
        ("UpperLeg", "LowerLeg", "Foot"),
        ("UpperArm", "LowerArm", "Hand"),
    )
)

#: VRM humanoid bone -> node name in a Mixamo rig, matched with or without Mixamo's
#: ``mixamorig:`` prefix, which some exporters drop.
HUMANOID_PRESETS: dict[str, dict[str, str]] = {
    "mixamo": {
        "hips": "Hips",
        "spine": "Spine",
        "chest": "Spine1",
        "upperChest": "Spine2",
        "neck": "Neck",
        "head": "Head",
        **{
            f"{side}{bone}": f"{side.capitalize()}{node}"
            for side in ("left", "right")
            for bone, node in (
                ("Shoulder", "Shoulder"),
                ("UpperArm", "Arm"),
                ("LowerArm", "ForeArm"),
                ("Hand", "Hand"),
                ("UpperLeg", "UpLeg"),
                ("LowerLeg", "Leg"),
                ("Foot", "Foot"),
                ("Toes", "ToeBase"),
                ("ThumbMetacarpal", "HandThumb1"),
                ("ThumbProximal", "HandThumb2"),
                ("ThumbDistal", "HandThumb3"),
                *(
                    (f"{finger}{joint}", f"Hand{mixamo}{n}")
                    for finger, mixamo in zip(
                        _FINGERS, ("Index", "Middle", "Ring", "Pinky")
                    )
                    for n, joint in enumerate(("Proximal", "Intermediate", "Distal"), 1)
                ),
            )
        },
    },
}

_MIXAMO_PREFIX = re.compile(r"^mixamorig\d*:")


def resolve_humanoid(
    humanoid: Mapping[str, str] | str, file: SkinFile
) -> dict[str, int]:
    """VRM bone -> node index for a glTF. A preset maps the bones whose nodes the file
    has; a table of the author's own must find every node it names."""
    if isinstance(humanoid, str):
        if humanoid not in HUMANOID_PRESETS:
            raise ValueError(
                f"No humanoid preset {humanoid!r}; there are {sorted(HUMANOID_PRESETS)}."
            )
        bare = {_MIXAMO_PREFIX.sub("", name): name for name in file.names}
        mapping = {
            bone: bare[node]
            for bone, node in HUMANOID_PRESETS[humanoid].items()
            if node in bare
        }
    else:
        unknown = sorted(set(humanoid) - VRM_HUMAN_BONES)
        if unknown:
            raise ValueError(
                f"humanoid names {unknown}, which are not VRM humanoid bones."
            )
        mapping = dict(humanoid)
    if "hips" not in mapping:
        raise ValueError(
            f"{file.path.name}: humanoid maps no hips, which carry the avatar."
        )
    return {bone: file.node(node) for bone, node in mapping.items()}


def bone_ancestors(bone: str) -> list[str]:
    """*bone*'s VRM ancestors, nearest first."""
    out = []
    parent = VRM_PARENTS.get(bone)
    while parent is not None:
        out.append(parent)
        parent = VRM_PARENTS[parent]
    return out


__all__ = [
    "HUMANOID_LIMBS",
    "HUMANOID_PRESETS",
    "VRM_HUMAN_BONES",
    "VRM_PARENTS",
    "bone_ancestors",
    "resolve_humanoid",
]
