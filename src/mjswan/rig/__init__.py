"""A skin's rig: its bones named (``humanoid``), measured against the robot (``pose``),
and drafted for an author (``guess``). Its file is read by ``document.skin``."""

from .humanoid import (
    HUMANOID_LIMBS,
    HUMANOID_PRESETS,
    VRM_HUMAN_BONES,
    VRM_PARENTS,
    bone_ancestors,
    resolve_humanoid,
)

__all__ = [
    "HUMANOID_LIMBS",
    "HUMANOID_PRESETS",
    "VRM_HUMAN_BONES",
    "VRM_PARENTS",
    "bone_ancestors",
    "resolve_humanoid",
]
