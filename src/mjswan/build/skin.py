"""A skin's manifest entry: what the browser needs to pose it, every name already a
node index and every measurement already made, so the engine reads no glTF names."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..skin import ResolvedSkin


def skin_entry(skin: ResolvedSkin) -> dict[str, Any]:
    """One entry of a scene's ``skins``; a key is left out when it holds the default."""
    config = skin.config
    return {
        "id": config.id,
        "name": config.name,
        "format": skin.file.format,
        "path": skin.path,
        "bones": dict(config.bones),
        **({"nodes": skin.nodes} if skin.nodes else {}),
        **({"limbs": [list(limb) for limb in skin.limbs]} if skin.limbs else {}),
        **({"reparent": [list(m) for m in skin.reparent]} if skin.reparent else {}),
        **({"scale": config.scale} if config.scale != 1 else {}),
        **({"facing": config.facing} if config.facing != "+z" else {}),
        **({"springs": skin.springs} if skin.springs else {}),
        **({"material": {"toon": asdict(config.material)}} if config.material else {}),
        **({"rest_offsets": skin.rest_offsets} if skin.rest_offsets else {}),
        **({"root_height": skin.root_height} if skin.root_height else {}),
        **({"height_ratio": config.height_ratio} if config.height_ratio else {}),
        **({"offset": list(config.offset)} if any(config.offset) else {}),
        **({"hips_offset": list(skin.hips_offset)} if any(skin.hips_offset) else {}),
        **({} if config.hide_robot else {"hide_robot": False}),
        **({"brightness": config.brightness} if config.brightness != 1.0 else {}),
        **({"mujoco_springs": True} if config.mujoco_springs else {}),
        **({"retarget": config.retarget} if config.retarget != "hybrid" else {}),
    }


__all__ = ["skin_entry"]
