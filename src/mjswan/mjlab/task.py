"""What a scene takes from an mjlab task's env config: the control rate, the entity
specs before the scene flattens them, the terrain's spawn patches, and the viewer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import mujoco

from ..viewer import ViewerConfig


def env_cfg_control_dt(env_cfg: Any) -> float | None:
    """An mjlab env config's seconds per control step, or ``None`` if it lacks one.

    Mirrors ``ManagerBasedRlEnv.step_dt`` (``sim.mujoco.timestep * decimation``) so the
    rate can be read off a config without paying to construct the env.
    """
    try:
        return float(env_cfg.sim.mujoco.timestep) * int(env_cfg.decimation)
    except (AttributeError, TypeError, ValueError):
        return None


def extract_terrain_data(scene: Any) -> dict[str, Any] | None:
    """Spawn positions from an mjlab Scene, for the browser's reset events.

    The terrain's named ``flat_patches`` when it samples them, else one
    ``terrain_origins`` point per sub-terrain tile.
    """
    terrain = getattr(scene, "terrain", None)
    if terrain is None:
        return None

    flat_patches = getattr(terrain, "flat_patches", None)
    if flat_patches:
        serialized: dict[str, list[list[float]]] = {}
        for name, patches in flat_patches.items():
            try:
                arr = patches.cpu().numpy()
                rows, cols, n, _ = arr.shape
                positions = arr.reshape(rows * cols * n, 3).tolist()
                serialized[name] = positions
            except Exception:
                pass
        if serialized:
            return {"flat_patches": serialized}

    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is not None:
        try:
            arr = terrain_origins.cpu().numpy()
            num_rows, num_cols, _ = arr.shape
            positions = arr.reshape(num_rows * num_cols, 3).tolist()
            return {"flat_patches": {"spawn": positions}}
        except Exception:
            pass

    return None


def entity_specs(scene_cfg: Any) -> Iterator[mujoco.MjSpec]:
    """The terrain's and each entity's own spec, before the scene flattens them."""
    spec_cfgs = [getattr(scene_cfg, "terrain", None)]
    entities = getattr(scene_cfg, "entities", {})
    if isinstance(entities, dict):
        spec_cfgs.extend(entities.values())

    for cfg in spec_cfgs:
        spec_fn = getattr(cfg, "spec_fn", None)
        if not callable(spec_fn):
            continue
        spec = spec_fn()
        if isinstance(spec, mujoco.MjSpec):
            yield spec


def adapt_viewer_config(config: Any | None) -> ViewerConfig | None:
    """Convert mjlab's ``ViewerConfig`` dataclass to mjswan's equivalent."""
    if config is None:
        return None

    defaults = ViewerConfig()
    entity_name = getattr(config, "entity_name", None)
    body_name = getattr(config, "body_name", None)
    if entity_name is None and body_name is not None:
        entity_name = "robot"
    origin_type_name = getattr(getattr(config, "origin_type", None), "name", None)
    if isinstance(origin_type_name, str):
        origin_type = getattr(ViewerConfig.OriginType, origin_type_name, None)
    else:
        origin_type = None

    return ViewerConfig(
        lookat=tuple(getattr(config, "lookat", (0.0, 0.0, 0.0))),
        distance=float(getattr(config, "distance", 4.0)),
        fovy=getattr(config, "fovy", None),
        elevation=float(getattr(config, "elevation", -30.0)),
        azimuth=float(getattr(config, "azimuth", 45.0)),
        origin_type=origin_type or defaults.origin_type,
        entity_name=entity_name,
        body_name=body_name,
        env_idx=int(getattr(config, "env_idx", 0)),
        max_extra_envs=int(getattr(config, "max_extra_envs", 2)),
        enable_reflections=bool(getattr(config, "enable_reflections", True)),
        enable_shadows=bool(getattr(config, "enable_shadows", True)),
        height=int(getattr(config, "height", 240)),
        width=int(getattr(config, "width", 320)),
    )


__all__ = [
    "adapt_viewer_config",
    "entity_specs",
    "env_cfg_control_dt",
    "extract_terrain_data",
]
