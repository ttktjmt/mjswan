"""mjlab event terms as mjswan's, in all three modes, and the terrain spawn swap.

Same resolution order as :mod:`.observation`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from ..envs.mdp.events import (
    EventBinding,
    reset_root_state_on_flat_patch,
)
from ..envs.mdp.events import _custom_registry as _custom_event_registry
from ..managers.event_manager import EventTermCfg as MjswanEventTermCfg
from .detect import is_from_mjlab


def _adapt_event_func(
    func: Any, term_name: str | None = None
) -> EventBinding | Callable[..., Any]:
    """Resolve the function an event term's ONNX graph is traced from."""
    if isinstance(func, EventBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_event_registry:
        return _custom_event_registry[name]
    if term_name and term_name in _custom_event_registry:
        return _custom_event_registry[term_name]
    return func


def _sanitize_event_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only event params while keeping joint scoping data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    joint_names = getattr(asset_cfg, "joint_names", None)
    if isinstance(joint_names, (list, tuple)):
        result["joint_names"] = [str(name) for name in joint_names]
    elif isinstance(joint_names, str):
        result["joint_names"] = [joint_names]

    joint_ids = getattr(asset_cfg, "joint_ids", None)
    if isinstance(joint_ids, (list, tuple)):
        result["joint_ids"] = [int(idx) for idx in joint_ids]

    return result


def _adapt_event_cfg(term: Any, term_name: str | None = None) -> MjswanEventTermCfg:
    """Convert a single mjlab ``EventTermCfg`` to mjswan.

    As in :func:`_adapt_obs_term`, params are sanitized only for the binding path.
    """
    func = _adapt_event_func(term.func, term_name=term_name)
    raw_params = dict(getattr(term, "params", None) or {})
    params = (
        _sanitize_event_params(raw_params)
        if isinstance(func, EventBinding)
        else raw_params
    )
    return MjswanEventTermCfg(
        func=func,
        mode=getattr(term, "mode", "reset"),
        params=params,
        interval_range_s=getattr(term, "interval_range_s", None),
        is_global_time=getattr(term, "is_global_time", False),
        min_step_count_between_reset=getattr(
            term, "min_step_count_between_reset", None
        ),
    )


def adapt_events(
    events: Mapping[str, Any] | None,
) -> dict[str, MjswanEventTermCfg] | None:
    """Adapt event configs, converting mjlab types if detected.

    Serialization (ONNX tracing included) happens later, at build time, once the scene's
    live env and output directory are known.
    """
    if not events:
        return None
    result: dict[str, MjswanEventTermCfg] = {}
    for key, term in events.items():
        if isinstance(term, MjswanEventTermCfg):
            result[key] = term
        elif is_from_mjlab(term):
            result[key] = _adapt_event_cfg(term, term_name=key)
        # non-mjlab, non-mjswan entries are skipped
    return result or None


def apply_terrain_spawn(scene: Any) -> None:
    """Swap a scene's ``reset_root_state_uniform`` for patch-based spawning, in place.

    mjlab spreads many envs over the terrain, so its uniform reset only jitters each
    around its own origin; with the browser's one env, only drawing a patch covers the
    terrain. A no-op unless the scene has both a flat-patch table and that mjlab term.
    """
    flat_patches = (scene.terrain_data or {}).get("flat_patches", {})
    events = scene.events
    if not flat_patches or not events:
        return
    patch_name = "spawn" if "spawn" in flat_patches else next(iter(flat_patches))
    patches = flat_patches[patch_name]
    if not patches:
        return

    # After the early returns: a terrain-free scene must not need mjlab importable.
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for key, event in events.items():
        if getattr(event.func, "__name__", None) != "reset_root_state_uniform":
            continue
        # The mjlab term's own yaw range, so the swap does not also widen it.
        pose_range = dict(event.params.get("pose_range") or {})
        entity = getattr(event.params.get("asset_cfg"), "name", None) or "robot"
        events[key] = MjswanEventTermCfg(
            func=reset_root_state_on_flat_patch,
            mode=event.mode,
            params={
                "asset_cfg": SceneEntityCfg(entity),
                "patches": patches,
                "yaw_range": tuple(pose_range.get("yaw", (-math.pi, math.pi))),
            },
        )
        return


__all__ = ["adapt_events", "apply_terrain_spawn"]
