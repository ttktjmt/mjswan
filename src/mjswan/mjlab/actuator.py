"""mjlab actuators whose torque law the browser runs as a traced graph.

mjlab computes an actuator's control in Python, every physics substep, unless MuJoCo
computes it from the target itself (the builtin and XML actuators). The build traces each
such law once per MDP, from that MDP's own env config, on an actuator built afresh on the
scene's trace env.

See ``context/tracing.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

#: Configs whose control MuJoCo computes from the target, which the browser writes as is.
_MUJOCO_COMPUTES = frozenset(
    {
        "BuiltinPositionActuatorCfg",
        "BuiltinPdActuatorCfg",
        "BuiltinMotorActuatorCfg",
        "BuiltinDcMotorActuatorCfg",
        "BuiltinVelocityActuatorCfg",
        "BuiltinMuscleActuatorCfg",
        "XmlActuatorCfg",
    }
)


def _mujoco_computes(cfg: Any) -> bool:
    return any(base.__name__ in _MUJOCO_COMPUTES for base in type(cfg).__mro__)


def _drives_joints(cfg: Any) -> bool:
    transmission = getattr(cfg, "transmission_type", None)
    return (
        transmission is None or getattr(transmission, "value", transmission) == "joint"
    )


def traced_actuator_cfgs(
    env_cfg: Any | None, entity_names: Iterable[str]
) -> tuple[tuple[str, Any], ...]:
    """``(entity name, cfg)`` for each actuator of *entity_names* in *env_cfg* whose law
    mjlab computes in Python, in config order.

    One driving tendons or sites warns and is left out: the browser feeds a traced law
    joint positions and velocities only, so it would be fed the wrong state.
    """
    entities = getattr(getattr(env_cfg, "scene", None), "entities", None) or {}
    found: list[tuple[str, Any]] = []
    for entity_name in dict.fromkeys(entity_names):
        articulation = getattr(entities.get(entity_name), "articulation", None)
        for cfg in getattr(articulation, "actuators", None) or ():
            if _mujoco_computes(cfg):
                continue
            if not _drives_joints(cfg):
                warnings.warn(
                    f"mjlab actuator '{type(cfg).__name__}' on {entity_name!r} drives "
                    f"{getattr(cfg.transmission_type, 'value', cfg.transmission_type)}s, "
                    "and the browser runs a traced actuator law on joints only, so it "
                    "is not run.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
                continue
            found.append((entity_name, cfg))
    return tuple(found)


def build_actuator(env: Any, entity_name: str, cfg: Any) -> tuple[Any, list[str]]:
    """*cfg* built and initialized on *env*'s entity, as mjlab builds it for a scene,
    with the names of the joints it drives in its target order."""
    entity = env.scene[entity_name]
    target_ids, target_names = entity.find_joints(cfg.target_names_expr)
    actuator = cfg.build(entity, target_ids, target_names)
    sim = env.sim
    actuator.initialize(sim.mj_model, sim.model, sim.data, str(sim.device))
    return actuator, list(target_names)


__all__ = ["build_actuator", "traced_actuator_cfgs"]
