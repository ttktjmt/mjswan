"""mjlab action terms as mjswan's.

Actions stay a fixed, native, non-traced set (ADR 0005 §7): ``type(cfg).__name__`` is
looked up on ``mjswan.envs.mdp.actions`` and the dataclass fields are copied. The regex
scale/offset patterns and the PD gains mjlab keeps on the actuator config are resolved
here against the policy's joint names, since the browser looks joints up by exact name.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import actions as _actions_module
from ..envs.mdp.actions.actions import (
    JointPositionActionCfg,
    ReferenceJointPositionActionCfg,
)
from ..managers.action_manager import ActionTermCfg as MjswanActionTermCfg
from .detect import is_from_mjlab

_ACTION_CLASS_ALIASES: dict[str, str] = {
    # myosuite's muscle action cfg sits outside mjlab's hierarchy; translate it.
    "MyoMuscleActivationActionCfg": "MuscleActivationActionCfg",
}


def _mjswan_action_class(term: Any) -> type[MjswanActionTermCfg] | None:
    """The mjswan action class matching *term*'s class name, or ``None``."""
    class_name = _ACTION_CLASS_ALIASES.get(type(term).__name__, type(term).__name__)
    cls = getattr(_actions_module, class_name, None)
    if isinstance(cls, type) and issubclass(cls, MjswanActionTermCfg):
        return cls
    return None


def _has_mjswan_action(term: Any) -> bool:
    """Whether *term* adapts by name — a task's own subclass is not in ``mjlab``."""
    return _mjswan_action_class(term) is not None


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    Looks up ``type(term).__name__`` on ``mjswan.envs.mdp.actions`` to
    find the corresponding mjswan class, then copies all matching
    dataclass fields automatically.

    Returns ``None`` if no mjswan equivalent exists; the caller is
    responsible for dropping the entry.
    """
    class_name = _ACTION_CLASS_ALIASES.get(type(term).__name__, type(term).__name__)
    mjswan_cls = _mjswan_action_class(term)

    if mjswan_cls is None:
        warnings.warn(
            f"mjlab action type '{class_name}' has no mjswan equivalent. "
            f"It will be skipped.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return None

    kwargs: dict[str, Any] = {}
    entity_name = getattr(term, "entity_name", None)
    for f in dataclasses.fields(mjswan_cls):
        if f.name == "unsupported_reason":
            continue
        val = getattr(term, f.name, dataclasses.MISSING)
        if val is not dataclasses.MISSING:
            kwargs[f.name] = val

    # Prefixed with the entity name, as mjlab does, to match policy_joint_names.
    if entity_name and "actuator_names" in kwargs:
        raw = kwargs["actuator_names"]
        if isinstance(raw, (list, tuple)):
            kwargs["actuator_names"] = tuple(f"{entity_name}/{n}" for n in raw)

    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif is_from_mjlab(term) or _has_mjswan_action(term):
            adapted = _adapt_action_cfg(term)
            if adapted is not None:
                result[key] = adapted
        else:
            # Copied: `resolve_action_scales` rewrites `scale` in place, and this is
            # the object a live mjlab env config holds.
            result[key] = copy.copy(term)
    return result


def resolve_action_scales(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
) -> None:
    """Resolve regex-pattern scale/offset dicts in action configs to literal joint names.

    mjlab stores per-joint scale as ``{".*_hip_joint": 0.37, ...}`` using regex
    patterns.  The browser runtime does exact string lookups, so patterns are
    expanded here against *joint_names* (the ordered list of joints the policy
    controls, prefixed with the entity name, e.g. ``"robot/left_hip_joint"``).

    Mutates the ``scale`` and ``offset`` fields of each action term in-place.
    """
    if not actions or not joint_names:
        return

    for term in actions.values():
        scale = getattr(term, "scale", None)
        if isinstance(scale, dict):
            setattr(term, "scale", _expand_patterns(scale, joint_names))
        offset = getattr(term, "offset", None)
        if isinstance(offset, dict):
            setattr(term, "offset", _expand_patterns(offset, joint_names))


def _expand_patterns(value: Any, joint_names: list[str]) -> Any:
    """A ``{pattern: value}`` mapping keyed by the joint names it matches."""
    if not isinstance(value, dict):
        return value
    resolved: dict[str, float] = {}
    for pattern, val in value.items():
        try:
            regex = re.compile(pattern)
        except re.error:
            resolved[pattern] = val
            continue
        for joint_name in joint_names:
            bare = joint_name.split("/")[-1] if "/" in joint_name else joint_name
            if regex.fullmatch(bare) or regex.fullmatch(joint_name):
                resolved[joint_name] = val
    return resolved if resolved else value


#: This family computes its PD in torch; builtin position actuators bake the gains
#: into the model and need nothing here.
_PYTHON_PD_ACTUATOR = "IdealPdActuatorCfg"


def resolve_pd_gains(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
    env_cfg: Any | None,
) -> None:
    """Fill a position term's ``stiffness``/``damping`` from the entity's actuators.

    The browser runs the PD for a ``biastype=none`` actuator itself and reads the gains
    off the *action* term, where mjlab keeps them on the *actuator* config.

    Mutates the two fields in-place, and only when the term sets neither.
    """
    entities = getattr(getattr(env_cfg, "scene", None), "entities", None)
    if not actions or not joint_names or not entities:
        return

    for term in actions.values():
        if not isinstance(
            term, (JointPositionActionCfg, ReferenceJointPositionActionCfg)
        ):
            continue
        if term.stiffness is not None or term.damping is not None:
            continue
        entity = entities.get(getattr(term, "entity_name", "robot"))
        actuators = [
            cfg
            for cfg in getattr(getattr(entity, "articulation", None), "actuators", ())
            if any(base.__name__ == _PYTHON_PD_ACTUATOR for base in type(cfg).__mro__)
        ]
        gains = {
            field: _expand_patterns(
                {
                    pattern: float(getattr(cfg, field))
                    for cfg in actuators
                    for pattern in cfg.target_names_expr
                },
                joint_names,
            )
            for field in ("stiffness", "damping")
        }
        if gains["stiffness"]:
            term.stiffness = gains["stiffness"]
            term.damping = gains["damping"]


__all__ = ["adapt_actions", "resolve_action_scales", "resolve_pd_gains"]
