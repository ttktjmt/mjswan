"""mjlab action terms as mjswan's.

mjlab's own action classes run natively in the browser, matched by class name; a task's
own class is traced at build time. The regex scale/offset patterns, and the PD gains and
command delays mjlab keeps on the actuator config, are resolved against the policy's
joint names, or a command term's own, since the browser looks joints up by exact name.
"""

from __future__ import annotations

import copy
import dataclasses
import re
import reprlib
import warnings
from collections.abc import Mapping
from typing import Any

from ..envs.mdp import actions as _actions_module
from ..envs.mdp.actions import is_traced, native_class
from ..envs.mdp.actions.actions import (
    BaseActionCfg,
    JointVelocityActionCfg,
    TracedActionCfg,
)
from ..managers.action_manager import ActionTermCfg as MjswanActionTermCfg
from .detect import differs, field_default, is_from_mjlab

#: Set by each mjlab action class in ``__post_init__`` (a tendon action's is ``TENDON``),
#: so the class name the mapping goes by already carries it.
_SET_BY_CLASS = frozenset({"transmission_type"})


def _set_fields(cfg: Any) -> dict[str, Any]:
    """*cfg*'s dataclass fields that hold other than their default, as a field without
    one always does. A duck-typed cfg has no fields to walk."""
    if not dataclasses.is_dataclass(cfg):
        return {}
    found: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name, dataclasses.MISSING)
        default = field_default(f)
        if default is dataclasses.MISSING or differs(value, default):
            found[f.name] = value
    return found


def _listing(fields: Mapping[str, Any]) -> str:
    return ", ".join(f"{name}={reprlib.repr(value)}" for name, value in fields.items())


def _prefixed(entity_name: str | None, names: Any) -> Any:
    """Patterns prefixed with the entity name, as mjlab does, to match
    ``policy_joint_names``. An alternation is grouped, or the prefix would bind to its
    first branch only."""
    if not entity_name or not isinstance(names, (list, tuple)):
        return names
    return tuple(
        f"{entity_name}/(?:{n})" if "|" in n else f"{entity_name}/{n}" for n in names
    )


def _traced_action_cfg(term: Any) -> TracedActionCfg:
    """A task's own action class, to be traced at build time."""
    entity_name = getattr(term, "entity_name", "robot")
    return TracedActionCfg(
        entity_name=entity_name,
        actuator_names=_prefixed(entity_name, getattr(term, "actuator_names", (".*",))),
        mjlab_cfg=term,
    )


def _adapt_action_cfg(term: Any) -> MjswanActionTermCfg | None:
    """Convert a single mjlab ``ActionTermCfg`` to mjswan.

    The class is looked up by name on ``mjswan.envs.mdp.actions`` and every field it
    shares with *term* is copied; a field set on *term* that the class lacks warns, since
    the browser ignores it. A task's own class with no mjswan equivalent is traced; one
    of mjlab's own without one (differential IK), or a cfg with no ``build`` to trace,
    is skipped and ``None`` returned.
    """
    mjswan_cls = native_class(term)

    if mjswan_cls is None:
        if is_traced(term):
            return _traced_action_cfg(term)
        traceable = callable(getattr(term, "build", None))
        warnings.warn(
            f"mjlab action type '{type(term).__name__}' has no mjswan equivalent"
            f"{'' if traceable else ' and no build() to trace'}. It will be skipped.",
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

    carried = {f.name for f in dataclasses.fields(mjswan_cls)} | _SET_BY_CLASS
    if dropped := {k: v for k, v in _set_fields(term).items() if k not in carried}:
        warnings.warn(
            f"mjlab action type '{type(term).__name__}' sets {_listing(dropped)}, which "
            f"mjswan's {mjswan_cls.__name__} does not have and the browser ignores.",
            category=RuntimeWarning,
            stacklevel=3,
        )

    if "actuator_names" in kwargs:
        kwargs["actuator_names"] = _prefixed(entity_name, kwargs["actuator_names"])

    return mjswan_cls(**kwargs)


def adapt_actions(
    actions: Mapping[str, Any] | None,
) -> Mapping[str, MjswanActionTermCfg] | None:
    """Adapt action configs, converting mjlab types if detected."""
    if actions is None:
        return None
    result: dict[str, MjswanActionTermCfg] = {}
    for key, term in actions.items():
        if term is None:
            continue
        if isinstance(term, MjswanActionTermCfg):
            result[key] = term
        elif (
            is_from_mjlab(term)
            or native_class(term) is not None
            or callable(getattr(term, "build", None))
        ):
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
    """Expand regex-keyed scale/offset dicts in action configs to literal joint names.

    mjlab keys per-joint values by regex (``{".*_hip_joint": 0.37}``) and the browser
    looks them up by exact name, so patterns are expanded against *joint_names*, the
    policy's joints or a command term's own, prefixed with the entity name
    (``"robot/left_hip_joint"``). Mutates each term in place.
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


def _entity_cfg(env_cfg: Any | None, term: MjswanActionTermCfg) -> Any | None:
    entities = getattr(getattr(env_cfg, "scene", None), "entities", None) or {}
    return entities.get(getattr(term, "entity_name", "robot"))


#: mjlab's ``ActuatorCfg.delay_*`` fields, by the name the manifest gives each.
_DELAY_FIELDS = {
    "min_lag": "delay_min_lag",
    "max_lag": "delay_max_lag",
    "hold_prob": "delay_hold_prob",
    "update_period": "delay_update_period",
    "per_env_phase": "delay_per_env_phase",
}


def _delay_groups(
    env_cfg: Any | None, term: Any, joint_names: list[str]
) -> dict[tuple[Any, ...], list[str]]:
    """*term*'s joints behind each mjlab delay buffer, by that buffer's key.

    mjlab fuses delayed actuators of one class, transmission and delay setting behind a
    single ``DelayBuffer``, which draws one lag for all of them.
    """
    entity = _entity_cfg(env_cfg, term)
    actuators = getattr(getattr(entity, "articulation", None), "actuators", ())
    targets = [re.compile(pattern) for pattern in term.actuator_names]
    driven = [n for n in joint_names if any(t.fullmatch(n) for t in targets)]
    groups: dict[tuple[Any, ...], list[str]] = {}
    for cfg in actuators:
        if getattr(cfg, "delay_max_lag", 0) <= 0:
            continue
        key = (
            type(cfg),
            getattr(cfg, "transmission_type", None),
            *(getattr(cfg, f) for f in _DELAY_FIELDS.values()),
        )
        joints = groups.setdefault(key, [])
        for name in _expand_patterns(
            {pattern: 0.0 for pattern in cfg.target_names_expr}, driven
        ):
            if name in driven and name not in joints:
                joints.append(name)
    return {
        key: [n for n in driven if n in joints]
        for key, joints in groups.items()
        if joints
    }


def _delayed_terms(
    actions: Mapping[str, MjswanActionTermCfg] | None,
) -> list[tuple[str, BaseActionCfg]]:
    return [
        (name, term)
        for name, term in (actions or {}).items()
        if isinstance(term, BaseActionCfg)
        and not isinstance(term, _actions_module.MuscleActivationActionCfg)
    ]


def resolve_actuator_delays(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
    env_cfg: Any | None,
) -> None:
    """Give each term the command delays of the actuators on its joints, in place.

    mjlab delays an actuator's command by ``delay_*`` physics steps (``ActuatorCfg``,
    any actuator type). One entry per mjlab delay buffer the term's joints sit behind,
    which draws one lag for them. A term whose ``delay`` the author set is left alone.
    """
    if not actions or not joint_names:
        return
    for _, term in _delayed_terms(actions):
        if term.delay is not None:
            continue
        term.delay = [
            {"joints": joints, **dict(zip(_DELAY_FIELDS, key[2:]))}
            for key, joints in _delay_groups(env_cfg, term, joint_names).items()
        ] or None


def warn_split_delays(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: Mapping[str, list[str]],
    env_cfg: Any | None,
) -> None:
    """Warn when two action terms drive joints behind one mjlab delay buffer whose lag
    is random: mjlab draws one lag for the buffer, the browser one per term.

    *joint_names* maps each term to the joints it was resolved against.
    """
    terms: dict[tuple[Any, ...], list[str]] = {}
    for name, term in _delayed_terms(actions):
        for key in _delay_groups(env_cfg, term, joint_names.get(name, [])):
            min_lag, max_lag, hold_prob = key[2], key[3], key[4]
            if min_lag != max_lag or hold_prob > 0:
                terms.setdefault(key, []).append(name)
    for key, names in terms.items():
        if len(names) > 1:
            warnings.warn(
                f"Action terms {names} drive joints behind one mjlab "
                f"'{key[0].__name__}' command delay with a random lag. mjlab draws one "
                "lag for them all; the browser draws each term's on its own.",
                category=RuntimeWarning,
                stacklevel=2,
            )


def resolve_default_joint_vel(
    actions: Mapping[str, MjswanActionTermCfg] | None,
    joint_names: list[str],
    env_cfg: Any | None,
) -> None:
    """Make a velocity term's ``offset`` the entity's default joint velocity.

    mjlab's ``use_default_offset`` replaces the offset with ``init_state.joint_vel``,
    which the browser has no copy of, so it is baked in here by exact joint name.
    """
    if not actions or not joint_names:
        return

    for term in actions.values():
        if not isinstance(term, JointVelocityActionCfg) or not term.use_default_offset:
            continue
        init_state = getattr(_entity_cfg(env_cfg, term), "init_state", None)
        joint_vel = getattr(init_state, "joint_vel", None)
        if not isinstance(joint_vel, dict):
            continue
        targets = [re.compile(pattern) for pattern in term.actuator_names]
        offset: dict[str, float] = {}
        for name in joint_names:
            if not any(target.fullmatch(name) for target in targets):
                continue
            bare = name.split("/")[-1]
            # First match wins, as in mjlab's `resolve_expr`.
            value = next((v for p, v in joint_vel.items() if re.match(p, bare)), 0.0)
            if value:
                offset[name] = float(value)
        term.offset = offset or 0.0


__all__ = [
    "adapt_actions",
    "resolve_action_scales",
    "resolve_actuator_delays",
    "resolve_default_joint_vel",
    "warn_split_delays",
]
