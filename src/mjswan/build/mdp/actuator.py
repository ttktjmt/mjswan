"""Actuators: each traced torque law's graph, and the MDP entry's ``actuators`` block.

See ``context/tracing.md``.
"""

from __future__ import annotations

import re
import reprlib
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...managers.action_manager import _actuator_registry
from . import graph
from .provenance import graph_meta, term_provenance

#: Action kinds whose apply runs no PD on the term's gains, law or none.
_NO_PD_KINDS = frozenset({"torque", "muscle_activation"})


def _model_joint_names(
    entity_name: str, targets: Sequence[str], policy_joint_names: Sequence[str]
) -> list[str]:
    """*targets* named as the policy names its joints: with the entity's prefix where
    the scene's model carries one, as an mjlab scene's does, or names none."""
    prefix = f"{entity_name}/"
    prefixed = not policy_joint_names or any(
        name.startswith(prefix) for name in policy_joint_names
    )
    return [f"{prefix}{t}" if prefixed else t for t in targets]


def serialize_actuators(
    actuators: Sequence[tuple[str, Any]],
    policy_joint_names: Sequence[str],
    *,
    env: Any,
    out_dir: Path,
    scope: str | None = None,
) -> dict[str, Any]:
    """The MDP entry's ``actuators`` block.

    Each ``(entity name, cfg)`` is built on *env*, its law (a registered one before its
    own ``compute``) traced and written under *scope*, and keyed by its class name and
    place in *actuators*. The entry lists the joints the graph's columns follow.
    """
    from ...compile.actuator import trace_actuator
    from ...mjlab.actuator import build_actuator

    entries: dict[str, Any] = {}
    for index, (entity_name, cfg) in enumerate(actuators):
        actuator, targets = build_actuator(env, entity_name, cfg)
        cls_name = type(actuator).__name__
        law = _actuator_registry.get(type(cfg).__name__)
        func = law or type(actuator)
        name = f"{cls_name}_{index}"
        export = trace_actuator(
            actuator,
            len(targets),
            what=(
                f"Actuator {cls_name} on {entity_name!r} "
                f"{reprlib.repr(tuple(cfg.target_names_expr))}"
            ),
            compute=law,
            num_envs=getattr(env, "num_envs", 1),
        )
        ref = graph.onnx_ref("actuator", name, scope)
        graph.write_onnx(
            out_dir, ref, export.onnx_bytes, meta=graph_meta("actuator", name, func)
        )
        entries[name] = {
            "joints": _model_joint_names(entity_name, targets, policy_joint_names),
            "onnx": ref,
            **term_provenance(func),
        }
    return entries


def _matches(pattern: str, name: str) -> bool:
    """*name* against an ``actuator_names`` pattern, anchored as the browser's are."""
    try:
        return re.fullmatch(pattern, name) is not None
    except re.error:
        return pattern == name


def warn_ignored_gains(
    actions: Mapping[str, Any], actuators: Mapping[str, Any]
) -> None:
    """Warn for each action entry whose ``stiffness`` or ``damping`` falls on joints a
    traced law in *actuators* drives: the browser runs the law there, not the term's PD.
    """
    driven = [joint for entry in actuators.values() for joint in entry["joints"]]
    for term_name, entry in actions.items():
        if not isinstance(entry, Mapping) or entry.get("type") in _NO_PD_KINDS:
            continue
        patterns = entry.get("actuator_names")
        if patterns is None:
            patterns = (".*",)
        covered = [j for j in driven if any(_matches(p, j) for p in patterns)]
        ignored: dict[str, list[str]] = {}
        for field in ("stiffness", "damping"):
            value = entry.get(field)
            if value is None:
                continue
            on = (
                [j for j in covered if j in value]
                if isinstance(value, Mapping)
                else covered
            )
            if on:
                ignored[field] = on
        if not ignored:
            continue
        joints = list(dict.fromkeys(j for on in ignored.values() for j in on))
        warnings.warn(
            f"Action term {term_name!r} sets {' and '.join(ignored)} on "
            f"{reprlib.repr(joints)}, which traced actuator laws drive: the browser runs "
            "each actuator's own law there and ignores the term's gains. Drop them, or "
            "trace the law you want with mjswan.register_actuator.",
            category=RuntimeWarning,
            stacklevel=2,
        )


__all__ = ["serialize_actuators", "warn_ignored_gains"]
