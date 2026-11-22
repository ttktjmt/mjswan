"""Action terms: the authored config to merge over, and the graphs of a traced one.

A term of an mjlab class mjswan runs natively serializes its own config. A task's own
class (:class:`TracedActionCfg`) is built on the trace env and its ``process_actions``
and ``reset`` traced; its entry names the native apply and its options.

See ``context/tracing.md``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...envs.mdp.actions import CommandJointPositionActionCfg, TracedActionCfg
from ...managers.action_manager import ActionBinding, _action_registry
from . import graph
from .provenance import graph_meta, term_provenance

if TYPE_CHECKING:
    from ...compile.action import ActionGraph
    from ...managers.action_manager import ActionTermCfg


def serialize_actions(
    actions: Mapping[str, ActionTermCfg],
    authored: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    commands: Collection[str] = (),
    widths: Mapping[str, int] | None = None,
    env: Any = None,
    out_dir: Path | None = None,
    scope: str | None = None,
) -> dict[str, Any]:
    """The MDP entry's ``actions`` block, keyed by term name.

    Each term's entry is merged field-wise over what the ``config_path`` sidecar authored
    for the same term, where a motor robot's PD gains live, so a scene can tweak the
    offset without restating them. A traced term needs *env*, and *out_dir* and *scope*
    to write its graphs.

    Raises ``ValueError`` if a command term's command is missing or the wrong width.
    """
    for name, cfg in actions.items():
        if not isinstance(cfg, CommandJointPositionActionCfg):
            continue
        if cfg.command_name not in commands:
            raise ValueError(
                f"Action term {name!r} reads command {cfg.command_name!r}, which the "
                f"MDP does not have (its commands: {sorted(commands)}). Add it to the "
                "policy's `commands`."
            )
        width = (widths or {}).get(cfg.command_name)
        if width is not None and width != len(cfg.actuator_names):
            raise ValueError(
                f"Action term {name!r} drives {len(cfg.actuator_names)} joint(s), but "
                f"command {cfg.command_name!r} is {width} wide: one column per joint."
            )
    authored = authored or {}
    return {
        name: {
            **authored.get(name, {}),
            **(
                traced_action_entry(name, cfg, env, out_dir, scope=scope)
                if isinstance(cfg, TracedActionCfg)
                else cfg.to_dict()
            ),
        }
        for name, cfg in actions.items()
    }


def inherited_apply(term: Any, *, unchanged: bool) -> str | None:
    """The native apply of *term*'s mjlab base: its own ``apply_actions`` when
    *unchanged*, else any it subclasses. ``None`` when there is none."""
    try:
        from mjlab.envs.mdp.actions import actions as mjlab_actions
    except ImportError:
        return None
    bases = (
        (mjlab_actions.RelativeJointPositionAction, "joint_position_relative"),
        (mjlab_actions.JointPositionAction, "joint_position"),
        (mjlab_actions.JointVelocityAction, "joint_velocity"),
        (mjlab_actions.JointEffortAction, "torque"),
    )
    for base, kind in bases:
        same = type(term).apply_actions is base.apply_actions
        if same or (not unchanged and isinstance(term, base)):
            return kind
    return None


#: The applies a ramp or a hold acts on: a position target, absolute or relative.
_POSITION_KINDS = frozenset({"joint_position", "joint_position_relative"})


def _resolve(value: Any, mjlab_cfg: Any) -> Any:
    """A binding field: a value, or a ``(mjlab_cfg) -> value`` callable."""
    return value(mjlab_cfg) if callable(value) else value


def traced_action_entry(
    name: str,
    cfg: TracedActionCfg,
    env: Any,
    out_dir: Path | None,
    *,
    scope: str | None = None,
) -> dict[str, Any]:
    """Build the term on *env*, trace it, write its graphs and return its entry."""
    from ...compile.action import joint_targets, trace_action_term
    from ...compile.slot import slots_json

    cls_name = type(cfg.mjlab_cfg).__name__
    if env is None or out_dir is None:
        raise ValueError(
            f"Action term {name!r} ({cls_name}) is a task's own action class, which the "
            "build traces against the scene's trace env. `add_scene_mjlab` supplies "
            "one; a plain scene needs `scene.set_trace_env(...)`."
        )
    term = cfg.mjlab_cfg.build(env)
    joint_targets(term, name)
    # Looked up here, so a binding registered after `add_policy` still counts.
    registered = cfg.binding or _action_registry.get(cls_name)
    binding = registered or ActionBinding()
    if binding.trace_override is not None:
        binding.trace_override(term)
    kind = binding.apply or inherited_apply(term, unchanged=registered is None)
    if kind is None:
        raise ValueError(
            f"Action term {name!r}: {type(term).__name__} overrides apply_actions, "
            "which runs every physics substep, so the browser runs it natively and "
            "needs to know what it does. Name it with mjswan.register_action("
            f"'{cls_name}', ActionBinding(apply='joint_position', ...)), adding "
            "`hold_s` or `interpolate` where it holds or ramps the target."
        )
    interpolate = bool(_resolve(binding.interpolate, cfg.mjlab_cfg))
    hold_s = _resolve(binding.hold_s, cfg.mjlab_cfg)
    if (interpolate or hold_s is not None) and kind not in _POSITION_KINDS:
        raise ValueError(
            f"Action term {name!r}: `interpolate` and `hold_s` ramp or hold a position "
            f"target, and its apply is {kind!r}. Drop them from its ActionBinding."
        )

    export = trace_action_term(term, name=name, ramp_from=interpolate)
    meta = graph_meta("action", name, type(term))

    def written(graph_name: str, traced: ActionGraph) -> dict[str, Any]:
        ref = graph.onnx_ref("action", graph_name, scope)
        graph.write_onnx(Path(out_dir), ref, traced.onnx_bytes, meta=meta)
        return {
            "onnx": ref,
            "input_slots": slots_json(traced),
            **(
                {"rand_dim": traced.rand_dim, "rand_ranges": traced.rand_ranges}
                if traced.rand_dim
                else {}
            ),
        }

    entry: dict[str, Any] = {
        "type": kind,
        "actuator_names": [re.escape(t) for t in export.target_names],
    }
    for field in ("stiffness", "damping", "saturation_effort", "velocity_limit"):
        if (value := getattr(cfg, field)) is not None:
            entry[field] = value
    if cfg.delay:
        entry["delay"] = cfg.delay
    if interpolate:
        entry["interpolate"] = True
    if hold_s is not None:
        entry["hold_s"] = float(hold_s)
    entry["process"] = {
        **written(name, export.process),
        "state_fields": export.state_fields,
    }
    if export.reset is not None:
        entry["reset"] = written(f"{name}_reset", export.reset)
    return {**entry, **term_provenance(type(term))}


__all__ = ["inherited_apply", "serialize_actions", "traced_action_entry"]
