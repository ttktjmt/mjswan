"""The base ``ActionTermCfg``, mirroring ``mjlab.managers.action_manager``.

The concrete action-term configs in ``mjswan.envs.mdp.actions`` derive from it, as
mjlab's do. Example::

    from mjswan.envs.mdp.actions import JointPositionActionCfg

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        ),
    }

See ``context/tracing.md``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Literal


@dataclass(kw_only=True)
class ActionTermCfg(abc.ABC):
    """Base configuration for an action term.

    Mirrors ``mjlab.managers.action_manager.ActionTermCfg``.
    """

    entity_name: str = "robot"
    """Accepted for mjlab compatibility; mjswan targets the single policy entity."""

    clip: dict[str, tuple] | None = None
    """Per-target ``(min, max)`` bounds, applied after scale/offset.

    Keys are joint-name patterns, resolved with ``re.fullmatch`` as mjlab does; a
    target no pattern matches is unbounded. As in mjlab's ``BaseActionCfg.clip``, the
    clamp hits ``raw * scale + offset``, before any encoder-bias subtraction."""

    unsupported_reason: str | None = None
    """If set, raises ``NotImplementedError`` at build time."""

    def _add_clip(self, entry: dict[str, Any]) -> None:
        """Attach ``clip`` to a serialized entry, if this term declares any.

        Emitted as patterns and resolved browser-side with mjlab's fullmatch, unlike
        ``stiffness``/``damping``, which are mjswan's own and keyed by exact joint name.
        """
        if self.clip is not None:
            entry["clip"] = {k: list(v) for k, v in self.clip.items()}

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS runtime."""
        raise NotImplementedError


#: What a traced term's ``apply_actions`` does each substep, as the browser runs it.
ApplyKind = Literal[
    "joint_position", "joint_position_relative", "joint_velocity", "torque"
]


@dataclass(frozen=True)
class ActionBinding:
    """How the browser runs a task's own mjlab action class.

    The build traces a task's own action class mjswan has no class of its own for: its
    ``process_actions`` and ``reset`` become graphs. Its ``apply_actions`` runs every
    physics substep, so the browser runs it natively, and a class that overrides it
    needs a binding naming what it does. One that inherits mjlab's
    ``JointPositionAction`` (or the relative, velocity or effort one) gets that base's
    apply without a binding.

    ``hold_s`` and ``interpolate`` may each be a value or a ``(mjlab_cfg) -> value``
    callable, to read the task's own field.
    """

    apply: ApplyKind | None = None
    """A position target, mjlab's relative one, a velocity target or a torque; ``None``
    infers it from the class's mjlab base."""

    hold_s: float | Callable[[Any], float | None] | None = None
    """Seconds from an episode's start during which each target is the joint's
    measured position, as a held get-up action does."""

    interpolate: bool | Callable[[Any], bool] = False
    """Ramp each target across the substeps, from the processed action before this
    control step to the new one, landing on it at the last substep."""

    trace_override: Callable[[Any], None] | None = None
    """Mutates the freshly built term before it is traced, e.g. to rebind a body the
    tracer cannot follow."""


_action_registry: dict[str, ActionBinding] = {}


def register_action(mjlab_name: str, binding: ActionBinding) -> None:
    """Bind a task's action cfg class, by its name, to how the browser runs it.

    The build looks the binding up, so this may come before or after ``add_policy``; a
    later call replaces an earlier one. For a term that holds each joint where the reset
    left it::

        mjswan.register_action(
            "HeldJointPositionActionCfg",
            ActionBinding(hold_s=lambda cfg: cfg.hold_duration_s),
        )
    """
    _action_registry[mjlab_name] = binding


__all__ = ["ActionBinding", "ActionTermCfg", "ApplyKind", "register_action"]
