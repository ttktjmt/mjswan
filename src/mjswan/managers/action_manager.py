"""Action manager configuration for mjswan.

Mirrors ``mjlab.managers.action_manager``: the base ``ActionTermCfg`` lives here, and the
concrete action-term configs in ``mjswan.envs.mdp.actions`` derive from it, the same
way mjlab's do. Example::

    from mjswan.envs.mdp.actions import JointPositionActionCfg

    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        ),
    }
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class ActionTermCfg(abc.ABC):
    """Base configuration for an action term.

    Mirrors ``mjlab.managers.action_manager.ActionTermCfg``.
    """

    entity_name: str = "robot"
    """Name of the entity in the scene.  Accepted for mjlab compatibility;
    mjswan targets the single policy entity."""

    clip: dict[str, tuple] | None = None
    """Per-target clipping bounds, applied after scale/offset.

    Keys are joint-name *patterns* (mjlab resolves them with ``re.fullmatch`` via
    ``resolve_matching_names_values``), values are ``(min, max)``. A target no
    pattern matches is unbounded. Mirrors ``BaseActionCfg.clip``: mjlab clamps
    ``raw * scale + offset`` — the *processed* action, before any encoder-bias
    subtraction — so the browser applies it at the same point."""

    unsupported_reason: str | None = None
    """If set, raises ``NotImplementedError`` at build time."""

    def _add_clip(self, entry: dict[str, Any]) -> None:
        """Attach ``clip`` to a serialized entry, if this term declares any.

        Emitted as patterns and resolved browser-side with mjlab's fullmatch — unlike
        ``stiffness``/``damping``, which are mjswan's own and keyed by exact joint name.
        """
        if self.clip is not None:
            entry["clip"] = {k: list(v) for k, v in self.clip.items()}

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict for the TS runtime."""
        raise NotImplementedError


def serialize_actions(actions: Mapping[str, ActionTermCfg]) -> dict[str, Any]:
    """Serialize a dict of action term configs to JSON-compatible format.

    Returns a dict keyed by term name, each value being the term's
    ``to_dict()`` output.
    """
    return {name: term_cfg.to_dict() for name, term_cfg in actions.items()}


__all__ = ["ActionTermCfg", "serialize_actions"]
