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
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


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


__all__ = ["ActionTermCfg"]
