"""Action terms: nothing to trace, only the authored config to merge over."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...managers.action_manager import ActionTermCfg


def serialize_actions(
    actions: Mapping[str, ActionTermCfg],
    authored: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The MDP entry's ``actions`` block, keyed by term name.

    Each term's ``to_dict()`` is merged field-wise over what the ``config_path`` sidecar
    authored for the same term, where a motor robot's PD gains live, so a scene can
    tweak the offset without restating them.
    """
    authored = authored or {}
    return {
        name: {**authored.get(name, {}), **cfg.to_dict()}
        for name, cfg in actions.items()
    }


__all__ = ["serialize_actions"]
