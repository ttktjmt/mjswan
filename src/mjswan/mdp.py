"""The MDP a policy runs against, as one unit.

mjlab keeps observations, actions, terminations, commands and events side by side on
``ManagerBasedRlEnvCfg`` and trains a checkpoint against all five at once, so
:class:`MdpConfig` holds them together (ADR 0006 §3).

Identity is by object: two policies handed the *same* config share one MDP, meaning one
``mdp/<id>/`` directory and one set of traced graphs, while two equal but separate
configs are two MDPs. The per-manager keyword arguments on
:meth:`~mjswan.scene.SceneHandle.add_policy` construct an anonymous one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.command_manager import CommandTermConfig
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg


@dataclass
class MdpConfig:
    """The five term sets a policy is run against, held as one unit.

    Each field means what the matching :meth:`~mjswan.scene.SceneHandle.add_policy`
    argument means: ``None`` is "take it from the scene's env config" (or from the
    scene's own events), ``{}`` is "this MDP genuinely has none". Mutated in place by the
    first policy to use it, which fills the ``None`` fields and adapts mjlab types.

    Example:
        mdp = mjswan.MdpConfig(observations=..., actions=..., terminations=...)
        scene.add_policy(name="model_1000", policy=..., mdp=mdp)
        scene.add_policy(name="model_2000", policy=..., mdp=mdp)  # shares it
    """

    observations: dict[str, ObservationGroupCfg] | Mapping[str, Any] | Any | None = None
    """Observation groups, keyed by the name the policy's ONNX slot table uses. A
    single group, mjlab's whole ``env_cfg.observations`` dict, or a dict already keyed."""

    actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None
    """Action terms, keyed by term name (e.g. ``"joint_pos"``)."""

    terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None
    """Termination terms, keyed by term name (e.g. ``"time_out"``, ``"fallen"``)."""

    commands: dict[str, CommandTermConfig] | Mapping[str, Any] | None = None
    """Command terms, keyed by their policy-visible names."""

    events: dict[str, EventTermCfg] | Mapping[str, Any] | None = None
    """Event terms, keyed by name, in any of the four modes. ``None`` takes the scene's
    events, so a policy that says nothing gets the ones it was trained with."""

    name: str | None = None
    """Optional name; its ``name2id`` is the MDP's id, its ``mdp/<id>/`` directory in the
    build. Unnamed ids are derived instead: see :meth:`~mjswan.scene.SceneConfig.mdp_id`."""

    _adapted: bool = field(default=False, init=False, repr=False, compare=False)
    """Set once the first policy has filled and adapted this config on a scene."""


__all__ = ["MdpConfig"]
