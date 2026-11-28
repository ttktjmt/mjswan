"""The MDP a policy runs against, as one unit.

mjlab keeps observations, actions, terminations, commands and events side by side on
``ManagerBasedRlEnvCfg`` and trains a checkpoint against all five at once, so
:class:`MdpConfig` holds them together.

Identity is by object: two policies handed the *same* config share one MDP, meaning one
``mdp/<id>/`` directory and one set of traced graphs, while two equal but separate
configs are two MDPs. The per-manager keyword arguments on
:meth:`~mjswan.scene.SceneHandle.add_policy` construct an anonymous one.

See ``context/object-model.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

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

    episode_length_s: float | None = None
    """Seconds an episode lasts, which a time limit ends it at. ``None`` takes the
    ``env_cfg`` the first policy passes to ``add_policy``, else the scene's."""

    _adapted: bool = field(default=False, init=False, repr=False, compare=False)
    """Set once the first policy has filled and adapted this config on a scene."""

    _actuators: tuple[tuple[str, Any], ...] = field(
        default=(), init=False, repr=False, compare=False
    )
    """``(entity name, mjlab actuator cfg)`` for each actuator whose torque law the build
    traces, set with the adaptation from the env config the actions came from."""


__all__ = ["MdpConfig"]


#: Past a minute a delay means "off" and a timed freeze "until reset", which both exist.
MAX_WAIT_S = 60.0


@dataclass(frozen=True)
class TerminationDefault:
    """What a termination does when the viewer opens, in place of mjlab's reset at once.

    Given to :meth:`SceneHandle.set_term_defaults`; it changes only the viewer, not what the
    build traces.

    Args:
        on_fire: ``"reset"``, mjlab's; ``"delay"``, the reset ``seconds`` of episode time
            after the term fires, the simulation running on; ``"freeze"``, the simulation
            stopped on the frame the term fires, for ``seconds`` of wall time or, with
            ``seconds=None``, until a reset.
        seconds: The wait, above 0 and at most 60. A delay needs it; a reset takes none.
        enabled: ``False`` keeps the term evaluated and shown but never ending the episode.

    Example::

        scene.set_term_defaults(terminations={
            "bad_orientation": mjswan.TerminationDefault("delay", seconds=0.2),
            "illegal_contact": mjswan.TerminationDefault("freeze", seconds=3.0),
            "time_out": mjswan.TerminationDefault(enabled=False),
        })
    """

    on_fire: Literal["reset", "delay", "freeze"] = "reset"
    seconds: float | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.on_fire not in ("reset", "delay", "freeze"):
            raise ValueError(
                f'on_fire must be "reset", "delay" or "freeze", not {self.on_fire!r}.'
            )
        if not isinstance(self.enabled, bool):
            raise TypeError(f"enabled must be a bool, not {self.enabled!r}.")
        if self.seconds is None:
            if self.on_fire == "delay":
                raise ValueError("A delay needs its seconds.")
            return
        if self.on_fire == "reset":
            raise ValueError('A reset at once takes no seconds: use on_fire="delay".')
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
            raise TypeError(f"seconds must be a number, not {self.seconds!r}.")
        if not 0 < self.seconds <= MAX_WAIT_S:
            raise ValueError(
                f"seconds must be above 0 and at most {MAX_WAIT_S:g}, not {self.seconds!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """The manifest's form of the engine's override: ``enabled`` and ``on_fire``."""
        if self.seconds is None:
            on_fire: Any = self.on_fire
        else:
            on_fire = {self.on_fire: float(self.seconds)}
        return {"enabled": self.enabled, "on_fire": on_fire}
