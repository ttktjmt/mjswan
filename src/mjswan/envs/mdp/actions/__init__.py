"""Action configuration classes for mjswan.

Mirrors ``mjlab.envs.mdp.actions`` so that the following import pattern
translates directly::

    # mjlab
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    # mjswan (identical)
    from mjswan.envs.mdp.actions import JointPositionActionCfg
"""

from .actions import (
    ActionTermCfg,
    BaseActionCfg,
    CommandJointPositionActionCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
    JointVelocityActionCfg,
    MuscleActivationActionCfg,
    ReferenceJointPositionActionCfg,
    RelativeJointPositionActionCfg,
    SiteEffortActionCfg,
    TendonEffortActionCfg,
    TendonLengthActionCfg,
    TendonVelocityActionCfg,
    TracedActionCfg,
)

#: An action cfg mjswan runs under another class name: myosuite's muscle action cfg
#: sits outside mjlab's hierarchy.
_ALIASES: dict[str, str] = {
    "MyoMuscleActivationActionCfg": "MuscleActivationActionCfg",
}


def native_class(cfg: object) -> type[ActionTermCfg] | None:
    """mjswan's own class for an mjlab action cfg, matched by class name, or ``None``."""
    name = type(cfg).__name__
    found = globals().get(_ALIASES.get(name, name))
    if isinstance(found, type) and issubclass(found, ActionTermCfg):
        return found
    return None


def is_traced(cfg: object) -> bool:
    """Whether the build traces *cfg*'s term: a task's own buildable class
    mjswan has none of. One of mjlab's own without one (differential IK) is skipped."""
    return (
        native_class(cfg) is None
        and callable(getattr(cfg, "build", None))
        and (type(cfg).__module__ or "").split(".")[0] != "mjlab"
    )


__all__ = [
    "ActionTermCfg",
    "BaseActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "ReferenceJointPositionActionCfg",
    "RelativeJointPositionActionCfg",
    "CommandJointPositionActionCfg",
    "JointEffortActionCfg",
    "MuscleActivationActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
    "TracedActionCfg",
    "is_traced",
    "native_class",
]
