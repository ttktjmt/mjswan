"""Manager configs, laid out like ``mjlab.managers``.

Import paths translate directly::

    # mjlab
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.command_manager import CommandTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    # mjswan (identical API, except the command config keeps its own name: it also
    # carries the browser-side UI and the pending trace)
    from mjswan.managers.action_manager import ActionTermCfg
    from mjswan.managers.command_manager import CommandTermConfig
    from mjswan.managers.event_manager import EventTermCfg
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjswan.managers.termination_manager import TerminationTermCfg
"""

from .action_manager import ActionTermCfg
from .command_manager import CommandTermConfig
from .event_manager import EventTermCfg
from .observation_manager import ObservationGroupCfg, ObservationTermCfg
from .termination_manager import TerminationTermCfg

__all__ = [
    "ActionTermCfg",
    "CommandTermConfig",
    "EventTermCfg",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "TerminationTermCfg",
]
