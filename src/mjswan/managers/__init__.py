"""Manager modules for mjswan.

Mirrors the ``mjlab.managers`` package layout so that mjlab import paths
translate directly::

    # mjlab
    from mjlab.managers.action_manager import ActionTermCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg

    # mjswan (identical API)
    from mjswan.managers.action_manager import ActionTermCfg
    from mjswan.managers.event_manager import EventTermCfg
    from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjswan.managers.termination_manager import TerminationTermCfg
"""

from .action_manager import ActionTermCfg
from .event_manager import EventTermCfg
from .observation_manager import ObservationGroupCfg, ObservationTermCfg
from .termination_manager import TerminationTermCfg

__all__ = [
    "ActionTermCfg",
    "EventTermCfg",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "TerminationTermCfg",
]
