"""The MDP half of a scene's manifest entry: one ``.onnx`` per traced term, and the
JSON that names it.

Each module here serializes one kind of term. A plain-callable term body is traced
against the scene's live env by :mod:`mjswan.compile`, written under
``<scene>/mdp/<mdp-id>/`` by :mod:`.graph`, and returned as the manifest-shaped entry
the runtime consumes; a ``*Binding``-typed term (a custom TS class) keeps serializing
through its own ``to_dict()``. Action terms trace nothing and only merge.

Called from :mod:`mjswan.build.manifest` once per MDP, after that scene's trace env
and directory are both known.
"""

from __future__ import annotations

from .action import serialize_actions
from .command import command_config, serialize_command, write_command_artifact
from .event import model_field_dr_descriptor, serialize_event, serialize_events
from .graph import onnx_ref, stamp_provenance, write_onnx
from .observation import (
    policy_native_sizes,
    serialize_observation_group,
    serialize_observation_term,
)
from .sensor import contact_sensor_descriptor, raycast_sensor_descriptor
from .termination import (
    FUSED_TERMINATION_KEY,
    serialize_termination,
    serialize_terminations,
)

__all__ = [
    "FUSED_TERMINATION_KEY",
    "command_config",
    "contact_sensor_descriptor",
    "model_field_dr_descriptor",
    "onnx_ref",
    "policy_native_sizes",
    "raycast_sensor_descriptor",
    "serialize_actions",
    "serialize_command",
    "serialize_event",
    "serialize_events",
    "serialize_observation_group",
    "serialize_observation_term",
    "serialize_termination",
    "serialize_terminations",
    "stamp_provenance",
    "write_command_artifact",
    "write_onnx",
]
