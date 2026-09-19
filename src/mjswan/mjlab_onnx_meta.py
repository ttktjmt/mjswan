"""Read the metadata mjlab bakes into an exported policy's ``.onnx``.

mjlab's ``attach_metadata_to_onnx`` writes the joint names, the rest pose, the action
scale and a description of every observation term into the file's ``metadata_props``, so
an mjlab policy travels self-describing. This module parses that block back out. It
needs neither mjlab nor torch installed — the encoding is plain strings — which is what
lets a Hub-fetched ONNX be added on the light path.

**The encoding is lossy.** mjlab formats every number in a list with ``{:.3f}``, so the
values read back are rounded to three decimals, and ints and bools arrive as ``"1.000"``
/ ``"0.000"`` (``isinstance(True, int)`` is true in Python). A scalar written outside a
list keeps its full precision, since only lists go through the formatter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import onnx

if TYPE_CHECKING:
    from .envs.mdp.actions.actions import ActionTermCfg

#: Keys every mjlab export carries (``mjlab.rl.exporter_utils.get_base_metadata``).
#: Both must be present for :func:`read_mjlab_metadata` to claim a model as mjlab's.
_REQUIRED_KEYS = ("joint_names", "default_joint_pos")

_DELIM = ","
_SUB_DELIM = ";"


def _split(value: str) -> list[str]:
    """mjlab's top-level CSV, with the empty string reading as the empty list."""
    return value.split(_DELIM) if value else []


def _floats(value: str) -> list[float]:
    return [float(item) for item in _split(value)]


def _ints(value: str) -> list[int]:
    # Written through the float formatter, so "3" arrives as "3.000".
    return [int(float(item)) for item in _split(value)]


def _bools(value: str) -> list[bool]:
    return [bool(int(float(item))) for item in _split(value)]


def _scalar_or_vector(item: str) -> float | list[float]:
    """One CSV entry: a plain number, or a per-dimension vector joined with ``;``."""
    if _SUB_DELIM in item:
        return [float(part) for part in item.split(_SUB_DELIM)]
    return float(item)


def _pairs(value: str) -> list[tuple[float, float]]:
    """``min;max`` entries. ``inf`` and ``-inf`` survive the round trip as written."""
    out: list[tuple[float, float]] = []
    for item in _split(value):
        parts = item.split(_SUB_DELIM)
        if len(parts) != 2:
            raise ValueError(f"Expected a 'min;max' clip range, got {item!r}.")
        out.append((float(parts[0]), float(parts[1])))
    return out


@dataclass(frozen=True)
class MjlabPolicyMetadata:
    """What mjlab wrote into one exported policy.

    ``joint_names`` and the two gain lists cover **every** joint of the robot, actuated
    or not, while the network drives only the actuated ones — see
    :func:`joint_mapping_usable` before mapping one onto the other.
    """

    joint_names: list[str]
    """Every joint of the robot, in the model's own order."""

    default_joint_pos: list[float]
    """The rest pose, aligned with :attr:`joint_names`. Rounded to three decimals."""

    joint_stiffness: list[float] = field(default_factory=list)
    """Position gain per joint. Informational: mjswan resolves gains from the model."""

    joint_damping: list[float] = field(default_factory=list)
    """Velocity gain per joint. Informational, like :attr:`joint_stiffness`."""

    action_scale: float | list[float] | None = None
    """The joint-position action term's scale — a scalar, or one value per action."""

    command_names: list[str] = field(default_factory=list)
    """The task's active command terms, by name."""

    observation_names: list[str] = field(default_factory=list)
    """The ``actor`` group's terms, in order. Names only: the functions behind them are
    not in the file, which is why an observation group cannot be rebuilt from this."""

    observation_terms_scale: list[float | list[float]] = field(default_factory=list)
    """Per term, aligned with :attr:`observation_names`."""

    observation_terms_clip: list[tuple[float, float]] = field(default_factory=list)
    """Per term. An unclipped term reads as ``(-inf, inf)``."""

    observation_terms_history_length: list[int] = field(default_factory=list)
    """Per term."""

    observation_terms_flatten_history_dim: list[bool] = field(default_factory=list)
    """Per term."""

    anchor_body_name: str | None = None
    """Tracking tasks only: the body the reference motion is anchored to."""

    body_names: list[str] = field(default_factory=list)
    """Tracking tasks only: the bodies the reference motion covers."""

    run_path: str | None = None
    """Where the export came from — a W&B run name, or ``"local"``."""

    raw: dict[str, str] = field(default_factory=dict)
    """Every ``metadata_props`` entry, unparsed, including keys this class has no field
    for (a task may add its own)."""


def read_mjlab_metadata(model: onnx.ModelProto) -> MjlabPolicyMetadata | None:
    """Parse an mjlab export's ``metadata_props``, or ``None`` if it has none.

    ``None`` means "not an mjlab export" — a hand-built graph, or one from a framework
    that writes no metadata — which is a normal case, not an error.
    """
    raw = {entry.key: entry.value for entry in model.metadata_props}
    if not all(key in raw for key in _REQUIRED_KEYS):
        return None

    action_scale_raw = raw.get("action_scale")
    action_scale: float | list[float] | None = None
    if action_scale_raw:
        # A list went through the CSV formatter; a scalar came straight from `str()`.
        action_scale = (
            _floats(action_scale_raw)
            if _DELIM in action_scale_raw
            else float(action_scale_raw)
        )

    return MjlabPolicyMetadata(
        joint_names=_split(raw["joint_names"]),
        default_joint_pos=_floats(raw["default_joint_pos"]),
        joint_stiffness=_floats(raw.get("joint_stiffness", "")),
        joint_damping=_floats(raw.get("joint_damping", "")),
        action_scale=action_scale,
        command_names=_split(raw.get("command_names", "")),
        observation_names=_split(raw.get("observation_names", "")),
        observation_terms_scale=[
            _scalar_or_vector(item)
            for item in _split(raw.get("observation_terms_scale", ""))
        ],
        observation_terms_clip=_pairs(raw.get("observation_terms_clip", "")),
        observation_terms_history_length=_ints(
            raw.get("observation_terms_history_length", "")
        ),
        observation_terms_flatten_history_dim=_bools(
            raw.get("observation_terms_flatten_history_dim", "")
        ),
        anchor_body_name=raw.get("anchor_body_name") or None,
        body_names=_split(raw.get("body_names", "")),
        run_path=raw.get("run_path") or None,
        raw=raw,
    )


def joint_mapping_usable(
    meta: MjlabPolicyMetadata, *, action_width: int | None = None
) -> bool:
    """Whether :attr:`~MjlabPolicyMetadata.joint_names` names what the network drives.

    mjlab writes every joint of the robot, the network outputs one action per *actuated*
    joint, and the two coincide only when the robot has no passive joints. Feeding a
    longer list to the runtime as ``policy_joint_names`` would shift every action onto
    the wrong actuator, and nothing at playback would say so — so the counts have to
    agree before the mapping is used.

    ``action_width`` unknown (a graph with a dynamic output shape) leaves only the
    internal consistency of the metadata to check.
    """
    if not meta.joint_names or not meta.default_joint_pos:
        return False
    if len(meta.joint_names) != len(meta.default_joint_pos):
        return False
    if action_width is not None and len(meta.joint_names) != action_width:
        return False
    if isinstance(meta.action_scale, list) and len(meta.action_scale) != len(
        meta.joint_names
    ):
        return False
    return True


def policy_kwargs_from_metadata(
    meta: MjlabPolicyMetadata, *, action_width: int | None = None
) -> dict[str, Any]:
    """The :meth:`~mjswan.scene.SceneHandle.add_policy` arguments this metadata fixes.

    ``{}`` when :func:`joint_mapping_usable` says the joint list is not the action list.
    Only the per-policy half is returned; the action term is
    :func:`action_cfg_from_metadata`, since that belongs to the MDP rather than the
    checkpoint.
    """
    if not joint_mapping_usable(meta, action_width=action_width):
        return {}
    return {
        "policy_joint_names": list(meta.joint_names),
        "default_joint_pos": list(meta.default_joint_pos),
    }


def action_cfg_from_metadata(
    meta: MjlabPolicyMetadata, *, action_width: int | None = None
) -> dict[str, ActionTermCfg]:
    """The joint-position action term this metadata describes, or ``{}``.

    mjlab's ``JointPositionAction`` adds the rest pose to the scaled action, which is
    mjswan's ``use_default_offset=True``. Guarded like
    :func:`policy_kwargs_from_metadata`, and empty when the export recorded no scale.
    """
    if meta.action_scale is None:
        return {}
    if not joint_mapping_usable(meta, action_width=action_width):
        return {}
    from .envs.mdp.actions import JointPositionActionCfg

    scale = (
        list(meta.action_scale)
        if isinstance(meta.action_scale, list)
        else meta.action_scale
    )
    return {
        "joint_pos": JointPositionActionCfg(scale=scale, use_default_offset=True),
    }


__all__ = [
    "MjlabPolicyMetadata",
    "action_cfg_from_metadata",
    "joint_mapping_usable",
    "policy_kwargs_from_metadata",
    "read_mjlab_metadata",
]
