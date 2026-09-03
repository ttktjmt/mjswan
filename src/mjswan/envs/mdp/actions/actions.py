"""Action term configurations for mjswan.

Mirrors ``mjlab.envs.mdp.actions.actions`` class hierarchy.  In mjswan
these configuration objects are **not** built into runtime ``ActionTerm``
instances — instead they serialize to JSON config entries consumed by the
browser-side ``runtime.ts``.

Usage (identical to mjlab)::

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
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mujoco


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


@dataclass(kw_only=True)
class BaseActionCfg(ActionTermCfg):
    """Configuration for actions that control actuator transmissions.

    Mirrors ``mjlab.envs.mdp.actions.actions.BaseActionCfg``.
    """

    actuator_names: tuple[str, ...] | list[str] = (".*",)
    """Actuator names (regex patterns) to control."""

    scale: float | list[float] | dict[str, float] = 1.0
    """Action scale applied to raw policy output."""

    offset: float | list[float] | dict[str, float] = 0.0
    """Action offset added after scaling."""

    preserve_order: bool = False
    """Accepted for mjlab compatibility; ignored in mjswan."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        self._add_clip(entry)
        return entry


@dataclass(kw_only=True)
class JointPositionActionCfg(BaseActionCfg):
    """Configuration for joint position control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointPositionActionCfg``.

    ``stiffness`` and ``damping`` are mjswan-specific fields for PD control
    in the browser runtime.  In mjlab these are actuator model properties,
    but mjswan needs them in the policy config because the TS runtime
    computes PD externally for motor actuators (biastype=none).
    """

    use_default_offset: bool = True
    """When True, action=0 commands the default joint pose."""

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp) for PD control.  Scalar, per-joint list, or dict
    mapping joint names (must match ``policy_joint_names``) to values.
    Only used by the TS runtime for motor actuators with external PD."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd) for PD control.  Scalar, per-joint list, or dict
    mapping joint names (must match ``policy_joint_names``) to values.
    Only used by the TS runtime for motor actuators with external PD."""

    ema_alpha: float | None = None
    """Exponential-moving-average factor on the processed target, in ``(0, 1]``.

    ``target = alpha * processed + (1 - alpha) * previous_target``, advanced once per
    control step — not per physics substep. ``None`` (or ``1.0``) is no smoothing."""

    warmup_time_s: float | None = None
    """Seconds at the start of an episode during which the default pose is held.

    Rounded up to whole control steps, matching mjlab's
    ``episode_length_buf * step_dt < warmup_time_s``. ``None`` is no warmup."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)
        if self.ema_alpha is not None and not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.warmup_time_s is not None and self.warmup_time_s < 0.0:
            raise ValueError(
                f"warmup_time_s must be non-negative, got {self.warmup_time_s}"
            )

        entry: dict[str, Any] = {"type": "joint_position"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        entry["use_default_offset"] = self.use_default_offset
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        if self.ema_alpha is not None and self.ema_alpha != 1.0:
            entry["ema_alpha"] = self.ema_alpha
        if self.warmup_time_s:
            entry["warmup_time_s"] = self.warmup_time_s
        self._add_clip(entry)
        return entry


@dataclass(kw_only=True)
class ReferenceJointPositionActionCfg(BaseActionCfg):
    """Joint position targets as a motion reference plus a scaled residual.

    ``q_cmd = q_ref(t) + scale * a - encoder_bias``. Identical to
    :class:`JointPositionActionCfg` except that the offset is the tracking command's
    reference pose for the current step, not the constant default pose.

    The policy must own a command publishing ``ref_joint_pos`` — ``TrackingCommand`` does.
    """

    command_name: str = "motion"
    """Command term supplying the reference joint positions."""

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp). mjswan-specific; see :class:`JointPositionActionCfg`."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd). mjswan-specific; see :class:`JointPositionActionCfg`."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "joint_position_reference"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            raise ValueError(
                "ReferenceJointPositionActionCfg takes no `offset`: the reference "
                "joint positions are the offset, and they come from the command."
            )
        entry["actuator_names"] = list(self.actuator_names)
        entry["command_name"] = self.command_name
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        self._add_clip(entry)
        return entry


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
    """Configuration for joint velocity control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointVelocityActionCfg``.

    .. note::
        Not supported in mjswan. Accepted for API compatibility.
    """

    use_default_offset: bool = True

    unsupported_reason: str | None = field(
        default=(
            "JointVelocityAction is not supported in mjswan: the browser "
            "runtime only supports joint_position and torque control types."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class JointEffortActionCfg(BaseActionCfg):
    """Configuration for joint effort (torque) control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointEffortActionCfg``.
    """

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp).  mjswan-specific; see ``JointPositionActionCfg``."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd).  mjswan-specific; see ``JointPositionActionCfg``."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "torque"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        self._add_clip(entry)
        return entry


@dataclass(kw_only=True)
class MuscleActivationActionCfg(BaseActionCfg):
    """MyoSuite-style muscle activation control.

    Writes excitation values to ``mjData.ctrl`` for the named MuJoCo muscle
    actuators (``dyntype=muscle``). Both modes apply ``raw = scale * a + offset``
    first, then:

    - ``normalize=True`` (default): applies the canonical MyoSuite sigmoid
      ``σ(5 * (raw - 0.5))`` to produce excitation in ``(0, 1)``.
    - ``normalize=False``: clips ``raw`` to ``[0, 1]`` for models that already
      output excitation in that range.

    Semantics mirror myosuite4 ``MuscleActionTermCfg.normalize``; see
    ``docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md``.
    """

    normalize: bool = True
    """Apply the MyoSuite sigmoid mapping. When False, clip ``raw`` to [0, 1]."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "muscle_activation"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        if not self.normalize:
            entry["normalize"] = False
        entry["actuator_names"] = list(self.actuator_names)
        self._add_clip(entry)
        return entry


def validate_muscle_actuators(
    model: "mujoco.MjModel",
    cfg: MuscleActivationActionCfg,
    term_name: str = "",
) -> list[int]:
    """Validate that ``cfg.actuator_names`` resolves to muscle actuators only.

    Patterns are full-match regexes (each wrapped as ``^(?:pattern)$``), mirroring
    the TS runtime's ``getCtrlMappingByActuatorNames``. Raises ``ValueError`` if
    any pattern matches no actuator or if any matched actuator has a non-muscle
    ``actuator_dyntype``. Returns the resolved actuator ids in original order
    (deduplicated).
    """
    import mujoco

    prefix = (
        f"MuscleActivationActionCfg {term_name!r}: "
        if term_name
        else "MuscleActivationActionCfg: "
    )
    if not cfg.actuator_names:
        raise ValueError(f"{prefix}actuator_names is empty")

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]

    matched: list[int] = []
    for pattern in cfg.actuator_names:
        regex = re.compile(f"^(?:{pattern})$")
        ids_for_pattern = [
            i for i, n in enumerate(actuator_names) if n is not None and regex.match(n)
        ]
        if not ids_for_pattern:
            available = [n for n in actuator_names if n]
            raise ValueError(
                f"{prefix}actuator_names pattern {pattern!r} matched no actuator. "
                f"Available actuators: {available}"
            )
        matched.extend(ids_for_pattern)

    seen: set[int] = set()
    unique_ids: list[int] = []
    for i in matched:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)

    muscle_dyn = int(mujoco.mjtDyn.mjDYN_MUSCLE)
    violations: list[tuple[str, int]] = []
    for i in unique_ids:
        dt = int(model.actuator_dyntype[i])
        if dt != muscle_dyn:
            violations.append((actuator_names[i] or f"#{i}", dt))
    if violations:
        details = ", ".join(f"{name!r} (dyntype={dt})" for name, dt in violations)
        raise ValueError(
            f"{prefix}actuators are not muscle dyntype: {details}. "
            f"Expected dyntype={muscle_dyn} (mjDYN_MUSCLE)."
        )

    return unique_ids


# ---------------------------------------------------------------------------
# Tendon actions (stubs — not supported in browser runtime)
# ---------------------------------------------------------------------------

_TENDON_UNSUPPORTED = (
    "Tendon actions are not supported in mjswan: the browser runtime does "
    "not expose tendon control APIs."
)


@dataclass(kw_only=True)
class TendonLengthActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class TendonVelocityActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


@dataclass(kw_only=True)
class TendonEffortActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(default=_TENDON_UNSUPPORTED)

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


# ---------------------------------------------------------------------------
# Site actions (stub — not supported in browser runtime)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SiteEffortActionCfg(BaseActionCfg):
    """Stub for mjlab compatibility. Not supported in mjswan."""

    unsupported_reason: str | None = field(
        default=(
            "SiteEffortAction is not supported in mjswan: the browser runtime "
            "does not expose site force/torque APIs."
        )
    )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(self.unsupported_reason)


__all__ = [
    "ActionTermCfg",
    "BaseActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "JointEffortActionCfg",
    "MuscleActivationActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
    "validate_muscle_actuators",
]
