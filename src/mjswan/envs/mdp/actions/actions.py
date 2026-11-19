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

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mjswan.managers.action_manager import ActionBinding, ActionTermCfg

if TYPE_CHECKING:
    import mujoco


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
    """Accepted for mjlab compatibility; ignored, as mjlab 1.6.0 ignores it for joint
    actions: targets follow the model's joint order."""

    delay: list[dict[str, Any]] | None = None
    """mjswan-specific: mjlab's actuator command delays (``ActuatorCfg.delay_*``) on this
    term's joints, one entry per mjlab delay buffer: ``{"joints", "min_lag", "max_lag",
    "hold_prob", "update_period", "per_env_phase"}``. Resolved from the actuator configs
    when unset."""

    saturation_effort: float | list[float] | dict[str, float] | None = None
    """mjswan-specific: mjlab's ``DcMotorActuatorCfg.saturation_effort`` on this term's
    joints. With ``velocity_limit``, the browser clips the motor torque it computes as
    mjlab's ``dc_motor_clip`` does. Resolved from the actuator configs when unset."""

    velocity_limit: float | list[float] | dict[str, float] | None = None
    """mjswan-specific: mjlab's ``DcMotorActuatorCfg.velocity_limit``. See
    ``saturation_effort``."""

    def _add_delay(self, entry: dict[str, Any]) -> None:
        if self.delay:
            entry["delay"] = self.delay

    def _add_torque_curve(self, entry: dict[str, Any]) -> None:
        if self.saturation_effort is not None and self.velocity_limit is not None:
            entry["saturation_effort"] = self.saturation_effort
            entry["velocity_limit"] = self.velocity_limit

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
        self._add_delay(entry)
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

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

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
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
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
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
        return entry


@dataclass(kw_only=True)
class CommandJointPositionActionCfg(BaseActionCfg):
    """Joint targets: ``scale * c + offset``, with ``c`` from ``command_name``.

    One command value per matched actuated joint, in joint order; no policy output.
    """

    command_name: str

    stiffness: float | list[float] | dict[str, float] | None = None
    """Motor PD gain (kp): a scalar, a list in column order, or a dict by joint name."""

    damping: float | list[float] | dict[str, float] | None = None
    """Motor PD gain (kd): a scalar, a list in column order, or a dict by joint name."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "joint_position_command"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        entry["command_name"] = self.command_name
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
        return entry


@dataclass(kw_only=True)
class RelativeJointPositionActionCfg(BaseActionCfg):
    """Joint targets relative to the measured position: ``q + scale * a``.

    Mirrors ``mjlab.envs.mdp.actions.actions.RelativeJointPositionActionCfg``. The
    browser reads ``q`` every substep, as mjlab's ``apply_actions`` does; ``clip``
    bounds the delta, and there is no offset and no encoder bias.
    """

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp). mjswan-specific; see :class:`JointPositionActionCfg`."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd). mjswan-specific; see :class:`JointPositionActionCfg`."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)
        if self.offset != 0.0:
            raise ValueError(
                "RelativeJointPositionActionCfg takes no `offset`: the target is the "
                "measured position plus the scaled action, as in mjlab."
            )

        entry: dict[str, Any] = {"type": "joint_position_relative"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        entry["actuator_names"] = list(self.actuator_names)
        if self.stiffness is not None:
            entry["stiffness"] = self.stiffness
        if self.damping is not None:
            entry["damping"] = self.damping
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
        return entry


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
    """Configuration for joint velocity control.

    Mirrors ``mjlab.envs.mdp.actions.actions.JointVelocityActionCfg``. The target
    ``scale * a + offset`` is ``ctrl`` for a velocity actuator (``biastype=affine``);
    for a motor the browser applies ``damping * (target - qvel)``.
    """

    use_default_offset: bool = True
    """When True, the build sets ``offset`` to the entity's default joint velocity."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd). mjswan-specific; see ``JointPositionActionCfg``."""

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "joint_velocity"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        entry["actuator_names"] = list(self.actuator_names)
        if self.damping is not None:
            entry["damping"] = self.damping
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
        return entry


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
        self._add_torque_curve(entry)
        self._add_clip(entry)
        self._add_delay(entry)
        return entry


@dataclass(kw_only=True)
class TracedActionCfg(BaseActionCfg):
    """A task's own mjlab action term, traced at build time (ADR 0009).

    The adapter makes one for a task's own action class mjswan has no class of its own
    for: its ``process_actions`` and ``reset`` become graphs, and its ``apply_actions`` is
    the native kind and options its :class:`~mjswan.ActionBinding` names, or its mjlab
    base's. ``actuator_names`` are the mjlab cfg's, entity-prefixed, for resolving the
    gains and delays of its joints; the build writes the exact joints the term drives.
    """

    mjlab_cfg: Any = None
    """The task's mjlab action cfg, built on the trace env."""

    binding: ActionBinding | None = None

    stiffness: float | list[float] | dict[str, float] | None = None
    """Position gain (kp). mjswan-specific; see :class:`JointPositionActionCfg`."""

    damping: float | list[float] | dict[str, float] | None = None
    """Velocity gain (kd). mjswan-specific; see :class:`JointPositionActionCfg`."""

    def to_dict(self) -> dict[str, Any]:
        raise TypeError(
            f"TracedActionCfg({type(self.mjlab_cfg).__name__}) is traced at build time; "
            "use mjswan.build.mdp.serialize_actions(..., env=...) instead of to_dict() "
            "directly (the Builder does this automatically)."
        )


MuscleActionMode = Literal["sigmoid", "direct", "excitation"]
"""How a muscle term maps a raw action to a control; see
:class:`MuscleActivationActionCfg`."""


@dataclass(kw_only=True)
class MuscleActivationActionCfg(BaseActionCfg):
    """MyoSuite-style muscle activation control.

    Writes excitation values to ``mjData.ctrl`` for the named MuJoCo muscle
    actuators (``dyntype=muscle``). Every mode applies ``raw = scale * a + offset``
    first, then:

    - ``"sigmoid"`` (default): the canonical MyoSuite sigmoid
      ``σ(5 * (raw - 0.5))``, producing excitation in ``(0, 1)``.
    - ``"excitation"``: clips ``raw`` to ``[0, 1]`` for models that already output
      excitation in that range.
    - ``"direct"``: clips ``raw`` to each actuator's own ``ctrlrange`` and writes it
      unchanged, for a policy trained against the raw control. A muscle model whose
      ``ctrlrange`` is ``[-1, 1]`` really does use the negative half, so this is not
      ``"excitation"`` by another name.

    Semantics mirror myosuite4 ``MyoMuscleActivationActionCfg.action_mode``; see
    ``docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md``.
    """

    normalize: bool = True
    """Legacy spelling of ``action_mode``: True is ``"sigmoid"``, False ``"excitation"``.
    Ignored when ``action_mode`` is set."""

    action_mode: MuscleActionMode | None = None
    """Named after mjlab/myosuite's own field, so the adapter copies it verbatim."""

    @property
    def mode(self) -> MuscleActionMode:
        """The mapping this term applies, with ``normalize`` as the fallback."""
        if self.action_mode is not None:
            if self.action_mode not in ("sigmoid", "direct", "excitation"):
                raise ValueError(
                    f"MuscleActivationActionCfg.action_mode must be 'sigmoid', "
                    f"'direct' or 'excitation', got {self.action_mode!r}."
                )
            return self.action_mode
        return "sigmoid" if self.normalize else "excitation"

    def to_dict(self) -> dict[str, Any]:
        if self.unsupported_reason is not None:
            raise NotImplementedError(self.unsupported_reason)

        entry: dict[str, Any] = {"type": "muscle_activation"}
        if self.scale != 1.0:
            entry["scale"] = self.scale
        if self.offset != 0.0:
            entry["offset"] = self.offset
        if self.mode != "sigmoid":
            entry["mode"] = self.mode
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
    "CommandJointPositionActionCfg",
    "JointPositionActionCfg",
    "JointVelocityActionCfg",
    "JointEffortActionCfg",
    "MuscleActivationActionCfg",
    "RelativeJointPositionActionCfg",
    "TracedActionCfg",
    "TendonLengthActionCfg",
    "TendonVelocityActionCfg",
    "TendonEffortActionCfg",
    "SiteEffortActionCfg",
    "validate_muscle_actuators",
]
