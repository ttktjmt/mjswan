"""Policy configuration and management.

This module defines the PolicyConfig dataclass and PolicyHandle class for
ONNX policy configuration and command management.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco

from .document.manifest import DEFAULT_IN_KEYS, DEFAULT_OUT_KEYS, RUNTIME_INPUT_SLOTS
from .managers.command_manager import CommandTermConfig
from .mdp import MdpConfig
from .motion import MotionConfig, MotionHandle

if TYPE_CHECKING:
    import onnx

    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .scene import SceneHandle


def onnx_io_names(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    """The network's real input and output names, initializers excluded."""
    initializers = {init.name for init in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    return inputs, [o.name for o in model.graph.output]


def check_slot_tables(
    name: str,
    model: onnx.ModelProto,
    in_keys: Sequence[str] | None,
    out_keys: Sequence[str | Sequence[str]] | None,
) -> tuple[list[str] | None, list[str | list[str]] | None]:
    """Check a policy's slot tables against its network and return them as lists.

    ``in_keys[i]`` fills the *i*-th input and ``out_keys[i]`` names the *i*-th output, so
    each table must be exactly as long as what it indexes. A network with several inputs
    must declare ``in_keys``: nothing else records where the runtime-synthesized tensors
    sit relative to the observation groups (ADR 0006 §5). Several *outputs* cannot be
    refused the same way, since which one is the action is unknowable here, so the
    default (the first) is announced instead.
    """
    inputs, outputs = onnx_io_names(model)
    if in_keys is None:
        if len(inputs) > 1:
            raise ValueError(
                f"Policy {name!r} has {len(inputs)} ONNX inputs ({inputs}) but declares "
                "no in_keys. Pass in_keys naming, per input in order, the observation "
                "group or runtime tensor (is_init, adapt_hx, time_step) that fills it."
            )
        checked_in = None
    else:
        checked_in = [str(k) for k in in_keys]
        if len(checked_in) != len(inputs):
            raise ValueError(
                f"Policy {name!r} declares {len(checked_in)} in_keys {checked_in} but its "
                f"ONNX has {len(inputs)} inputs ({inputs}). in_keys[i] fills the i-th "
                "input, so the two must have the same length."
            )
    if out_keys is None:
        if len(outputs) > 1:
            warnings.warn(
                f"Policy {name!r} has {len(outputs)} ONNX outputs ({outputs}) but "
                f"declares no out_keys, so the runtime drives the actuators from the "
                f"first one, {outputs[0]!r}. If the action is a different output, pass "
                "out_keys naming each output in order (ADR 0006 §5).",
                category=RuntimeWarning,
                stacklevel=3,
            )
        checked_out: list[str | list[str]] | None = None
    else:
        checked_out = [
            k if isinstance(k, str) else [str(p) for p in k] for k in out_keys
        ]
        if len(checked_out) != len(outputs):
            raise ValueError(
                f"Policy {name!r} declares {len(checked_out)} out_keys but its ONNX has "
                f"{len(outputs)} outputs ({outputs}). out_keys[i] names the i-th output, "
                "so the two must have the same length."
            )
    return checked_in, checked_out


def onnx_output_width(model: onnx.ModelProto) -> int | None:
    """The last dim of the graph's first output, or ``None`` when it is not static."""
    if not model.graph.output:
        return None
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(dims) < 2:
        return None
    width = dims[-1].dim_value
    return int(width) if width > 0 else None


def actuated_joint_names(model: mujoco.MjModel | None) -> list[str] | None:
    """The joint each actuator drives, in actuator order.

    ``None`` when the model does not give one unambiguously: no actuators, a
    transmission that is not a joint (tendon, site, body), an unnamed joint, or two
    actuators on the same joint. A wrong answer would be silent at playback.
    """
    if model is None or model.nu == 0:
        return None
    names: list[str] = []
    for index in range(model.nu):
        if int(model.actuator_trntype[index]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            return None
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0])
        )
        if not name:
            return None
        names.append(name)
    return names if len(set(names)) == len(names) else None


def actuated_joints_in_joint_order(model: mujoco.MjModel | None) -> list[str] | None:
    """Every joint an actuator drives, in the model's own **joint** order.

    mjlab resolves an action term through ``Entity.find_joints_by_actuator_names``,
    which narrows ``joint_names`` to the actuated ones and keeps their order, so this is
    the order actions come out in.

    ``None`` on the same terms as :func:`actuated_joint_names`, which decides them.
    """
    if actuated_joint_names(model) is None or model is None:
        return None
    driven = {int(model.actuator_trnid[index, 0]) for index in range(model.nu)}
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        for joint in range(model.njnt)
        if joint in driven
    ]
    return None if any(not name for name in names) else names


def drives_joints(actions: Mapping[str, Any] | None) -> bool:
    """Whether any action term reaches its actuator through a joint.

    A muscle term names actuators directly and needs no ``policy_joint_names``; every
    other kind the browser looks up by joint name, so it needs them or it drives nothing.
    """
    return any(
        type(term).__name__ != "MuscleActivationActionCfg"
        for term in (actions or {}).values()
    )


def action_term_joint_names(
    actions: Mapping[str, Any] | None, model: mujoco.MjModel | None
) -> list[str] | None:
    """The joints an adapted action-term set drives, in the order the actions come out.

    What an mjlab export's ``joint_names`` would say, recovered from the action terms.
    ``actuator_names`` holds *joint* patterns despite its name (ADR 0006;
    ``JointPositionAction`` matches them against joint names), so each term's patterns
    are matched against the actuated joints in joint order, term by term.

    ``None`` unless every term answers: a muscle term names actuators rather than joints,
    a pattern may match nothing, and two terms may claim one joint.
    """
    joints = actuated_joints_in_joint_order(model)
    if not actions or joints is None:
        return None
    names: list[str] = []
    for term in actions.values():
        if type(term).__name__ == "MuscleActivationActionCfg":
            return None
        patterns = getattr(term, "actuator_names", None)
        if not patterns:
            return None
        try:
            regexes = [re.compile(f"(?:{pattern})") for pattern in patterns]
        except re.error:
            return None
        matched = [name for name in joints if any(r.fullmatch(name) for r in regexes)]
        if not matched:
            return None
        names.extend(matched)
    return names if len(set(names)) == len(names) else None


@dataclass
class PolicyConfig:
    """Configuration for an ONNX policy."""

    name: str
    """Name of the policy."""

    model: onnx.ModelProto
    """ONNX model for the policy."""

    id: str = ""
    """Sanitized name, unique within the scene: the ``.onnx`` file's stem and the
    ``?policy=`` value (ADR 0006 §4). Assigned by :meth:`~mjswan.scene.SceneHandle.add_policy`;
    defaults to ``name2id(name)``."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the policy."""

    source_path: str | None = None
    """Optional source path for the policy ONNX file."""

    config_path: str | None = None
    """Optional source path for the policy config JSON file."""

    mdp: MdpConfig = field(default_factory=MdpConfig)
    """The MDP this policy runs against: its observations, actions, terminations,
    commands and events as one unit (ADR 0006 §3). Two policies given the same object
    share one MDP: one set of traced graphs, one ``mdp/<id>/`` directory. The five
    read-only properties below are views onto it."""

    policy_joint_names: list[str] | None = None
    """Ordered list of joint names controlled by the policy.

    Required by the browser-side ``PolicyRunner`` to map policy outputs to
    the correct actuators in the MuJoCo model.  When set, serialized as
    ``policy_joint_names`` at the top level of the policy JSON config.
    """

    policy_num_actions: int | None = None
    """Explicit number of policy output actions.

    Use for policies (e.g. muscle-driven) where ``policy_joint_names`` is
    empty and the output size cannot be inferred from joint names.  When set,
    serialized as ``policy_num_actions`` in the policy JSON and used by the
    TS ``PolicyRunner`` instead of ``policy_joint_names.length``.
    """

    default_joint_pos: list[float] | None = None
    """Default joint positions corresponding to ``policy_joint_names``.

    Used by the browser runtime when ``use_default_offset=True``: action=0
    commands this pose.  Must be in the same order as ``policy_joint_names``.
    """

    encoder_bias: list[float] | None = None
    """Per-joint encoder bias corresponding to ``policy_joint_names``.

    Used by the browser runtime to mirror mjlab's joint-position action path:
    the final target written to actuators is ``processed_action - encoder_bias``.
    """

    in_keys: list[str] | None = None
    """The ONNX input slot table: ``in_keys[i]`` names the tensor that fills the
    network's *i*-th input, an observation group or one the runtime synthesizes
    (:data:`RUNTIME_INPUT_SLOTS`). The mapping is positional, so the network's own input
    names never matter. ``None`` for a single-input policy, whose one input takes its one
    observation group; required beyond one input (ADR 0006 §5)."""

    out_keys: list[str | list[str]] | None = None
    """The ONNX output slot table, positional like ``in_keys``: ``out_keys[i]`` names the
    network's *i*-th output. The runtime reads ``action`` and, for a recurrent policy,
    the ``["next", "adapt_hx"]`` carry; the rest are labels. ``None`` means
    ``["action"]``."""

    clip_actions: float | None = None
    """Symmetric bound the raw policy output is clamped to, or ``None`` for unbounded.

    rsl-rl's ``RslRlVecEnvWrapper.step`` clamps to ``[-clip_actions, +clip_actions]``
    *before* ``env.step``, so the action manager — and therefore any ``last_action``
    observation — sees the clamped vector. The browser mirrors that placement: the clamp
    lands on the ONNX output before the action terms or the ``prev_action`` slot read it.

    Not ``ActionTermCfg.clip``, which bounds ``raw * scale + offset`` per target and
    lives on the action term.
    """

    initial_qpos: list[float] | None = None
    """Optional initial qpos samples or defaults for runtime reset logic."""

    initial_qvel: list[float] | None = None
    """Optional initial qvel samples or defaults for runtime reset logic."""

    extras: dict[str, Any] | None = None
    """Optional extra policy config payload serialized verbatim into JSON."""

    motions: list[MotionConfig] = field(default_factory=list)
    """Reference motions available for this policy."""

    default: bool = False
    """Whether this policy should be the initially selected one in the viewer.

    At most one policy in a scene may set it; when none does, the first added wins.
    """

    def __post_init__(self) -> None:
        if not self.id:
            from .document.ids import name2id

            self.id = name2id(self.name)

    # Views onto `mdp`, so a reader that wants one term set need not know where it lives.

    @property
    def observations(
        self,
    ) -> dict[str, ObservationGroupCfg] | Mapping[str, Any] | Any | None:
        """The MDP's observation groups, keyed by the name the policy's slot table uses."""
        return self.mdp.observations

    @property
    def actions(self) -> Mapping[str, ActionTermCfg] | None:
        """The MDP's action terms, keyed by term name."""
        return self.mdp.actions

    @property
    def terminations(self) -> dict[str, TerminationTermCfg] | None:
        """The MDP's termination terms, keyed by term name."""
        return self.mdp.terminations

    @property
    def commands(self) -> dict[str, CommandTermConfig] | Mapping[str, Any]:
        """The MDP's command terms, keyed by their policy-visible names (``{}`` if none)."""
        return self.mdp.commands or {}

    @property
    def events(self) -> dict[str, EventTermCfg] | Mapping[str, Any] | None:
        """The MDP's event terms, keyed by name."""
        return self.mdp.events


class PolicyHandle:
    """Handle for configuring a policy and its commands.

    Commands should be passed via the ``commands=`` parameter of
    :meth:`~mjswan.scene.SceneHandle.add_policy`.

    Example:
        policy = scene.add_policy(
            policy=onnx.load("locomotion.onnx"),
            name="Locomotion",
            config_path="locomotion.json",
            commands={"velocity": mjswan.velocity_command()},
        )
    """

    def __init__(self, policy_config: PolicyConfig, scene: SceneHandle) -> None:
        self._config = policy_config
        self._scene = scene

    @property
    def name(self) -> str:
        """Name of the policy."""
        return self._config.name

    @property
    def model(self) -> onnx.ModelProto:
        """ONNX model for the policy."""
        return self._config.model

    def set_metadata(self, key: str, value: Any) -> PolicyHandle:
        """Set metadata for this policy.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self

    def _append_motion(self, motion: MotionConfig) -> MotionHandle:
        if motion.default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)

    def add_motion(
        self,
        *,
        name: str,
        source: str,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Add a bundled ``.npz`` reference motion to this policy."""
        motion = MotionConfig(
            name=name,
            source=source,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        return self._append_motion(motion)

    def add_motion_wandb(
        self,
        *,
        name: str | None = None,
        run_path: str | None = None,
        run_id: str | None = None,
        entity: str | None = None,
        project: str | None = None,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Download a motion artifact from W&B and attach it to this policy."""
        from . import source

        resolved_run_path = source.wandb.resolve_run_path(
            run_path=run_path,
            run_id=run_id,
            entity=entity,
            project=project,
        )
        motion_name, payload = source.wandb.fetch_motion_npz(resolved_run_path)
        motion = MotionConfig(
            name=name or motion_name,
            data=payload,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        if default:
            for existing in self._config.motions:
                existing.default = False
        self._config.motions.append(motion)
        return MotionHandle(motion, self)

    def add_motion_hf(
        self,
        repo_id: str,
        filename: str,
        *,
        name: str | None = None,
        revision: str | None = None,
        repo_type: str = "dataset",
        token: str | None = None,
        fps: float = 50.0,
        anchor_body_name: str,
        body_names: tuple[str, ...] | list[str],
        dataset_joint_names: list[str] | None = None,
        default: bool = False,
        loop: bool = True,
    ) -> MotionHandle:
        """Download a ``.npz`` reference motion from the Hugging Face Hub.

        The Hub counterpart of :meth:`add_motion_wandb`, with the clip named by path.

        Args:
            repo_id: Hub repository, ``"<owner>/<name>"``.
            filename: Path to the ``.npz`` within the repository.
            name: Display name. Defaults to the file's stem.
            revision: Branch, tag or commit. ``None`` takes the default branch.
            repo_type: ``"dataset"`` by default, as motion clips are usually published.
            token: Hub token for a gated or private repository (several public motion
                datasets are gated behind a license agreement).
            fps: Playback frame rate.
            anchor_body_name: Reference anchor body for the tracking observations.
            body_names: Ordered body names the clip covers.
            dataset_joint_names: Joint ordering in the clip. Defaults to the policy's.
            default: Select this motion when the policy loads.
            loop: Restart from the first frame after the last.
        """
        from . import source

        motion_name, payload = source.hf.fetch_motion_npz(
            repo_id,
            filename,
            revision=revision,
            repo_type=repo_type,
            token=token,
        )
        motion = MotionConfig(
            name=name or motion_name,
            data=payload,
            fps=fps,
            anchor_body_name=anchor_body_name,
            body_names=tuple(body_names),
            dataset_joint_names=(
                list(dataset_joint_names)
                if dataset_joint_names is not None
                else (
                    list(self._config.policy_joint_names)
                    if self._config.policy_joint_names is not None
                    else None
                )
            ),
            default=default,
            loop=loop,
        )
        return self._append_motion(motion)


__all__ = [
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "RUNTIME_INPUT_SLOTS",
    "PolicyConfig",
    "PolicyHandle",
    "actuated_joint_names",
    "check_slot_tables",
    "onnx_io_names",
    "onnx_output_width",
]
