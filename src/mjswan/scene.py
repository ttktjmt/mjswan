"""Scene configuration and management.

This module defines the SceneConfig dataclass and SceneHandle class for
managing MuJoCo scenes and their associated policies.
"""

from __future__ import annotations

import copy
import os
import re
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from .document.ids import assign_id, name2id, unique_id
from .license import Attribution, resolve_license, resolve_notice
from .mdp import MdpConfig
from .mjlab import (
    adapt_actions,
    adapt_commands,
    adapt_events,
    adapt_observations,
    adapt_terminations,
    resolve_action_scales,
    resolve_pd_gains,
    resolve_runner_defaults,
)
from .mjlab.task import env_cfg_control_dt
from .motion import attach_tracking_motion, tracking_motion_term
from .policy import (
    PolicyConfig,
    PolicyHandle,
    action_term_joint_names,
    actuated_joint_names,
    check_slot_tables,
    drives_joints,
    onnx_output_width,
)
from .splat import SplatConfig, SplatHandle
from .viewer import ViewerConfig

if TYPE_CHECKING:
    import onnx

    from .envs.mdp.actions.actions import ActionTermCfg
    from .managers.event_manager import EventTermCfg
    from .managers.observation_manager import ObservationGroupCfg
    from .managers.termination_manager import TerminationTermCfg
    from .project import ProjectHandle


def _get_scene_model(scene_config: SceneConfig) -> mujoco.MjModel | None:
    if scene_config.model is not None:
        return scene_config.model
    if scene_config.spec is None:
        return None
    try:
        return scene_config.spec.compile()
    except Exception:
        return None


def _get_default_qpos(model: mujoco.MjModel) -> list[float]:
    if model.nkey > 0:
        try:
            key_qpos = np.asarray(model.key_qpos).reshape(model.nkey, model.nq)
            return [float(v) for v in key_qpos[0]]
        except Exception:
            pass
    return [float(v) for v in np.asarray(model.qpos0).reshape(model.nq)]


def _resolve_observation_joints(
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> tuple[list[str], list[float]] | None:
    joint_names_cfg = config.get("joint_names")
    entity_name = config.get("entity_name")
    if joint_names_cfg is None and entity_name is None:
        return None

    default_qpos = _get_default_qpos(model)
    prefix = f"{entity_name}/" if entity_name else ""
    joints: list[tuple[str, int]] = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model.joint(i).name
        if prefix and not name.startswith(prefix):
            continue
        joints.append((name, int(model.jnt_qposadr[i])))

    if not joints:
        return None

    if joint_names_cfg in (None, "all"):
        selected = joints
    else:
        patterns = (
            list(joint_names_cfg)
            if isinstance(joint_names_cfg, (list, tuple))
            else [joint_names_cfg]
        )
        regexes = []
        for pattern in patterns:
            try:
                regexes.append(re.compile(f"^(?:{pattern})$"))
            except re.error:
                continue
        if not regexes:
            return None

        def _matches(name: str) -> bool:
            bare = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            return any(rex.fullmatch(bare) or rex.fullmatch(name) for rex in regexes)

        selected = [(name, adr) for name, adr in joints if _matches(name)]

    if not selected:
        return None

    names = [name for name, _ in selected]
    defaults = [
        default_qpos[adr] if adr < len(default_qpos) else 0.0 for _, adr in selected
    ]
    return names, defaults


def _enrich_joint_observations(
    scene_config: SceneConfig,
    observations: dict[str, Any] | None,
) -> None:
    """Resolve joint_names/default_joint_pos from the scene spec, for ``ts_name``-keyed
    joint terms only.

    Those serialize straight into JSON for a native TS class, which needs the literals
    up front. A plain-callable term is traced instead, and the tracer reads whatever the
    function itself needs off the live env.
    """
    if observations is None:
        return
    model = _get_scene_model(scene_config)
    if model is None:
        return

    legacy_pos = {"JointPos", "JointPositions"}
    legacy_vel = {"JointVelocities"}

    for group in observations.values():
        terms = getattr(group, "terms", None)
        if not isinstance(terms, dict):
            continue
        for term in terms.values():
            func = getattr(term, "func", None)
            ts_name = getattr(func, "ts_name", None)

            is_pos = ts_name in legacy_pos
            is_vel = ts_name in legacy_vel
            if not (is_pos or is_vel):
                continue

            params = dict(getattr(term, "params", {}) or {})
            defaults = getattr(func, "defaults", {})
            merged = {**defaults, **params}
            if merged.get("joint_name") is not None:
                continue
            resolved = _resolve_observation_joints(model, merged)
            if resolved is None:
                continue
            joint_names, default_joint_pos = resolved
            params["joint_names"] = joint_names
            if is_pos:
                params["default_joint_pos"] = default_joint_pos
            term.params = params


def _hf_driven_joints(
    meta: Any, width: int | None, actuated: list[str] | None
) -> tuple[list[str] | None, str | None]:
    """The joints an mjlab export drives in action order, or why its metadata cannot say.

    Returns ``(names, None)`` spelled as this scene's model spells them, or
    ``(None, reason)``. Action order is *joint* order: ``JointPositionAction`` resolves its
    term through ``Entity.find_joints_by_actuator_names``, which keeps the actuated joints
    in the model's joint order. So the metadata's order is right and it can only list
    extra, unactuated joints (mjlab's YAM ties its second finger to the first), as long as
    the task has a single action term. ``actuated`` supplies the model's spelling; its own
    order is the actuator block's, not the action order (the two differ on the Unitree G1).
    """
    if actuated is None:
        return None, "this scene's model does not drive each actuator through one joint"
    if not meta.joint_names or len(meta.default_joint_pos) != len(meta.joint_names):
        return None, "its joint list and rest pose do not line up"
    # mjlab exports the bare joint name; a scene built from an mjlab task carries it
    # namespaced (`robot/hip`). Match on the tail, return the model's spelling.
    spelling = {name.rsplit("/", 1)[-1]: name for name in actuated}
    if len(spelling) != len(actuated):
        return None, "two joints this scene's model actuates share a bare name"
    driven = [spelling[name] for name in meta.joint_names if name in spelling]
    if len(driven) != len(actuated):
        return None, "it does not list every joint this scene's model actuates"
    if width is not None and len(driven) != width:
        return (
            None,
            f"the network has {width} actions for {len(driven)} actuated joints",
        )
    if isinstance(meta.action_scale, list) and len(meta.action_scale) != len(driven):
        return None, (
            f"it scales {len(meta.action_scale)} actions for {len(driven)} actuated "
            "joints; fewer means the task has other action terms, whose order the "
            "metadata does not record"
        )
    return driven, None


def _metadata_rest_pose(meta: Any, names: list[str]) -> list[float] | None:
    """``names``' rest pose from an mjlab export, matched on the bare joint name."""
    if len(meta.default_joint_pos) != len(meta.joint_names):
        return None
    pose = dict(zip(meta.joint_names, meta.default_joint_pos))
    try:
        return [pose[name.rsplit("/", 1)[-1]] for name in names]
    except KeyError:
        return None


def _model_rest_pose(
    model: mujoco.MjModel | None, names: list[str]
) -> list[float] | None:
    """``names``' positions in the model's first keyframe (mjlab's ``init_state``)."""
    if model is None:
        return None
    qpos = _get_default_qpos(model)
    pose: list[float] = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or int(model.jnt_type[joint]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            return None
        pose.append(qpos[int(model.jnt_qposadr[joint])])
    return pose


def _default_to_latest(handles: list[PolicyHandle], scene: SceneConfig) -> None:
    """Open the scene on the highest-step checkpoint these handles brought.

    Skipped when the scene already has a default: several calls may add policies to one
    scene, and each marking its own best would leave several defaults, which the build
    refuses (ADR 0006 §4).
    """
    if not handles or any(policy.default for policy in scene.policies):
        return

    def _step(handle: PolicyHandle) -> int:
        match = re.search(r"_(\d+)", handle._config.name)
        return int(match.group(1)) if match else -1

    max(handles, key=_step)._config.default = True


@dataclass
class PendingConversion:
    """One scene's W&B → ONNX conversion, held until the build reaches that scene.

    ``run`` reports each run id as it starts, for the build's progress line.
    """

    run_paths: list[str]
    run: Callable[[Callable[[str], None]], None]


@dataclass
class SceneConfig:
    """Configuration for a MuJoCo scene."""

    name: str
    """Name of the scene."""

    id: str = ""
    """Sanitized name, unique within the project: the scene's directory in the build and
    its ``?scene=`` value (ADR 0006 §4). Assigned by :meth:`ProjectHandle.add_scene`;
    defaults to ``name2id(name)``."""

    model: mujoco.MjModel | None = None
    """MuJoCo model for the scene (saved as .mjb)."""

    spec: mujoco.MjSpec | None = None
    """MuJoCo spec for the scene (saved as .mjz)."""

    policies: list[PolicyConfig] = field(default_factory=list)
    """List of policies available for this scene."""

    pending_conversions: list[PendingConversion] = field(default_factory=list)
    """W&B checkpoint conversions ``Builder.build`` runs when it reaches this scene, so
    each scene converts, then traces, before the next one starts."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the scene."""

    attributions: list[Attribution] = field(default_factory=list, repr=False)
    """The third-party components this scene contains, each written to the scene
    directory as ``LICENSE.<component>`` / ``NOTICE.<component>`` (ADR 0007 §2)."""

    splats: list[SplatConfig] = field(default_factory=list)
    """Gaussian Splat backgrounds available for this scene."""

    splat_section: bool = False
    """Show the Splat section in the control panel even when no splats are defined."""

    viewer: ViewerConfig | None = None
    """Optional viewer configuration for this scene."""

    events: dict[str, Any] | None = None
    """The scene's event terms (mjswan or mjlab ``EventTermCfg``), adapted.

    The default a policy's MDP takes when it says nothing about events (ADR 0006 §3).
    Serialized at build time, when ONNX tracing has the live env and output directory."""

    events_explicit: bool = field(default=False, repr=False, compare=False)
    """Whether :attr:`events` were set by the author rather than inherited from a task's
    env config; only an author's are worth warning about when no policy carries them."""

    mdps: list[MdpConfig] = field(default_factory=list, repr=False, compare=False)
    """Every distinct :class:`MdpConfig` a policy on this scene uses, in first-use order.
    Index *n* here is what makes an unnamed config ``mdp_<n>``."""

    mdp_ids: list[str] = field(default_factory=list, repr=False, compare=False)
    """The id of each entry in :attr:`mdps`, unique within the scene."""

    terrain_data: dict[str, Any] | None = None
    """Optional terrain data (e.g. flat_patches) for browser-side event execution."""

    control_dt: float | None = None
    """Seconds per control step, mjlab's ``env.step_dt`` (``timestep * decimation``).

    The rate the policy acts at, and the ``dt`` every runtime timer counts in (physics
    substeps per step, command resampling, interval events). Set by
    :meth:`ProjectHandle.add_scene_mjlab`. The model carries only ``timestep``, so a
    plain :meth:`ProjectHandle.add_scene` scene with a policy must set it; the build
    refuses to guess, as a wrong rate plays without error at the wrong speed. Never
    read from a trace env, whose ``decimation=1`` is a tracing placeholder."""

    mjlab_env: Any = field(default=None, repr=False, compare=False)
    """Live env the ONNX tracer (ADR 0005) runs term bodies against. Build-time only.

    Built lazily from :attr:`mjlab_env_cfg` (see ``mjswan.build.pipeline``), so a
    tracking task's env exists only once its clip is bundled, unless
    :meth:`SceneHandle.add_policy_wandb` hands over the one it built for export. A
    plain :meth:`ProjectHandle.add_scene` scene has none: set one with
    :meth:`SceneHandle.set_trace_env` if it uses plain-callable (non-``Binding``)
    terms. It needs only ``env.scene[name].data.<field>`` (and entity write methods
    for events), see :func:`mjswan.mjlab.env.build_single_entity_trace_env`."""

    mjlab_env_cfg: Any = field(default=None, repr=False, compare=False)
    """The mjlab env config this scene was built from, when it came from a task.

    The source every policy on the scene falls back on for its observations, commands,
    actions and terminations — mjlab keeps all four on the env config, while mjswan puts
    them on the policy so that one scene can host several. Set automatically by
    :meth:`ProjectHandle.add_scene_mjlab`; ``None`` for a plain
    :meth:`ProjectHandle.add_scene` scene, which then has nothing to derive from and
    needs each term set passed explicitly.

    Held rather than re-loaded because ``load_env_cfg`` returns a deepcopy: a second call
    yields an equal but separate config, so edits made to one (a tracking task's
    ``motion_file``, a task-side param injection) would be invisible to the other.
    Python-build-time-only state; never serialized."""

    mjlab_task_id: str | None = field(default=None, repr=False, compare=False)
    """The mjlab task id behind this scene, when it came from one.

    Reaches the task's runner config for the actor's observation group and
    ``clip_actions``, see :func:`mjswan.mjlab.resolve_runner_defaults`."""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = name2id(self.name)
        # The scene asset's filename, from whichever of spec/model was given. Fixed now
        # rather than a property: the build drops both right after writing the asset.
        self.scene_filename = "scene.mjz" if self.spec is not None else "scene.mjb"

    def mdp_id(self, mdp: MdpConfig, *, policy_id: str | None = None) -> str:
        """The id of ``mdp`` on this scene, registering it on first use.

        By object identity, not equality (ADR 0006 §3): the same config on two policies
        is one MDP, two equal configs are two.

        A named config takes ``name2id(name)``. One the term-set sugar built for a single
        policy takes that policy's id, so ``mdp/locomotion/`` sits beside
        ``policy/locomotion.onnx``. Only a config shared by hand falls back to
        ``mdp_<n>``, numbered by first use within this scene, so adding a scene in front
        of this one moves none of its ids.
        """
        for known, ident in zip(self.mdps, self.mdp_ids):
            if known is mdp:
                return ident
        source = mdp.name if mdp.name is not None else policy_id
        if source is not None:
            ident = assign_id(source, set(self.mdp_ids), kind="mdp", stacklevel=4)
        else:
            # No warning on a rename here: the author never chose this spelling.
            ident = unique_id(f"mdp_{len(self.mdps)}", self.mdp_ids)
        self.mdps.append(mdp)
        self.mdp_ids.append(ident)
        return ident


_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SceneHandle:
    """Handle for adding policies and configuring a scene.

    This class provides methods for adding policies and customizing scene properties.
    Similar to viser's client handles, this allows for a fluent API pattern.
    """

    def __init__(self, scene_config: SceneConfig, project: ProjectHandle) -> None:
        self._config = scene_config
        self._project = project

    @property
    def name(self) -> str:
        """Name of the scene."""
        return self._config.name

    def _resolve_env_cfg(self, env_cfg: Any | None) -> Any | None:
        """The env config a policy's unset term sets come from, or ``None``.

        An explicit one wins over the scene's, but its control rate must match: the
        runtime derives every timer from one ``control_dt`` per scene, so a policy
        wanting a different rate needs its own scene rather than a silent demotion.
        """
        if env_cfg is None:
            return self._config.mjlab_env_cfg

        policy_dt = env_cfg_control_dt(env_cfg)
        scene_dt = self._config.control_dt
        if (
            policy_dt is not None
            and scene_dt is not None
            and abs(policy_dt - scene_dt) > 1e-12
        ):
            raise ValueError(
                f"Scene {self._config.name!r} runs at control_dt={scene_dt} s, but the "
                f"env_cfg passed for this policy implies {policy_dt} s "
                "(sim.mujoco.timestep * decimation). The control rate is per scene, not "
                "per policy — put this policy on its own scene."
            )
        return env_cfg

    def _derive_term_sets(
        self,
        env_cfg: Any | None,
        observations: Any,
        commands: Any,
        actions: Any,
        terminations: Any,
        *,
        events: Any = None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Fill each unset term set from an mjlab env config, and return all five.

        Per field, so "the task's observations but my own terminations" needs one
        override rather than all five. ``{}`` is not ``None``, so an explicitly empty
        term set still reads as "this policy has none".

        Events default differently from the other four: a per-policy ``env_cfg`` supplies
        its own, but otherwise they come from the scene's, which carry the terrain-spawn
        patch.

        Shared with :meth:`add_policy_wandb`, which needs the resolved ``commands`` to
        find the tracking term that names the motion clip to fetch.
        """
        # Returned unadapted, like the other four: `_resolve_mdp` adapts all five.
        if events is None:
            events = (
                getattr(env_cfg, "events", None)
                if env_cfg is not None
                else self._config.events
            )
        source_cfg = self._resolve_env_cfg(env_cfg)
        if source_cfg is None:
            return observations, commands, actions, terminations, events
        if observations is None:
            observations = getattr(source_cfg, "observations", None)
        if commands is None:
            commands = getattr(source_cfg, "commands", None)
        if actions is None:
            actions = getattr(source_cfg, "actions", None)
        if terminations is None:
            terminations = getattr(source_cfg, "terminations", None)
        return observations, commands, actions, terminations, events

    def _resolve_mdp(
        self,
        mdp: MdpConfig,
        *,
        env_cfg: Any | None,
        task_id: str | None,
        policy_joint_names: list[str] | None,
    ) -> None:
        """Fill ``mdp``'s unset fields and adapt its mjlab types, in place, once.

        Runs on the first policy to use the config; later ones find ``_adapted`` set and
        share the result, including action scales and PD gains resolved against the first
        policy's joint names, which the checkpoints of one run share.
        """
        if mdp._adapted:
            return
        (
            observations,
            commands,
            actions,
            terminations,
            events,
        ) = self._derive_term_sets(
            env_cfg,
            mdp.observations,
            mdp.commands,
            mdp.actions,
            mdp.terminations,
            events=mdp.events,
        )
        runner = resolve_runner_defaults(
            task_id if task_id is not None else self._config.mjlab_task_id
        )
        mdp.observations = adapt_observations(
            observations, obs_groups=runner.obs_groups
        )
        mdp.commands = adapt_commands(commands) or {}
        mdp.actions = adapt_actions(actions)
        mdp.terminations = adapt_terminations(terminations)
        mdp.events = adapt_events(events)
        _enrich_joint_observations(self._config, mdp.observations)
        if mdp.actions and policy_joint_names:
            resolve_action_scales(mdp.actions, policy_joint_names)
            resolve_pd_gains(
                mdp.actions, policy_joint_names, self._resolve_env_cfg(env_cfg)
            )
        mdp._adapted = True

    def add_policy(
        self,
        name: str,
        policy: onnx.ModelProto,
        *,
        metadata: dict[str, Any] | None = None,
        source_path: str | None = None,
        config_path: str | None = None,
        env_cfg: Any | None = None,
        task_id: str | None = None,
        mdp: MdpConfig | None = None,
        observations: ObservationGroupCfg | Mapping[str, Any] | Any | None = None,
        commands: Mapping[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
        events: dict[str, EventTermCfg] | Mapping[str, Any] | None = None,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str | Sequence[str]] | None = None,
        policy_joint_names: list[str] | None = None,
        policy_num_actions: int | None = None,
        default_joint_pos: list[float] | None = None,
        encoder_bias: list[float] | None = None,
        clip_actions: float | None = None,
        initial_qpos: list[float] | None = None,
        initial_qvel: list[float] | None = None,
        extras: dict[str, Any] | None = None,
        default: bool = False,
    ) -> PolicyHandle:
        """Add an ONNX policy to this scene.

        The policy runs against an MDP: observations, actions, terminations, commands
        and events (ADR 0006 §3). Pass one as ``mdp`` to share it between policies (the
        checkpoints of one run), or pass the five term sets directly and an anonymous
        :class:`~mjswan.mdp.MdpConfig` is built from them; not both.

        Each term set defaults to the matching field of an mjlab env config, when one
        is available: the ``env_cfg`` passed here, else the one the scene was built
        from by :meth:`ProjectHandle.add_scene_mjlab`. Events default to the scene's own.
        Pass a term set to override that field; pass ``{}`` to say the policy genuinely
        has none. A plain :meth:`ProjectHandle.add_scene` scene has no config to fall
        back on, so there each field means exactly what it says.

        Args:
            policy: ONNX model containing the policy.
            name: Name for the policy (displayed in the UI).
            metadata: Optional metadata dictionary for the policy.
            source_path: Optional source path for the policy ONNX file.
            config_path: Optional source path for the policy config JSON file.
            env_cfg: mjlab env config to take this policy's unset term sets from,
                instead of the scene's — for one scene hosting policies trained against
                different configs. Its control rate must match the scene's
                ``control_dt``, which the scene owns.
            task_id: mjlab task id whose *runner* config supplies the actor's
                observation group and ``clip_actions``. Defaults to the scene's task.
            mdp: The MDP to run against, shared by every policy handed the same object.
                Its unset fields are filled and its mjlab types adapted, in place, by
                the first policy to use it.
            observations: A single observation group, mjlab's whole
                ``env_cfg.observations`` dict, or a dict keyed by the slot names
                ``in_keys`` uses. Prefer the first two: a lone group lands under
                ``"actor"``, the default slot, and needs no ``in_keys`` at all. A group
                named for a training-only network (``"critic"``) is dropped.
            commands: Command term configurations. Custom mjlab terms are converted
                through the Python command-term registry.
            actions: Action term configurations.
            terminations: Termination term configurations.
            events: Event term configurations, in any of the four modes.
            in_keys: The network's input slot table: ``in_keys[i]`` names what fills its
                *i*-th input, an observation group or a tensor the runtime synthesizes
                (``is_init``, ``adapt_hx``, ``time_step``). The mapping is positional, so
                the network's own input names never matter. Required when the network has
                more than one input; a single-input one takes its one observation group.
            out_keys: The output slot table, ``out_keys[i]`` naming the *i*-th output.
                ``action`` is the one the runtime drives the actuators from; a recurrent
                policy also carries ``["next", "adapt_hx"]``. Defaults to ``["action"]``.
            policy_num_actions: Output width for policies whose action count cannot be
                inferred from ``policy_joint_names`` (e.g. muscle-driven ones).
            clip_actions: Symmetric bound on the raw policy output, before any action
                term sees it, mirroring rsl-rl's ``RslRlVecEnvWrapper``. Distinct from
                ``ActionTermCfg.clip``, which bounds ``raw * scale + offset``. Defaults
                to the task's runner config; ``0.0`` is a real bound, not "unset".
            initial_qpos: Initial qpos serialized into the policy config JSON.
            initial_qvel: Initial qvel serialized into the policy config JSON.
            extras: Extra JSON payload merged into the policy config.

        Returns:
            PolicyHandle for configuring the policy (adding commands, etc.)

        Example:
            from mjlab.envs.mdp import observations as obs_fns
            from mjswan.managers.observation_manager import (
                ObservationGroupCfg,
                ObservationTermCfg,
            )

            policy = scene.add_policy(
                policy=onnx.load("locomotion.onnx"),
                name="Locomotion",
                config_path="locomotion.json",
                commands={"velocity": mjswan.velocity_command()},
                observations=ObservationGroupCfg(
                    terms={
                        "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                        "joint_pos": ObservationTermCfg(
                            func=obs_fns.joint_pos_rel, scale=0.5
                        ),
                    },
                ),
            )
        """
        if metadata is None:
            metadata = {}

        given = {
            k: v
            for k, v in (
                ("observations", observations),
                ("commands", commands),
                ("actions", actions),
                ("terminations", terminations),
                ("events", events),
            )
            if v is not None
        }
        if mdp is not None and given:
            raise ValueError(
                f"Policy {name!r} was given mdp= and also {sorted(given)}. An MdpConfig "
                "carries all five term sets; pass either it or the term sets, not both."
            )
        # An MDP built here belongs to this policy alone, so it takes the policy's id;
        # one passed in may be shared, and is numbered unless it carries a name.
        sugar_built = mdp is None
        if mdp is None:
            mdp = MdpConfig(
                observations=observations,
                commands=commands,
                actions=actions,
                terminations=terminations,
                events=events,
            )
        self._resolve_mdp(
            mdp,
            env_cfg=env_cfg,
            task_id=task_id,
            policy_joint_names=policy_joint_names,
        )
        policy_id = assign_id(
            name, {p.id for p in self._config.policies}, kind="policy", stacklevel=3
        )
        self._config.mdp_id(mdp, policy_id=policy_id if sugar_built else None)

        runner = resolve_runner_defaults(
            task_id if task_id is not None else self._config.mjlab_task_id
        )
        if clip_actions is None:
            clip_actions = runner.clip_actions
        slot_in, slot_out = check_slot_tables(name, policy, in_keys, out_keys)
        if policy_num_actions is None and not policy_joint_names:
            # A muscle policy has no joint transmission to count, so take the action
            # count from the network's own output width.
            policy_num_actions = onnx_output_width(policy, slot_out)

        policy_config = PolicyConfig(
            name=name,
            id=policy_id,
            model=policy,
            metadata=metadata,
            source_path=source_path,
            config_path=config_path,
            mdp=mdp,
            in_keys=slot_in,
            out_keys=slot_out,
            policy_joint_names=policy_joint_names,
            policy_num_actions=policy_num_actions,
            default_joint_pos=default_joint_pos,
            encoder_bias=encoder_bias,
            clip_actions=clip_actions,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
            extras=extras,
            default=default,
        )
        self._config.policies.append(policy_config)
        return PolicyHandle(policy_config, self)

    def add_policy_wandb(
        self,
        run_path: str | list[str],
        *,
        only_latest: bool = False,
        task_id: str | None = None,
        config_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        env_cfg: Any | None = None,
        observations: ObservationGroupCfg | Mapping[str, Any] | Any | None = None,
        commands: Mapping[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str | Sequence[str]] | None = None,
        clip_actions: float | None = None,
        extras: dict[str, Any] | None = None,
    ) -> list[PolicyHandle]:
        """Add ONNX policies fetched from one or more W&B runs to this scene.

        ``config_path``, ``observations``, ``commands``, ``actions``, and
        ``terminations`` are applied identically to every policy fetched from every run,
        and each defaults to the scene's mjlab env config exactly as in
        :meth:`add_policy` — so for a scene from
        :meth:`ProjectHandle.add_scene_mjlab` the run path alone is enough.

        Args:
            run_path: W&B run path in the format ``"entity/project/run_id"``, or
                a list of such paths to fetch policies from multiple runs.
            only_latest: If ``False`` (default), fetches all ``model_*.pt``
                checkpoints and converts each to ONNX via mjlab — requires
                ``mjlab`` and ``torch`` to be installed and ``task_id`` to be
                provided.  If ``True``, fetches only the ``.onnx`` file from
                each run (the latest exported checkpoint).
            task_id: mjlab task identifier (e.g. ``"go2_flat"``). Defaults to the
                scene's task when it came from :meth:`ProjectHandle.add_scene_mjlab`,
                so it only has to be given for a plain scene. Required when
                ``only_latest=False``; when ``only_latest=True`` it is still used, if
                known, to read the task's runner config.
            config_path: Optional path to a policy config JSON file applied to
                all fetched policies.
            metadata: Optional metadata dictionary applied to all fetched
                policies.
            env_cfg: mjlab env config the unset term sets below are taken from,
                instead of the scene's. See :meth:`add_policy`.
            observations: Observation groups applied to all fetched policies —
                a single group (``env_cfg.observations["actor"]``) or a dict of
                them; see :meth:`add_policy`.
            commands: Command term configurations applied to all fetched policies.
            actions: Action term configurations applied to all fetched policies.
            terminations: Termination term configurations applied to all fetched
                policies.
            in_keys: ONNX input slot table applied to every fetched policy; see
                :meth:`add_policy`. Required for a run whose export takes more than
                one input (an observation group plus a runtime tensor such as
                ``time_step``).
            out_keys: ONNX output slot table applied to every fetched policy; see
                :meth:`add_policy`.
            clip_actions: Overrides the raw-action bound that would otherwise be
                read from the task's mjlab runner config. Only the
                ``only_latest=True`` path needs it explicitly — that path skips
                mjlab entirely, so there is no runner config to read.
            extras: Optional extra JSON payload applied to every fetched policy.

        Returns:
            Flat list of :class:`PolicyHandle` instances across all runs, in the order
            the runs were provided — **empty when ``only_latest=False``**, since that
            path defers its ``.pt`` conversion to ``Builder.build`` and the checkpoint
            names are not known until then.

        Raises:
            ValueError: If ``only_latest=False`` and ``task_id`` is not provided,
                or if no matching files are found in a W&B run.
            ImportError: If ``only_latest=False`` and ``mjlab``/``torch`` are not
                installed.

        Example — all logged checkpoints from a single run (default):
            ```python
            scene.add_policy_wandb(
                run_path="my-org/my-project/run-id",
                task_id="go2_flat",
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```

        Example — latest checkpoint only:
            ```python
            scene.add_policy_wandb(
                run_path="my-org/my-project/run-id",
                only_latest=True,
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```

        Example — multiple runs:
            ```python
            scene.add_policy_wandb(
                run_path=[
                    "my-org/my-project/run-id-1",
                    "my-org/my-project/run-id-2",
                ],
                only_latest=True,
                config_path="assets/locomotion.json",
                actions={"joint_pos": JointPositionActionCfg(scale=1.0)},
            )
            ```
        """
        if task_id is None:
            task_id = self._config.mjlab_task_id
        if not only_latest and task_id is None:
            raise ValueError(
                "task_id is required when only_latest=False and the scene did not come "
                "from an mjlab task. Provide the mjlab task identifier, e.g. "
                "task_id='go2_flat'."
            )

        run_paths = [run_path] if isinstance(run_path, str) else run_path

        # Resolved here, not left to `add_policy`: the tracking clip is found by scanning
        # `commands`, which would miss one that came from the scene's env config.
        observations, commands, actions, terminations, events = self._derive_term_sets(
            env_cfg, observations, commands, actions, terminations
        )
        tracking_term = tracking_motion_term(commands)
        tracking_motion_cache: dict[str, tuple[str, bytes]] = {}
        # One MDP for every checkpoint this call adds: they were trained against one
        # env config, so they share its graphs rather than tracing them once each.
        shared_mdp = MdpConfig(
            observations=observations,
            commands=commands,
            actions=actions,
            terminations=terminations,
            events=events,
        )

        from . import source

        handles = []
        seen_names: set[str] = set()
        if only_latest:
            for path in run_paths:
                name, model = source.wandb.fetch_onnx(path)
                if name not in seen_names:
                    seen_names.add(name)
                    handle = self.add_policy(
                        name=name,
                        policy=model,
                        config_path=config_path,
                        metadata=metadata,
                        env_cfg=env_cfg,
                        task_id=task_id,
                        mdp=shared_mdp,
                        in_keys=in_keys,
                        out_keys=out_keys,
                        clip_actions=clip_actions,
                        extras=extras,
                    )
                    attach_tracking_motion(
                        handle,
                        path,
                        tracking_term,
                        tracking_motion_cache,
                    )
                    handles.append(handle)
        else:
            # Deferred to build time so each scene converts and traces before the next
            # starts; converting up front held one mjlab env per scene alive.
            assert task_id is not None

            def _convert(on_run: Callable[[str], None] = lambda _: None) -> None:
                from .mjlab.runner import (
                    create_pt_onnx_export_context,
                    export_checkpoint,
                )

                with tempfile.TemporaryDirectory() as staging_dir:
                    # The config mjlab's *export* env is built from: the scene's own, so
                    # the observation widths that env reports are the ones the exported
                    # ONNX takes. Copied, since the tracking branch below writes a
                    # staging path into it that dies with this block.
                    source_cfg = self._resolve_env_cfg(env_cfg)
                    export_env_cfg: Any = (
                        copy.deepcopy(source_cfg) if source_cfg is not None else None
                    )
                    if tracking_term is not None:
                        existing_file = getattr(tracking_term, "motion_file", None)
                        if existing_file and Path(existing_file).is_file():
                            motion_name = Path(existing_file).stem
                            motion_bytes = Path(existing_file).read_bytes()
                            motion_file_for_env = existing_file
                        else:
                            motion_name, motion_bytes = source.wandb.fetch_motion_npz(
                                run_paths[0]
                            )
                            staged = Path(staging_dir) / f"{motion_name}.npz"
                            staged.write_bytes(motion_bytes)
                            motion_file_for_env = str(staged)
                        for rp in run_paths:
                            tracking_motion_cache.setdefault(
                                rp, (motion_name, motion_bytes)
                            )

                        if export_env_cfg is None:
                            # A plain scene has no config to follow, so fall back on the
                            # play config `add_scene_mjlab` would have chosen.
                            try:
                                from mjlab.tasks.registry import (
                                    load_env_cfg as _load_env_cfg,
                                )
                            except ImportError as exc:
                                raise ImportError(
                                    "mjlab is required to resolve the tracking motion for "
                                    "export."
                                ) from exc
                            export_env_cfg = _load_env_cfg(task_id, play=True)
                        export_env_cfg.commands[
                            "motion"
                        ].motion_file = motion_file_for_env  # type: ignore[attr-defined]

                    export_context = create_pt_onnx_export_context(
                        task_id, env_cfg=export_env_cfg
                    )

                    def _adopt(path: str, name: str, model: onnx.ModelProto) -> None:
                        handle = self.add_policy(
                            name=name,
                            policy=model,
                            config_path=config_path,
                            metadata=metadata,
                            env_cfg=env_cfg,
                            task_id=task_id,
                            mdp=shared_mdp,
                            policy_joint_names=export_context.joint_names or None,
                            default_joint_pos=export_context.default_joint_pos or None,
                            encoder_bias=export_context.encoder_bias or None,
                            in_keys=in_keys,
                            out_keys=out_keys,
                            clip_actions=clip_actions,
                            extras=extras,
                        )
                        attach_tracking_motion(
                            handle,
                            path,
                            tracking_term,
                            tracking_motion_cache,
                            dataset_joint_names=export_context.joint_names or None,
                        )
                        handles.append(handle)

                    try:
                        for path in run_paths:
                            on_run(path.rsplit("/", 1)[-1])
                            with source.wandb.fetch_checkpoints(path) as checkpoints:
                                for name, pt_path in checkpoints:
                                    if name in seen_names:
                                        continue
                                    seen_names.add(name)
                                    model = export_checkpoint(export_context, pt_path)
                                    _adopt(path, name, model)
                    finally:
                        # Keep it as the scene's trace env rather than building a second
                        # one from the same config.
                        if (
                            env_cfg is None
                            and self._config.mjlab_env_cfg is not None
                            and self._config.mjlab_env is None
                        ):
                            self._config.mjlab_env = export_context.env.unwrapped
                        else:
                            export_context.close()

                # Here, not after the call: these handles exist only once this runs.
                _default_to_latest(handles, self._config)

            self._config.pending_conversions.append(
                PendingConversion(run_paths=list(run_paths), run=_convert)
            )

        _default_to_latest(handles, self._config)
        return handles

    def add_policy_hf(
        self,
        repo_id: str,
        *,
        filename: str | list[str] | None = None,
        revision: str | None = None,
        repo_type: str = "model",
        token: str | None = None,
        name: str | None = None,
        use_metadata: bool = True,
        task_id: str | None = None,
        config_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        env_cfg: Any | None = None,
        observations: ObservationGroupCfg | Mapping[str, Any] | Any | None = None,
        commands: Mapping[str, Any] | None = None,
        actions: Mapping[str, ActionTermCfg] | Mapping[str, Any] | None = None,
        terminations: dict[str, TerminationTermCfg] | dict[str, Any] | None = None,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str | Sequence[str]] | None = None,
        policy_joint_names: list[str] | None = None,
        default_joint_pos: list[float] | None = None,
        encoder_bias: list[float] | None = None,
        clip_actions: float | None = None,
        extras: dict[str, Any] | None = None,
    ) -> list[PolicyHandle]:
        """Add ONNX policies fetched from a Hugging Face Hub repository.

        The light counterpart of :meth:`add_policy_wandb`: it downloads the exported
        ``.onnx`` and reads the metadata mjlab baked into it, so neither mjlab nor torch
        is needed and ``task_id`` is optional.

        What the caller does not pass is filled as mjlab has it. ``policy_joint_names``
        are the joints the task's action terms name, in mjlab's action order, when the
        scene or ``env_cfg`` has those terms. Otherwise, with ``use_metadata`` on, they
        come from an mjlab export's metadata, and so does the joint-position action term.
        The metadata lists every joint of the robot, so it is used only when it covers
        every joint **this scene's own model** actuates, one per action; anything else
        would misdrive every actuator with nothing at playback to say so, so a mismatch
        warns and fills nothing. It does not record the order of several action terms,
        so such a task needs its env config. ``default_joint_pos`` is looked up by joint
        name in the metadata, else in the scene model's first keyframe, which is mjlab's
        ``init_state``.

        Observation terms are *never* reconstructed: the metadata names them but does
        not carry the functions mjswan traces, so ``observations`` stays the caller's
        (or the scene's) to supply.

        Args:
            repo_id: Hub repository, ``"<owner>/<name>"``.
            filename: Path within the repository, or a list to add several policies from
                one repository. ``None`` resolves it: ``policy.onnx``, then
                ``final.onnx``, then the single ``.onnx`` if that is all there is.
            revision: Branch, tag or commit. ``None`` takes the default branch, so the
                build follows the repository; pass a commit to pin it.
            repo_type: ``"model"`` (default), ``"dataset"`` or ``"space"``.
            token: Hub token for a gated or private repository. ``None`` uses the
                locally stored login, then anonymous access.
            name: Display name, for a single file only. Omitted, the name is the file's
                stem, or the repository's own name when the stem is a generic one such
                as ``policy``.
            use_metadata: Read mjlab's ``metadata_props``, as described above. ``False``
                ignores them entirely.
            task_id: mjlab task whose runner config supplies defaults such as
                ``clip_actions``. Optional; defaults to the scene's, when it has one.
            config_path: Optional policy config JSON applied to every fetched policy.
            metadata: Optional metadata dictionary applied to every fetched policy.
            env_cfg: mjlab env config the unset term sets are taken from, instead of the
                scene's. See :meth:`add_policy`.
            observations: Observation groups applied to every fetched policy.
            commands: Command term configurations applied to every fetched policy.
            actions: Action term configurations applied to every fetched policy. Given
                here, the metadata's action term is not consulted.
            terminations: Termination terms applied to every fetched policy.
            in_keys: ONNX input slot table; see :meth:`add_policy`.
            out_keys: ONNX output slot table; see :meth:`add_policy`.
            policy_joint_names: The joints the actions drive, in the order they come
                out. Overrides what the action terms or the metadata would supply.
            default_joint_pos: Their rest pose, in the same order. Overrides the lookup.
            encoder_bias: Per-joint encoder bias; the metadata carries none.
            clip_actions: Raw-action bound. Unset, it is read from ``task_id``'s runner
                config when mjlab and that task are installed.
            extras: Optional extra JSON payload applied to every fetched policy.

        Returns:
            One :class:`PolicyHandle` per fetched file, in the order given.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
            ValueError: If ``name`` is given for more than one file, or the repository
                has no unambiguous ``.onnx`` and none was named.

        Example:
            ```python
            scene.add_policy_hf("my-org/g1-velocity-flat")
            ```

        Example: several policies from one repository, pinned to a commit:
            ```python
            scene.add_policy_hf(
                "my-org/microduck",
                filename=["policies/walk.onnx", "policies/stand.onnx"],
                revision="9a1c2f0",
            )
            ```
        """
        from . import source
        from .mjlab.onnx_meta import action_cfg_from_metadata, read_mjlab_metadata

        if filename is None:
            filenames = [
                source.hf.resolve_policy_filename(
                    repo_id, revision=revision, repo_type=repo_type, token=token
                )
            ]
        elif isinstance(filename, str):
            filenames = [filename]
        else:
            filenames = list(filename)
        if name is not None and len(filenames) != 1:
            raise ValueError(
                f"add_policy_hf({repo_id!r}) was given name={name!r} for "
                f"{len(filenames)} files. A name applies to one policy; drop it and "
                "each file is named after itself."
            )

        fetched = [
            source.hf.fetch_onnx(
                repo_id,
                fname,
                revision=revision,
                repo_type=repo_type,
                token=token,
            )
            for fname in filenames
        ]

        metas = [
            read_mjlab_metadata(model) if use_metadata else None for _, model in fetched
        ]
        widths = [onnx_output_width(model, out_keys) for _, model in fetched]
        # Compiled once: every fetched policy is checked against the same scene.
        scene_model = _get_scene_model(self._config)
        actuated = actuated_joint_names(scene_model)

        if task_id is None:
            task_id = self._config.mjlab_task_id
        observations, commands, actions, terminations, events = self._derive_term_sets(
            env_cfg, observations, commands, actions, terminations
        )
        # mjlab's action order is its action terms', one after another, and the metadata
        # records it only when there is one term, so the terms answer first. Adapted on
        # a copy: `_resolve_mdp` adapts the shared MDP itself, later and once.
        adapted_actions = adapt_actions(actions)
        term_joint_names = (
            None
            if policy_joint_names is not None
            else action_term_joint_names(adapted_actions, scene_model)
        )
        first_meta = metas[0] if metas else None
        if actions is None and first_meta is not None:
            # `_derive_term_sets` has run, so neither the caller nor an env config gave
            # actions, and the export's joint-position term is all there is.
            names, reason = _hf_driven_joints(first_meta, widths[0], actuated)
            if names is not None:
                actions = (
                    action_cfg_from_metadata(first_meta, num_actions=len(names)) or None
                )
            elif policy_joint_names is not None and config_path is None:
                # Without the caller's names, `_hf_joint_kwargs` reports this instead.
                warnings.warn(
                    f"Policy {fetched[0][0]!r} from {repo_id!r} carries mjlab metadata, "
                    f"but {reason}, so its joint-position action term was not taken "
                    "from it and the policy has no action term. Pass actions.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )

        # One MDP for all fetched policies: they describe one task, so share its graphs.
        shared_mdp = MdpConfig(
            observations=observations,
            commands=commands,
            actions=actions,
            terminations=terminations,
            events=events,
        )

        handles: list[PolicyHandle] = []
        for (policy_name, model), meta, width in zip(fetched, metas, widths):
            joint_kwargs = self._hf_joint_kwargs(
                meta,
                width,
                actuated=actuated,
                term_joint_names=term_joint_names,
                scene_model=scene_model,
                policy_name=policy_name,
                repo_id=repo_id,
                policy_joint_names=policy_joint_names,
                default_joint_pos=default_joint_pos,
                drives_joints=drives_joints(adapted_actions),
                has_sidecar=config_path is not None,
            )
            handles.append(
                self.add_policy(
                    name=name or policy_name,
                    policy=model,
                    config_path=config_path,
                    metadata=metadata,
                    env_cfg=env_cfg,
                    task_id=task_id,
                    mdp=shared_mdp,
                    in_keys=in_keys,
                    out_keys=out_keys,
                    encoder_bias=encoder_bias,
                    clip_actions=clip_actions,
                    extras=extras,
                    **joint_kwargs,
                )
            )

        _default_to_latest(handles, self._config)
        return handles

    @staticmethod
    def _hf_joint_kwargs(
        meta: Any | None,
        width: int | None,
        *,
        actuated: list[str] | None,
        term_joint_names: list[str] | None,
        scene_model: mujoco.MjModel | None,
        policy_name: str,
        repo_id: str,
        policy_joint_names: list[str] | None,
        default_joint_pos: list[float] | None,
        drives_joints: bool,
        has_sidecar: bool,
    ) -> dict[str, Any]:
        """``policy_joint_names`` / ``default_joint_pos`` for one policy, as mjlab has them.

        The names are the caller's, else the joints the task's action terms name, else
        the export's metadata. The rest pose is the caller's, else looked up by joint
        name: in the metadata, then in the scene model's first keyframe, which is
        mjlab's ``init_state``. Names that cannot be filled warn, or the policy would
        drive the wrong actuators, or none, with nothing at playback to say so.
        """
        names = policy_joint_names
        if names is None and term_joint_names is not None:
            if width is None or len(term_joint_names) == width:
                names = list(term_joint_names)
            else:
                warnings.warn(
                    f"Policy {policy_name!r} from {repo_id!r} has {width} actions, but "
                    f"the task's action terms name {len(term_joint_names)} joints, so "
                    "policy_joint_names was left unset. Pass it explicitly, in the "
                    "order the actions come out.",
                    category=RuntimeWarning,
                    stacklevel=3,
                )
        elif names is None and meta is not None:
            names, reason = _hf_driven_joints(meta, width, actuated)
            # Spared: a sidecar, which may still carry the names.
            if names is None and not has_sidecar:
                warnings.warn(
                    f"Policy {policy_name!r} from {repo_id!r} carries mjlab metadata, "
                    f"but {reason}, so policy_joint_names, default_joint_pos and the "
                    "joint-position action term were not taken from it. Pass them "
                    "explicitly, in the order the policy's actions come out.",
                    category=RuntimeWarning,
                    stacklevel=3,
                )
        elif names is None and drives_joints and not has_sidecar:
            # Warn here: the browser skips an action term it cannot map with only a
            # `console.warn`, which release bundles strip. Spared: a policy with no
            # joint action term, and a sidecar, which may still carry the names.
            warnings.warn(
                f"Policy {policy_name!r} from {repo_id!r} carries no mjlab metadata, "
                "and this scene's action terms do not say which joints it drives, so "
                "policy_joint_names is unset and the browser will write no control at "
                "all. mjlab attaches that metadata from its velocity, manipulation and "
                "tracking runners only. Pass policy_joint_names, in the order the "
                "actions come out.",
                category=RuntimeWarning,
                stacklevel=3,
            )

        pose = default_joint_pos
        if names is not None and pose is None and meta is not None:
            pose = _metadata_rest_pose(meta, names)
        if names is not None and pose is None:
            pose = _model_rest_pose(scene_model, names)
        return {
            key: value
            for key, value in (
                ("policy_joint_names", names),
                ("default_joint_pos", pose),
            )
            if value is not None
        }

    def actuated_joint_names(self) -> list[str] | None:
        """The joint each of this scene's actuators drives, in **actuator** order.

        A starting point for ``policy_joint_names``, not its value: an mjlab policy's
        actions come out in *joint* order (``Entity.find_joints_by_actuator_names``
        keeps ``joint_names`` order), and on some robots (Unitree G1) the two differ.

        ``None`` when the model does not answer unambiguously: no model, no actuators, a
        transmission that is not a joint (a tendon, a site, a body), an unnamed joint, or
        two actuators on one joint.
        """
        return actuated_joint_names(_get_scene_model(self._config))

    def add_splat(
        self,
        name: str,
        *,
        source: str | None = None,
        url: str | None = None,
        scale: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        collider_url: str | None = None,
        control: bool = False,
    ) -> SplatHandle:
        """Add a Gaussian Splat background to this scene.

        Provide either ``source`` (recommended) or ``url`` — not both.

        Using ``source`` copies the .spz file into the built application so it
        is served locally, giving you a fully self-contained deployment with no
        external dependencies. This is the recommended approach.

        Using ``url`` keeps the .spz file on an external server. The app stays
        smaller, but requires network access at runtime and will not work
        offline.

        Args:
            name: Display name shown in the viewer control panel.
            source: Local path to a .spz splat file to bundle into the app.
                The file is copied during :meth:`Builder.build`.
            url: URL to an external .spz splat file. The browser fetches it at
                runtime; the file is not bundled.
            scale: Metric scale factor. Use ``metric_scale_factor`` from your
                capture metadata if available.
            x_offset: X-axis position offset (in scaled splat units).
            y_offset: Y-axis position offset (in scaled splat units).
            z_offset: Vertical position offset. Use ``ground_plane_offset`` from
                your capture metadata if available.
            roll: Roll rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            pitch: Pitch rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            yaw: Yaw rotation in degrees applied on top of the COLMAP→Three.js
                base rotation.
            collider_url: Optional URL or local path to a .glb collision mesh.
            control: If True, shows scale and offset controls in the viewer
                control panel. Defaults to False.

        Returns:
            SplatHandle for further configuration.

        Example:
            # Recommended: bundle the .spz file into the app
            scene.add_splat(
                "Outdoor",
                source="background.spz",
                scale=1.35,
                z_offset=1.0,
            )

            # Alternative: reference an external URL
            scene.add_splat(
                "Outdoor",
                url="https://cdn.example.com/background.spz",
                scale=1.35,
                z_offset=1.0,
            )
        """
        if source is None and url is None:
            raise ValueError(
                "Provide either 'source' (local .spz file path to bundle) "
                "or 'url' (external URL)."
            )
        if source is not None and url is not None:
            raise ValueError("Provide either 'source' or 'url', not both.")

        splat_config = SplatConfig(
            name=name,
            id=assign_id(
                name, {s.id for s in self._config.splats}, kind="splat", stacklevel=3
            ),
            source=source,
            url=url,
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            collider_url=collider_url,
            control=control,
        )
        self._config.splats.append(splat_config)
        return SplatHandle(splat_config, self)

    def add_splat_hf(
        self,
        repo_id: str,
        filename: str,
        *,
        name: str | None = None,
        revision: str | None = None,
        repo_type: str = "model",
        token: str | None = None,
        scale: float = 1.0,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        collider_url: str | None = None,
        control: bool = False,
    ) -> SplatHandle:
        """Add a Gaussian Splat background fetched from a Hugging Face Hub repository.

        The Hub counterpart of :meth:`add_splat`: the ``.spz`` is downloaded and bundled
        like a local ``source=``, so the deployed app needs no network. The placement
        arguments mean what they do in :meth:`add_splat` and stay the caller's to
        supply, since no file on the Hub says how a capture lines up with a model.

        Args:
            repo_id: Hub repository, ``"<owner>/<name>"``.
            filename: Path to the ``.spz`` within the repository.
            name: Display name shown in the viewer control panel. Omitted, the name is
                the file's stem, or the repository's own name when the stem is a generic
                one such as ``background``.
            revision: Branch, tag or commit. ``None`` takes the default branch, so the
                build follows the repository; pass a commit to pin it.
            repo_type: ``"model"`` (default), ``"dataset"`` or ``"space"``.
            token: Hub token for a gated or private repository. ``None`` uses the
                locally stored login, then anonymous access.
            scale: Metric scale factor. See :meth:`add_splat`.
            x_offset: X-axis position offset (in scaled splat units).
            y_offset: Y-axis position offset (in scaled splat units).
            z_offset: Vertical position offset.
            roll: Roll rotation in degrees.
            pitch: Pitch rotation in degrees.
            yaw: Yaw rotation in degrees.
            collider_url: Optional URL to a ``.glb`` collision mesh. A collider is not
                bundled, so one living on the Hub is named by its ``resolve`` URL
                (``https://huggingface.co/<repo>/resolve/<rev>/<path>``).
            control: If True, shows scale and offset controls in the viewer.

        Returns:
            SplatHandle for further configuration.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.

        Example:
            ```python
            scene.add_splat_hf(
                "my-org/assets", "splats/street.spz", scale=3.275, z_offset=0.708
            )
            ```
        """
        from . import source

        local_path = source.hf.fetch_file(
            repo_id,
            filename,
            revision=revision,
            repo_type=repo_type,
            token=token,
        )
        return self.add_splat(
            name or source.hf.splat_name_for(repo_id, filename),
            source=str(local_path),
            scale=scale,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            collider_url=collider_url,
            control=control,
        )

    def enable_splat_section(self) -> SceneHandle:
        """Show the Splat section in the control panel even when no splats are defined.

        This allows users to load splats by pasting a .spz URL directly in the
        control panel, without requiring any pre-configured splats.

        Returns:
            Self for method chaining.

        Example:
            scene.enable_splat_section()
        """
        self._config.splat_section = True
        return self

    def set_viewer(self, config: ViewerConfig) -> SceneHandle:
        """Set viewer configuration for this scene.

        Args:
            config: A :class:`ViewerConfig` instance describing the camera
                position, tracking mode, and rendering settings.

        Returns:
            Self for method chaining.

        Example::

            from mjswan import ViewerConfig
            scene.set_viewer(ViewerConfig(
                lookat=(0.0, 0.0, 0.7),
                distance=4.3,
                elevation=-33,
                azimuth=-34,
                origin_type=ViewerConfig.OriginType.ASSET_BODY,
                body_name="torso_link",
            ))
        """
        self._config.viewer = config
        return self

    def set_events(
        self, events: Mapping[str, Any], *, _explicit: bool = True
    ) -> SceneHandle:
        """Set the scene's events, the default for every policy's MDP on it.

        Accepts mjswan or mjlab ``EventTermCfg`` instances in any of the four modes.
        ONNX tracing happens at build time, as for observations and terminations. Events
        belong to an MDP (ADR 0006 §3), so a scene with events and no policy has nowhere
        to put them, and the build says so.

        Args:
            events: Dict mapping event names to ``EventTermCfg`` instances.

        Returns:
            Self for method chaining.
        """
        self._config.events = adapt_events(events)
        self._config.events_explicit = _explicit
        return self

    def set_trace_env(self, env: Any) -> SceneHandle:
        """Set the live env ONNX tracing runs authored term bodies against.

        Required for a plain :meth:`ProjectHandle.add_scene` scene with plain-callable
        term functions, which has no task env of its own. The env only has to satisfy
        ``env.scene[name].data.<field>`` (plus the entity write methods for write-side
        terms); :func:`mjswan.mjlab.env.build_single_entity_trace_env` builds a minimal
        one from a single entity's spec.

        An :meth:`ProjectHandle.add_scene_mjlab` scene builds its own at build time;
        setting one here pre-empts that.

        Args:
            env: A live env satisfying the tracer's read/write contract.

        Returns:
            Self for method chaining.
        """
        self._config.mjlab_env = env
        return self

    def add_attribution(
        self,
        component: str,
        *,
        license: str | os.PathLike[str] | None = None,
        notice: str | os.PathLike[str] | None = None,
        copyright: str | None = None,
    ) -> SceneHandle:
        """Declare a third-party component this scene contains (ADR 0007 §2).

        Written to the scene directory as ``LICENSE.<component>`` and
        ``NOTICE.<component>``, and shown on mjswan Cloud beside the work's own license.
        Motion clips and policy weights are only ever declared this way: there is no
        file on disk to detect them from. A component already declared, by detection or
        an earlier call, is replaced.

        Args:
            component: Label for the component, e.g. ``"lafan1"``; letters, digits,
                ``_`` and ``-``.
            license: A path to the license text, copied verbatim, or a generatable SPDX
                id (``"BSD-3-Clause"``, ``"CC-BY-4.0"``, …) for the standard text.
            notice: A path to the notice, copied verbatim, or the text itself.
            copyright: The holder line of a generated license text.

        Returns:
            Self for method chaining.
        """
        if license is None and notice is None:
            raise ValueError(
                f"Attribution {component!r} needs a license, a notice, or both."
            )
        if not _COMPONENT.match(component):
            raise ValueError(
                f"Component {component!r} must be 1-64 letters, digits, '_' or '-'; "
                "it names the file (LICENSE.<component>)."
            )
        attribution = Attribution(
            component=component,
            license=(
                None
                if license is None
                else resolve_license(license, copyright=copyright)
            ),
            notice=None if notice is None else resolve_notice(notice),
            origin="author",
        )
        self._config.attributions = [
            a for a in self._config.attributions if a.component != component
        ] + [attribution]
        return self

    def clear_attributions(self) -> SceneHandle:
        """Drop every attribution on this scene, detected or declared. Returns self for
        method chaining."""
        self._config.attributions.clear()
        return self

    def set_metadata(self, key: str, value: Any) -> SceneHandle:
        """Set metadata for this scene.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self


__all__ = ["ViewerConfig", "SceneConfig", "SceneHandle", "SplatConfig", "SplatHandle"]
