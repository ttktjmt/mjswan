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
import onnx

from .adapters import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
    resolve_action_scales,
    resolve_pd_gains,
    resolve_runner_defaults,
)
from .licenses import Attribution, resolve_license, resolve_notice
from .mdp import MdpConfig
from .motion import MotionConfig
from .policy import PolicyConfig, PolicyHandle
from .splat import SplatConfig, SplatHandle
from .utils import assign_id, name2id, unique_id
from .viewer import ViewerConfig

if TYPE_CHECKING:
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


def _env_cfg_control_dt(env_cfg: Any) -> float | None:
    """An mjlab env config's seconds-per-control-step, or ``None`` if it carries neither.

    Mirrors ``ManagerBasedRlEnv.step_dt`` (``sim.mujoco.timestep * decimation``) so the
    rate can be read off a config without paying to construct the env.
    """
    try:
        return float(env_cfg.sim.mujoco.timestep) * int(env_cfg.decimation)
    except (AttributeError, TypeError, ValueError):
        return None


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


def _default_to_latest(handles: list[PolicyHandle]) -> None:
    """Open the scene on the highest-step checkpoint."""
    if not handles:
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
    """Seconds per control step — mjlab's ``env.step_dt`` (``timestep * decimation``).

    The rate the policy was trained to act at, and the ``dt`` every timer in the
    runtime counts in: the physics substep count per step, the command resample
    schedule, the interval-event triggers. It cannot be inferred from the model,
    which carries only the physics ``timestep``.

    Set automatically by :meth:`ProjectHandle.add_scene_mjlab` from the task's live
    env. **Required** for a scene built via plain :meth:`ProjectHandle.add_scene`
    that carries a policy — the build fails rather than defaulting, because a wrong
    control rate raises no error at playback, it just runs the policy at a speed it
    was not trained for. Deliberately *not* read from a trace env: the one
    :func:`mjswan.trace_env.build_single_entity_trace_env` builds declares
    ``decimation=1`` as a tracing placeholder, which is not anybody's control rate."""

    mjlab_env: Any = field(default=None, repr=False, compare=False)
    """Live env ONNX tracing (ADR 0005) runs authored observation/termination/
    event/command term bodies against. Built at build time from
    :attr:`mjlab_env_cfg` when the scene came from a task (see
    ``builder._scene_trace_env``), so a tracking task's env is constructed only once
    its clip is in the bundle — unless :meth:`SceneHandle.add_policy_wandb` already
    built one to export the checkpoints, which it hands over rather than closing. A
    scene built via plain :meth:`ProjectHandle.add_scene` (no mjlab task) has none by
    default — set
    one explicitly with :meth:`SceneHandle.set_trace_env` if it uses
    plain-callable (non-``Binding``) term functions. Only needs
    ``env.scene[name].data.<field>`` (and, for events, entity write methods) —
    doesn't have to be a full ``ManagerBasedRlEnv``, see
    :func:`mjswan.trace_env.build_single_entity_trace_env`. Python-build-time-
    only state; never part of the scene's serialized JSON output."""

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

    Used to reach the task's *runner* config for the two things playback needs from it
    (which observation group the actor reads, and ``clip_actions``) — see
    :func:`mjswan.adapters.resolve_runner_defaults`."""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = name2id(self.name)
        # The scene asset's filename, from whichever of spec/model was given. Fixed now
        # rather than a property: `_save_web` drops both right after writing the asset.
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


def _onnx_io_names(model: onnx.ModelProto) -> tuple[list[str], list[str]]:
    """The network's real input and output names, initializers excluded."""
    initializers = {init.name for init in model.graph.initializer}
    inputs = [i.name for i in model.graph.input if i.name not in initializers]
    return inputs, [o.name for o in model.graph.output]


def _check_slot_tables(
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
    inputs, outputs = _onnx_io_names(model)
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

        policy_dt = _env_cfg_control_dt(env_cfg)
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
        from .adapters.mjlab_adapter import adapt_events

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
        slot_in, slot_out = _check_slot_tables(name, policy, in_keys, out_keys)

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
        tracking_motion_term = _extract_tracking_motion_term(commands)
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

        handles = []
        seen_names: set[str] = set()
        if only_latest:
            for path in run_paths:
                from .wandb_io import fetch_onnx_from_wandb_run

                name, model = fetch_onnx_from_wandb_run(path)
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
                    _attach_tracking_motion(
                        handle,
                        path,
                        tracking_motion_term,
                        tracking_motion_cache,
                    )
                    handles.append(handle)
        else:
            # Deferred to build time so each scene converts and traces before the next
            # starts; converting up front held one mjlab env per scene alive.
            assert task_id is not None

            def _convert(on_run: Callable[[str], None] = lambda _: None) -> None:
                from .wandb_io import (
                    create_pt_onnx_export_context,
                    fetch_motion_npz_from_wandb_run,
                    fetch_pt_onnx_from_wandb_run,
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
                    if tracking_motion_term is not None:
                        existing_file = getattr(
                            tracking_motion_term, "motion_file", None
                        )
                        if existing_file and Path(existing_file).is_file():
                            motion_name = Path(existing_file).stem
                            motion_bytes = Path(existing_file).read_bytes()
                            motion_file_for_env = existing_file
                        else:
                            motion_name, motion_bytes = fetch_motion_npz_from_wandb_run(
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
                    try:
                        for path in run_paths:
                            on_run(path.rsplit("/", 1)[-1])
                            for name, model in fetch_pt_onnx_from_wandb_run(
                                path, task_id, export_context=export_context
                            ):
                                if name in seen_names:
                                    continue
                                seen_names.add(name)
                                handle = self.add_policy(
                                    name=name,
                                    policy=model,
                                    config_path=config_path,
                                    metadata=metadata,
                                    env_cfg=env_cfg,
                                    task_id=task_id,
                                    mdp=shared_mdp,
                                    policy_joint_names=export_context.joint_names
                                    or None,
                                    default_joint_pos=export_context.default_joint_pos
                                    or None,
                                    encoder_bias=export_context.encoder_bias or None,
                                    in_keys=in_keys,
                                    out_keys=out_keys,
                                    clip_actions=clip_actions,
                                    extras=extras,
                                )
                                _attach_tracking_motion(
                                    handle,
                                    path,
                                    tracking_motion_term,
                                    tracking_motion_cache,
                                    dataset_joint_names=export_context.joint_names
                                    or None,
                                )
                                handles.append(handle)
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
                _default_to_latest(handles)

            self._config.pending_conversions.append(
                PendingConversion(run_paths=list(run_paths), run=_convert)
            )

        _default_to_latest(handles)
        return handles

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
        from .adapters.mjlab_adapter import adapt_events

        self._config.events = adapt_events(events)
        self._config.events_explicit = _explicit
        return self

    def set_trace_env(self, env: Any) -> SceneHandle:
        """Set the live env ONNX tracing runs authored term bodies against.

        Required for a plain :meth:`ProjectHandle.add_scene` scene with plain-callable
        term functions, which has no task env of its own. The env only has to satisfy
        ``env.scene[name].data.<field>`` (plus the entity write methods for write-side
        terms) — see :func:`mjswan.trace_env.build_single_entity_trace_env` for a minimal
        one built from a single entity's spec.

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


def _extract_tracking_motion_term(commands: Mapping[str, Any] | None) -> Any | None:
    if not commands:
        return None
    for term in commands.values():
        if type(term).__name__ == "MotionCommandCfg":
            return term
        if hasattr(term, "anchor_body_name") and hasattr(term, "body_names"):
            return term
    return None


def _attach_tracking_motion(
    handle: PolicyHandle,
    run_path: str,
    tracking_motion_term: Any | None,
    cache: dict[str, tuple[str, bytes]],
    *,
    dataset_joint_names: list[str] | None = None,
) -> None:
    if tracking_motion_term is None:
        return

    from .wandb_io import fetch_motion_npz_from_wandb_run

    motion_file = getattr(tracking_motion_term, "motion_file", None)
    motion_path = Path(motion_file).expanduser() if motion_file else None
    if motion_path is not None and motion_path.is_file():
        motion_name = motion_path.stem or "motion"
        payload = motion_path.read_bytes()
    else:
        if run_path not in cache:
            cache[run_path] = fetch_motion_npz_from_wandb_run(run_path)
        motion_name, payload = cache[run_path]

    resolved_joint_names = (
        dataset_joint_names
        if dataset_joint_names is not None
        else (
            list(handle._config.policy_joint_names)
            if handle._config.policy_joint_names is not None
            else None
        )
    )
    motion = MotionConfig(
        name=motion_name,
        data=payload,
        anchor_body_name=getattr(tracking_motion_term, "anchor_body_name", ""),
        body_names=tuple(getattr(tracking_motion_term, "body_names", ()) or ()),
        dataset_joint_names=resolved_joint_names,
        default=True,
    )
    handle._append_motion(motion)
