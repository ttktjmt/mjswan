"""Build a minimal live env for ONNX tracing of non-mjlab scenes.

``add_scene_mjlab`` gets a tracing env for free; a plain ``add_scene()`` scene has none.
Tracing only ever needs ``env.scene[name].data.<field>`` and the entity write methods,
so this builds exactly that much out of mjlab's own ``Entity``/``Scene`` rather than
reimplementing entity-frame kinematics.
"""

from __future__ import annotations

import contextlib
import io
import re
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock


def _required_capacity(message: str, name: str) -> int | None:
    match = re.search(rf"{name} overflow \({name} must be >= (\d+)\)", message)
    return None if match is None else int(match.group(1))


def _next_capacity(required: int) -> int:
    return required + max(32, required // 8)


def _quiet_warp_module_loads() -> None:
    """Drop warp's per-kernel ``Module … load on device`` lines.

    Only the default level is nudged, so setting ``warp.config.log_level`` before the
    build brings them back.
    """
    import warp

    # warp 1.12 has no `log_level`; its module loads are quiet by default.
    if getattr(warp.config, "log_level", None) == getattr(warp, "LOG_INFO", object()):
        warp.config.log_level = warp.LOG_WARNING


def build_mjlab_env(env_cfg: Any, *, device: str = "cpu") -> Any:
    """Build a ``ManagerBasedRlEnv``, growing ``nconmax``/``njmax`` until it fits.

    Those buffers are sized for the training scene, and a single-env re-use can need
    more — which mujoco_warp only reports once the build fails.

    mjlab's manager tables are held back so they do not bury the build's progress, and
    replayed if the build fails.
    """
    from mjlab.envs import ManagerBasedRlEnv

    _quiet_warp_module_loads()
    tables = io.StringIO()
    while True:
        try:
            with contextlib.redirect_stdout(tables):
                return ManagerBasedRlEnv(cfg=env_cfg, device=device)
        except ValueError as exc:
            nconmax = _required_capacity(str(exc), "nconmax")
            njmax = _required_capacity(str(exc), "njmax")
            if nconmax is None and njmax is None:
                print(tables.getvalue(), end="")
                raise
            if nconmax is not None:
                env_cfg.sim.nconmax = _next_capacity(nconmax)
            if njmax is not None:
                env_cfg.sim.njmax = _next_capacity(njmax)
            tables.seek(0)
            tables.truncate(0)
        except Exception:
            print(tables.getvalue(), end="")
            raise


class TraceCommandManager:
    """Stand-in ``CommandManager`` serving trace-time values for browser-side commands.

    A traced term may read a command the browser owns and the trace env cannot build (a
    ``UiCommand``, a native ``TrackingCommand``). Only the tensor shapes matter: the
    values bake nothing, becoming graph inputs the runtime serves from the live command.
    *fallback* (the env's own manager) answers for the rest.
    """

    def __init__(self, terms: dict[str, Any], fallback: Any = None):
        self._terms = dict(terms)
        self._fallback = fallback

    def get_term(self, name: str) -> Any:
        if name in self._terms:
            return self._terms[name]
        term = _command_term(self._fallback, name)
        if term is None:
            known = {*self._terms, *getattr(self._fallback, "active_terms", ())}
            raise KeyError(
                f"Trace env has no command {name!r}; it knows {sorted(known)}. Pass it "
                "to `build_single_entity_trace_env(commands=...)`."
            )
        return term

    def get_command(self, name: str) -> Any:
        return self.get_term(name).command


def _command_term(manager: Any, name: str) -> Any:
    """*manager*'s term *name*, or ``None``: mjlab's ``NullCommandManager`` answers
    ``None`` where its ``CommandManager`` raises."""
    try:
        return manager.get_term(name)
    except (AttributeError, KeyError):
        return None


class TraceActionManager:
    """Stand-in ``ActionManager`` for a trace env with no action terms of its own.

    A traced ``last_action`` reads the policy's output, whose width a plain scene's env
    cannot know. Only the shape matters: the value becomes a graph input the runtime
    serves from the policy's last action.
    """

    def __init__(self, num_actions: int, *, num_envs: int = 1, device: Any = "cpu"):
        import torch

        self.action = torch.zeros((num_envs, num_actions), device=device)
        self.total_action_dim = num_actions
        # No terms, so a term-scoped `last_action` still fails, naming none.
        self.active_terms: list[str] = []
        self.action_term_dim: list[int] = []


class _MismatchedActions:
    """*real*, refusing a read of its action vector, which is not the policy's width."""

    def __init__(self, real: Any, num_actions: int):
        self._real = real
        self._num_actions = num_actions

    def _refuse(self) -> ValueError:
        return ValueError(
            f"The trace env's action terms are {_action_width(self._real)} wide, but the "
            f"policy outputs {self._num_actions} (`policy_num_actions`, else "
            "`policy_joint_names`). A traced last-action read would take the env's width "
            "while the browser holds the policy's."
        )

    @property
    def action(self) -> Any:
        raise self._refuse()

    def get_term(self, name: str) -> Any:
        raise self._refuse()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _action_width(manager: Any) -> int:
    """The width of *manager*'s action vector, as a traced read sees it."""
    action = getattr(manager, "action", None)
    return int(action.shape[-1]) if action is not None else 0


@contextlib.contextmanager
def policy_actions(env: Any, num_actions: int) -> Iterator[None]:
    """Give *env* a :class:`TraceActionManager` of the policy's width while tracing,
    unless it has action terms of its own, as an mjlab task's env does; those must be
    the policy's width."""
    real = getattr(env, "action_manager", None)
    width = _action_width(real)
    if env is None or not num_actions or width == num_actions:
        yield
        return
    if width:
        stand_in: Any = _MismatchedActions(real, num_actions)
    else:
        stand_in = TraceActionManager(
            num_actions,
            num_envs=getattr(env, "num_envs", 1),
            device=getattr(env, "device", "cpu"),
        )
    with mock.patch.object(env, "action_manager", stand_in, create=True):
        yield


@contextlib.contextmanager
def mdp_commands(env: Any, widths: Mapping[str, int]) -> Iterator[None]:
    """Give *env* a zero stand-in of each width in *widths* for a command it has no term
    for, as a plain scene's env has none for a command only the browser holds."""
    real = getattr(env, "command_manager", None)
    missing = {n: w for n, w in widths.items() if _command_term(real, n) is None}
    if env is None or not missing:
        yield
        return
    import torch

    num_envs = getattr(env, "num_envs", 1)
    device = getattr(env, "device", "cpu")
    stand_ins = {
        n: SimpleNamespace(command=torch.zeros((num_envs, w), device=device))
        for n, w in missing.items()
    }
    stand_in = TraceCommandManager(stand_ins, real)
    with mock.patch.object(env, "command_manager", stand_in, create=True):
        yield


#: mjlab's play configs' episode length for "no time limit".
_NO_TIME_LIMIT_S = 1e9


def build_single_entity_trace_env(
    spec_fn: Callable[[], Any],
    *,
    entity_name: str = "robot",
    device: str = "cpu",
    zero_geom_margins: bool = True,
    commands: dict[str, Any] | None = None,
    control_dt: float | None = None,
    episode_length_s: float | None = None,
) -> Any:
    """Build a minimal single-entity ``ManagerBasedRlEnv`` for ONNX tracing.

    The env configures no managers and is never stepped — it is only the tracer's
    ``env.scene[entity_name]`` read/write target. Returns it already ``reset()``; pass
    it to :meth:`mjswan.SceneHandle.set_trace_env`.

    Args:
        spec_fn: Zero-arg callable returning a fresh ``mujoco.MjSpec`` (mjlab's
            ``EntityCfg.spec_fn`` contract, so it must not share mutable state).
        entity_name: Match whatever the traced functions use as ``asset_cfg.name``.
        device: Torch device for the entity's tensors.
        zero_geom_margins: Zero every geom's contact margin before compiling, which
            mujoco_warp's collision backend requires of some robot XMLs. Safe here
            since nothing is simulated; set ``False`` to leave the geoms untouched.
        commands: Trace-time stand-ins for commands the browser owns, keyed by the name
            traced terms read. See :class:`TraceCommandManager`.
        control_dt: The scene's ``control_dt``, which the env's ``step_dt`` then equals,
            so a term counting control steps bakes the browser's count.
        episode_length_s: When ``time_out`` ends an episode. Unset means no time limit,
            as in mjlab's play configs; set, it needs ``control_dt``.
    """
    from mjlab.entity import EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg

    def _spec_fn():
        spec = spec_fn()
        if zero_geom_margins:
            for geom in spec.geoms:
                geom.margin = 0.0
        return spec

    # The browser resets to the keyframe, so `default_joint_pos` must match it; mjlab's
    # `{".*": 0.0}` would bake a zero default into every `*_rel` observation.
    keyframe_pos = _keyframe_joint_pos(_spec_fn())
    init_state = EntityCfg.InitialStateCfg()
    if keyframe_pos:
        init_state = EntityCfg.InitialStateCfg(joint_pos=keyframe_pos)
    entity_cfg = EntityCfg(spec_fn=_spec_fn, init_state=init_state)
    scene_cfg = SceneCfg(num_envs=1, entities={entity_name: entity_cfg})
    if episode_length_s is None:
        episode_length_s = _NO_TIME_LIMIT_S
    elif control_dt is None:
        raise ValueError(
            "episode_length_s is counted in control steps, so it needs the scene's "
            "control_dt as well."
        )
    env_cfg = ManagerBasedRlEnvCfg(
        decimation=1, scene=scene_cfg, episode_length_s=episode_length_s
    )
    if control_dt is not None:
        if not control_dt > 0:
            raise ValueError(f"control_dt must be positive, got {control_dt!r}.")
        # decimation=1 keeps step_dt exact; an ulp off can shift the horizon a step.
        env_cfg.sim.mujoco.timestep = control_dt
    # Nothing traced reads a contact, and more collision candidates than the one-env
    # `nconmax` holds overrun mujoco_warp's narrowphase buffer and can crash `reset()`.
    env_cfg.sim.mujoco.disableflags = ("contact",)
    # Through `build_mjlab_env` for its quieting.
    env = build_mjlab_env(env_cfg, device=device)
    env.reset()
    if commands:
        # After reset(), since mjlab builds its own empty manager during construction.
        env.command_manager = TraceCommandManager(commands)
    return env


def _keyframe_joint_pos(spec: Any) -> dict[str, float]:
    """Per-joint positions from the model's first keyframe, as ``EntityCfg`` wants them.

    Not mjlab's ``init_state.joint_pos = None``, which builds ``default_joint_pos`` from
    the float64 keyframe and then fails writing it into float32 ``qpos``. One-dof joints
    only, as ``InitialStateCfg.joint_pos`` assumes anyway.

    Empty when the model has no keyframe, leaving mjlab's ``{".*": 0.0}`` in place.
    """
    import re

    import mujoco

    if not spec.keys:
        return {}
    model = spec.compile()
    qpos = model.key_qpos[0]
    positions: dict[str, float] = {}
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            continue  # the root pose, not a joint position
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        # Keys are regexes to mjlab (`resolve_expr`), and a joint name is not one.
        positions[re.escape(name)] = float(qpos[model.jnt_qposadr[joint]])
    return positions


__all__ = ["TraceCommandManager", "build_single_entity_trace_env"]
