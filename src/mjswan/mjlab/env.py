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
from typing import Any, Callable


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
    """

    def __init__(self, terms: dict[str, Any]):
        self._terms = dict(terms)

    def get_term(self, name: str) -> Any:
        if name not in self._terms:
            raise KeyError(
                f"Trace env has no command {name!r}; it knows "
                f"{sorted(self._terms)}. Pass it to "
                "`build_single_entity_trace_env(commands=...)`."
            )
        return self._terms[name]

    def get_command(self, name: str) -> Any:
        return self.get_term(name).command


def build_single_entity_trace_env(
    spec_fn: Callable[[], Any],
    *,
    entity_name: str = "robot",
    device: str = "cpu",
    zero_geom_margins: bool = True,
    commands: dict[str, Any] | None = None,
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
    env_cfg = ManagerBasedRlEnvCfg(decimation=1, scene=scene_cfg)
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
