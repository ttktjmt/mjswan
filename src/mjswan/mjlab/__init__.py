"""The mjlab side of mjswan: mjlab's objects and output formats, in mjswan's terms.

Not the real ``mjlab``. That is imported lazily, inside the functions that need it, so a
build that never touches an mjlab task never pays for it, and :mod:`.onnx_meta` needs it
not at all. Three kinds of things live here:

- adapters, one per manager kind (:mod:`.observation`, :mod:`.termination`,
  :mod:`.command`, :mod:`.action`, :mod:`.event`): an env config's term sets become
  mjswan cfgs, mjlab's own term functions kept for tracing;
- the task and the runner: what a scene takes from ``env_cfg`` (:mod:`.task`), what
  playback takes from ``rl_cfg`` and how a checkpoint becomes ONNX (:mod:`.runner`);
- the live-env side: envs to trace against (:mod:`.env`), sim options applied to a spec
  (:mod:`.sim`), the viser GUI recorded as a control-panel descriptor (:mod:`.gui`), and
  the metadata mjlab bakes into an exported ``.onnx`` (:mod:`.onnx_meta`).

``mjswan.managers`` and ``mjswan.envs.mdp`` are different: they *mirror* mjlab's import
paths rather than translate its objects, and stay where mjlab has them.
"""

from .action import adapt_actions, resolve_action_scales, resolve_pd_gains
from .command import adapt_commands, default_viz
from .detect import is_from_mjlab
from .env import build_mjlab_env, build_single_entity_trace_env
from .event import adapt_events, apply_terrain_spawn
from .observation import DEFAULT_OBS_GROUP_KEY, adapt_observations
from .runner import MjlabRunnerDefaults, resolve_runner_defaults
from .sim import apply_mjlab_sim_options, ensure_mjlab_extensions
from .termination import adapt_terminations

__all__ = [
    "DEFAULT_OBS_GROUP_KEY",
    "MjlabRunnerDefaults",
    "adapt_actions",
    "adapt_commands",
    "adapt_events",
    "adapt_observations",
    "adapt_terminations",
    "apply_mjlab_sim_options",
    "apply_terrain_spawn",
    "build_mjlab_env",
    "build_single_entity_trace_env",
    "default_viz",
    "ensure_mjlab_extensions",
    "is_from_mjlab",
    "resolve_action_scales",
    "resolve_pd_gains",
    "resolve_runner_defaults",
]
