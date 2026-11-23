"""The manifest entries below the document header, and the write of ``manifest.json``.

The one place that knows the shape of a project, scene, MDP and policy entry.
Every key is ``snake_case``; a path resolves against the directory of the level that
declares it: the scene directory for everything under a scene entry, the document root
for the top-level ``plugins``.

See ``context/document.md``.
"""

from __future__ import annotations

import contextlib
import json
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import mujoco

from .._version import __version__
from ..document import DOCUMENT_FORMAT
from ..document.manifest import DEFAULT_IN_KEYS, DEFAULT_OUT_KEYS, RUNTIME_INPUT_SLOTS
from ..envs.mdp.actions import TracedActionCfg
from ..envs.mdp.commands import TRACKING_STATE_FIELDS
from ..mjlab.env import mdp_commands, policy_actions
from ..mjlab.gui import record_scene_gui
from ..viewer import ColorSpace, ViewerConfig
from .asset import motion_key
from .credit import LicenseNames, scene_credits, work_entry
from .frontend import uses_custom_js
from .mdp import (
    command_widths,
    debug_vis_entry,
    serialize_actions,
    serialize_command,
    serialize_events,
    serialize_observation_group,
    serialize_terminations,
)
from .skin import skin_entry

if TYPE_CHECKING:
    from ..mdp import MdpConfig
    from ..policy import PolicyConfig
    from ..project import ProjectConfig
    from ..scene import SceneConfig
    from ..splat import SplatConfig

# Keys of a `config_path` sidecar that describe the MDP rather than the checkpoint.
MDP_SIDECAR_KEYS = (
    "observations",
    "actions",
    "terminations",
    "commands",
    "events",
)


def load_sidecar(policy: PolicyConfig) -> dict[str, Any]:
    """The policy's authored ``config_path`` JSON, or ``{}``.

    A missing file warns and reads as empty: the policy still ships, with whatever
    the Python side declared.
    """
    config_path = getattr(policy, "config_path", None)
    if not config_path:
        return {}
    config_src = Path(config_path).expanduser()
    if not config_src.is_absolute():
        config_src = (Path.cwd() / config_src).resolve()
    if not config_src.exists():
        warnings.warn(
            f"Policy config path not found: {config_src}",
            category=RuntimeWarning,
            stacklevel=2,
        )
        return {}
    with open(config_src, "r") as f:
        sidecar = json.load(f)
    return _strip_slot_tables(sidecar, config_src)


def _strip_slot_tables(sidecar: dict, config_src: Path) -> dict:
    """Drop a sidecar's ``onnx`` block and slot tables; those are declared in Python.

    ``in_keys``/``out_keys``, top-level or under ``onnx.meta``, belong on ``add_policy``
    where the build checks them against the network. Found here they are ignored with a
    warning, so a stale table cannot quietly win over the code.
    """
    onnx_block = sidecar.get("onnx")
    meta = (onnx_block.get("meta") or {}) if isinstance(onnx_block, dict) else {}
    found = [k for k in ("in_keys", "out_keys") if k in sidecar or k in meta]
    if found:
        warnings.warn(
            f"{config_src.name} carries {found}; slot tables in a config_path sidecar "
            "are ignored. Declare them on add_policy(in_keys=..., out_keys=...); a "
            "single-input policy needs neither.",
            category=RuntimeWarning,
            stacklevel=2,
        )
    return {
        k: v for k, v in sidecar.items() if k not in ("onnx", "in_keys", "out_keys")
    }


def mdp_entry(
    mdp: MdpConfig,
    mdp_id: str,
    owners: list[PolicyConfig],
    *,
    sidecars: dict[str, dict[str, Any]],
    env: Any,
    scene_dir: Path,
    on_term: Callable[[str], None],
) -> dict[str, Any]:
    """Trace one MDP's terms into ``<scene>/mdp/<mdp_id>/`` and return its entry.

    ``owners`` are the policies that run against it, in order. The first supplies the
    per-policy context a trace needs: its action count fixes the width of a last-action
    read, its sidecar's ``actions`` block carries the authored PD gains. The rest must
    agree with it, a disagreement being a config mistake rather than a second MDP.
    """
    first = owners[0]
    first_sidecar = sidecars[first.id]
    for other in owners[1:]:
        disagreements = [
            field
            for field in ("policy_joint_names", "policy_num_actions")
            if getattr(other, field) != getattr(first, field)
        ]
        disagreements += [
            f"config_path.{key}"
            for key in MDP_SIDECAR_KEYS
            if sidecars[other.id].get(key) != first_sidecar.get(key)
        ]
        if disagreements:
            raise ValueError(
                f"Policies {first.name!r} and {other.name!r} on scene "
                f"{scene_dir.name!r} share one MdpConfig but disagree on "
                f"{disagreements}. Policies trained against one MDP share these; "
                "give the odd one its own MdpConfig."
            )

    scope = f"mdp/{mdp_id}"
    if env is None and (mdp.observations or mdp.terminations):
        raise ValueError(
            f"MDP {mdp_id!r} on scene {scene_dir.name!r} has observation/termination "
            "terms to trace, but the scene has no trace env to trace them against. "
            "`add_scene_mjlab` supplies one; a plain `add_scene` scene needs it "
            "explicitly: `scene.set_trace_env(build_single_entity_trace_env(spec_fn))`."
        )

    entry: dict = {"id": mdp_id}
    if mdp.commands:
        on_term("commands")
        entry["commands"] = {
            name: serialize_command(name, cmd, env, scene_dir, scope=scope)
            for name, cmd in mdp.commands.items()
        }
    policy = {
        k: first_sidecar.get(k) if getattr(first, k) is None else getattr(first, k)
        for k in ("policy_joint_names", "policy_num_actions")
    }
    num_actions = policy["policy_num_actions"] or len(
        policy["policy_joint_names"] or ()
    )
    widths = command_widths(mdp.commands, entry.get("commands", {}))
    # A plain scene's env lacks the widths a last-action or command read needs.
    with policy_actions(env, num_actions), mdp_commands(env, widths):
        # Before the observations, which may read a traced action term's state.
        if mdp.actions:
            if any(isinstance(a, TracedActionCfg) for a in mdp.actions.values()):
                on_term("actions")
            entry["actions"] = serialize_actions(
                mdp.actions,
                first_sidecar.get("actions"),
                commands=entry.get("commands", {}),
                widths=widths,
                env=env,
                out_dir=scene_dir,
                scope=scope,
            )
        elif first_sidecar.get("actions"):
            entry["actions"] = first_sidecar["actions"]
        action_state = {
            name: [f["name"] for f in action["process"]["state_fields"]]
            for name, action in entry.get("actions", {}).items()
            if "process" in action
        }
        reading: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
        if action_state:
            # Lazy: a core install has no torch, which only a traced term needs.
            from ..compile.slot import traced_action_state

            reading = traced_action_state(action_state)
        with reading:
            _serialize_reading_actions(
                mdp,
                entry,
                first_sidecar,
                env=env,
                scene_dir=scene_dir,
                scope=scope,
                on_term=on_term,
            )
    if mdp.events:
        events = serialize_events(
            mdp.events,
            env,
            scene_dir,
            on_term=lambda name: on_term(f"event/{name}"),
            scope=scope,
        )
        if events:
            entry["events"] = events
    elif first_sidecar.get("events"):
        entry["events"] = first_sidecar["events"]
    if debug_vis := debug_vis_entry(env):
        entry["debug_vis"] = debug_vis
    return entry


def _serialize_reading_actions(
    mdp: MdpConfig,
    entry: dict[str, Any],
    sidecar: dict[str, Any],
    *,
    env: Any,
    scene_dir: Path,
    scope: str,
    on_term: Callable[[str], None],
) -> None:
    """The observation and termination terms, which may read the action terms."""
    if mdp.observations:
        on_term("observations")
        # Authored groups first, never overwritten: the key names the fused graph.
        obs_config = dict(sidecar.get("observations") or {})
        for key, group in mdp.observations.items():
            target_key = f"{key}_monitor" if key in obs_config else key
            obs_config[target_key] = serialize_observation_group(
                group, env, scene_dir, target_key, scope=scope
            )
        entry["observations"] = obs_config
    elif sidecar.get("observations"):
        entry["observations"] = sidecar["observations"]
    if mdp.terminations:
        on_term("terminations")
        terminations = serialize_terminations(
            mdp.terminations, env, scene_dir, scope=scope
        )
        if terminations:
            entry["terminations"] = terminations
    elif sidecar.get("terminations"):
        entry["terminations"] = sidecar["terminations"]


def policy_entry(
    policy: PolicyConfig,
    mdp: Mapping[str, Any],
    sidecar: dict[str, Any],
    motion_files: dict[str, str],
) -> dict[str, Any]:
    """One policy's manifest entry: the checkpoint's own metadata plus its MDP ref.

    The sidecar's keys pass through except the MDP sections, which
    :func:`mdp_entry` merges; Python-side fields win over it. ``mdp`` is the MDP's
    entry, which the slot table and the motions are checked against. A table equal to
    the runtime's default is omitted.
    """
    _check_motion_fields(policy, mdp)
    entry: dict = {
        "id": policy.id,
        "name": policy.name,
        **({"default": True} if policy.default else {}),
        "mdp": mdp["id"],
        "onnx": f"policy/{policy.id}.onnx",
    }
    skip = {*MDP_SIDECAR_KEYS, "id", "name", "mdp", "default", "motions"}
    entry.update({k: v for k, v in sidecar.items() if k not in skip})
    in_keys = _input_slots(policy, list(mdp.get("observations") or {}))
    if in_keys != list(DEFAULT_IN_KEYS):
        entry["in_keys"] = in_keys
    if policy.out_keys is not None and policy.out_keys != list(DEFAULT_OUT_KEYS):
        entry["out_keys"] = policy.out_keys

    if policy.policy_joint_names:
        entry["policy_joint_names"] = policy.policy_joint_names
    if policy.policy_num_actions:
        entry["policy_num_actions"] = policy.policy_num_actions
    if policy.default_joint_pos:
        entry["default_joint_pos"] = policy.default_joint_pos
    if policy.encoder_bias:
        entry["encoder_bias"] = policy.encoder_bias
    # Not `if policy.clip_actions:`: 0.0 is a legal bound, not "unset".
    if policy.clip_actions is not None:
        entry["clip_actions"] = float(policy.clip_actions)
    if getattr(policy, "initial_qpos", None):
        entry["initial_qpos"] = policy.initial_qpos
    if getattr(policy, "initial_qvel", None):
        entry["initial_qvel"] = policy.initial_qvel
    if getattr(policy, "extras", None):
        entry["extras"] = policy.extras
    if policy.motions:
        entry["motions"] = [
            motion.to_dict(f"assets/{motion_files[motion_key(motion)]}")
            for motion in policy.motions
        ]
    return entry


def _check_motion_fields(policy: PolicyConfig, mdp: Mapping[str, Any]) -> None:
    """Refuse a graph reading a tracking command field no motion of the policy serves.

    ``TrackingCommand`` answers a field it neither computes nor finds in the selected
    motion's ``frame_fields`` with nothing, and the graph then holds a stale value.
    """
    tracking = {
        name
        for name, command in (mdp.get("commands") or {}).items()
        if command.get("name") == "TrackingCommand"
    }
    read = {
        slot["field"]
        for slot in _graph_slots(mdp)
        if slot.get("command") in tracking and "field" in slot
    }
    for field in sorted(read - TRACKING_STATE_FIELDS):
        lacking = [m.name for m in policy.motions if field not in m.frame_fields]
        if lacking or not policy.motions:
            whose = f"motions {lacking} do not" if lacking else "no motion does"
            raise ValueError(
                f"Policy {policy.name!r} has a graph reading tracking command field "
                f"{field!r}, which the browser serves only from a motion's "
                f"frame_fields, and {whose} list it. Give each motion a {field!r} "
                f"array, one row per frame, and add_motion(frame_fields=[{field!r}])."
            )


def _graph_slots(node: Any) -> Iterator[dict[str, Any]]:
    """Every graph input slot in an MDP entry, however deep its graph sits."""
    if isinstance(node, Mapping):
        yield from node.get("input_slots") or ()
        for value in node.values():
            yield from _graph_slots(value)
    elif isinstance(node, list):
        for value in node:
            yield from _graph_slots(value)


def _input_slots(policy: PolicyConfig, obs_keys: list[str]) -> list[str]:
    """The policy's effective ``in_keys``, checked against its MDP's observation groups.

    Undeclared, the network has one input (``add_policy`` refuses a multi-input policy
    without a table) and its one observation group fills it, whatever the group is
    called; with several groups the default slot must be one of them. Every slot must
    then name a group that exists or a tensor the runtime supplies, since anything else
    surfaces only at playback, as a missing input with the policy silently inert.
    """
    keys: list[str]
    if policy.in_keys is not None:
        keys = [str(k) for k in policy.in_keys]
    elif len(obs_keys) == 1:
        keys = list(obs_keys)
    elif not obs_keys or DEFAULT_IN_KEYS[0] in obs_keys:
        keys = list(DEFAULT_IN_KEYS)
    else:
        raise ValueError(
            f"Policy {policy.name!r} has one ONNX input but {len(obs_keys)} observation "
            f"groups ({obs_keys}) and no in_keys; pass in_keys=[<group>] naming the one "
            "that feeds it, or hand over just that group."
        )
    unknown = [k for k in keys if k not in obs_keys and k not in RUNTIME_INPUT_SLOTS]
    if unknown and obs_keys:
        raise ValueError(
            f"Policy {policy.name!r}: in_keys names {unknown}, which is neither one of "
            f"its observation groups ({obs_keys}) nor a tensor the runtime supplies "
            f"({sorted(RUNTIME_INPUT_SLOTS)}). Playback would find no input for it."
        )
    return keys


def scene_entry(
    scene: SceneConfig,
    mdps: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    licenses: LicenseNames,
    graphics: dict[str, Any],
) -> dict[str, Any]:
    """A scene's manifest entry; its paths resolve against the scene directory.
    ``licenses`` names the project's license files (:func:`.credit.license_names`), and
    ``graphics`` is :func:`graphics_entry`'s, read while the scene still has its model."""
    return {
        "id": scene.id,
        "name": scene.name,
        "scene": scene.scene_filename,
        **({"control_dt": _require_control_dt(scene)} if scene.policies else {}),
        "camera": (scene.viewer or ViewerConfig()).to_dict(),
        "graphics": graphics,
        **({"terrain_data": scene.terrain_data} if scene.terrain_data else {}),
        **({"splat_section": True} if scene.splat_section and not scene.splats else {}),
        **({"splats": [splat_entry(s) for s in scene.splats]} if scene.splats else {}),
        **(
            {"skins": [skin_entry(skin) for skin in scene.skins]} if scene.skins else {}
        ),
        **({"credits": credits} if (credits := scene_credits(scene, licenses)) else {}),
        **(
            {"term_defaults": term_defaults_entry(scene, mdps)}
            if scene.term_defaults
            else {}
        ),
        "mdps": mdps,
        "policies": policies,
    }


def term_defaults_entry(
    scene: SceneConfig, mdps: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """A scene's ``term_defaults`` (format 10), warning for a name none of its MDPs has:
    the viewer would carry it for nothing."""
    for kind in ("terminations", "events"):
        known = {name for mdp in mdps for name in _term_names(mdp.get(kind))}
        for name in scene.term_defaults.get(kind, {}):
            if name not in known:
                warnings.warn(
                    f"Scene {scene.name!r} sets a default for {kind[:-1]} {name!r}, which "
                    f"none of its MDPs has (they have {sorted(known) or 'none'}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return scene.term_defaults


def _term_names(terms: Any) -> list[str]:
    if isinstance(terms, Mapping):
        return list(terms)
    return [t["name"] for t in terms or [] if isinstance(t, Mapping) and "name" in t]


#: MuJoCo's Filament renderer's own values for the flags the build reads, and the one it
#: falls back to on a value it does not know (`scene_view.cc`, MuJoCo 4.0).
_FILAMENT_TEXTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "tone_mapping": (
        "filament.cg.tone_mapping",
        ("pbr_neutral", "aces", "aces_legacy", "filmic", "linear"),
    ),
    "output_color_space": ("filament.cg.output_color_space", get_args(ColorSpace)),
    "base_color_type": ("filament.material.base_color_type", get_args(ColorSpace)),
}


def graphics_entry(
    viewer: ViewerConfig | None, model: mujoco.MjSpec | mujoco.MjModel
) -> dict[str, Any]:
    """A scene's ``graphics``, complete: the viewer's settings, what it leaves unset from
    the model's ``filament.*`` (MuJoCo's Filament renderer's flags, read under MuJoCo
    4.0's names), then MuJoCo's defaults. Every scene gets the Filament ones, so the
    viewer can switch a classic scene to Filament and see what MuJoCo would draw."""
    viewer = viewer or ViewerConfig()
    texts = {
        field: _filament_text(model, name, known, getattr(viewer, field))
        for field, (name, known) in _FILAMENT_TEXTS.items()
    }
    exposure = viewer.exposure
    if exposure is None:
        exposure = _custom_number(model, "filament.cg.exposure")
        # JSON has no infinity or NaN: the browser would refuse the whole manifest.
        if exposure is not None and not math.isfinite(exposure):
            warnings.warn(
                f"The model's filament.cg.exposure is {exposure}, not a finite number of "
                "stops; the viewer draws 0.",
                category=RuntimeWarning,
                stacklevel=2,
            )
            exposure = None
    return {
        "color_management": viewer.color_management,
        "tone_mapping": texts["tone_mapping"],
        "exposure": float(exposure) if exposure is not None else 0.0,
        "output_color_space": texts["output_color_space"],
        "base_color_type": texts["base_color_type"],
        # The browser takes a switch as a boolean alone, and keeps its default otherwise.
        "enable_shadows": bool(viewer.enable_shadows),
        "enable_reflections": bool(viewer.enable_reflections),
        "enable_haze": bool(viewer.enable_haze),
    }


def _filament_text(
    model: mujoco.MjSpec | mujoco.MjModel,
    name: str,
    known: tuple[str, ...],
    value: str | None,
) -> str:
    """The viewer's value, else the model's flag as MuJoCo reads it, else MuJoCo's default."""
    if value is not None:
        return value
    value = _custom_text(model, name)
    if value is None:
        return known[0]
    if value not in known:
        warnings.warn(
            f"The model's {name} is {value!r}, which MuJoCo's Filament renderer does not "
            f"know ({', '.join(known)}); it draws {known[0]!r}, and so does the viewer.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        return known[0]
    return value


def _custom_text(model: mujoco.MjSpec | mujoco.MjModel, name: str) -> str | None:
    """A ``<custom><text>``'s data, looked up by its full name as MuJoCo's renderer does."""
    if isinstance(model, mujoco.MjSpec):
        return next((t.data for t in model.texts if t.name == name), None)
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    start = model.text_adr[i]
    return bytes(model.text_data[start : start + model.text_size[i] - 1]).decode()


def _custom_number(model: mujoco.MjSpec | mujoco.MjModel, name: str) -> float | None:
    """A ``<custom><numeric>``'s first value."""
    if isinstance(model, mujoco.MjSpec):
        data = next((list(n.data) for n in model.numerics if n.name == name), None)
        return data[0] if data else None
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    return float(model.numeric_data[model.numeric_adr[i]]) if i >= 0 else None


def _require_control_dt(scene: SceneConfig) -> float:
    """A scene's seconds-per-control-step, or fail naming the scene.

    Raises rather than defaulting, because a wrong rate raises no error at playback: the
    substep count, resample schedule, and interval timers all just count in the wrong
    unit, and it reads as a policy that behaves badly.

    Only reached for a scene carrying a policy; a viewer-only scene has no rate.
    """
    if scene.control_dt is None:
        raise ValueError(
            f"Scene {scene.name!r} has policies but no control_dt. Pass it to "
            "add_scene(control_dt=...) as seconds per control step: mjlab's "
            "`timestep * decimation`, the rate the policy was trained to act at. The "
            "model carries only the physics timestep, so this cannot be inferred, and "
            "guessing it wrong is silent. add_scene_mjlab() fills it in from the task."
        )
    if scene.control_dt <= 0:
        raise ValueError(
            f"Scene {scene.name!r} has control_dt={scene.control_dt!r}; it must be a "
            "positive number of seconds."
        )
    return float(scene.control_dt)


def splat_entry(splat: SplatConfig) -> dict[str, Any]:
    """A splat's manifest entry; ``path`` is the bundled copy under ``assets/``."""
    d = splat.to_dict()
    if splat.source is not None:
        d["path"] = f"assets/{splat.id}.spz"
    return d


def write_manifest(
    output_path: Path,
    projects: list[ProjectConfig],
    scene_entries: dict[tuple[str, str], dict[str, Any]],
) -> None:
    """Write the document's one descriptor, ``manifest.json``, at its root."""
    custom_js = uses_custom_js()
    manifest = {
        "format": DOCUMENT_FORMAT,
        "version": __version__,
        "uses_custom_js": custom_js,
        # Author custom-MDP terms, loaded by the app in trusted contexts only.
        **({"plugins": "assets/plugins.js"} if custom_js else {}),
        **({"viewer_gui": viewer_gui} if (viewer_gui := record_scene_gui()) else {}),
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                **({"default": True} if project.default else {}),
                **({"work": work} if (work := work_entry(project)) else {}),
                "scenes": [
                    scene_entries[(project.id, scene.id)] for scene in project.scenes
                ],
            }
            for project in projects
        ],
    }
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)


__all__ = [
    "MDP_SIDECAR_KEYS",
    "load_sidecar",
    "mdp_entry",
    "policy_entry",
    "scene_entry",
    "splat_entry",
    "write_manifest",
]
