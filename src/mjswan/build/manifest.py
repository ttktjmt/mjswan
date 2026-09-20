"""The manifest entries below the document header, and the one write of ``manifest.json``.

The one place that knows the shape of a project, scene, MDP and policy entry (ADR 0006).
Every key is ``snake_case``; a path resolves against the directory of the level that
declares it: the scene directory for everything under a scene entry, the document root
for the top-level ``plugins`` (manifest rules 1 and 2).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .._version import __version__
from ..document import DOCUMENT_FORMAT
from ..document.manifest import DEFAULT_IN_KEYS, DEFAULT_OUT_KEYS, RUNTIME_INPUT_SLOTS
from ..viewer import ViewerConfig
from .asset import motion_key
from .frontend import uses_custom_js
from .mdp import (
    policy_native_sizes,
    serialize_actions,
    serialize_command,
    serialize_events,
    serialize_observation_group,
    serialize_terminations,
)

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

    ``in_keys`` / ``out_keys`` (top-level or under ``onnx.meta``) belong on ``add_policy``,
    where the build checks them against the network (ADR 0006 §5). Found here they are
    ignored with a warning, so a stale table cannot quietly win over the code.
    """
    onnx_block = sidecar.get("onnx")
    meta = (onnx_block.get("meta") or {}) if isinstance(onnx_block, dict) else {}
    found = [k for k in ("in_keys", "out_keys") if k in sidecar or k in meta]
    if found:
        warnings.warn(
            f"{config_src.name} carries {found}; slot tables in a config_path sidecar "
            "are ignored. Declare them on add_policy(in_keys=..., out_keys=...); a "
            "single-input policy needs neither (ADR 0006 §5).",
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
    """Trace one MDP's five term sets into ``<scene>/mdp/<mdp_id>/`` and return its entry.

    ``owners`` are the policies that run against it, in order. The first supplies the
    per-policy context a trace needs: its joint names fix the native widths, its
    sidecar's ``actions`` block carries the authored PD gains. The rest must agree
    with it, a disagreement being a config mistake rather than a second MDP.
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
            "explicitly: `scene.set_trace_env(build_single_entity_trace_env(spec_fn))` "
            "(ADR 0005 §6)."
        )

    entry: dict = {"id": mdp_id}
    if mdp.commands:
        on_term("commands")
        entry["commands"] = {
            name: serialize_command(name, cmd, env, scene_dir, scope=scope)
            for name, cmd in mdp.commands.items()
        }
    if mdp.observations:
        on_term("observations")
        native_sizes = policy_native_sizes(
            {
                "policy_joint_names": first.policy_joint_names,
                "policy_num_actions": first.policy_num_actions,
                **{
                    k: first_sidecar[k]
                    for k in ("policy_joint_names", "policy_num_actions")
                    if k in first_sidecar and getattr(first, k) is None
                },
            },
            mdp.commands,
        )
        # Authored groups first, never overwritten: the key names the fused graph too.
        obs_config = dict(first_sidecar.get("observations") or {})
        for key, group in mdp.observations.items():
            target_key = f"{key}_monitor" if key in obs_config else key
            obs_config[target_key] = serialize_observation_group(
                group, env, scene_dir, target_key, native_sizes, scope=scope
            )
        entry["observations"] = obs_config
    elif first_sidecar.get("observations"):
        entry["observations"] = first_sidecar["observations"]
    if mdp.actions:
        entry["actions"] = serialize_actions(mdp.actions, first_sidecar.get("actions"))
    elif first_sidecar.get("actions"):
        entry["actions"] = first_sidecar["actions"]
    if mdp.terminations:
        on_term("terminations")
        terminations = serialize_terminations(
            mdp.terminations, env, scene_dir, scope=scope
        )
        if terminations:
            entry["terminations"] = terminations
    elif first_sidecar.get("terminations"):
        entry["terminations"] = first_sidecar["terminations"]
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
    return entry


def policy_entry(
    policy: PolicyConfig,
    mdp_id: str,
    sidecar: dict[str, Any],
    motion_files: dict[str, str],
    *,
    obs_keys: list[str],
) -> dict[str, Any]:
    """One policy's manifest entry: the checkpoint's own metadata plus its MDP ref.

    The sidecar's keys pass through except the MDP sections, which
    :func:`mdp_entry` merges; Python-side fields win over it. ``obs_keys`` are
    the MDP entry's observation groups, which the slot table is checked against. A
    table equal to the runtime's default is omitted (ADR 0006 §5).
    """
    entry: dict = {
        "id": policy.id,
        "name": policy.name,
        **({"default": True} if policy.default else {}),
        "mdp": mdp_id,
        "onnx": f"policy/{policy.id}.onnx",
    }
    skip = {*MDP_SIDECAR_KEYS, "id", "name", "mdp", "default", "motions"}
    entry.update({k: v for k, v in sidecar.items() if k not in skip})
    in_keys = _input_slots(policy, obs_keys)
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
    scene: SceneConfig, mdps: list[dict[str, Any]], policies: list[dict[str, Any]]
) -> dict[str, Any]:
    """A scene's manifest entry; every path in it resolves against the scene directory."""
    return {
        "id": scene.id,
        "name": scene.name,
        "scene": scene.scene_filename,
        **({"control_dt": _require_control_dt(scene)} if scene.policies else {}),
        "camera": (scene.viewer or ViewerConfig()).to_dict(),
        **({"terrain_data": scene.terrain_data} if scene.terrain_data else {}),
        **({"splat_section": True} if scene.splat_section and not scene.splats else {}),
        **({"splats": [splat_entry(s) for s in scene.splats]} if scene.splats else {}),
        "mdps": mdps,
        "policies": policies,
    }


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
            "add_scene(control_dt=...) as seconds per control step — mjlab's "
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
    """Write the one descriptor of the document, ``manifest.json``, at its root.

    Every key is ``snake_case``; a path resolves against the directory of the level
    that declares it: the scene directory for everything under a scene entry, the
    document root for the top-level ``plugins`` (ADR 0006, manifest rules 1 and 2).
    """
    custom_js = uses_custom_js()
    manifest = {
        "format": DOCUMENT_FORMAT,
        "version": __version__,
        "uses_custom_js": custom_js,
        # Author custom-MDP terms, loaded by the app in trusted contexts only.
        **({"plugins": "assets/plugins.js"} if custom_js else {}),
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                **({"default": True} if project.default else {}),
                "scenes": [
                    scene_entries[(project.id, scene.id)] for scene in project.scenes
                ],
            }
            for project in projects
        ],
    }
    with open(output_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


__all__ = [
    "MDP_SIDECAR_KEYS",
    "load_sidecar",
    "mdp_entry",
    "policy_entry",
    "scene_entry",
    "splat_entry",
    "write_manifest",
]
