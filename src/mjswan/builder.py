"""Builder class for constructing mjswan applications.

This module provides the main Builder class which serves as the entry point
for programmatically creating interactive MuJoCo simulations.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import mujoco
import onnx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from . import __version__
from ._build_client import ClientBuilder, install_spa
from .app import MjswanApp
from .document import DOCUMENT_FORMAT
from .envs.mdp.actions.actions import (
    MuscleActivationActionCfg,
    validate_muscle_actuators,
)
from .policy import DEFAULT_IN_KEYS, DEFAULT_OUT_KEYS, RUNTIME_INPUT_SLOTS
from .project import ProjectConfig, ProjectHandle
from .scene import SceneConfig
from .splat import SplatConfig
from .utils import assign_id, collect_spec_assets, name2id, to_zip_deflated, unique_id
from .viewer import ViewerConfig


def _build_uses_custom_js() -> bool:
    """Whether the current build embeds author-supplied TypeScript.

    Surfaced at the top of ``manifest.json`` so a consumer can enforce a
    declarative-only policy without inspecting the bundled engine (ADR 0003).
    """
    from .command import _custom_registry as _command_registry
    from .envs.mdp.events import _custom_registry as _event_registry
    from .envs.mdp.observations import _custom_registry as _obs_registry
    from .envs.mdp.terminations import _custom_registry as _term_registry

    for registry in (_obs_registry, _term_registry, _event_registry, _command_registry):
        for sentinel in registry.values():
            if getattr(sentinel, "ts_src", None) is not None:
                return True
    return False


def _scene_trace_env(scene: SceneConfig) -> Any | None:
    """The env ONNX tracing runs term bodies against, built on first use.

    Deferred to build time so a tracking task's env is constructed only once its clip is
    on disk, and so a scene that traces nothing never pays for one at all.
    """
    if scene.mjlab_env is not None:
        return scene.mjlab_env
    if scene.mjlab_env_cfg is None:
        return None
    from .trace_env import build_mjlab_env

    scene.mjlab_env = build_mjlab_env(scene.mjlab_env_cfg)
    scene.mjlab_env.reset()
    return scene.mjlab_env


def _motion_key(motion: Any) -> str:
    """Identity of a clip's content, so two policies sharing one share its file."""
    if motion.data is not None:
        return hashlib.sha256(motion.data).hexdigest()
    if motion.source is not None:
        src = _resolve_motion_source(motion.source)
        try:
            return hashlib.sha256(src.read_bytes()).hexdigest()
        except OSError:
            return f"src:{src}"  # missing; the copy below warns about it
    return f"empty:{motion.name}"


def _resolve_motion_source(source: str) -> Path:
    src = Path(source).expanduser()
    return src if src.is_absolute() else (Path.cwd() / src).resolve()


def _write_scene_motions(scene: SceneConfig, scene_dir: Path) -> dict[str, str]:
    """Write each distinct clip in the scene once; return content key -> filename.

    Scene-scoped rather than per-policy: the checkpoints of one run share a clip, and a
    copy per policy meant N copies of the same megabytes. Same name and same content
    collapse to one file; same name and different content get a ``_1``/``_2`` suffix.
    """
    files: dict[str, str] = {}
    used: set[str] = set()
    for policy in scene.policies:
        for motion in policy.motions:
            key = _motion_key(motion)
            if key in files:
                continue
            stem = unique_id(name2id(motion.name), used)
            used.add(stem)
            filename = f"{stem}.npz"
            files[key] = filename

            target = scene_dir / filename
            if motion.data is not None:
                target.write_bytes(motion.data)
            elif motion.source is not None:
                src = _resolve_motion_source(motion.source)
                if src.exists():
                    shutil.copy2(str(src), str(target))
                else:
                    warnings.warn(
                        f"Motion source file not found: {src}",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
    return files


def _point_env_cfg_at_bundled_motion(
    scene: SceneConfig, scene_dir: Path, motion_files: dict[str, str]
) -> None:
    """Aim a tracking task's ``motion_file`` at the clip just written to the bundle.

    mjlab registers tracking tasks with ``motion_file=""`` and ``MotionLoader`` reads the
    path when the env is constructed, so the bundled copy is what the trace env loads —
    no second copy anywhere.
    """
    env_cfg = scene.mjlab_env_cfg
    if env_cfg is None or scene.mjlab_env is not None or not motion_files:
        return
    for term in (getattr(env_cfg, "commands", None) or {}).values():
        if not hasattr(term, "motion_file"):
            continue
        existing = getattr(term, "motion_file", "") or ""
        if existing and Path(existing).expanduser().is_file():
            continue
        term.motion_file = str(scene_dir / next(iter(motion_files.values())))
        return


def _require_control_dt(scene: Any) -> float:
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


class _SceneSteps:
    """The sub-step line under a project's progress bar: one phase at a time, counting
    its own units and naming the one it is on."""

    def __init__(self, progress: Progress):
        self._progress = progress
        self._task = progress.add_task("", total=1, scene="")
        self._started = False

    def begin(self, label: str, total: int) -> None:
        self._started = False
        # `total=None` means "keep the current total" to rich, so it is always passed.
        self._progress.reset(
            self._task, total=total, description=f"   └ {label}", scene=""
        )

    def on(self, name: str) -> None:
        """Name the unit now starting; the previous one counts as done."""
        self._progress.update(self._task, scene=name, advance=1 if self._started else 0)
        self._started = True

    def close(self) -> None:
        self._progress.remove_task(self._task)


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


def _input_slots(policy, obs_keys: list[str]) -> list[str]:
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


class Builder:
    """Builder for creating mjswan applications.

    The Builder class provides a fluent API for programmatically constructing
    interactive MuJoCo simulations with ONNX policies. It handles projects, scenes, and policies hierarchically.
    """

    def __init__(
        self,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize a new Builder instance.

        Args:
            base_path: Base path for subdirectory deployment (e.g., '/mjswan/').
            gtm_id: Google Tag Manager ID (e.g., 'GTM-XXXXXXX'). Injects GTM snippet if set.
            mt: Enable multi-threaded MuJoCo WASM. Requires COOP/COEP headers — these are
                written as a ``_headers`` file (Netlify/Cloudflare Pages/Vercel) and a
                service worker (required for GitHub Pages hosting). Defaults to False.
            debug: Keep browser console messages in the built app. Defaults to False
                (console messages are stripped from the production bundle).
        """
        self._projects: list[ProjectConfig] = []
        self._base_path = base_path
        self._gtm_id = gtm_id
        self._mt = mt
        self._debug = debug

    @classmethod
    def from_mjlab(
        cls,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> Builder:
        """Create a Builder pre-configured with a single mjlab task.

        This is a convenience factory for the common pattern of visualizing one
        mjlab task. The returned Builder can be further modified before calling
        :meth:`build`.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all ``model_*.pt``
                checkpoints from each run are fetched and converted to ONNX
                via mjlab+torch (both required). ``task_id`` above is reused
                for the conversion. Defaults to ``None`` (no policy attached).
                For finer control (e.g. ``only_latest=True``, custom
                observations/actions), build manually with
                :meth:`add_project` → :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`
                → :meth:`~mjswan.scene.SceneHandle.add_policy_wandb`.
            project_name: Name for the auto-created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config, for a task whose
                registered one is incomplete — mjlab's tracking tasks ship
                ``commands["motion"].motion_file = ""``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            base_path: Base path for the application (e.g., ``"/mjswan/"``).
            gtm_id: Optional Google Tag Manager container ID.

        Returns:
            Builder with one project and one scene already configured.

        Example:
            ```python
            # Minimal usage
            app = mjswan.Builder.from_mjlab("go2_flat").build()
            app.launch()

            # Load mjlab scene and attach all checkpoints from a W&B run
            app = mjswan.Builder.from_mjlab(
                "Mjlab-Velocity-Flat-Anymal-C",
                run_path="ttktjmt-org/mjlab/dqxvf0eb",
            ).build()

            # Customise before building
            builder = mjswan.Builder.from_mjlab("go2_flat")
            scene = builder.get_projects()[0].scenes[0]  # access SceneConfig
            app = builder.build()
            ```
        """
        builder = cls(base_path=base_path, gtm_id=gtm_id, mt=mt, debug=debug)
        builder.add_project_mjlab(
            task_id,
            run_path=run_path,
            project_name=project_name,
            play=play,
            env_cfg=env_cfg,
        )
        return builder

    def add_project_mjlab(
        self,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
    ) -> ProjectHandle:
        """Add a project pre-configured with a single mjlab task.

        Convenience for the common pattern of visualizing one mjlab task:
        creates a project, adds the mjlab scene, and (optionally) attaches all
        ``model_*.pt`` checkpoints from one or more W&B runs as ONNX policies.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all checkpoints are fetched
                and converted to ONNX via mjlab+torch (both required) using
                ``task_id``. Defaults to ``None`` (no policy attached).
            project_name: Name for the created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.

        Returns:
            ProjectHandle for the created project.
        """
        project = self.add_project(name=project_name)
        # `play` stays unresolved: `add_scene_mjlab` rejects it alongside `env_cfg`, so
        # materialising the default here would trip that guard for every caller.
        scene = project.add_scene_mjlab(task_id, play=play, env_cfg=env_cfg)
        if run_path is not None:
            scene.add_policy_wandb(run_path, task_id=task_id)
        return project

    def add_project(self, name: str, *, default: bool = False) -> ProjectHandle:
        """Add a new project to the builder.

        The project's id (its directory in the build and its ``?project=`` value) is
        ``name2id(name)``, made unique within the document: a second project that
        sanitizes to the same id is stored as ``<id>_1`` with a warning (ADR 0006 §4).

        Args:
            name: Name for the project (displayed in the UI).
            default: Open the app on this project. At most one project may set it;
                when none does, the first added is the default.

        Returns:
            ProjectHandle for adding scenes and further configuration.
        """
        if default:
            taken = next((p for p in self._projects if p.default), None)
            if taken is not None:
                raise ValueError(
                    f"Project {name!r} cannot be the default: {taken.name!r} already "
                    "is. Exactly one project may set default=True."
                )
        project = ProjectConfig(
            name=name,
            id=assign_id(name, {p.id for p in self._projects}, kind="project"),
            default=default,
        )
        self._projects.append(project)
        return ProjectHandle(project, self)

    def build(
        self,
        output_dir: str | Path | None = None,
        build_frontend: bool | None = None,
    ) -> MjswanApp:
        """Build the application from the configured projects.

        This method finalizes the configuration and creates a MjswanApp
        instance. If output_dir is provided, it also saves the application
        to that directory. If output_dir is not provided, it defaults to
        'dist' in the caller's directory.

        Args:
            output_dir: Optional directory to save the application files.
                       If None, defaults to 'dist' in the caller's directory.

        Returns:
            MjswanApp instance ready to be launched.
        """
        if not self._projects:
            raise ValueError(
                "Cannot build an empty application. "
                "You must add at least one project using builder.add_project() before building.\n"
                "Example:\n"
                "  builder = mwx.Builder()\n"
                "  project = builder.add_project(name='My Project')\n"
                "  scene = project.add_scene(spec=mujoco_spec, name='Scene 1')\n"
                "  app = builder.build()"
            )

        # Get caller's file path
        frame = inspect.stack()[1]
        caller_file = frame.filename
        # Handle REPL or interactive mode where filename might be <stdin> or similar
        if caller_file.startswith("<") and caller_file.endswith(">"):
            base_dir = Path.cwd()
        else:
            base_dir = Path(caller_file).parent

        if output_dir is None:
            output_path = base_dir / "dist"
        else:
            # Resolve relative paths against the caller's directory
            output_path = base_dir / Path(output_dir)

        # TODO: Build with separate function (and then save the web app with _save_web). And set scene.path and policy.path after building.
        self._save_web(output_path, build_frontend=build_frontend)

        return MjswanApp(output_path)

    # Keys of a `config_path` sidecar that describe the MDP rather than the checkpoint.
    _MDP_SIDECAR_KEYS = (
        "observations",
        "actions",
        "terminations",
        "commands",
        "events",
    )

    def _load_sidecar(self, policy) -> dict:
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

    def _serialize_mdp(
        self,
        mdp,
        mdp_id: str,
        owners: list,
        *,
        sidecars: dict[str, dict],
        env,
        scene_dir: Path,
        on_term,
    ) -> dict:
        """Trace one MDP's five term sets into ``<scene>/mdp/<mdp_id>/`` and return its entry.

        ``owners`` are the policies that run against it, in order. The first supplies the
        per-policy context a trace needs: its joint names fix the native widths, its
        sidecar's ``actions`` block carries the authored PD gains. The rest must agree
        with it, a disagreement being a config mistake rather than a second MDP.
        """
        from ._onnx_build import (
            policy_native_sizes,
            serialize_command,
            serialize_events,
            serialize_observation_group,
            serialize_terminations,
        )

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
                for key in self._MDP_SIDECAR_KEYS
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
            # Merged field-wise over the authored config, where a motor robot's PD gains
            # live, so a scene can tweak the offset without restating them.
            authored = first_sidecar.get("actions") or {}
            entry["actions"] = {
                name: {**authored.get(name, {}), **cfg.to_dict()}
                for name, cfg in mdp.actions.items()
            }
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

    def _serialize_policy_entry(
        self,
        policy,
        mdp_id: str,
        sidecar: dict,
        motion_files: dict[str, str],
        *,
        obs_keys: list[str],
    ) -> dict:
        """One policy's manifest entry: the checkpoint's own metadata plus its MDP ref.

        The sidecar's keys pass through except the MDP sections, which
        :meth:`_serialize_mdp` merges; Python-side fields win over it. ``obs_keys`` are
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
        skip = {*self._MDP_SIDECAR_KEYS, "id", "name", "mdp", "default", "motions"}
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
                motion.to_dict(f"assets/{motion_files[_motion_key(motion)]}")
                for motion in policy.motions
            ]
        return entry

    def _scene_entry(
        self, scene: SceneConfig, mdps: list[dict], policies: list[dict]
    ) -> dict:
        """A scene's manifest entry; every path in it resolves against the scene directory."""
        return {
            "id": scene.id,
            "name": scene.name,
            "scene": scene.scene_filename,
            **({"control_dt": _require_control_dt(scene)} if scene.policies else {}),
            "camera": (scene.viewer or ViewerConfig()).to_dict(),
            **({"terrain_data": scene.terrain_data} if scene.terrain_data else {}),
            **(
                {"splat_section": True}
                if scene.splat_section and not scene.splats
                else {}
            ),
            **(
                {"splats": [self._build_splat_config_dict(s) for s in scene.splats]}
                if scene.splats
                else {}
            ),
            "mdps": mdps,
            "policies": policies,
        }

    def _save_manifest(
        self, output_path: Path, scene_entries: dict[tuple[str, str], dict]
    ) -> None:
        """Write the one descriptor of the document, ``manifest.json``, at its root.

        Every key is ``snake_case``; a path resolves against the directory of the level
        that declares it: the scene directory for everything under a scene entry, the
        document root for the top-level ``plugins`` (ADR 0006, manifest rules 1 and 2).
        """
        uses_custom_js = _build_uses_custom_js()
        manifest = {
            "format": DOCUMENT_FORMAT,
            "version": __version__,
            "uses_custom_js": uses_custom_js,
            # Author custom-MDP terms, loaded by the app in trusted contexts only.
            **({"plugins": "assets/plugins.js"} if uses_custom_js else {}),
            "projects": [
                {
                    "id": project.id,
                    "name": project.name,
                    **({"default": True} if project.default else {}),
                    "scenes": [
                        scene_entries[(project.id, scene.id)]
                        for scene in project.scenes
                    ],
                }
                for project in self._projects
            ],
        }
        with open(output_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _save_mt_headers(self, output_path: Path) -> None:
        """Write COOP/COEP response headers needed by multi-threaded MuJoCo.

        Two mechanisms are written so the output works on any static host:
        - ``_headers``: honored by Netlify, Cloudflare Pages, and Vercel.
        - ``coi-serviceworker.js`` (emitted by the Vite build only when mt=True): used by the
          injected inline script for GitHub Pages, which cannot set response headers.
        """
        headers_content = (
            "/*\n"
            "  Cross-Origin-Opener-Policy: same-origin\n"
            "  Cross-Origin-Embedder-Policy: require-corp\n"
            "\n"
        )
        (output_path / "_headers").write_text(headers_content)

    def _build_splat_config_dict(self, splat: SplatConfig) -> dict:
        """A splat's manifest entry; ``path`` is the bundled copy under ``assets/``."""
        d = splat.to_dict()
        if splat.source is not None:
            d["path"] = f"assets/{splat.id}.spz"
        return d

    def _check_defaults(self) -> None:
        """Refuse two siblings both marked default (ADR 0006, manifest rule 3).

        None set is fine: the first in document order is then the default.
        """
        defaults = [p.name for p in self._projects if p.default]
        if len(defaults) > 1:
            raise ValueError(
                f"Projects {defaults!r} are all marked default=True; at most one may be."
            )
        for project in self._projects:
            for scene in project.scenes:
                names = [p.name for p in scene.policies if p.default]
                if len(names) > 1:
                    raise ValueError(
                        f"Scene {scene.name!r} has policies {names!r} all marked "
                        "default=True; at most one may be."
                    )

    def _policy_filename(self, name: str) -> str:
        if not name or name.strip() == "":
            raise ValueError("Policy name must be a non-empty string.")
        if "/" in name or "\\" in name:
            raise ValueError(
                "Policy name cannot contain path separators ('/' or '\\')."
            )
        return name

    def _validate_muscle_action_terms(self, scene: SceneConfig) -> None:
        """Validate every ``MuscleActivationActionCfg`` in the scene's policies.

        Each term's ``actuator_names`` must resolve to muscle-dyntype actuators
        in the scene's MuJoCo model. Raises ``ValueError`` on the first violation
        so users see configuration mistakes before deployment rather than at
        runtime in the browser.
        """
        muscle_terms: list[tuple[str, MuscleActivationActionCfg]] = []
        for policy in scene.policies:
            actions = getattr(policy, "actions", None) or {}
            for term_name, cfg in actions.items():
                if isinstance(cfg, MuscleActivationActionCfg):
                    muscle_terms.append((term_name, cfg))
        if not muscle_terms:
            return

        model = scene.model
        if model is None and scene.spec is not None:
            model = scene.spec.compile()
        if model is None:
            return

        for term_name, cfg in muscle_terms:
            validate_muscle_actuators(model, cfg, term_name=term_name)

    def _save_web(self, output_path: Path, build_frontend: bool | None = None) -> None:
        """Save as a complete web application.

        Output structure (ADR 0006 §2):
            dist/
            ├── index.html, logo.svg, robots.txt
            ├── assets/            (compiled js/css/wasm, plugins.js)
            ├── manifest.json      (the one descriptor; every key snake_case)
            └── <project-id>/
                └── <scene-id>/
                    ├── scene.mjz | scene.mjb
                    ├── mdp/<mdp-id>/{obs,term,command,event}/<name>.onnx
                    ├── policy/<policy-id>.onnx
                    └── assets/    (<motion>.npz, <splat>.spz)
        """
        self._check_defaults()
        if output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Copy template directory
        template_dir = Path(__file__).parent / "template"
        client_builder: ClientBuilder | None = None
        if template_dir.exists():
            # Build client first (reuses a cached SPA build when it matches)
            package_json = template_dir / "package.json"
            if package_json.exists():
                print("Building the mjswan application...")
                client_builder = ClientBuilder(template_dir)
                client_builder.build(
                    base_path=self._base_path,
                    gtm_id=self._gtm_id,
                    mt=self._mt,
                    debug=self._debug,
                    build_frontend=build_frontend,
                )

            if not install_spa(output_path, template_dir):
                warnings.warn(
                    f"No built SPA found at {template_dir / 'dist'}; the output will be "
                    "missing the web application.",
                    category=RuntimeWarning,
                )
        else:
            warnings.warn(
                f"Template directory not found at {template_dir}.",
                category=RuntimeWarning,
            )

        # The SPA's own directory; `plugins.js` lives beside the bundle it extends.
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Author custom-MDP terms compile to a runtime ESM the manifest points at, via esbuild.
        if _build_uses_custom_js() and client_builder is not None:
            print("Compiling custom-MDP term module (plugins.js)...")
            client_builder.build_plugins_module(assets_dir / "plugins.js")

        # Write COOP/COEP headers for multi-threaded MuJoCo (SharedArrayBuffer)
        if self._mt:
            self._save_mt_headers(output_path)

        scene_entries: dict[tuple[str, str], dict] = {}
        max_name_len = max(len(p.name) for p in self._projects)
        for project in self._projects:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[dim]{task.fields[scene]}"),
            ) as progress:
                task = progress.add_task(
                    project.name.ljust(max_name_len),
                    total=len(project.scenes),
                    scene="",
                )
                steps = _SceneSteps(progress)
                for scene in project.scenes:
                    progress.update(task, scene=scene.name)
                    scene_dir = output_path / project.id / scene.id
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    scene_path = scene_dir / scene.scene_filename

                    # First: the conversion is what creates this scene's policies (and
                    # hands over its export env as the trace env).
                    for pending in scene.pending_conversions:
                        steps.begin(
                            "converting checkpoints", total=len(pending.run_paths)
                        )
                        pending.run(steps.on)
                    scene.pending_conversions.clear()

                    self._validate_muscle_action_terms(scene)

                    steps.begin("building scene", total=3)
                    steps.on("packaging scene")
                    if scene.spec is not None:
                        scene.spec.assets.update(collect_spec_assets(scene.spec))
                        to_zip_deflated(scene.spec, str(scene_path))  # Saves as .mjz
                        scene.spec = None
                    else:
                        if scene.model is None:
                            raise RuntimeError(
                                f"Scene '{scene.name}' has no model to save as .mjb"
                            )
                        mujoco.mj_saveModel(
                            scene.model, str(scene_path)
                        )  # Saves as .mjb
                        scene.model = None
                    gc.collect()

                    # Before anything needing the trace env: a tracking task's env loads
                    # its clip from disk, and the bundled copy is that file.
                    steps.on("bundling clips")
                    scene_assets_dir = scene_dir / "assets"
                    scene_assets_dir.mkdir(exist_ok=True)
                    motion_files = _write_scene_motions(scene, scene_assets_dir)
                    _point_env_cfg_at_bundled_motion(
                        scene, scene_assets_dir, motion_files
                    )

                    steps.on("copying splats")
                    for splat in scene.splats:
                        if splat.source is not None:
                            src = Path(splat.source).expanduser()
                            if not src.is_absolute():
                                src = (Path.cwd() / src).resolve()
                            if src.exists():
                                shutil.copy2(
                                    str(src),
                                    str(scene_assets_dir / f"{splat.id}.spz"),
                                )
                            else:
                                warnings.warn(
                                    f"Splat source file not found: {src}",
                                    category=RuntimeWarning,
                                    stacklevel=2,
                                )
                    if not any(scene_assets_dir.iterdir()):
                        scene_assets_dir.rmdir()

                    if scene.events and not scene.policies and scene.events_explicit:
                        warnings.warn(
                            f"Scene {scene.name!r} has events but no policy. Events "
                            "belong to a policy's MDP (ADR 0006 §3), so with no policy "
                            "to carry them they are not written.",
                            category=RuntimeWarning,
                            stacklevel=2,
                        )

                    steps.begin(
                        "tracing mdps", total=len(scene.mdps) + len(scene.policies)
                    )
                    sidecars = {p.id: self._load_sidecar(p) for p in scene.policies}
                    mdp_entries = []
                    for mdp, mdp_id in zip(scene.mdps, scene.mdp_ids):
                        owners = [p for p in scene.policies if p.mdp is mdp]
                        steps.on(f"mdp/{mdp_id}")
                        mdp_entries.append(
                            self._serialize_mdp(
                                mdp,
                                mdp_id,
                                owners,
                                sidecars=sidecars,
                                env=_scene_trace_env(scene),
                                scene_dir=scene_dir,
                                on_term=lambda name, mdp_id=mdp_id: steps.on(
                                    f"mdp/{mdp_id}/{name}"
                                ),
                            )
                        )

                    policy_entries = []
                    if scene.policies:
                        (scene_dir / "policy").mkdir(exist_ok=True)
                    mdp_entry_by_id = {entry["id"]: entry for entry in mdp_entries}
                    for policy in scene.policies:
                        steps.on(f"policy/{policy.name}")
                        onnx.save(
                            policy.model,
                            str(scene_dir / "policy" / f"{policy.id}.onnx"),
                        )
                        mdp_id = scene.mdp_id(policy.mdp)
                        policy_entries.append(
                            self._serialize_policy_entry(
                                policy,
                                mdp_id,
                                sidecars[policy.id],
                                motion_files,
                                obs_keys=list(
                                    mdp_entry_by_id[mdp_id].get("observations") or {}
                                ),
                            )
                        )

                    scene_entries[(project.id, scene.id)] = self._scene_entry(
                        scene, mdp_entries, policy_entries
                    )
                    progress.advance(task)
                steps.close()

        self._save_manifest(output_path, scene_entries)

        print(f"✓ Saved mjswan application to: {output_path}")

    def get_projects(self) -> list[ProjectConfig]:
        """Get a copy of all project configurations.

        Returns:
            List of ProjectConfig objects.
        """
        return self._projects.copy()


__all__ = ["Builder"]
