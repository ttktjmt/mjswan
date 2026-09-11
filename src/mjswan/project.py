"""Project configuration and management.

This module defines the ProjectConfig dataclass and ProjectHandle class for
managing projects containing multiple scenes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco

from .adapters import apply_mjlab_sim_options, ensure_mjlab_extensions
from .envs.mdp.events import apply_terrain_spawn
from .licenses import (
    detect_attributions,
    known_attribution,
    resolve_license,
    resolve_notice,
    spec_asset_directories,
)
from .scene import SceneConfig, SceneHandle, _env_cfg_control_dt
from .utils import assign_id, collect_spec_assets, name2id
from .viewer import ViewerConfig

if TYPE_CHECKING:
    from .builder import Builder


@dataclass
class ProjectConfig:
    """Configuration for a project containing multiple scenes."""

    name: str
    """Name of the project."""

    id: str = ""
    """Sanitized name, unique within the document: the project's directory in the build
    and its ``?project=`` value (ADR 0006 §4). Assigned by :meth:`Builder.add_project`;
    defaults to ``name2id(name)``."""

    scenes: list[SceneConfig] = field(default_factory=list)
    """List of scenes in the project."""

    default: bool = False
    """Open the app on this project. At most one project may set it; when none does,
    the first added is the default."""

    license: bytes | None = field(default=None, repr=False)
    """The work's ``LICENSE``, written to the project directory (ADR 0007 §1)."""

    notice: bytes | None = field(default=None, repr=False)
    """The work's ``NOTICE``, beside it."""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = name2id(self.name)


class ProjectHandle:
    """Handle for adding scenes and configuring a project.

    This class provides methods for adding scenes and customizing project properties.
    Similar to viser's server handle, this allows for hierarchical configuration.
    """

    def __init__(self, project_config: ProjectConfig, builder: Builder) -> None:
        self._config = project_config
        self._builder = builder

    @property
    def name(self) -> str:
        """Name of the project."""
        return self._config.name

    @property
    def id(self) -> str:
        """The project's id: its directory in the build and its ``?project=`` value."""
        return self._config.id

    def set_license(
        self, license: str | os.PathLike[str], *, copyright: str | None = None
    ) -> ProjectHandle:
        """Set the work's license, written as ``<project-id>/LICENSE`` (ADR 0007 §1).

        Args:
            license: A generatable SPDX id (one of
                :data:`mjswan.licenses.GENERATABLE_LICENSES`) for the standard text, or
                the path to a license text, copied verbatim.
            copyright: The holder line of a generated text, e.g. ``"2026 Example"``.

        Returns:
            Self for method chaining.
        """
        self._config.license = resolve_license(license, copyright=copyright)
        return self

    def set_notice(self, notice: str | os.PathLike[str]) -> ProjectHandle:
        """Set the work's notice, written as ``<project-id>/NOTICE``: the path to a
        file, copied verbatim, or the text itself. Returns self for method chaining."""
        self._config.notice = resolve_notice(notice)
        return self

    def add_scene(
        self,
        name: str,
        *,
        model: mujoco.MjModel | None = None,
        spec: mujoco.MjSpec | None = None,
        metadata: dict[str, Any] | None = None,
        control_dt: float | None = None,
        events: Mapping[str, Any] | None = None,
    ) -> SceneHandle:
        """Add a MuJoCo scene to this project.

        Provide either ``model`` or ``spec`` (not both).

        Using ``model`` saves the scene as a binary ``.mjb`` file, which loads
        faster in the browser but produces larger files. This is recommended
        when loading speed is a priority and storage size is not a concern.

        Using ``spec`` saves the scene as a compressed ``.mjz`` file, which
        uses significantly less storage but may take slightly longer to load.
        This is recommended when the generated web app exceeds 1 GB of storage
        (e.g., the GitHub Pages deployment limit).

        Args:
            name: Name for the scene (displayed in the UI).
            model: MuJoCo model for the scene (saved as .mjb).
            spec: MuJoCo spec for the scene (saved as .mjz).
            metadata: Optional metadata dictionary for the scene.
            control_dt: Seconds per control step — the rate the policy acts at,
                mjlab's ``timestep * decimation``. Required once the scene carries a
                policy: the model supplies only the physics ``timestep``, so nothing
                else can supply this, and a wrong control rate produces no error at
                playback — only a policy running at a speed it was not trained for.
                :meth:`add_scene_mjlab` fills it in from the task.
            events: Optional dict of ``EventTermCfg`` instances (mjswan or mjlab).
                Equivalent to calling :meth:`~mjswan.scene.SceneHandle.set_events`
                afterwards: the default every policy's MDP on this scene takes for its
                events (ADR 0006 §3).

        Returns:
            SceneHandle for adding policies and further configuration.

        For a ``spec`` loaded from disk, ``LICENSE*`` / ``NOTICE*`` files beside the
        model or its meshes and textures are copied verbatim into the scene directory as
        ``LICENSE.<component>`` (ADR 0007 §2); a model in the known-assets table gets a
        generated file when none is found. Adjust with
        :meth:`~mjswan.scene.SceneHandle.add_attribution` and
        :meth:`~mjswan.scene.SceneHandle.clear_attributions`.

        Example:
            ```
            # Fast loading (larger files):
            project.add_scene(
                model=mujoco.MjModel.from_xml_path("scene.xml"),
                name="My Scene",
            )

            # Compact storage (slower loading):
            project.add_scene(
                spec=mujoco.MjSpec.from_file("scene.xml"),
                name="My Scene",
            )
            ```
        """
        if model is not None and spec is not None:
            raise ValueError("Provide either 'model' or 'spec', not both.")
        if model is None and spec is None:
            raise ValueError("Either 'model' or 'spec' must be provided.")

        if metadata is None:
            metadata = {}

        scene_config = SceneConfig(
            name=name,
            id=assign_id(
                name, {s.id for s in self._config.scenes}, kind="scene", stacklevel=4
            ),
            model=model,
            spec=spec,
            metadata=metadata,
            control_dt=None if control_dt is None else float(control_dt),
        )
        if spec is not None:
            scene_config.attributions = detect_attributions(
                spec_asset_directories(spec)
            )
            if not scene_config.attributions:
                known = known_attribution(spec.modelname)
                if known is not None:
                    scene_config.attributions.append(known)
        self._config.scenes.append(scene_config)
        handle = SceneHandle(scene_config, self)
        if events:
            handle.set_events(events)
        return handle

    def add_scene_mjlab(
        self,
        task_id: str,
        *,
        play: bool | None = None,
        env_cfg: Any | None = None,
        events: Mapping[str, Any] | None = None,
    ) -> SceneHandle:
        """Add a MuJoCo scene from an mjlab task.

        Loads the task's MuJoCo spec from the mjlab task registry and adds it
        as a scene to this project. ``mjlab`` must be installed.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            play: Which of the task's two registered configs to load, as mjlab's
                ``load_env_cfg(task_id, play=...)`` does.

                Unset means **play**, unlike mjlab's own default: the training config
                sets ``episode_length_s`` to 10-20 s, so a viewer built from it resets
                the robot every few seconds while someone is watching. Pass ``False``
                to reproduce training-time conditions.

                Mutually exclusive with ``env_cfg``, which is already one of the two.
            env_cfg: Pre-loaded (and possibly edited) env config, loaded with the
                ``play`` you want. A tracking task does not need this — the builder aims
                mjlab's empty ``motion_file`` at the clip it bundles.

                The scene keeps whichever config it used, and its policies default their
                term sets off it, so this is also how those defaults pick up your edits.
            events: Scene events, overriding the task's own ``env_cfg.events``. Omit to
                take the task's (the usual case); pass ``{}`` for a scene with none.

        Returns:
            SceneHandle for further configuration (add_policy, add_splat, etc.)

        Example:
            ```python
            builder = mjswan.Builder()
            project = builder.add_project(name="My App")
            scene = project.add_scene_mjlab("go2_flat")
            app = builder.build()
            ```
        """
        try:
            from mjlab.scene import Scene
            from mjlab.tasks.registry import load_env_cfg
        except ImportError as e:
            raise ImportError(
                "mjlab is required for add_scene_mjlab(). "
                "Install it with: pip install mjlab"
            ) from e

        if env_cfg is not None and play is not None:
            raise ValueError(
                "Provide either 'play' or 'env_cfg', not both. `env_cfg` is already one "
                "of the task's two registered configs, so `play` has nothing left to "
                "select — load it as `load_env_cfg(task_id, play=...)` instead."
            )

        ensure_mjlab_extensions()
        if env_cfg is None:
            env_cfg = load_env_cfg(task_id, play=True if play is None else play)
        # Always 1: `num_envs` only sets the batch dimension, `Scene.spec` is the same
        # either way, and traced graphs carry a dynamic batch axis.
        env_cfg.scene.num_envs = 1
        scene = Scene(env_cfg.scene, device="cpu")
        scene.spec.assets.update(_collect_mjlab_scene_assets(env_cfg.scene))
        apply_mjlab_sim_options(scene.spec, getattr(env_cfg, "sim", None))
        handle = self.add_scene(spec=scene.spec, name=task_id)
        # Kept so policies on this scene can default their term sets off the same config
        # the scene (and its tracing env) was built from.
        handle._config.mjlab_env_cfg = env_cfg
        handle._config.mjlab_task_id = task_id
        # The composed scene spec has no directory; the entities' specs do, and the
        # task id names the robot when nothing sits beside them (ADR 0007 §2).
        if not handle._config.attributions:
            handle._config.attributions = detect_attributions(
                d
                for entity_spec in _mjlab_entity_specs(env_cfg.scene)
                for d in spec_asset_directories(entity_spec)
            )
        if not handle._config.attributions:
            known = known_attribution(task_id)
            if known is not None:
                handle._config.attributions.append(known)

        # The trace env comes later, from `builder._scene_trace_env`: a tracking task
        # cannot build one until its clip has been written into the bundle.
        # Rates differ per task — Cartpole 0.05, locomotion 0.02 — so read it here.
        control_dt = _env_cfg_control_dt(env_cfg)
        if control_dt is None:
            raise ValueError(
                f"Could not read a control rate off task {task_id!r}'s env config "
                "(sim.mujoco.timestep * decimation). Pass `control_dt` to add_scene "
                "and build the scene manually."
            )
        handle._config.control_dt = control_dt
        viewer_cfg = _adapt_mjlab_viewer_config(getattr(env_cfg, "viewer", None))
        if viewer_cfg is not None:
            handle.set_viewer(viewer_cfg)
        terrain_data = _extract_terrain_data(scene)
        if terrain_data:
            handle._config.terrain_data = terrain_data
        if events is not None:
            handle.set_events(events)
        elif getattr(env_cfg, "events", None):
            handle.set_events(env_cfg.events, _explicit=False)
        # After both: it rewrites an adapted event term using the patch table.
        apply_terrain_spawn(handle._config)
        return handle


def _extract_terrain_data(scene: Any) -> dict[str, Any] | None:
    """Extract spawn positions from a mjlab Scene for browser-side event execution.

    Tries named flat_patches first (higher-quality sampled positions); falls back
    to terrain_origins (one per sub-terrain tile) when flat_patch_sampling is not
    configured on any sub-terrain.
    """
    terrain = getattr(scene, "terrain", None)
    if terrain is None:
        return None

    # Try explicit flat_patches (only present when flat_patch_sampling is configured).
    flat_patches = getattr(terrain, "flat_patches", None)
    if flat_patches:
        serialized: dict[str, list[list[float]]] = {}
        for name, patches in flat_patches.items():
            # patches: (num_rows, num_cols, num_patches, 3) tensor
            try:
                arr = patches.cpu().numpy()
                rows, cols, n, _ = arr.shape
                positions = arr.reshape(rows * cols * n, 3).tolist()
                serialized[name] = positions
            except Exception:
                pass
        if serialized:
            return {"flat_patches": serialized}

    # Fall back to terrain_origins (one spawn point per sub-terrain tile).
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_origins is not None:
        try:
            arr = terrain_origins.cpu().numpy()
            # shape: (num_rows, num_cols, 3)
            num_rows, num_cols, _ = arr.shape
            positions = arr.reshape(num_rows * num_cols, 3).tolist()
            return {"flat_patches": {"spawn": positions}}
        except Exception:
            pass

    return None


def _mjlab_entity_specs(scene_cfg: Any) -> Iterator[mujoco.MjSpec]:
    """The terrain's and each entity's own spec, before the scene flattens them."""
    spec_cfgs = [getattr(scene_cfg, "terrain", None)]
    entities = getattr(scene_cfg, "entities", {})
    if isinstance(entities, dict):
        spec_cfgs.extend(entities.values())

    for cfg in spec_cfgs:
        spec_fn = getattr(cfg, "spec_fn", None)
        if not callable(spec_fn):
            continue
        spec = spec_fn()
        if isinstance(spec, mujoco.MjSpec):
            yield spec


def _collect_mjlab_scene_assets(scene_cfg: Any) -> dict[str, bytes]:
    """Collect assets from mjlab scene component specs before they are flattened."""
    assets: dict[str, bytes] = {}
    for spec in _mjlab_entity_specs(scene_cfg):
        assets.update(collect_spec_assets(spec))
    return assets


def _adapt_mjlab_viewer_config(config: Any | None) -> ViewerConfig | None:
    """Convert mjlab's ``ViewerConfig`` dataclass to mjswan's equivalent."""
    if config is None:
        return None

    defaults = ViewerConfig()
    entity_name = getattr(config, "entity_name", None)
    body_name = getattr(config, "body_name", None)
    if entity_name is None and body_name is not None:
        entity_name = "robot"
    origin_type_name = getattr(getattr(config, "origin_type", None), "name", None)
    if isinstance(origin_type_name, str):
        origin_type = getattr(ViewerConfig.OriginType, origin_type_name, None)
    else:
        origin_type = None

    return ViewerConfig(
        lookat=tuple(getattr(config, "lookat", (0.0, 0.0, 0.0))),
        distance=float(getattr(config, "distance", 4.0)),
        fovy=getattr(config, "fovy", None),
        elevation=float(getattr(config, "elevation", -30.0)),
        azimuth=float(getattr(config, "azimuth", 45.0)),
        origin_type=origin_type or defaults.origin_type,
        entity_name=entity_name,
        body_name=body_name,
        env_idx=int(getattr(config, "env_idx", 0)),
        max_extra_envs=int(getattr(config, "max_extra_envs", 2)),
        enable_reflections=bool(getattr(config, "enable_reflections", True)),
        enable_shadows=bool(getattr(config, "enable_shadows", True)),
        height=int(getattr(config, "height", 240)),
        width=int(getattr(config, "width", 320)),
    )


__all__ = ["ProjectConfig", "ProjectHandle"]
