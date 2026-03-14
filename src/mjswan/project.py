"""Project configuration and management.

This module defines the ProjectConfig dataclass and ProjectHandle class for
managing projects containing multiple scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco

from .scene import SceneConfig, SceneHandle

if TYPE_CHECKING:
    from .builder import Builder


@dataclass
class ProjectConfig:
    """Configuration for a project containing multiple scenes."""

    name: str
    """Name of the project."""

    id: str | None = None
    """Optional ID for the project used in URL routing (e.g., 'menagerie' for /#/menagerie/)."""

    scenes: list[SceneConfig] = field(default_factory=list)
    """List of scenes in the project."""


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
    def id(self) -> str | None:
        """Optional ID of the project for URL routing."""
        return self._config.id

    def add_scene(
        self,
        name: str,
        *,
        model: mujoco.MjModel | None = None,
        spec: mujoco.MjSpec | None = None,
        metadata: dict[str, Any] | None = None,
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

        Returns:
            SceneHandle for adding policies and further configuration.

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
            model=model,
            spec=spec,
            metadata=metadata,
        )
        self._config.scenes.append(scene_config)
        return SceneHandle(scene_config, self)


    def add_mjlab_scene(self, task_id: str) -> SceneHandle:
        """Add a MuJoCo scene from an mjlab task.

        Loads the task's MuJoCo spec from the mjlab task registry and adds it
        as a scene to this project. ``mjlab`` must be installed.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).

        Returns:
            SceneHandle for further configuration (add_policy, add_splat, etc.)

        Example:
            ```python
            builder = mjswan.Builder()
            project = builder.add_project(name="My App")
            scene = project.add_mjlab_scene("go2_flat")
            app = builder.build()
            ```
        """
        try:
            from mjlab.scene import Scene
            from mjlab.tasks.registry import load_env_cfg
        except ImportError as e:
            raise ImportError(
                "mjlab is required for add_mjlab_scene(). "
                "Install it with: pip install mjlab"
            ) from e

        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = 1
        scene = Scene(env_cfg.scene, device="cpu")
        return self.add_scene(spec=scene.spec, name=task_id)


__all__ = ["ProjectConfig", "ProjectHandle"]
