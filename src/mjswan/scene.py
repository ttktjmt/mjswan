"""Scene configuration and management.

This module defines the SceneConfig dataclass and SceneHandle class for
managing MuJoCo scenes and their associated policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mujoco
import onnx

from .policy import PolicyConfig, PolicyHandle
from .splat import SplatConfig, SplatHandle

if TYPE_CHECKING:
    from .project import ProjectHandle


@dataclass
class SceneConfig:
    """Configuration for a MuJoCo scene."""

    name: str
    """Name of the scene."""

    model: mujoco.MjModel | None = None
    """MuJoCo model for the scene (saved as .mjb)."""

    spec: mujoco.MjSpec | None = None
    """MuJoCo spec for the scene (saved as .mjz)."""

    policies: list[PolicyConfig] = field(default_factory=list)
    """List of policies available for this scene."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the scene."""

    splat: SplatConfig | None = None
    """Optional Gaussian Splat background configuration."""

    @property
    def scene_filename(self) -> str:
        """Return the scene filename based on which field is set."""
        return "scene.mjz" if self.spec is not None else "scene.mjb"


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

    def add_policy(
        self,
        policy: onnx.ModelProto,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        source_path: str | None = None,
        config_path: str | None = None,
    ) -> PolicyHandle:
        """Add an ONNX policy to this scene.

        Args:
            policy: ONNX model containing the policy.
            name: Name for the policy (displayed in the UI).
            metadata: Optional metadata dictionary for the policy.
            source_path: Optional source path for the policy ONNX file.
            config_path: Optional source path for the policy config JSON file.

        Returns:
            PolicyHandle for configuring the policy (adding commands, etc.)

        Example:
            policy = scene.add_policy(
                policy=onnx.load("locomotion.onnx"),
                name="Locomotion",
                config_path="locomotion.json",
            )
            policy.add_velocity_command()
        """
        if metadata is None:
            metadata = {}

        policy_config = PolicyConfig(
            name=name,
            model=policy,
            metadata=metadata,
            source_path=source_path,
            config_path=config_path,
        )
        self._config.policies.append(policy_config)
        return PolicyHandle(policy_config, self)

    def add_splat(
        self,
        url: str,
        *,
        scale: float = 1.0,
        ground_offset: float = 0.0,
        collider_url: str | None = None,
        control: bool = False,
    ) -> SplatHandle:
        """Add a Gaussian Splat background to this scene.

        The splat is baked into the app's config during build() and loaded
        automatically when the scene is opened in the viewer.

        Args:
            url: URL or local path to the .spz splat file.
            scale: Metric scale factor. Use ``metric_scale_factor`` from your
                capture metadata if available.
            ground_offset: Ground plane offset in the splat's coordinate system.
                Use ``ground_plane_offset`` from your capture metadata if available.
            collider_url: Optional URL or local path to a .glb collision mesh.
            control: If True, shows scale and ground offset controls in the viewer
                control panel. Defaults to False.

        Returns:
            SplatHandle for further configuration.

        Example:
            scene.add_splat(
                "https://cdn.example.com/background.spz",
                scale=1.35,
                ground_offset=1.0,
            )
        """
        splat_config = SplatConfig(
            url=url,
            scale=scale,
            ground_offset=ground_offset,
            collider_url=collider_url,
            control=control,
        )
        self._config.splat = splat_config
        return SplatHandle(splat_config, self)

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


__all__ = ["SceneConfig", "SceneHandle", "SplatConfig", "SplatHandle"]
