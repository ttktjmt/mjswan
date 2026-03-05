"""Gaussian Splat configuration and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scene import SceneHandle


@dataclass
class SplatConfig:
    """Configuration for a Gaussian Splat scene background."""

    url: str
    """URL or local path to the .spz splat file."""

    scale: float = 1.0
    """Metric scale factor (converts splat units to meters)."""

    ground_offset: float = 0.0
    """Ground plane offset in the splat's coordinate system."""

    collider_url: str | None = None
    """Optional URL or local path to a .glb collider mesh."""

    dev: bool = False
    """Expose calibration controls (scale, offset) in the viewer UI."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the splat."""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "url": self.url,
            "scale": self.scale,
            "groundOffset": self.ground_offset,
        }
        if self.collider_url is not None:
            d["colliderUrl"] = self.collider_url
        if self.dev:
            d["dev"] = True
        return d


class SplatHandle:
    """Handle for configuring a Gaussian Splat scene background.

    This class provides a fluent API for configuring a splat after it has been
    added to a scene, mirroring the pattern used by PolicyHandle.

    Example:
        splat = scene.add_splat(
            "https://cdn.example.com/scene.spz",
            scale=1.35,
            ground_offset=1.0,
        )
    """

    def __init__(self, splat_config: SplatConfig, scene: SceneHandle) -> None:
        self._config = splat_config
        self._scene = scene

    @property
    def url(self) -> str:
        """URL or local path to the .spz splat file."""
        return self._config.url

    @property
    def scale(self) -> float:
        """Metric scale factor."""
        return self._config.scale

    @property
    def ground_offset(self) -> float:
        """Ground plane offset."""
        return self._config.ground_offset

    def set_metadata(self, key: str, value: Any) -> SplatHandle:
        """Set metadata for this splat.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            Self for method chaining.
        """
        self._config.metadata[key] = value
        return self


__all__ = ["SplatConfig", "SplatHandle"]
