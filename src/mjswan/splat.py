"""Gaussian Splat configuration and management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .scene import SceneHandle


@dataclass
class SplatConfig:
    """Configuration for a Gaussian Splat scene background."""

    name: str
    """Display name shown in the viewer control panel."""

    source: str | None = None
    """Local path to a .spz splat file to bundle into the app (recommended).
    The file is copied into the built application during :meth:`Builder.build`.
    Mutually exclusive with ``url``."""

    url: str | None = None
    """URL to an external .spz splat file (alternative to ``source``).
    The file is not bundled; the browser fetches it at runtime.
    Mutually exclusive with ``source``."""

    scale: float = 1.0
    """Metric scale factor (converts splat units to meters)."""

    x_offset: float = 0.0
    """X-axis position offset (in scaled splat units)."""

    y_offset: float = 0.0
    """Y-axis position offset (in scaled splat units)."""

    z_offset: float = 0.0
    """Z-axis position offset (vertical). Use ``ground_plane_offset`` from capture metadata if available."""

    roll: float = 0.0
    """Roll rotation in degrees applied on top of the COLMAP→Three.js base rotation."""

    pitch: float = 0.0
    """Pitch rotation in degrees applied on top of the COLMAP→Three.js base rotation."""

    yaw: float = 0.0
    """Yaw rotation in degrees applied on top of the COLMAP→Three.js base rotation."""

    collider_url: str | None = None
    """Optional URL or local path to a .glb collider mesh."""

    control: bool = False
    """Show scale and offset controls in the viewer control panel."""

    id: str = ""
    """Sanitized name, unique within the scene; the ``.spz`` file's stem. Assigned by
    :meth:`~mjswan.scene.SceneHandle.add_splat`; defaults to ``name2id(name)``."""

    def __post_init__(self) -> None:
        if not self.id:
            from .document.ids import name2id

            self.id = name2id(self.name)

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata for the splat."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a manifest entry. ``path`` is added externally when ``source`` is set."""
        d: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "scale": self.scale,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "z_offset": self.z_offset,
        }
        if self.url is not None:
            d["url"] = self.url
        if self.roll != 0.0:
            d["roll"] = self.roll
        if self.pitch != 0.0:
            d["pitch"] = self.pitch
        if self.yaw != 0.0:
            d["yaw"] = self.yaw
        if self.collider_url is not None:
            d["collider_url"] = self.collider_url
        if self.control:
            d["control"] = True
        return d


class SplatHandle:
    """Handle for configuring a Gaussian Splat scene background.

    This class provides a fluent API for configuring a splat after it has been
    added to a scene, mirroring the pattern used by PolicyHandle.

    Example:
        scene.add_splat(
            "Outdoor",
            source="background.spz",
            scale=1.35,
            z_offset=1.0,
        )
    """

    def __init__(self, splat_config: SplatConfig, scene: SceneHandle) -> None:
        self._config = splat_config
        self._scene = scene

    @property
    def source(self) -> str | None:
        """Local path to the .spz splat file (bundled into the app)."""
        return self._config.source

    @property
    def url(self) -> str | None:
        """URL to an external .spz splat file."""
        return self._config.url

    @property
    def scale(self) -> float:
        """Metric scale factor."""
        return self._config.scale

    @property
    def x_offset(self) -> float:
        """X-axis position offset."""
        return self._config.x_offset

    @property
    def y_offset(self) -> float:
        """Y-axis position offset."""
        return self._config.y_offset

    @property
    def z_offset(self) -> float:
        """Z-axis position offset (vertical)."""
        return self._config.z_offset

    @property
    def roll(self) -> float:
        """Roll rotation in degrees."""
        return self._config.roll

    @property
    def pitch(self) -> float:
        """Pitch rotation in degrees."""
        return self._config.pitch

    @property
    def yaw(self) -> float:
        """Yaw rotation in degrees."""
        return self._config.yaw

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
