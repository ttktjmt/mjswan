"""Viewer configuration for mjswan scenes."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ViewerConfig:
    """Viewer configuration for a scene.

    Matches the API of ``mjlab.viewer.ViewerConfig``.
    """

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at point in MuJoCo coordinates (x forward, y left, z up)."""

    distance: float = 4.0
    """Distance from the look-at point to the viewer."""

    fovy: float | None = None
    """Vertical field of view in degrees. Defaults to 45."""

    elevation: float = -30.0
    """Viewer elevation in degrees (negative = viewer above the look-at point)."""

    azimuth: float = 45.0
    """Viewer azimuth in degrees measured from the x-axis (forward) CCW."""

    class OriginType(enum.Enum):
        """The frame in which the viewer position and target are defined."""

        AUTO = enum.auto()
        """Track the first non-fixed body, or fall back to a free viewer."""
        WORLD = enum.auto()
        """Free viewer at the configured lookat point."""
        ASSET_ROOT = enum.auto()
        """Track the root body of the asset defined by entity_name."""
        ASSET_BODY = enum.auto()
        """Track the body defined by body_name in the asset defined by entity_name."""

    origin_type: OriginType = OriginType.AUTO
    """How the viewer origin is determined."""

    entity_name: str | None = None
    """Name of the asset/entity (unused in single-entity scenes)."""

    body_name: str | None = None
    """Body to track when origin_type is ASSET_BODY."""

    env_idx: int = 0
    """Environment index to follow."""

    max_extra_envs: int = 2
    """Number of neighboring environments to render around env_idx."""

    enable_reflections: bool = True
    """Whether to enable reflections."""

    enable_shadows: bool = True
    """Whether to enable shadows."""

    height: int = 240
    """Viewer canvas height in pixels."""

    width: int = 320
    """Viewer canvas width in pixels."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the manifest's ``camera`` entry (snake_case, ADR 0006)."""
        d: dict[str, Any] = {
            "lookat": list(self.lookat),
            "distance": self.distance,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "origin_type": self.origin_type.name,
            "enable_reflections": self.enable_reflections,
            "enable_shadows": self.enable_shadows,
            "height": self.height,
            "width": self.width,
            # None means "45" to Python; the document says what it means.
            "fovy": self.fovy if self.fovy is not None else 45.0,
        }
        if self.entity_name is not None:
            d["entity_name"] = self.entity_name
        if self.body_name is not None:
            d["body_name"] = self.body_name
        return d

    @staticmethod
    def from_position(
        position: tuple[float, float, float],
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        *,
        fovy: float | None = None,
        origin_type: "ViewerConfig.OriginType" = None,  # type: ignore[assignment]
        body_name: str | None = None,
    ) -> "ViewerConfig":
        """Create a ViewerConfig from an explicit viewer position and look-at target.

        Computes lookat, distance, elevation, and azimuth from MuJoCo-coordinate
        position/target vectors.

        Args:
            position: Viewer position in MuJoCo coordinates (x forward, y left, z up).
            target: Look-at point in MuJoCo coordinates.
            fovy: Vertical field of view in degrees.
            origin_type: Tracking mode (defaults to ASSET_BODY if body_name given, else WORLD).
            body_name: Body to track.
        """
        dx = position[0] - target[0]
        dy = position[1] - target[1]
        dz = position[2] - target[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        elevation = math.degrees(math.asin(-dz / distance))
        azimuth = math.degrees(math.atan2(dy, dx))
        if origin_type is None:
            origin_type = (
                ViewerConfig.OriginType.ASSET_BODY
                if body_name is not None
                else ViewerConfig.OriginType.WORLD
            )
        return ViewerConfig(
            lookat=target,
            distance=distance,
            fovy=fovy,
            elevation=elevation,
            azimuth=azimuth,
            origin_type=origin_type,
            body_name=body_name,
        )


__all__ = ["ViewerConfig"]
