"""Viewer configuration for mjswan scenes."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Any, Literal, get_args

Renderer = Literal["classic", "filament"]

ToneMapping = Literal[
    # three.js's operators
    "none",
    "linear",
    "reinhard",
    "cineon",
    "aces_filmic",
    "agx",
    "neutral",
    # MuJoCo Filament's, as it means them: `pbr_neutral` is `neutral`, `aces_legacy` is
    # `aces_filmic`, `aces` is `aces_filmic` at 0.6 of the exposure, `filmic` Narkowicz's
    "pbr_neutral",
    "aces_legacy",
    "aces",
    "filmic",
]

ColorSpace = Literal["srgb", "linear"]

#: The settings only the Filament renderer reads.
_FILAMENT_FIELDS = ("tone_mapping", "exposure", "output_color_space", "base_color_type")


@dataclass
class ViewerConfig:
    """Viewer configuration for a scene: the camera, and how the scene is drawn.

    Matches the API of ``mjlab.viewer.ViewerConfig``, plus the graphics settings mjlab's
    viewer does not have (``renderer`` onwards).
    """

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Look-at point in MuJoCo coordinates (x forward, y left, z up)."""

    distance: float = 4.0
    """Distance from the look-at point to the viewer."""

    fovy: float | None = None
    """Vertical field of view in degrees. Defaults to 45."""

    elevation: float = -30.0
    """Viewer elevation in degrees (negative = viewer above the look-at point)."""

    azimuth: float = -135.0
    """Viewer azimuth in degrees: the direction the viewer looks, CCW from the x-axis
    (forward), as MuJoCo's free camera and mjlab measure it."""

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
    """Reflections, as MuJoCo's ``mjRND_REFLECTION``: the first opaque plane or box top
    with a material ``reflectance`` mirrors the scene."""

    enable_shadows: bool = True
    """Shadows, as MuJoCo's ``mjRND_SHADOW``: cast from each light with ``castshadow``."""

    height: int = 240
    """Viewer canvas height in pixels."""

    width: int = 320
    """Viewer canvas width in pixels."""

    renderer: Renderer = "classic"
    """``"classic"`` draws as MuJoCo's classic renderer: ``rgba`` as written, an sRGB 2D
    texture decoded and the output not encoded, so darker than the file, a cube or skybox
    as stored, and no tone mapping. ``"filament"`` draws as its Filament renderer: colour managed, tone mapped
    and exposed. The settings below apply to it alone."""

    tone_mapping: ToneMapping | None = None
    """Every operator three.js has, and MuJoCo's names with their meaning. Unset, the
    model's ``filament.cg.tone_mapping``, else ``"pbr_neutral"``."""

    exposure: float | None = None
    """In stops, before tone mapping. Unset, the model's ``filament.cg.exposure``, else 0."""

    output_color_space: ColorSpace | None = None
    """``"srgb"`` encodes the output, ``"linear"`` leaves it. Unset, the model's
    ``filament.cg.output_color_space``, else ``"srgb"``."""

    base_color_type: ColorSpace | None = None
    """How an ``rgba`` is read: ``"srgb"`` decodes it, ``"linear"`` takes it as written.
    Unset, the model's ``filament.material.base_color_type``, else ``"srgb"``."""

    enable_haze: bool = True
    """Haze where the first infinite plane meets the skybox, as MuJoCo's ``mjRND_HAZE``:
    the model's ``<visual>`` ``haze`` and its colour, drawn only with a skybox."""

    def __post_init__(self) -> None:
        _check_choice("renderer", self.renderer, Renderer)
        _check_choice("tone_mapping", self.tone_mapping, ToneMapping)
        _check_choice("output_color_space", self.output_color_space, ColorSpace)
        _check_choice("base_color_type", self.base_color_type, ColorSpace)
        # JSON has no infinity: the browser would refuse the whole document.
        if self.exposure is not None and not math.isfinite(self.exposure):
            raise ValueError(
                f"ViewerConfig exposure={self.exposure!r} is not a finite number of stops."
            )
        if self.renderer == "classic":
            set_here = [
                name for name in _FILAMENT_FIELDS if getattr(self, name) is not None
            ]
            if set_here:
                raise ValueError(
                    f"ViewerConfig sets {', '.join(set_here)} with renderer='classic', "
                    "which draws with no tone mapping or colour management, as MuJoCo's "
                    "classic renderer does. Pass renderer='filament' to use them."
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the manifest's ``camera`` entry (snake_case, ADR 0006)."""
        d: dict[str, Any] = {
            "lookat": list(self.lookat),
            "distance": self.distance,
            "elevation": self.elevation,
            "azimuth": self.azimuth,
            "origin_type": self.origin_type.name,
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
        azimuth = math.degrees(math.atan2(-dy, -dx))
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


def _check_choice(name: str, value: Any, choices: Any) -> None:
    if value is not None and value not in get_args(choices):
        raise ValueError(
            f"ViewerConfig {name}={value!r} is not one of {', '.join(get_args(choices))}."
        )


__all__ = ["ColorSpace", "Renderer", "ToneMapping", "ViewerConfig"]
