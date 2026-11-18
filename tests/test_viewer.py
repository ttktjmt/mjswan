"""ViewerConfig's spherical pose is MuJoCo's free camera's, so mjlab's values carry over."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjswan.mjlab.task import adapt_viewer_config
from mjswan.viewer import ViewerConfig

_MODEL = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom size='0.1'/></worldbody></mujoco>"
)


def _mujoco_camera_position(lookat, distance, elevation, azimuth) -> np.ndarray:
    """Where ``mjv_updateScene`` puts the free camera: midway between its two eyes."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
    scene = mujoco.MjvScene(_MODEL, maxgeom=10)
    mujoco.mjv_updateScene(
        _MODEL,
        mujoco.MjData(_MODEL),
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    return (scene.camera[0].pos + scene.camera[1].pos) / 2


@pytest.mark.parametrize(
    ("elevation", "azimuth"), [(-10.0, 90.0), (-30.0, -135.0), (20.0, 170.0)]
)
def test_from_position_inverts_mujocos_free_camera(elevation, azimuth):
    lookat = (0.5, -1.0, 0.3)
    position = _mujoco_camera_position(lookat, 1.5, elevation, azimuth)

    config = ViewerConfig.from_position(tuple(position), lookat)

    assert config.distance == pytest.approx(1.5)
    assert config.elevation == pytest.approx(elevation)
    assert config.azimuth == pytest.approx(azimuth)


def test_an_mjlab_viewer_config_keeps_its_azimuth():
    mjlab_viewer = SimpleNamespace(distance=1.5, elevation=-10.0, azimuth=90.0)
    adapted = adapt_viewer_config(mjlab_viewer)
    assert adapted is not None
    assert (adapted.elevation, adapted.azimuth) == (-10.0, 90.0)


_FILAMENT_XML = """<mujoco><custom>
  <text name="filament.cg.tone_mapping" data="aces"/>
  <numeric name="filament.cg.exposure" data="1.5"/>
  <text name="filament.cg.output_color_space" data="linear"/>
</custom><worldbody><geom size="0.1"/></worldbody></mujoco>"""


def test_a_filament_setting_on_a_classic_scene_is_refused():
    # Classic draws with no tone mapping or colour management: the setting would do nothing.
    with pytest.raises(ValueError, match="color_management=.filament."):
        ViewerConfig(tone_mapping="agx")
    ViewerConfig(color_management="filament", tone_mapping="agx", exposure=0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color_management": "pbr"},
        {"color_management": "filament", "tone_mapping": "acesfilmic"},
        {"color_management": "filament", "output_color_space": "rec709"},
        {"color_management": "filament", "exposure": float("inf")},
    ],
)
def test_an_unknown_value_is_refused(kwargs):
    with pytest.raises(ValueError, match="is not"):
        ViewerConfig(**kwargs)


def test_a_switch_reaches_the_document_as_a_boolean():
    from mjswan.build.manifest import graphics_entry

    entry = graphics_entry(ViewerConfig(enable_shadows=0), _MODEL)  # type: ignore[arg-type]
    assert entry["enable_shadows"] is False


@pytest.mark.parametrize(
    "model",
    [
        mujoco.MjSpec.from_string(_FILAMENT_XML),
        mujoco.MjModel.from_xml_string(_FILAMENT_XML),
    ],
    ids=["spec", "model"],
)
def test_graphics_fall_back_on_the_models_filament_flags(model):
    from mjswan.build.manifest import graphics_entry

    viewer = ViewerConfig(color_management="filament", exposure=-1.0, enable_haze=False)
    assert graphics_entry(viewer, model) == {
        "color_management": "filament",
        "tone_mapping": "aces",
        "exposure": -1.0,
        "output_color_space": "linear",
        "base_color_type": "srgb",
        "enable_shadows": True,
        "enable_reflections": True,
        "enable_haze": False,
    }


def test_graphics_take_mujocos_defaults_and_its_fallback():
    from mjswan.build.manifest import graphics_entry

    entry = graphics_entry(None, _MODEL)
    assert (entry["color_management"], entry["tone_mapping"], entry["exposure"]) == (
        "classic",
        "pbr_neutral",
        0.0,
    )
    # MuJoCo draws an unknown tone mapper as PBR Neutral, so the viewer does too.
    unknown = mujoco.MjSpec.from_string(
        '<mujoco><custom><text name="filament.cg.tone_mapping" data="agx"/></custom>'
        "<worldbody/></mujoco>"
    )
    with pytest.warns(RuntimeWarning, match="does not know"):
        assert graphics_entry(None, unknown)["tone_mapping"] == "pbr_neutral"


@pytest.mark.parametrize("data", ["inf", "-inf", "nan"])
def test_a_non_finite_model_exposure_draws_zero(data):
    from mjswan.build.manifest import graphics_entry

    xml = (
        f'<mujoco><custom><numeric name="filament.cg.exposure" data="{data}"/>'
        "</custom><worldbody/></mujoco>"
    )
    for model in (mujoco.MjSpec.from_string(xml), mujoco.MjModel.from_xml_string(xml)):
        with pytest.warns(RuntimeWarning, match="not a finite"):
            assert graphics_entry(None, model)["exposure"] == 0.0


def test_an_mjlab_viewer_config_brings_its_shadow_and_reflection_switches():
    adapted = adapt_viewer_config(
        SimpleNamespace(enable_shadows=False, enable_reflections=False)
    )
    assert adapted is not None
    assert (adapted.enable_shadows, adapted.enable_reflections) == (False, False)
    assert adapted.color_management == "classic"
