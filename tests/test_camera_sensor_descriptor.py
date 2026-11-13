"""The descriptor the browser renders a `CameraSensor` from.

Layer: L1 (a fake sensor; the runtime half is `core/onnx/__tests__/camera.test.ts`).

The camera's pose, field of view and intrinsics are the model's, which the browser
compiles too; what travels is the image size and what the renderer draws into it. A
wrong group list is a policy shown geometry it never saw in training, or none it did.
"""

from __future__ import annotations

import warnings

import pytest

pytest.importorskip("torch")

from mjswan.build.mdp import camera_sensor_descriptor  # noqa: E402
from mjswan.build.mdp.sensor import structured_sensor_descriptors  # noqa: E402
from mjswan.compile.slot import _SENSOR_NS  # noqa: E402


def _cfg(**over):
    fields = dict(
        width=32,
        height=32,
        data_types=("rgb",),
        enabled_geom_groups=(0, 3),
        use_textures=True,
        use_shadows=False,
        orthographic=False,
    )
    fields.update(over)
    return type("CameraSensorCfg", (), fields)()


class _FakeEnv:
    def __init__(self, **sensors):
        self.scene = type("Scene", (), {"sensors": sensors})()


def _camera(camera_name: str = "robot/camera_d405", **over):
    return type("CameraSensor", (), {"cfg": _cfg(**over), "camera_name": camera_name})()


def test_the_model_camera_and_what_to_draw_travel():
    descriptor = camera_sensor_descriptor(
        _FakeEnv(camera_d405=_camera()), "camera_d405"
    )

    assert descriptor == {
        "kind": "camera",
        "camera": "robot/camera_d405",
        "width": 32,
        "height": 32,
        "enabled_geom_groups": [0, 3],
        "use_shadows": False,
    }


def test_a_sensor_that_is_not_a_camera_is_not_described():
    other = type("RayCastSensor", (), {"cfg": object()})()
    assert camera_sensor_descriptor(_FakeEnv(scan=other), "scan") is None


def test_an_orthographic_camera_is_refused():
    env = _FakeEnv(cam=_camera(orthographic=True))
    with pytest.raises(ValueError, match="perspective"):
        camera_sensor_descriptor(env, "cam")


def test_an_untextured_camera_warns_that_the_browser_draws_textures():
    env = _FakeEnv(cam=_camera(use_textures=False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        camera_sensor_descriptor(env, "cam")
    assert any("textures" in str(w.message) for w in caught)


def test_a_graph_reading_a_cameras_depth_is_refused():
    """Only `rgb` renders; a depth slot would be fed nothing at all."""
    export = type("Export", (), {"input_slots": [(_SENSOR_NS, "cam.depth")]})()
    with pytest.raises(ValueError, match="renders only"):
        structured_sensor_descriptors(export, _FakeEnv(cam=_camera()), owner="camera")


def test_a_graph_reading_a_cameras_rgb_carries_its_descriptor():
    export = type("Export", (), {"input_slots": [(_SENSOR_NS, "cam.rgb")]})()
    descriptors = structured_sensor_descriptors(
        export, _FakeEnv(cam=_camera()), owner="camera"
    )
    assert descriptors["cam"]["kind"] == "camera"
