"""The descriptor the browser re-casts a `RayCastSensor`'s rays from.

Layer: L1 (a fake sensor; the runtime half is `core/onnx/__tests__/raycast.test.ts`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from mjswan.build.mdp import raycast_sensor_descriptor  # noqa: E402


class _Cfg:
    def __init__(self, include_geom_groups):
        self.ray_alignment = "yaw"
        self.max_distance = 5.0
        self.exclude_parent_body = True
        self.include_geom_groups = include_geom_groups


class _FakeEnv:
    """Just enough env: `scene.sensors` and `sim.mj_model.body(id).name`."""

    def __init__(self, sensor):
        self.scene = type("Scene", (), {"sensors": {"terrain_scan": sensor}})()
        named = lambda _self, _id: type("E", (), {"name": "robot/pelvis"})()  # noqa: E731
        mj_model = type("M", (), {"body": named, "site": named, "geom": named})()
        self.sim = type("Sim", (), {"mj_model": mj_model})()


def _sensor(include_geom_groups):
    return type(
        "RayCastSensor",
        (),
        {
            "_local_offsets": torch.zeros(1, 3),
            "_local_directions": torch.tensor([[0.0, 0.0, -1.0]]),
            "_frame_infos": [("body", 1, None)],
            "cfg": _Cfg(include_geom_groups),
        },
    )()


def test_the_sensors_geom_groups_travel():
    """Terrain-only is `(0,)`; dropped, the rays stop on the robot's own legs."""
    descriptor = raycast_sensor_descriptor(_FakeEnv(_sensor((0,))), "terrain_scan")

    assert descriptor is not None
    assert descriptor["include_geom_groups"] == [0]


def test_no_groups_means_every_group():
    descriptor = raycast_sensor_descriptor(_FakeEnv(_sensor(None)), "terrain_scan")

    assert descriptor is not None
    assert descriptor["include_geom_groups"] is None
