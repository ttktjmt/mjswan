"""The descriptor the browser reproduces a `ContactSensor` from.

Layer: L1 (a fake sensor; the runtime half is `core/onnx/__tests__/contact.test.ts`).

Only the layout travels — which sensor windows make up a field, in which order, how many
slots each packs, how deep the history goes. Get it wrong and the runtime reads real
numbers in the wrong order, which nothing in the browser would catch.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from mjswan.build.mdp import contact_sensor_descriptor  # noqa: E402


class _Cfg:
    def __init__(self, num_slots: int = 1, history_length: int = 4):
        self.num_slots = num_slots
        self.history_length = history_length


class _Slot:
    def __init__(self, primary: str, field: str):
        self.primary_name = primary
        self.field_name = field
        self.sensor_name = f"ee_ground_collision_{primary}_{field}"


class _FakeEnv:
    """Just enough env: `scene.sensors` and `sim.mj_model.sensor(name).dim`."""

    def __init__(self, sensor, dims: dict[str, int]):
        env = self

        class _MjModel:
            def sensor(self, name: str):
                return type("S", (), {"dim": [dims[name]]})()

        self.scene = type("Scene", (), {"sensors": {"ee_ground_collision": sensor}})()
        self.sim = type("Sim", (), {"mj_model": _MjModel()})()
        del env


def _sensor(*slots: _Slot, num_slots: int = 1, history_length: int = 4):
    return type(
        "ContactSensor",
        (),
        {"_slots": list(slots), "cfg": _Cfg(num_slots, history_length)},
    )()


def test_a_field_lists_its_windows_in_primary_order():
    """`ContactData` is primary-major, so the order is data the browser needs."""
    sensor = _sensor(
        _Slot("left", "found"),
        _Slot("left", "force"),
        _Slot("right", "found"),
        _Slot("right", "force"),
    )
    dims = {
        "ee_ground_collision_left_found": 1,
        "ee_ground_collision_left_force": 3,
        "ee_ground_collision_right_found": 1,
        "ee_ground_collision_right_force": 3,
    }

    descriptor = contact_sensor_descriptor(
        _FakeEnv(sensor, dims), "ee_ground_collision"
    )

    assert descriptor is not None
    assert descriptor["fields"]["force"]["sensors"] == [
        "ee_ground_collision_left_force",
        "ee_ground_collision_right_force",
    ]
    assert descriptor["fields"]["force"]["dim"] == 3
    assert descriptor["fields"]["found"]["dim"] == 1


def test_the_per_slot_dim_is_the_window_divided_by_the_slot_count():
    """One window packs `num_slots * dim`; the runtime needs `dim` to walk it."""
    sensor = _sensor(_Slot("link_6", "force"), num_slots=2)
    descriptor = contact_sensor_descriptor(
        _FakeEnv(sensor, {"ee_ground_collision_link_6_force": 6}), "ee_ground_collision"
    )

    assert descriptor is not None
    assert descriptor["num_slots"] == 2
    assert descriptor["fields"]["force"]["dim"] == 3


def test_only_the_buffered_fields_are_named_as_such():
    """mjlab buffers force/torque/dist alone (`ContactSensor.initialize`)."""
    sensor = _sensor(_Slot("link_6", "found"), _Slot("link_6", "force"))
    descriptor = contact_sensor_descriptor(
        _FakeEnv(
            sensor,
            {
                "ee_ground_collision_link_6_found": 1,
                "ee_ground_collision_link_6_force": 3,
            },
        ),
        "ee_ground_collision",
    )

    assert descriptor is not None
    assert descriptor["history_fields"] == ["force"]


def test_no_history_length_means_no_buffered_fields():
    sensor = _sensor(_Slot("link_6", "force"), history_length=0)
    descriptor = contact_sensor_descriptor(
        _FakeEnv(sensor, {"ee_ground_collision_link_6_force": 3}), "ee_ground_collision"
    )

    assert descriptor is not None
    assert descriptor["history_fields"] == []


def test_a_sensor_that_is_not_a_contact_sensor_is_not_described():
    """The caller falls through to the raycast descriptor, then fails the build."""
    raycast = type("RayCastSensor", (), {"cfg": _Cfg()})()
    assert (
        contact_sensor_descriptor(_FakeEnv(raycast, {}), "ee_ground_collision") is None
    )
