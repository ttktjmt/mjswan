"""What the browser needs to reproduce a structured sensor a graph reads.

A builtin MuJoCo sensor is a ``sensordata`` window and needs nothing; mjlab's raycast
and contact sensors have their own state, so a descriptor of it travels with any graph
whose slots name one.
"""

from __future__ import annotations

from typing import Any


def _mj_element_name(env: Any, obj_type: str, obj_id: int) -> str:
    """Model name of a body/site/geom the sensor's rays are attached to.

    Names travel, not ids: the browser's model is compiled separately, so an id
    from the build env means nothing there.
    """
    mj_model = env.sim.mj_model
    return {"body": mj_model.body, "site": mj_model.site, "geom": mj_model.geom}[
        obj_type
    ](obj_id).name


_CONTACT_HISTORY_FIELDS = ("force", "torque", "dist")
"""Fields mjlab buffers (``ContactSensor.initialize``)."""


def contact_sensor_descriptor(env: Any, sensor_name: str) -> dict[str, Any] | None:
    """What the browser needs to reproduce one ``ContactSensor``, or None if not one.

    mjlab adds a real MuJoCo sensor per ``(primary, field)`` pair, so the values are
    already in ``sensordata`` — only the layout to read them back travels, plus the
    ring buffer the runtime owns.
    """
    sensor = env.scene.sensors.get(sensor_name)
    slots = getattr(sensor, "_slots", None)
    if not slots:
        return None
    fields: dict[str, Any] = {}
    for slot in slots:
        entry = fields.setdefault(slot.field_name, {"sensors": []})
        entry["sensors"].append(slot.sensor_name)
    for entry in fields.values():
        window = env.sim.mj_model.sensor(entry["sensors"][0])
        # `num_slots * dim` per window; the runtime reshapes with `dim`.
        entry["dim"] = int(window.dim[0]) // int(sensor.cfg.num_slots)
    history = [f for f in _CONTACT_HISTORY_FIELDS if f in fields]
    return {
        "kind": "contact",
        "num_slots": int(sensor.cfg.num_slots),
        "history_length": int(sensor.cfg.history_length),
        # Only these have a buffer, whatever `history_length` says.
        "history_fields": history if sensor.cfg.history_length > 0 else [],
        "fields": fields,
    }


def raycast_sensor_descriptor(env: Any, sensor_name: str) -> dict[str, Any] | None:
    """Everything the browser needs to reproduce one ``RayCastSensor``'s readings.

    There is no ``sensordata`` window for a raycast sensor, so the browser casts the
    rays itself with ``mj_ray``. The offsets and directions are baked from the live
    sensor rather than re-implementing mjlab's pattern generators, which means an
    unknown pattern works for free.

    Returns ``None`` if *sensor_name* is not a raycast sensor.
    """
    sensor = env.scene.sensors.get(sensor_name)
    offsets = getattr(sensor, "_local_offsets", None)
    if offsets is None:
        return None
    return {
        "kind": "raycast",
        # [N, 3] each, in the frame's local coordinates.
        "local_offsets": offsets.detach().cpu().tolist(),
        "local_directions": sensor._local_directions.detach().cpu().tolist(),
        "frames": [
            {"type": obj_type, "name": _mj_element_name(env, obj_type, obj_id)}
            for obj_type, obj_id, _ in sensor._frame_infos
        ],
        # "base" | "yaw" | "world" — how the frame's rotation reaches the rays.
        "ray_alignment": sensor.cfg.ray_alignment,
        "max_distance": float(sensor.cfg.max_distance),
        # mjlab excludes each frame's own parent body so a ray cannot self-hit.
        "exclude_parent_body": bool(sensor.cfg.exclude_parent_body),
        # A terrain scan is `(0,)`: without it the rays hit the robot's own legs.
        "include_geom_groups": (
            None
            if sensor.cfg.include_geom_groups is None
            else [int(g) for g in sensor.cfg.include_geom_groups]
        ),
    }


def structured_sensor_descriptors(
    export: Any, env: Any, *, owner: str
) -> dict[str, Any]:
    """Descriptors for the structured sensors a graph's slots name.

    Without one the runtime would hold a stale value rather than fail, so an
    undescribable sensor fails the build instead.
    """
    from ...compile.slot import _SENSOR_NS

    descriptors: dict[str, Any] = {}
    for namespace, name_part in export.input_slots:
        if namespace != _SENSOR_NS or "." not in name_part:
            continue
        sensor_name, field = name_part.split(".", 1)
        if sensor_name in descriptors:
            continue
        descriptor = raycast_sensor_descriptor(
            env, sensor_name
        ) or contact_sensor_descriptor(env, sensor_name)
        if descriptor is None:
            sensor = env.scene.sensors.get(sensor_name)
            raise ValueError(
                f"{owner} reads {sensor_name!r}.{field}, but the browser has no "
                f"implementation for a {type(sensor).__name__} — only raycast and "
                "contact sensors can serve fields. Implement it in the runtime and emit "
                "a descriptor here, hand the term to the browser as a TS class, or drop "
                "it from the exported set."
            )
        descriptors[sensor_name] = descriptor
    return descriptors
