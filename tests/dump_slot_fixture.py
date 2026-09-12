"""Dump a slot-reader parity fixture from live mjlab envs.

The TypeScript slot reader (``core/onnx/slotReader/``) reimplements mjlab's
``EntityData`` field semantics against raw ``mjModel``/``mjData``. Nothing in the
Python parity harness covers that reimplementation: the harness proves each
traced *graph* matches mjlab, while the reader decides what numbers go *into* the
graph, browser-side. A wrong address or element order there produces a policy
that runs happily on the wrong state.

Regenerate with::

    MUJOCO_GL=disable .venv/bin/python tests/dump_slot_fixture.py

Ground truth is read off ``env.scene[entity].data`` — the same property the tracer
recorded — so the fixture stays honest even if mjlab changes a definition. Every field
in ``READER_FIELDS`` is dumped, and each entity's ``EntityIndexing`` with it, so the
browser's element resolution is checked against mjlab's and not only its arithmetic.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

from mjswan.compile.tracer import READER_FIELDS  # noqa: E402

OUT = (
    Path(__file__).resolve().parents[1]
    / "src/mjswan/template/src/core/onnx/__tests__/fixtures/slotFields.json"
)

# Fixed so a regeneration is a reviewable diff rather than a fixture of fresh numbers.
SEED = 0

# One task with prefixed entities, sites and a free joint; one with sensors and 29 joints.
TASKS = ("Mjlab-Lift-Cube-Yam", "Mjlab-Velocity-Flat-Unitree-G1")

# What neither task has: tendons and tendon-driven actuators, an unnamed geom and site,
# and a plain-MJCF (unprefixed) single-entity model, the `set_trace_env` shape.
SYNTHETIC = "synthetic/tendon-arm"
SYNTHETIC_MODEL = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom name="base_box" type="box" size="0.1 0.1 0.1"/>
      <geom type="sphere" size="0.05" pos="0.15 0 0"/>
      <site name="s_a" pos="0 0 0.1"/>
      <body name="link" pos="0 0 0.3">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <site name="s_b" pos="0 0 -0.1"/>
        <site pos="0.05 0 0"/>
        <body name="tip" pos="0 0 0.2">
          <joint name="wrist" type="hinge" axis="1 0 0"/>
          <geom name="tip_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
          <site name="s_c" pos="0 0 0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_ab"><site site="s_a"/><site site="s_b"/></spatial>
    <spatial name="t_bc"><site site="s_b"/><site site="s_c"/></spatial>
  </tendon>
  <actuator>
    <muscle name="m_ab" tendon="t_ab" lengthrange="0 1"/>
    <muscle name="m_bc" tendon="t_bc" lengthrange="0 1"/>
  </actuator>
</mujoco>
"""

# Every field the reader implements, so a newly-added one cannot skip the check.
FIELDS = tuple(sorted(READER_FIELDS))

MODEL_INTS = ("nq", "nv", "nu", "njnt", "nbody", "ngeom", "nsite", "ntendon", "nsensor")
MODEL_INT_ARRAYS = (
    "jnt_type",
    "jnt_qposadr",
    "jnt_dofadr",
    "name_jntadr",
    "name_bodyadr",
    "name_geomadr",
    "name_siteadr",
    "name_tendonadr",
    "name_actuatoradr",
    "name_sensoradr",
    "sensor_adr",
    "sensor_dim",
    "geom_bodyid",
    "site_bodyid",
)
MODEL_FLOAT_ARRAYS = ("body_iquat",)
# mjData fields, each sliced to env 0 (the browser runs a single env), with the mjModel
# count for that element — consulted rather than caught, since warp raises on a
# zero-length vector array (a task with no tendons).
DATA_ARRAYS = (
    ("qpos", "nq"),
    ("qvel", "nv"),
    ("qacc", "nv"),
    ("qfrc_actuator", "nv"),
    ("qfrc_smooth", "nv"),
    ("qfrc_applied", "nv"),
    ("qfrc_passive", "nv"),
    ("qfrc_bias", "nv"),
    ("actuator_force", "nu"),
    ("xpos", "nbody"),
    ("xquat", "nbody"),
    ("xipos", "nbody"),
    ("cvel", "nbody"),
    ("subtree_com", "nbody"),
    ("xfrc_applied", "nbody"),
    ("geom_xpos", "ngeom"),
    ("geom_xmat", "ngeom"),
    ("site_xpos", "nsite"),
    ("site_xmat", "nsite"),
    ("ten_length", "ntendon"),
    ("ten_velocity", "ntendon"),
    ("sensordata", "nsensordata"),
)
INDEXING = (
    "body_ids",
    "geom_ids",
    "site_ids",
    "tendon_ids",
    "ctrl_ids",
    "joint_q_adr",
    "joint_v_adr",
)


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _data_field(env: Any, name: str, count_attr: str) -> list[float]:
    if int(getattr(env.sim.mj_model, count_attr)) == 0:
        return []
    return _flat(getattr(env.sim.data, name)[0])


def _push(env: Any) -> None:
    """Load every body with a wrench, so the external-force fields have numbers in them.

    Deterministic per body, and applied before the forward pass so `qfrc_smooth`
    carries the Jacobian-mapped part `qfrc_external` recovers.
    """
    import torch

    for entity in env.scene.entities.values():
        n = int(entity.indexing.body_ids.numel())
        if n == 0:
            continue
        rows = torch.arange(1, n + 1, dtype=torch.float32).reshape(1, n, 1)
        offsets = torch.tensor([[[0.1, -0.2, 0.3]]])
        entity.data.write_external_wrench(rows * offsets, rows * offsets.flip(-1))
    env.sim.forward()


def _settle(env: Any) -> None:
    # A few steps, so a reader that just returned qpos0 cannot pass at t=0.
    for _ in range(3):
        env.sim.forward()
        env.scene.update(env.step_dt)
    _push(env)


def _dump_env(env: Any) -> dict[str, Any]:
    mj_model = env.sim.mj_model
    model: dict[str, Any] = {name: int(getattr(mj_model, name)) for name in MODEL_INTS}
    for name in MODEL_INT_ARRAYS:
        model[name] = [int(v) for v in getattr(mj_model, name).reshape(-1).tolist()]
    for name in MODEL_FLOAT_ARRAYS:
        model[name] = [float(v) for v in getattr(mj_model, name).reshape(-1).tolist()]
    # The NUL-separated name blob ships as a list of ints, to keep the fixture plain JSON.
    model["names"] = list(bytes(mj_model.names))

    data = {name: _data_field(env, name, count) for name, count in DATA_ARRAYS}

    entities: dict[str, dict[str, Any]] = {}
    for entity_name in env.scene.entities:
        entity = env.scene[entity_name]
        entity_data = entity.data
        fields: dict[str, list[float]] = {}
        for field in FIELDS:
            try:
                value = getattr(entity_data, field)
            except Exception:  # noqa: BLE001 — a field this entity cannot report
                continue
            fields[field] = _flat(value[0] if value.ndim > 0 else value)
        # The walking tasks randomize `encoder_bias`, so the reader needs the same bias.
        bias = _flat(entity_data.encoder_bias[0])
        indexing = entity.indexing
        ids = {
            name: [int(v) for v in getattr(indexing, name).tolist()]
            for name in INDEXING
        }
        # `root_body_id` indexes `bodies[0]`, which a body-less terrain has not got.
        ids["root_body_id"] = int(indexing.root_body_id) if indexing.bodies else -1
        entities[entity_name] = {
            "fields": fields,
            "encoder_bias": dict(zip(entity.joint_names, bias, strict=True)),
            "indexing": ids,
        }

    # Only builtin sensors are `sensordata` windows; the tracer already rejects the rest.
    sensors: dict[str, list[float]] = {}
    for name, sensor in env.scene.sensors.items():
        value = sensor.data
        if hasattr(value, "ndim") and hasattr(value, "detach"):
            sensors[name] = _flat(value[0])

    return {"model": model, "data": data, "entities": entities, "sensors": sensors}


def _dump_task(task_id: str) -> dict[str, Any]:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=True)
    # Before construction, not just on `reset`: startup events run in `__init__`.
    cfg.seed = SEED
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset(seed=SEED)
        _settle(env)
    return _dump_env(env)


def _dump_synthetic() -> dict[str, Any]:
    """The plain-MJCF shape, with what the tasks lack, built as `set_trace_env` would
    but with its XML actuators wrapped so mjlab counts the entity as actuated."""
    import mujoco
    import torch
    from mjlab.actuator import XmlActuatorCfg
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg

    from mjswan.trace_env import build_mjlab_env

    entity_cfg = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_string(SYNTHETIC_MODEL),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                XmlActuatorCfg(
                    target_names_expr=(".*",),
                    transmission_type=TransmissionType.TENDON,
                ),
            )
        ),
    )
    env_cfg = ManagerBasedRlEnvCfg(
        decimation=1, scene=SceneCfg(num_envs=1, entities={"robot": entity_cfg})
    )
    with contextlib.redirect_stdout(io.StringIO()):
        env = build_mjlab_env(env_cfg)
        env.reset(seed=SEED)
        # Never stepped, so put it somewhere every field has a value worth checking.
        gen = torch.Generator().manual_seed(SEED)
        data = env.scene["robot"].data
        quat = torch.rand(1, 4, generator=gen) - 0.5
        data.write_root_pose(
            torch.cat([torch.rand(1, 3, generator=gen), quat / quat.norm()], dim=-1)
        )
        data.write_root_velocity(torch.rand(1, 6, generator=gen) - 0.5)
        data.write_joint_position(torch.rand(1, 2, generator=gen) - 0.5)
        data.write_joint_velocity(torch.rand(1, 2, generator=gen) - 0.5)
        data.write_ctrl(torch.rand(1, 2, generator=gen))
        _settle(env)
    return _dump_env(env)


def main() -> None:
    fixture = {task_id: _dump_task(task_id) for task_id in TASKS}
    fixture[SYNTHETIC] = _dump_synthetic()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    for task_id, payload in fixture.items():
        entities = payload["entities"]
        fields = {f for e in entities.values() for f in e["fields"]}
        print(
            f"{task_id}: {len(entities)} entities "
            f"({', '.join(sorted(entities))}), {len(fields)} fields, "
            f"{len(payload['sensors'])} sensors"
        )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
