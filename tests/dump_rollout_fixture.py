"""Dump an N-step rollout-parity fixture from live mjlab tasks, for
``rolloutParity.test.ts`` — the only check that runs the whole chain composed:
reader → fused graph → clip/scale → concat → group vector, and reader → termination
graph → lanes → OR-reduce.

**States are replayed, not co-simulated.** mjlab integrates with ``mujoco_warp`` and
the browser with MuJoCo's WASM, so a free-running comparison would measure MuJoCo
against itself. Each step's ``mjModel``/``mjData`` arrays are captured alongside
mjlab's own observation vector and termination verdicts *at that state*.

Only ``actor`` is dumped; ``critic`` is training-only and never reaches a bundle.

Regenerate with::

    MUJOCO_GL=disable .venv/bin/python tests/dump_rollout_fixture.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "disable")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "src/mjswan/template/src/core/engine/__tests__/fixtures/rollout"

# Cartpole is the minimal shape (two entity-data slots, one native termination);
# G1-Velocity-Flat the wide one (builtin sensors, `joint_pos_biased`, seven terms in a
# 99-wide fused group, native inputs, a traced termination beside `time_out`).
#
# Velocity-Rough's raycast slot and Lift-Cube-Yam's command-state slot are covered by
# `raycast.test.ts` and the `OnnxCommand` tests; both need a stub this format lacks.
TASKS = ("Mjlab-Cartpole-Balance", "Mjlab-Velocity-Flat-Unitree-G1")

STEPS = 20

# Fixed so a regeneration is a reviewable diff rather than a fixture of fresh numbers.
SEED = 0

# `mjModel` fields the slot reader indexes (mirrors `dump_slot_fixture.py`).
MODEL_INTS = ("njnt", "nbody", "nsite", "nsensor")
MODEL_ARRAYS = (
    "jnt_type",
    "jnt_qposadr",
    "jnt_dofadr",
    "name_jntadr",
    "name_bodyadr",
    "name_siteadr",
    "name_sensoradr",
    "sensor_adr",
    "sensor_dim",
)
# `mjData` fields sliced to env 0, each with the `mjModel` count for that element — consulted
# rather than caught, since warp raises on a zero-length vector array (Cartpole has no sites).
DATA_ARRAYS = (
    ("qpos", "nq"),
    ("qvel", "nv"),
    ("xpos", "nbody"),
    ("xquat", "nbody"),
    ("cvel", "nbody"),
    ("subtree_com", "nbody"),
    ("site_xpos", "nsite"),
    ("sensordata", "nsensordata"),
)


def _flat(value: Any) -> list[float]:
    return [float(v) for v in value.detach().cpu().reshape(-1).tolist()]


def _data_field(env: Any, name: str, count_attr: str) -> list[float]:
    if int(getattr(env.sim.mj_model, count_attr)) == 0:
        return []
    return _flat(getattr(env.sim.data, name)[0])


def _action(step: int, num_actions: int) -> list[float]:
    """Bounded so the robot neither freezes nor flails off the map — the states have to
    stay in the region the observation terms are meaningful in."""
    return [0.4 * math.sin(0.35 * step + 0.17 * j) for j in range(num_actions)]


def _termination_verdicts(env: Any, cfg: Any) -> dict[str, bool]:
    """Each termination term's verdict at the *current* state.

    Re-evaluated rather than read off `termination_manager`: that one ran before
    mjlab's single `forward()`, so its derived quantities are a substep stale. What
    is under test here is whether the graph agrees with the term, so both sides
    must see the same state.
    """
    from mjswan.build.mdp.provenance import resolved_params

    verdicts: dict[str, bool] = {}
    for name, term_cfg in cfg.terminations.items():
        params = resolved_params(term_cfg.params, env)
        value = term_cfg.func(env, **params)
        verdicts[name] = bool(value.reshape(-1)[0].item())
    return verdicts


def _native_inputs(env: Any, group_entry: dict[str, Any]) -> dict[str, list[float]]:
    """The values the orchestrator supplies natively, not through a graph.

    `prev_action` and `command` are computed browser-side (the policy's own last
    output; a live command term), so feeding mjlab's value here keeps the
    comparison about the graph and the pipeline. Both are covered separately by
    the `PolicyRunner` and `OnnxCommand` suites.
    """
    natives: dict[str, list[float]] = {}
    for native in group_entry.get("native_inputs", []):
        kind = native["native"]
        if kind == "prev_action":
            natives[native["input"]] = _flat(env.action_manager.action)
        elif kind == "command":
            natives[native["input"]] = _flat(
                env.command_manager.get_command(native["command_name"])
            )
        else:  # pragma: no cover — a new native marker needs a decision here
            raise ValueError(f"no fixture source for native input {kind!r}")
    return natives


# Bracketing an orientation limit from both sides, then back under it.
TILT_PITCHES = (0.0, 0.4, 1.2, 1.6, 2.2, 2.8, 0.2)


def _append_tilt_sweep(env: Any, cfg: Any, record: Any, num_actions: int) -> None:
    """Force the root orientation through a range, recording each state.

    A natural rollout stays upright, so every verdict reads False and a graph hardwired
    to False would pass. Only for a floating base, whose root quaternion sits at
    ``qpos[3:7]``; a task without one has no orientation term to trip.

    Written straight into ``qpos`` rather than reached by stepping — walking a robot
    into a fall takes a controller, and what needs covering is the verdict at a tilted
    state, not how it got there.
    """
    import mujoco
    import torch

    mj_model = env.sim.mj_model
    if mj_model.njnt == 0 or mj_model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        return

    for pitch in TILT_PITCHES:
        half = pitch / 2.0
        # mjlab's (w, x, y, z) order.
        quat = [math.cos(half), 0.0, math.sin(half), 0.0]
        qpos = env.sim.data.qpos
        qpos[0, 3:7] = torch.tensor(quat, dtype=qpos.dtype)
        env.sim.forward()
        env.scene.update(env.step_dt)
        record(_action(0, num_actions))


def _dump_task(task_id: str, out_dir: Path) -> dict[str, Any]:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.build.mdp import serialize_observation_group, serialize_terminations
    from mjswan.mjlab.observation import _adapt_obs_group
    from mjswan.mjlab.termination import _adapt_term_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 1
    # Before construction, not just on `reset`: startup events run in `__init__`.
    cfg.seed = SEED
    with contextlib.redirect_stdout(io.StringIO()):
        env = ManagerBasedRlEnv(cfg, device="cpu")
        env.reset(seed=SEED)

    # The Builder's own serializers, so the test loads the bytes that really ship.
    group_entry = serialize_observation_group(
        _adapt_obs_group(cfg.observations["actor"]), env, out_dir, group_name="policy"
    )
    if not isinstance(group_entry, dict) or "fused" not in group_entry:
        raise RuntimeError(
            f"{task_id}: the actor group did not fuse; this fixture only covers "
            "the fused path."
        )
    termination_entries = serialize_terminations(
        {name: _adapt_term_cfg(term) for name, term in cfg.terminations.items()},
        env,
        out_dir,
    )

    mj_model = env.sim.mj_model
    model: dict[str, Any] = {name: int(getattr(mj_model, name)) for name in MODEL_INTS}
    for name in MODEL_ARRAYS:
        model[name] = [int(v) for v in getattr(mj_model, name).reshape(-1).tolist()]
    model["names"] = list(bytes(mj_model.names))

    num_actions = env.action_manager.total_action_dim
    steps: list[dict[str, Any]] = []

    def record(action: list[float]) -> None:
        steps.append(
            {
                "action": action,
                "data": {
                    name: _data_field(env, name, count) for name, count in DATA_ARRAYS
                },
                "native": _native_inputs(env, group_entry),
                "obs": _flat(env.observation_manager.compute_group("actor")),
                "terminations": _termination_verdicts(env, cfg),
            }
        )

    for step in range(STEPS):
        action = _action(step, num_actions)
        env.step(torch.tensor([action], dtype=torch.float32))
        record(action)

    _append_tilt_sweep(env, cfg, record, num_actions)

    # The walking tasks randomize `encoder_bias`, so the reader needs the same lookup.
    encoder_bias: dict[str, float] = {}
    for entity_name in env.scene.entities:
        entity = env.scene[entity_name]
        with contextlib.suppress(AttributeError):
            bias = _flat(entity.data.encoder_bias[0])
            encoder_bias.update(zip(entity.joint_names, bias, strict=True))

    payload = {
        "group": group_entry,
        "terminations": termination_entries,
        "model": model,
        "encoder_bias": encoder_bias,
        "num_actions": num_actions,
        "steps": steps,
    }
    env.close()
    return payload


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    fixture: dict[str, Any] = {}
    for task_id in TASKS:
        graph_dir = OUT_DIR / task_id
        graph_dir.mkdir(parents=True, exist_ok=True)
        fixture[task_id] = _dump_task(task_id, graph_dir)

    index = OUT_DIR / "rollout.json"
    index.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")

    for task_id, payload in fixture.items():
        graphs = sorted(
            p.relative_to(OUT_DIR).as_posix()
            for p in (OUT_DIR / task_id).rglob("*.onnx")
        )
        traced = [n for n, e in payload["terminations"].items() if "onnx" in e]
        print(
            f"{task_id}: {len(payload['steps'])} steps, "
            f"obs width {payload['group']['size']}, "
            f"{len(payload['group']['input_slots'])} slots + "
            f"{len(payload['group']['native_inputs'])} native, "
            f"{len(payload['terminations'])} terminations "
            f"({len(traced)} traced), graphs: {', '.join(graphs)}"
        )
    print(f"wrote {index} ({index.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
