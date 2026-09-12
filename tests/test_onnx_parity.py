"""ADR 0005 parity: exported ONNX term graphs vs the live mjlab env.

Layer: L3 (requires the ``examples`` extras — mjlab / torch / onnxruntime — and a
one-time warp CPU-kernel compile, so it is marked ``slow``/``mjlab`` and skipped
when those deps are absent).

Two scopes. **Cartpole** is asserted in detail — every observation term traced,
``time_out`` classified native, both reset Events replayed against recorded RNG draws
— because it is small enough for each claim to be specific. **Every other reference
task** goes through :func:`parity_sweep`, the same harness over a wider term set.

Run the whole thing with::

    MUJOCO_GL=disable pytest tests/test_onnx_parity.py -m mjlab

Note that the default CI job runs ``-m "not slow"`` *and* installs only the
``dev`` extra, so nothing in this file runs there — the sweep has its own
workflow (``.github/workflows/parity.yml``).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

pytestmark = [pytest.mark.slow, pytest.mark.mjlab]


@pytest.fixture(scope="module")
def cartpole_report():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

    from mjswan.compile import run_parity

    cfg = cartpole_balance_env_cfg(play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    return run_parity(env, obs_group="actor", n_steps=16, seed=0)


def test_all_terms_pass_parity(cartpole_report):
    assert cartpole_report.passed, "\n" + cartpole_report.summary()


def test_observation_terms_are_onnx(cartpole_report):
    onnx_terms = {t.name for t in cartpole_report.terms if t.representation == "onnx"}
    # The four Cartpole actor observation terms must all be ONNX graphs.
    assert {"cart_pos", "pole_angle", "cart_vel", "pole_vel"} <= onnx_terms


def test_time_out_is_native(cartpole_report):
    time_out = next(t for t in cartpole_report.terms if t.name == "time_out")
    assert time_out.representation == "native"


def test_every_onnx_obs_term_checked_every_step(cartpole_report):
    for t in cartpole_report.terms:
        if t.representation == "onnx" and t.kind == "observation":
            assert t.steps_checked == cartpole_report.n_steps
            assert t.max_abs_diff <= cartpole_report.atol


def test_reset_events_are_onnx_and_match(cartpole_report):
    events = {t.name: t for t in cartpole_report.terms if t.kind == "event"}
    assert {"reset_slider", "reset_hinge"} <= set(events)
    for name in ("reset_slider", "reset_hinge"):
        ev = events[name]
        assert ev.representation == "onnx"
        # reset_joints_by_offset draws one position + one velocity offset.
        assert ev.rand_dim == 2
        assert ev.steps_checked > 0
        assert ev.max_abs_diff <= cartpole_report.atol


# ---------------------------------------------------------------------------
# The sweep: the same harness over every reference task, not just Cartpole, whose four
# scalar observations miss the bugs the wide tasks had — a frozen `height_scan` on
# Velocity-Rough, an unresolved `SceneEntityCfg` widening `ee_to_cube` on Lift.
# ---------------------------------------------------------------------------

# The reference tasks, with why each one earns its place in the sweep.
SWEEP_TASKS = [
    pytest.param("Mjlab-Cartpole-Swingup", id="cartpole-swingup"),
    # Command-state slots and site-indexed reads, where the `SceneEntityCfg` bug hid.
    pytest.param("Mjlab-Lift-Cube-Yam", id="lift-cube-yam"),
    # Builtin-sensor slots, `joint_pos_biased`, a traced termination.
    pytest.param("Mjlab-Velocity-Flat-Unitree-G1", id="velocity-flat-g1"),
    pytest.param("Mjlab-Velocity-Flat-Unitree-Go1", id="velocity-flat-go1"),
    # `height_scan`: a structured `RayCastSensor`, once baked as 187 constants.
    pytest.param("Mjlab-Velocity-Rough-Unitree-G1", id="velocity-rough-g1"),
    pytest.param("Mjlab-Velocity-Rough-Unitree-Go1", id="velocity-rough-go1"),
]

# Out of the sweep: the Tracking tasks need a W&B motion clip and cannot be built offline,
# and the Lift camera variants observe rendered images, which mjswan does not serve.


@pytest.fixture(scope="module")
def sweep_report(request):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.compile import run_parity

    cfg = load_env_cfg(request.param, play=True)
    cfg.scene.num_envs = 1
    # The Rough terrain makes far more contacts than mjlab's one-env `nconmax` allows,
    # so without this they raise `nconmax overflow` at construction.
    cfg.sim.nconmax = 200_000
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        yield run_parity(env, obs_group="actor", n_steps=8, seed=0)
    finally:
        env.close()


@pytest.mark.parametrize("sweep_report", SWEEP_TASKS, indirect=True)
def test_every_term_matches_mjlab(sweep_report):
    assert sweep_report.passed, "\n" + sweep_report.summary()


@pytest.mark.parametrize("sweep_report", SWEEP_TASKS, indirect=True)
def test_no_term_is_silently_unchecked(sweep_report):
    """A term reported as a graph must have been *compared*, every step.

    Without this the suite above passes on a task whose terms all traced and none
    of which was ever run: `passed` is an AND over comparisons, and an empty AND
    is True. `steps_checked` is what distinguishes "agreed" from "never asked".
    """
    graphs = [
        t
        for t in sweep_report.terms
        if t.representation == "onnx" and t.kind == "observation"
    ]
    assert graphs, "no observation term traced to a graph — the task serialized empty"
    for term in graphs:
        assert term.steps_checked == sweep_report.n_steps, (
            f"{term.name} compared on {term.steps_checked} of "
            f"{sweep_report.n_steps} steps"
        )
        assert term.max_abs_diff <= sweep_report.atol


# ---------------------------------------------------------------------------
# The traced path: every `EntityData` property recomputed in the graph from the raw sim
# fields it reads, mjlab's own math included. The shortcut — value slots the browser's
# reader fills — is checked against mjlab by `slotReaderParity.test.ts`; this is the
# other half, so a property with no reader is proven the same way.
# ---------------------------------------------------------------------------

_TRACED_TASKS = [
    # Site-indexed reads and a command; `root_link_*` through `compute_velocity_from_cvel`.
    pytest.param("Mjlab-Lift-Cube-Yam", id="lift-cube-yam-traced"),
    # `joint_pos_biased` (encoder bias baked), `projected_gravity_b`, builtin sensors.
    pytest.param("Mjlab-Velocity-Flat-Unitree-G1", id="velocity-flat-g1-traced"),
]


@pytest.fixture(scope="module")
def traced_report(request):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan.compile import run_parity

    cfg = load_env_cfg(request.param, play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        yield run_parity(env, obs_group="actor", n_steps=8, seed=0, reader_fields=())
    finally:
        env.close()


@pytest.mark.parametrize("traced_report", _TRACED_TASKS, indirect=True)
def test_every_property_traced_through_matches_mjlab(traced_report):
    from mjlab.entity.data import EntityData

    assert traced_report.passed, "\n" + traced_report.summary()
    properties = {n for n, v in vars(EntityData).items() if isinstance(v, property)}
    graphs = [
        t
        for t in traced_report.terms
        if t.representation == "onnx" and t.kind == "observation"
    ]
    assert graphs, "no observation term traced to a graph"
    for term in graphs:
        assert term.steps_checked == traced_report.n_steps
        assert term.max_abs_diff <= traced_report.atol
        # No property survives as a value slot: each was recomputed from `sim:*`.
        entity_fields = [
            label.split(".", 1)[1]
            for label in term.input_slots
            if not label.startswith(("sensor:", "command:", "sim:"))
        ]
        assert not (set(entity_fields) & properties), term.input_slots


# ---------------------------------------------------------------------------
# `push_robot`: the interval-mode event, which the sweep above cannot reach.
#
# mjlab's play configs pop it (`go1/env_cfgs.py`: `cfg.events.pop("push_robot")`), and
# the sweep builds every task with `play=True`, so re-adding it here is the only way to
# exercise an event that *reads* live state — `root_link_vel_w` as a dynamic slot — and
# writes root velocity back as an `entity_write`. Every other traced event on the
# reference tasks is reset-mode and reads only baked constants.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def push_robot_report():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs.mdp import push_by_setting_velocity
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

    from mjswan.compile import run_parity

    cfg = unitree_go1_flat_env_cfg(play=True)
    cfg.events["push_robot"] = EventTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        yield run_parity(
            env,
            # Stepped first, so `root_link_vel_w` is non-zero and the dynamic slot
            # carries a value a baked constant could not fake.
            n_steps=20,
            seed=0,
            event_modes=("interval", "reset"),
            n_event_draws=16,
            # Observations need Command handling, covered by the sweep and
            # `test_onnx_command_parity.py`.
            include_obs=False,
        )
    finally:
        env.close()


def test_push_robot_matches_mjlab(push_robot_report):
    assert push_robot_report.passed, "\n" + push_robot_report.summary()


def test_push_robot_is_a_graph_over_a_dynamic_slot(push_robot_report):
    push = next((t for t in push_robot_report.terms if t.name == "push_robot"), None)
    assert push is not None, "push_robot missing — did play mode pop it again?"
    assert push.representation == "onnx"
    # Six draws (x, y, z, roll, pitch, yaw) against the live root velocity it adds to.
    assert push.rand_dim == 6
    assert push.input_slots, (
        "push_robot reported no dynamic slot, so the graph baked `root_link_vel_w` "
        "as a constant — the failure this task is in the suite to catch"
    )
    assert push.steps_checked > 0
    assert push.max_abs_diff <= push_robot_report.atol


# ---------------------------------------------------------------------------
# The Builder serializes from the task config while the harness above reads the env's
# prepared managers — two sources for the same terms, which diverged once when an
# unresolved `SceneEntityCfg` widened `ee_to_cube` from 3 to 6. Pinned here to the width
# mjlab itself computes.
# ---------------------------------------------------------------------------

_SIZE_TASKS = [
    "Mjlab-Cartpole-Balance",
    "Mjlab-Lift-Cube-Yam",
    "Mjlab-Velocity-Flat-Unitree-G1",
]


def _mjlab_observation_widths(env, group: str) -> dict[str, int]:
    """Per-term output width as mjlab's own resolved manager computes it."""
    om = env.observation_manager
    widths = {}
    for term_name in om.active_terms[group]:
        cfg = om.get_term_cfg(group, term_name)
        value = cfg.func(env, **cfg.params)
        widths[term_name] = int(value.reshape(1, -1).shape[-1])
    return widths


@pytest.mark.parametrize("task_id", _SIZE_TASKS)
def test_serialized_observation_widths_match_mjlab(task_id, tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from mjswan._onnx_build import serialize_observation_group
    from mjswan.adapters.mjlab_adapter import adapt_observations

    cfg = load_env_cfg(task_id, play=True)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    expected = _mjlab_observation_widths(env, "actor")

    groups = adapt_observations({"policy": cfg.observations["actor"]})
    assert groups is not None
    entries = serialize_observation_group(groups["policy"], env, tmp_path, "policy")

    # These groups all fuse, so the per-term widths live in the group's `layout`.
    assert isinstance(entries, dict), "expected a fused group"
    actual = {term["name"]: term["size"] for term in entries["layout"]}
    assert actual == expected
    assert entries["size"] == sum(expected.values())
    # And the concatenated group width the policy network sees.
    assert sum(actual.values()) == sum(expected.values())
