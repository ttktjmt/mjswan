"""OnnxCommand config serialization (ADR 0005 §3).

Layer: L1 (pure Python — no mjlab/torch/onnxruntime; builds a CommandExport by
hand). Verifies the config emitted from a traced command carries everything the
runtime needs.
"""

from __future__ import annotations

import json
import warnings

import pytest

# Pure-Python serialization, but `mjswan.compile` imports torch at load time.
torch = pytest.importorskip("torch")

from mjswan.compile import command_config, write_command_artifact  # noqa: E402
from mjswan.compile.tracer import (  # noqa: E402
    _COMMAND_NS,
    _SENSOR_NS,
    CommandExport,
    _is_dynamic_field,
    slot_to_json,
)


def _make_export() -> CommandExport:
    """A velocity-shaped CommandExport (as trace_command_term would return)."""
    return CommandExport(
        name="twist",
        onnx_bytes=b"\x08\x01onnx-graph-bytes",
        state_fields=[
            {
                "name": "vel_command_b",
                "shape": [1, 3],
                "dtype": "float32",
                "init": [0.0, 0.0, 0.0],
            },
            {"name": "heading_target", "shape": [1], "dtype": "float32", "init": [0.0]},
            {"name": "is_heading_env", "shape": [1], "dtype": "bool", "init": [False]},
            {"name": "is_standing_env", "shape": [1], "dtype": "bool", "init": [False]},
        ],
        command_field="vel_command_b",
        input_slots=[("robot", "heading_w")],
        input_names=["robot__heading_w"],
        rand_dim=6,
        # Six draws of mjlab's velocity ranges, as `DrawRecorder` records them.
        rand_ranges=[[-1.0, 1.0]] * 3 + [[-3.14, 3.14]] * 3,
        output_names=["next_vel_command_b"],
        write_targets=[],
        reference_rand=torch.zeros(6),
    )


def test_command_config_shape():
    export = _make_export()
    cfg = command_config(
        export, onnx_ref="command/twist.onnx", resampling_time_range=(3.0, 8.0)
    )
    # "name" is the registry key (always "OnnxCommand"); the term's own id is the
    # caller's dict key, kept as "term_id" for diagnostics.
    assert cfg["name"] == "OnnxCommand"
    assert cfg["term_id"] == "twist"
    assert cfg["onnx"] == "command/twist.onnx"
    assert cfg["command_field"] == "vel_command_b"
    assert cfg["rand_dim"] == 6
    assert cfg["resampling_time_range"] == [3.0, 8.0]
    # `input` names the graph input, so the runtime never re-derives it
    assert {
        "entity": "robot",
        "field": "heading_w",
        "input": "robot__heading_w",
    } in cfg["input_slots"]
    # The runtime allocates from shape + dtype and starts the term at `init`, rather
    # than zero-filling and relying on the first resample.
    for sf in cfg["state_fields"]:
        assert set(sf) == {"name", "shape", "dtype", "init"}
        expected = 1
        for dim in sf["shape"]:
            expected *= dim
        assert len(sf["init"]) == expected


def test_viz_reading_an_undeclared_state_field_warns():
    """A stale field name draws nothing, and the browser cannot tell you why."""
    export = _make_export()
    viz = [{"shape": "sphere", "origin": {"state": "vel_command"}}]
    with pytest.warns(RuntimeWarning, match="vel_command"):
        command_config(export, onnx_ref="command/twist.onnx", viz=viz)


def test_viz_on_a_declared_state_field_is_quiet():
    export = _make_export()
    viz = [{"shape": "sphere", "origin": {"state": "vel_command_b"}}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        command_config(export, onnx_ref="command/twist.onnx", viz=viz)


def test_command_config_round_trips_through_json():
    export = _make_export()
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    assert json.loads(json.dumps(cfg)) == cfg


def test_lifting_command_with_entity_write():
    export = CommandExport(
        name="lift_height",
        onnx_bytes=b"onnx",
        state_fields=[
            {
                "name": "target_pos",
                "shape": [1, 3],
                "dtype": "float32",
                "init": [0.0, 0.0, 0.0],
            }
        ],
        command_field="target_pos",
        input_slots=[],
        input_names=[],
        rand_dim=7,
        rand_ranges=[[-0.5, 0.5]] * 7,
        output_names=["next_target_pos", "root_pose__pose", "root_velocity__velocity"],
        write_targets=[
            {"kind": "root_pose", "entity": "cube", "fields": ["pose"]},
            {"kind": "root_velocity", "entity": "cube", "fields": ["velocity"]},
        ],
        reference_rand=torch.zeros(7),
    )
    cfg = command_config(export, onnx_ref="command/lift_height.onnx")
    kinds = {w["kind"] for w in cfg["write_targets"]}
    assert kinds == {"root_pose", "root_velocity"}
    # The command field must name a declared state field, or the runtime allocates
    # a command of width 0 and every consumer silently reads zeros.
    assert cfg["command_field"] in {sf["name"] for sf in cfg["state_fields"]}


def test_slot_to_json_entity_data():
    assert slot_to_json(("robot", "joint_pos")) == {
        "entity": "robot",
        "field": "joint_pos",
        "input": "robot__joint_pos",
    }


def test_slot_to_json_sensor():
    # A whole-sensor read is its own slot shape, its MJCF path folded to an identifier.
    assert slot_to_json((_SENSOR_NS, "robot/imu_lin_vel")) == {
        "sensor": "robot/imu_lin_vel",
        "input": "sensor__robot_imu_lin_vel",
    }


def test_slot_to_json_command_state():
    # A read of another command's state (mjlab's object_to_goal_distance).
    assert slot_to_json((_COMMAND_NS, "lift_height.target_pos")) == {
        "command": "lift_height",
        "field": "target_pos",
        "input": "command__lift_height_target_pos",
    }


def test_slots_json_drops_a_slot_the_exporter_folded_away():
    """A declared slot the graph does not take must not reach the config.

    `torch.onnx.export` bakes an integer index tensor into the Gather it feeds, so
    a term reading one (mjlab's `MotionCommand.body_indexes`) exports a graph
    without that input. ORT rejects a feed that is not a graph input outright, so
    shipping the slot would break every run of the graph.
    """
    import onnx
    from onnx import TensorProto, helper

    kept = helper.make_tensor_value_info("robot__heading_w", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["robot__heading_w"], ["value"])],
        "g",
        [kept],
        [helper.make_tensor_value_info("value", TensorProto.FLOAT, [1])],
    )
    export = _make_export()
    export.onnx_bytes = onnx.helper.make_model(graph).SerializeToString()
    export.input_slots = [("robot", "heading_w"), (_COMMAND_NS, "motion.body_indexes")]
    export.input_shapes = [[1], [14]]

    cfg = command_config(export, onnx_ref="command/twist.onnx")
    assert [slot["input"] for slot in cfg["input_slots"]] == ["robot__heading_w"]


def test_unknown_data_fields_default_to_dynamic():
    # Only model-derived constants are listed, so anything else errs toward dynamic:
    # baking a field that varies is silent corruption.
    assert _is_dynamic_field("site_pos_w")
    assert _is_dynamic_field("some_future_mjlab_field")
    assert not _is_dynamic_field("default_joint_pos")
    assert not _is_dynamic_field("soft_joint_pos_limits")


def test_command_config_accepts_a_sensor_slot():
    export = _make_export()
    export.input_slots = [(_SENSOR_NS, "robot/imu_ang_vel")]
    cfg = command_config(export, onnx_ref="command/twist.onnx")
    assert cfg["input_slots"] == [
        {"sensor": "robot/imu_ang_vel", "input": "sensor__robot_imu_ang_vel"}
    ]


def test_write_command_artifact(tmp_path):
    export = _make_export()
    cfg = write_command_artifact(export, tmp_path, resampling_time_range=(3.0, 8.0))
    written = tmp_path / "command" / "twist.onnx"
    assert written.read_bytes() == export.onnx_bytes
    assert cfg["onnx"] == "command/twist.onnx"


# ---------------------------------------------------------------------------
# A native command's traced reset graph: `MotionCommand`'s RSI jitter. The clip lookup
# stays native, the `sample_uniform` around it is traced.
# ---------------------------------------------------------------------------


def _write_capture_env():
    """Minimal env satisfying the event tracer: `scene[name].data.<field>` + writes."""

    class _Data:
        def __init__(self, **fields):
            for key, value in fields.items():
                setattr(self, key, value)

    class _Entity:
        def __init__(self, data):
            self.data = data

        def write_joint_state_to_sim(self, position, velocity, **_):
            pass

    class _Scene:
        def __init__(self, entities):
            self._entities = entities
            self.env_origins = torch.zeros((1, 3))

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self, entities):
            self.scene = _Scene(entities)
            self.num_envs = 1  # read by the tracer's single-env guard

    data = _Data(
        joint_pos=torch.tensor([[0.1, 0.2]]),
        joint_vel=torch.tensor([[0.0, 0.0]]),
    )
    return _Env({"robot": _Entity(data)})


class _AssetCfg:
    """Stands in for mjlab's SceneEntityCfg (only `.name` is read here)."""

    def __init__(self, name):
        self.name = name


def test_the_export_filters_match_the_exporters_wording(monkeypatch):
    """String matches against torch's wording, so drift is silent.

    What keeps an unvetted warning — a `copy_` removal, another constructor's bake —
    from being swallowed with the three that are safe. `catch_warnings` stops the
    process-wide install from leaking into the rest of the session.
    """
    from mjswan.compile import tracer

    monkeypatch.setattr(tracer, "_EXPORT_FILTERS_INSTALLED", False)
    vetted = [
        "ONNX Preprocess - Removing mutation from node aten::index_put_ on block "
        "input: 'value.1'. This changes graph semantics.",
        "Using len to get tensor shape might cause the trace to be incorrect. "
        "Recommended usage would be tensor.shape[0].",
        "torch.tensor results are registered as constants in the trace. You can "
        "safely ignore this warning if you use this function to create tensors out "
        "of constant variables that would be the same every time you call this "
        "function.",
    ]
    others = [
        "ONNX Preprocess - Removing mutation from node aten::copy_ on block input",
        "Converting a tensor to a Python boolean might cause the trace to be incorrect",
        "torch.as_tensor results are registered as constants in the trace",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tracer._prepare_single_env_export(1)
        for message in vetted + others:
            warnings.warn(message, UserWarning, stacklevel=1)
    assert [str(w.message) for w in caught] == others


def test_a_batched_trace_is_refused():
    """Silencing `len(env_ids)` is only sound while the baked row count is 1."""
    from mjswan.compile.tracer import _prepare_single_env_export

    with pytest.raises(ValueError, match="num_envs=4"):
        _prepare_single_env_export(4)


def test_native_command_emits_a_traced_reset_graph(tmp_path):
    pytest.importorskip("mjlab")
    from rsi_body_fixture import rsi_joint_offset

    from mjswan._onnx_build import serialize_command
    from mjswan.command import CommandTermConfig, PendingResetTrace

    cfg = CommandTermConfig(
        term_name="TrackingCommand",
        params={"sampling_mode": "start"},
        pending_reset_trace=PendingResetTrace(
            func=rsi_joint_offset,
            params={"asset_cfg": _AssetCfg("robot"), "offset": 0.1},
        ),
    )
    entry = serialize_command("motion", cfg, _write_capture_env(), tmp_path)

    # The native term's own params survive untouched...
    assert entry["name"] == "TrackingCommand"
    assert entry["sampling_mode"] == "start"
    # ...and the graph rides alongside in the shape `OnnxEvent` consumes.
    graph = entry["reset_graph"]
    assert graph["mode"] == "reset"
    assert graph["onnx"] == "command/motion_reset.onnx"
    assert graph["rand_dim"] == 2  # one draw per joint
    # The graph consumes the sampler's output, so without these the runtime draws
    # [0, 1) — a radian of joint jitter where the term asked for ±0.1.
    flat = [v for pair in graph["rand_ranges"] for v in pair]
    assert flat == pytest.approx([-0.1, 0.1, -0.1, 0.1], abs=1e-7)
    assert [t["kind"] for t in graph["write_targets"]] == ["joint_state"]
    assert [s["field"] for s in graph["input_slots"]] == ["joint_pos", "joint_vel"]
    assert (tmp_path / graph["onnx"]).exists()


def test_element_bounds_broadcasts_mjlab_per_axis_ranges():
    """A per-axis range column has to land on the right elements.

    mjlab's `_resample_command` draws a 6-dof pose offset with `ranges[:, 0]`,
    `ranges[:, 1]` — tensors, not scalars — and an omitted `pose_range` key means a
    zero-width range for that axis: mjlab draws exactly 0.0 there. Broadcasting the
    columns wrong would hand those axes somebody else's range.
    """
    from mjswan.compile.rng import _element_bounds

    lower = torch.tensor([0.0, 0.0, 0.0, -0.2, -0.2, -0.2])
    upper = torch.tensor([0.0, 0.0, 0.0, 0.2, 0.2, 0.2])
    bounds = _element_bounds(torch.Size([1, 6]), lower, upper, (1, 6))
    assert bounds.flatten().tolist() == pytest.approx(
        [0, 0, 0, 0, 0, 0, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2], abs=1e-7
    )
    # Scalar bounds broadcast to every element of the draw.
    scalar = _element_bounds(torch.Size([1, 2]), -0.1, 0.1, (1, 2))
    assert scalar.flatten().tolist() == pytest.approx([-0.1, 0.1, -0.1, 0.1], abs=1e-7)


def test_command_without_a_reset_trace_is_unchanged(tmp_path):
    from mjswan._onnx_build import serialize_command
    from mjswan.command import CommandTermConfig

    cfg = CommandTermConfig(
        term_name="TrackingCommand", params={"sampling_mode": "start"}
    )
    # No env is touched at all when there is nothing to trace.
    entry = serialize_command("motion", cfg, object(), tmp_path)
    assert entry == {"name": "TrackingCommand", "sampling_mode": "start"}


# ---------------------------------------------------------------------------
# An untraceable observation must fail the build: dropping it shortens the vector the
# network was trained on, and baking it freezes a live input (187 frozen terrain heights,
# in `height_scan`'s case).
# ---------------------------------------------------------------------------


def _opaque_state_env():
    """An env whose only readable state is not a tensor, so no slot can carry it."""

    class _Data:
        def __init__(self):
            # Real state the term branches on, but nothing a graph input can hold.
            self.contact_mode = "soft"

    class _Entity:
        def __init__(self):
            self.data = _Data()

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": _Entity()}

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()

    return _Env()


def _reads_opaque_state(env, *, entity_name="robot"):
    """A term whose only env read yields no tensor — the untraceable shape.

    mjlab's `height_scan` used to land here (its `RayCastSensor.data` is a struct
    of ray hits); the tracer now follows that per field, so this fixture uses a
    non-tensor field instead. The failure mode being guarded is unchanged: state
    was read, none of it can become a graph input, and baking the result would
    freeze whatever it varies with.
    """
    mode = env.scene[entity_name].data.contact_mode
    return torch.full((1, 3), 1.0 if mode == "soft" else 2.0)


def _reads_nothing(env, *, width=3):
    """A genuine constant — a fixed-size padding term with no env dependency."""
    del env
    return torch.zeros((1, width))


def test_untraceable_observation_fails_the_build():
    from mjswan.compile.tracer import UntraceableTerm, trace_term

    with pytest.raises(UntraceableTerm) as excinfo:
        trace_term(_reads_opaque_state, {}, _opaque_state_env(), name="contact_obs")
    # The message has to name what it could not follow, or nobody can act on it.
    assert "contact_obs" in str(excinfo.value)
    assert "robot.contact_mode" in str(excinfo.value)
    assert excinfo.value.touched == ["robot.contact_mode"]


def test_term_reading_nothing_is_a_constant_not_untraceable():
    from mjswan.compile.tracer import ConstantTerm, UntraceableTerm, trace_term

    with pytest.raises(ConstantTerm) as excinfo:
        trace_term(_reads_nothing, {}, _opaque_state_env(), name="padding")
    # A `ConstantTerm` is safe to bake, an `UntraceableTerm` is not, and "no graph
    # inputs" alone cannot tell them apart.
    assert not isinstance(excinfo.value, UntraceableTerm)


def test_serializer_bakes_a_constant_but_refuses_an_untraceable_term(tmp_path):
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.compile.tracer import UntraceableTerm
    from mjswan.managers.observation_manager import ObservationTermCfg

    env = _opaque_state_env()
    baked = serialize_observation_term(
        "padding", ObservationTermCfg(func=_reads_nothing), env, tmp_path, None
    )
    assert baked["native"] == "constant"
    assert baked["size"] == 3

    with pytest.raises(UntraceableTerm):
        serialize_observation_term(
            "contact_obs",
            ObservationTermCfg(func=_reads_opaque_state),
            env,
            tmp_path,
            None,
        )


def test_observation_binding_without_ts_src_fails_rather_than_dropping():
    """A binding names a TS class; without `ts_src` there is no class.

    mjswan ships no built-in TS observation classes, so such a binding resolves to
    nothing in the browser and the term goes missing from a bundle that reports
    itself complete — shortening the vector the policy was trained on.
    """
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.envs.mdp.observations import ObservationBinding
    from mjswan.managers.observation_manager import ObservationTermCfg

    term = ObservationTermCfg(func=ObservationBinding(ts_name="HeightScan"))
    with pytest.raises(ValueError, match="no built-in TS term classes"):
        serialize_observation_term("height_scan", term, object(), None, None)


def test_termination_binding_without_ts_src_fails_rather_than_dropping():
    """Same treatment as the observation above, for the same reason.

    An unresolvable termination used to be dropped from the config with nothing
    logged — and dropped *at build time*, so the runtime never saw the term either
    and could not warn the way it does for a term whose graph failed to load. The
    episode then silently never checks a reset condition it is configured to have.
    """
    from mjswan._onnx_build import serialize_terminations
    from mjswan.envs.mdp.terminations import TerminationBinding
    from mjswan.managers.termination_manager import TerminationTermCfg

    terms = {
        "illegal_contact": TerminationTermCfg(
            func=TerminationBinding(ts_name="IllegalContact")
        )
    }
    with pytest.raises(ValueError, match="no built-in TS term classes"):
        serialize_terminations(terms, object(), None)


def test_a_binding_with_ts_src_serializes(tmp_path):
    """The supported shape: a class the builder will inject."""
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.envs.mdp.observations import ObservationBinding
    from mjswan.managers.observation_manager import ObservationTermCfg

    term = ObservationTermCfg(
        func=ObservationBinding(ts_name="MyObs", ts_src=str(tmp_path / "MyObs.ts"))
    )
    entry = serialize_observation_term("my_obs", term, object(), tmp_path, None)
    assert entry is not None and entry["name"] == "MyObs"


def test_structured_sensor_fields_become_one_slot_each():
    """mjlab's `RayCastSensor` traces per field, not as one opaque blob.

    A height scan reads `distances` / `frame_pos_w` / `hit_pos_w` off a sensor whose
    `.data` is a struct. Logging the struct wholesale made the term look
    untraceable, which is what silently froze `height_scan` on both Velocity-Rough
    tasks. Each field is its own slot now, so the arithmetic traces and the runtime
    is told exactly which readings to supply.
    """
    from mjswan.compile.tracer import slot_to_json, trace_term

    class _RayData:
        def __init__(self):
            self.distances = torch.tensor([[1.0, 2.0]])
            self.hit_pos_w = torch.tensor([[[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]]])

    class _Sensor:
        def __init__(self):
            self.data = _RayData()

    class _Scene:
        def __init__(self):
            self.sensors = {"terrain_scan": _Sensor()}

        def __getitem__(self, name):
            return self.sensors[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()

    def height_scan(env, *, sensor_name="terrain_scan"):
        data = env.scene[sensor_name].data
        heights = -data.hit_pos_w[..., 2]
        return torch.where(data.distances < 0, torch.zeros_like(heights), heights)

    export = trace_term(height_scan, {}, _Env(), name="height_scan")
    assert [slot_to_json(k) for k in export.input_slots] == [
        {
            "sensor": "terrain_scan",
            "input": "sensor__terrain_scan_distances",
            "field": "distances",
        },
        {
            "sensor": "terrain_scan",
            "input": "sensor__terrain_scan_hit_pos_w",
            "field": "hit_pos_w",
        },
    ]
    # Only the fields the term touched — `normals_w` and the rest stay out.
    assert export.reference_output.shape == (1, 2)


# ---------------------------------------------------------------------------
# Termination fusion, whose payoff scales with the traced-term count: the locomotion and
# manipulation tasks have 0-1, the tracking tasks three beside the native `time_out`.
# ---------------------------------------------------------------------------


def _term_env():
    """Two entity fields, so a group can read one, the other, or both."""

    class _Data:
        def __init__(self):
            self.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.4]])
            self.projected_gravity_b = torch.tensor([[0.0, 0.1, -0.99]])

    class _Entity:
        def __init__(self):
            self.data = _Data()

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": _Entity()}

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()
            self.max_episode_length_s = 20.0

    return _Env()


def _too_low(env, *, minimum_height=0.5):
    return env.scene["robot"].data.root_link_pos_w[:, 2] < minimum_height


def _tipped(env, *, limit=0.5):
    return env.scene["robot"].data.projected_gravity_b[:, 2] > -limit


def _time_out(env):
    """Native by construction: reads nothing off the env."""
    del env
    return torch.zeros(1, dtype=torch.bool)


def test_terminations_fuse_into_one_graph_with_one_lane_per_term(tmp_path):
    pytest.importorskip("mjlab")
    from mjswan._onnx_build import FUSED_TERMINATION_KEY, serialize_terminations
    from mjswan.managers.termination_manager import TerminationTermCfg

    entries = serialize_terminations(
        {
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
            "too_low": TerminationTermCfg(func=_too_low),
            "tipped": TerminationTermCfg(func=_tipped),
        },
        _term_env(),
        tmp_path,
    )

    # The native marker keeps its own entry: it reads no state to fuse.
    assert entries["time_out"]["native"] == "elapsed_s >= episode_length_s"
    assert entries["time_out"]["episode_length_s"] == 20.0

    fused = entries[FUSED_TERMINATION_KEY]
    assert fused["fused"] == "term/terminations.onnx"
    assert (tmp_path / fused["fused"]).exists()
    # One lane per term in graph output order, each flagged truncation or not, and
    # naming the function it traces (#118).
    assert [(lane["name"], lane["time_out"]) for lane in fused["lanes"]] == [
        ("too_low", False),
        ("tipped", False),
    ]
    assert fused["lanes"][0]["func"] == f"{__name__}:_too_low"
    # Slots are the union of what the two terms read, deduplicated.
    assert sorted(s["field"] for s in fused["input_slots"]) == [
        "projected_gravity_b",
        "root_link_pos_w",
    ]


def test_a_lone_traced_termination_is_not_fused(tmp_path):
    """Fusing one term buys no `ort.run()` and costs a wire shape."""
    pytest.importorskip("mjlab")
    from mjswan._onnx_build import FUSED_TERMINATION_KEY, serialize_terminations
    from mjswan.managers.termination_manager import TerminationTermCfg

    entries = serialize_terminations(
        {
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
            "too_low": TerminationTermCfg(func=_too_low),
        },
        _term_env(),
        tmp_path,
    )
    assert FUSED_TERMINATION_KEY not in entries
    assert entries["too_low"]["onnx"] == "term/too_low.onnx"


def test_fused_lanes_match_the_terms_run_individually(tmp_path):
    """The graph's lane *i* must be term *i* — a swap would be silent."""
    pytest.importorskip("mjlab")
    onnxruntime = pytest.importorskip("onnxruntime")
    from mjswan.compile.tracer import (
        GroupTermSpec,
        read_slot,
        trace_termination_group,
    )

    env = _term_env()
    specs = [
        GroupTermSpec("too_low", _too_low, {"minimum_height": 0.5}),
        GroupTermSpec("never", _too_low, {"minimum_height": 0.1}),
        GroupTermSpec("tipped", _tipped, {"limit": 0.5}),
    ]
    export = trace_termination_group(specs, env, name="terminations")
    session = onnxruntime.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    feeds = {
        name: read_slot(env, key).detach().numpy()
        for name, key in zip(export.input_names, export.input_slots)
    }
    lanes = session.run([export.output_name], feeds)[0].reshape(-1).astype(bool)
    expected = [bool(spec.func(env, **spec.params).item()) for spec in specs]

    assert export.lanes == ["too_low", "never", "tipped"]
    assert lanes.tolist() == expected
    # The lanes differ, so agreement is no coincidence: z=0.4, upright.
    assert expected == [True, False, False]


# ---------------------------------------------------------------------------
# Stateful-term initial values: without them the runtime zero-fills, which is wrong for
# a term carrying a counter or a held value rather than resampling every field.
# ---------------------------------------------------------------------------


class _StatefulTerm:
    """A traceable command term whose state does *not* start at zero."""

    num_envs = 1

    def __init__(self):
        self.cfg = type("_Cfg", (), {"entity_name": None})()
        # Neither is re-drawn on resample, so zero-filling would start the term elsewhere.
        self.bias = torch.tensor([[0.25, -0.5, 1.75]])
        self.latched = torch.tensor([True])
        self.command = torch.tensor([[0.0, 0.0, 0.0]])

    def _resample_command(self, rand):
        # Only `command` is resampled; `bias`/`latched` carry over untouched.
        self.command = self.bias + rand.reshape(1, -1)[:, :3]

    def _update_command(self):
        pass


def test_traced_state_fields_carry_the_terms_initial_values(tmp_path):
    pytest.importorskip("mjlab")
    from mjswan.compile import trace_command_term

    term = _StatefulTerm()
    export = trace_command_term(
        term,
        ["bias", "latched", "command"],
        name="held",
        command_field="command",
    )
    specs = {sf["name"]: sf for sf in export.state_fields}

    # The values `build()` left, not zeros — and flattened to the declared width.
    assert specs["bias"]["init"] == [0.25, -0.5, 1.75]
    assert specs["bias"]["shape"] == [1, 3]
    # A bool field round-trips as a bool rather than as 1.0.
    assert specs["latched"]["init"] == [True]
    assert specs["latched"]["dtype"] == "bool"
    # Even an overwritten field reports its pre-resample value: frame 0 needs a start.
    assert specs["command"]["init"] == [0.0, 0.0, 0.0]


def test_state_field_init_is_restored_not_post_trace(tmp_path):
    """Tracing mutates the term; the emitted init must be the pre-trace value.

    `trace_command_term` snapshots the state, runs discovery and the export (both
    of which call `_resample_command`), then restores. If `init` were read after
    tracing instead of after the restore, it would ship whatever the last traced
    resample happened to leave.
    """
    pytest.importorskip("mjlab")
    from mjswan.compile import trace_command_term

    term = _StatefulTerm()
    export = trace_command_term(
        term, ["bias", "command"], name="held", command_field="command"
    )
    specs = {sf["name"]: sf for sf in export.state_fields}
    assert specs["command"]["init"] == [0.0, 0.0, 0.0]
    # And the term itself is back where it started, so a second trace agrees.
    assert term.command.reshape(-1).tolist() == [0.0, 0.0, 0.0]


class TestATermThatReadsNumEnvs:
    """`env.num_envs` is forwarded to the real env, not stood in for.

    mjlab's tracking observations end in `pos.view(env.num_envs, -1)`. The replay env
    served the recorded slots but had no `num_envs` at all, so tracing such a term died
    with `AttributeError`; the *event* replay env meanwhile defaulted it to 1, so
    discovery saw the real N and replay silently saw 1.
    """

    @staticmethod
    def _env(num_envs):
        class _Data:
            def __init__(self):
                self.root_pos_w = torch.zeros(num_envs, 3)

        class _Entity:
            def __init__(self):
                self.data = _Data()

        class _Scene:
            def __getitem__(self, name):
                return _Entity()

        class _Env:
            def __init__(self):
                self.scene = _Scene()
                self.num_envs = num_envs
                self.device = "cpu"

        return _Env()

    @staticmethod
    def _term(env, asset_name="robot"):
        return env.scene[asset_name].data.root_pos_w.view(env.num_envs, -1)

    def test_a_term_reading_num_envs_traces(self):
        from mjswan.compile import trace_term

        export = trace_term(self._term, {}, self._env(1), name="root_pos")
        assert export.onnx_bytes

    def test_the_forwarded_value_is_the_real_envs(self):
        """Not a hardcoded 1: an env with N reaches the term as N."""
        from mjswan.compile import trace_term

        export = trace_term(self._term, {}, self._env(4), name="root_pos")
        # 4 rows in, 4 rows out — a baked `view(1, -1)` would collapse them to 1.
        assert export.reference_output.shape[0] == 4

    def test_an_undeclared_env_attr_still_raises(self):
        """Only num_envs/device forward; everything else must stay an error."""
        from mjswan.compile import trace_term

        def reads_episode_length(env, asset_name="robot"):
            return env.scene[asset_name].data.root_pos_w * env.max_episode_length

        with pytest.raises(AttributeError, match="max_episode_length"):
            trace_term(reads_episode_length, {}, self._env(1), name="bad")


def test_a_command_terms_replay_env_does_not_forward_back_into_the_term():
    """`_EventReplayEnv` must forward to the env it replaced, not to the term.

    `ManagerTermBase.num_envs` is a property returning `self._env.num_envs`, and the
    command tracer swaps `term._env` for the replay env — so pointing the replay env's
    forwarding at the term makes `num_envs` recurse until the stack dies. Only the
    slow command-parity suite caught this, which is too late to be useful.
    """
    from mjswan.compile.tracer import _EventReplayEnv

    class _RealEnv:
        num_envs = 4
        device = "cpu"

    class _Term:
        """Shaped like mjlab's `ManagerTermBase`: num_envs reads through `_env`."""

        def __init__(self):
            self._env = _RealEnv()

        @property
        def num_envs(self):
            return self._env.num_envs

    term = _Term()
    original = term._env
    term._env = _EventReplayEnv({}, {}, real_env=original)
    assert term.num_envs == 4
