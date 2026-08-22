"""Startup domain-randomization events that write `mjModel` fields (ADR 0005 §5).

Layer: L1 (pure Python — `model_field_dr_descriptor` touches neither torch nor
onnxruntime), plus one L3 integration test behind `importorskip`.

These events perturb the *model* rather than `mjData`, so the `entity_write` tracer
sees nothing and there is no graph to compare against mjlab numerically. What can be
checked is the description itself, and the two ways it has gone wrong:

* **Scope.** An unresolved `SceneEntityCfg` has `geom_ids=slice(None)`, so reading
  the raw params described *every* geom in the scene — 56 instead of Lift's 12
  fingertip geoms. The event still "worked"; it just also roughened the floor.
* **Defaults.** mjlab's wrappers do not share an `operation` default
  (`geom_friction` is `abs`, `body_com_offset` `add`, `body_mass` `scale`), so a
  single hardcoded default would describe an omitted `operation` as *replacing*
  a body's mass with a number in [0.8, 1.2] rather than scaling by it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mjswan._onnx_build import model_field_dr_descriptor

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "mjswan" / "template"

# A tiny model: the world owns geom 0, the robot owns 1..5 and bodies 1..2.
GEOM_NAMES = [
    "world/floor",
    "robot/torso_collision",
    "robot/lf_tip_collision",
    "robot/rf_tip_collision",
    "robot/lf_pad_collision",
    "robot/rf_pad_collision",
]
BODY_NAMES = ["world", "robot/torso", "robot/hand"]
ROBOT_GEOM_IDS = [1, 2, 3, 4, 5]
ROBOT_BODY_IDS = [1, 2]


class _Ids(list):
    """Stands in for the int tensor `EntityIndexing` holds."""

    def tolist(self):
        return list(self)


class _Named:
    def __init__(self, name):
        self.name = name


class _MjModel:
    """Only the name accessors `_dr_entity_names` reaches for."""

    def geom(self, i):
        return _Named(GEOM_NAMES[i])

    def body(self, i):
        return _Named(BODY_NAMES[i])

    def site(self, i):
        raise AssertionError("no sites in this fixture")


class _EntityCfg:
    """Stands in for mjlab's `SceneEntityCfg`, unresolved until `resolve()`.

    The `slice(None)` default is the point: it means "every element", which is why
    describing an event from the raw params silently widens its scope.
    """

    def __init__(self, name, *, geom_names=(), body_names=()):
        self.name = name
        self.geom_names = geom_names
        self.body_names = body_names
        self.geom_ids = slice(None)
        self.body_ids = slice(None)

    def resolve(self, scene):
        asset = scene[self.name]
        for kind, names in (("geom", self.geom_names), ("body", self.body_names)):
            if not names:
                continue
            owned = [
                {"geom": GEOM_NAMES, "body": BODY_NAMES}[kind][i]
                for i in getattr(asset.indexing, f"{kind}_ids").tolist()
            ]
            setattr(self, f"{kind}_ids", [owned.index(n) for n in names])


class _Env:
    def __init__(self):
        indexing = type(
            "_Indexing",
            (),
            {"geom_ids": _Ids(ROBOT_GEOM_IDS), "body_ids": _Ids(ROBOT_BODY_IDS)},
        )()
        asset = type("_Asset", (), {"indexing": indexing})()
        self.scene = {"robot": asset}
        self.sim = type("_Sim", (), {"mj_model": _MjModel()})()


class _TermCfg:
    """Stands in for `EventTermCfg` (only `.func` / `.params` / `.mode` are read)."""

    def __init__(self, func, params, mode="startup"):
        self.func = func
        self.params = params
        self.mode = mode


# mjlab's wrappers with their real signatures: the defaults are what is under test, so
# they must be defaults here too. The bodies write nothing, which is faithful — the real
# ones write `env.sim.model`, which the recording proxy does not wrap, and that is why
# these events fall through to the descriptor.


def geom_friction(  # noqa: PLR0917 — mjlab's own arity; the signature is the fixture
    env,
    env_ids,
    ranges,
    asset_cfg=None,
    distribution="uniform",
    operation="abs",
    axes=None,
    shared_random=False,
):
    del env, env_ids, ranges, asset_cfg, distribution, operation, axes, shared_random


def body_com_offset(  # noqa: PLR0917 — mjlab's own arity; the signature is the fixture
    env,
    env_ids,
    ranges,
    asset_cfg=None,
    distribution="uniform",
    operation="add",
    axes=None,
    shared_random=False,
):
    del env, env_ids, ranges, asset_cfg, distribution, operation, axes, shared_random


def body_mass(  # noqa: PLR0917 — mjlab's own arity; the signature is the fixture
    env,
    env_ids,
    ranges,
    asset_cfg=None,
    distribution="uniform",
    operation="scale",
    axes=None,
    shared_random=False,
):
    del env, env_ids, ranges, asset_cfg, distribution, operation, axes, shared_random


# `requires_model_fields(..., recompute=RecomputeLevel.set_const)` sets this.
body_com_offset.recompute = 3
body_mass.recompute = 3
geom_friction.recompute = 0


def _fingertips():
    return _EntityCfg(
        "robot", geom_names=("robot/lf_tip_collision", "robot/rf_tip_collision")
    )


class TestScope:
    """The 56-instead-of-12 regression."""

    def test_a_scoped_event_describes_only_its_own_elements(self):
        term = _TermCfg(
            geom_friction, {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)}
        )
        descriptor = model_field_dr_descriptor(term, _Env())
        assert descriptor["entity_names"] == [
            "robot/lf_tip_collision",
            "robot/rf_tip_collision",
        ]

    def test_it_resolves_the_cfg_itself_when_given_raw_params(self):
        # The descriptor is public, so it cannot assume `serialize_event` resolved first.
        term = _TermCfg(
            geom_friction, {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)}
        )
        assert len(model_field_dr_descriptor(term, _Env())["entity_names"]) == 2

    def test_an_unscoped_cfg_still_means_the_whole_entity(self):
        term = _TermCfg(
            geom_friction, {"asset_cfg": _EntityCfg("robot"), "ranges": (0.3, 1.5)}
        )
        descriptor = model_field_dr_descriptor(term, _Env())
        # The robot's five geoms — but never the world's floor.
        assert descriptor["entity_names"] == GEOM_NAMES[1:]

    def test_names_not_ids_because_the_browser_compiles_its_own_model(self):
        term = _TermCfg(
            body_com_offset,
            {
                "asset_cfg": _EntityCfg("robot", body_names=("robot/torso",)),
                "ranges": (0.0, 0.1),
            },
        )
        descriptor = model_field_dr_descriptor(term, _Env())
        assert descriptor["entity_names"] == ["robot/torso"]
        assert descriptor["entity_type"] == "body"


class TestWrapperDefaults:
    """Each mjlab wrapper carries its own defaults; they are read, not assumed."""

    def test_an_omitted_operation_comes_from_the_wrapper(self):
        # The severe case: `abs` would replace a body's mass with the range value.
        mass = model_field_dr_descriptor(
            _TermCfg(
                body_mass, {"asset_cfg": _EntityCfg("robot"), "ranges": (0.8, 1.2)}
            ),
            _Env(),
        )
        assert mass["operation"] == "scale"
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    body_com_offset,
                    {"asset_cfg": _EntityCfg("robot"), "ranges": (0.0, 0.1)},
                ),
                _Env(),
            )["operation"]
            == "add"
        )
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    geom_friction,
                    {"asset_cfg": _EntityCfg("robot"), "ranges": (0.3, 1.5)},
                ),
                _Env(),
            )["operation"]
            == "abs"
        )

    def test_an_explicit_operation_wins(self):
        descriptor = model_field_dr_descriptor(
            _TermCfg(
                body_mass,
                {
                    "asset_cfg": _EntityCfg("robot"),
                    "ranges": (0.8, 1.2),
                    "operation": "abs",
                },
            ),
            _Env(),
        )
        assert descriptor["operation"] == "abs"

    def test_uses_defaults_tracks_the_operation(self):
        # `Operation.uses_defaults`: `add`/`scale` use the compiled default, `abs` overwrites.
        def described(operation):
            return model_field_dr_descriptor(
                _TermCfg(
                    geom_friction,
                    {
                        "asset_cfg": _EntityCfg("robot"),
                        "ranges": (0.3, 1.5),
                        "operation": operation,
                    },
                ),
                _Env(),
            )["uses_defaults"]

        assert described("add") is True
        assert described("scale") is True
        assert described("abs") is False

    def test_an_operation_instance_is_named_not_stringified(self):
        # mjlab accepts an `Operation` instance as well as a string.
        instance = type("Operation", (), {"name": "scale"})()
        descriptor = model_field_dr_descriptor(
            _TermCfg(
                geom_friction,
                {
                    "asset_cfg": _EntityCfg("robot"),
                    "ranges": (1.0, 2.0),
                    "operation": instance,
                },
            ),
            _Env(),
        )
        assert descriptor["operation"] == "scale"

    def test_set_const_comes_from_the_recompute_level(self):
        # mjlab records whether `mj_setConst` is owed on the function, so it is read.
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    body_com_offset,
                    {"asset_cfg": _EntityCfg("robot"), "ranges": (0.0, 0.1)},
                ),
                _Env(),
            )["set_const"]
            is True
        )
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    geom_friction,
                    {"asset_cfg": _EntityCfg("robot"), "ranges": (0.3, 1.5)},
                ),
                _Env(),
            )["set_const"]
            is False
        )


class TestAxes:
    """mjlab's `_determine_target_axes` precedence, and per-axis ranges."""

    def _described(self, params, func=geom_friction):
        params = {"asset_cfg": _EntityCfg("robot"), **params}
        return model_field_dr_descriptor(_TermCfg(func, params), _Env())

    def test_a_tuple_range_broadcasts_over_the_wrappers_default_axes(self):
        # `geom_friction` defaults to axis 0 alone (tangential friction).
        assert self._described({"ranges": (0.3, 1.5)})["axis_ranges"] == {0: [0.3, 1.5]}
        # `body_com_offset` defaults to all three.
        assert self._described({"ranges": (-0.02, 0.02)}, body_com_offset)[
            "axis_ranges"
        ] == {
            0: [-0.02, 0.02],
            1: [-0.02, 0.02],
            2: [-0.02, 0.02],
        }

    def test_explicit_axes_win_over_the_default(self):
        # Lift's three friction events each name one axis, so they compose.
        assert self._described({"ranges": (1e-4, 2e-2), "axes": [1]})[
            "axis_ranges"
        ] == {1: [1e-4, 2e-2]}

    def test_int_keyed_ranges_target_exactly_those_axes(self):
        # Velocity's `base_com` form.
        described = self._described(
            {"ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}},
            body_com_offset,
        )
        assert described["axis_ranges"] == {
            0: [-0.025, 0.025],
            1: [-0.025, 0.025],
            2: [-0.03, 0.03],
        }

    def test_axes_narrow_a_keyed_range_rather_than_widening_it(self):
        # `_prepare_axis_ranges` drops a range for an axis nobody targets.
        assert self._described(
            {"ranges": {0: (0.3, 1.5), 2: (1e-5, 5e-3)}, "axes": [0]}
        )["axis_ranges"] == {0: [0.3, 1.5]}

    def test_a_target_axis_with_no_range_is_left_undescribed(self):
        # The browser cannot draw for an unbounded axis, so it stays a native marker.
        assert self._described({"ranges": {0: (0.3, 1.5)}, "axes": [0, 1]}) is None


class TestNotDescribable:
    """What must stay a native marker rather than be half-described."""

    def test_a_func_that_is_not_a_known_model_field_helper(self):
        def encoder_bias(env, env_ids, asset_cfg=None, bias_range=(0.0, 0.0)):
            raise AssertionError("never called")

        # `encoder_bias` is not an `mjModel` field: the runtime applies it from the config.
        assert (
            model_field_dr_descriptor(
                _TermCfg(encoder_bias, {"asset_cfg": _EntityCfg("robot")}), _Env()
            )
            is None
        )

    def test_string_keyed_ranges(self):
        # mjlab resolves these per element by name pattern; not reproduced here.
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    geom_friction,
                    {
                        "asset_cfg": _EntityCfg("robot"),
                        "ranges": {".*_tip.*": (0.3, 1.5)},
                    },
                ),
                _Env(),
            )
            is None
        )

    def test_a_missing_range(self):
        assert (
            model_field_dr_descriptor(
                _TermCfg(geom_friction, {"asset_cfg": _EntityCfg("robot")}), _Env()
            )
            is None
        )

    def test_an_entity_type_that_cannot_be_enumerated(self):
        def dof_damping(env, env_ids, ranges, asset_cfg=None):
            raise AssertionError("never called")

        # `dof`-indexed fields address by joint dof, which is not a name table.
        dof_damping._mjswan_dr_field = ("dof_damping", "dof", [0])
        assert (
            model_field_dr_descriptor(
                _TermCfg(
                    dof_damping,
                    {"asset_cfg": _EntityCfg("robot"), "ranges": (0.1, 0.5)},
                ),
                _Env(),
            )
            is None
        )


def test_an_author_wrapper_opts_in_with_mjswan_dr_field():
    """A wrapper mjswan has never heard of, declaring its own field."""

    def stiffen_pads(env, env_ids, ranges, asset_cfg=None, operation="scale"):
        raise AssertionError("never called")

    stiffen_pads._mjswan_dr_field = ("geom_solref", "geom", [0, 1])
    descriptor = model_field_dr_descriptor(
        _TermCfg(
            stiffen_pads, {"asset_cfg": _EntityCfg("robot"), "ranges": (0.9, 1.1)}
        ),
        _Env(),
    )
    assert descriptor["field"] == "geom_solref"
    assert descriptor["axis_ranges"] == {0: [0.9, 1.1], 1: [0.9, 1.1]}
    # No `recompute` attribute, so `set_const` falls back to the field list.
    assert descriptor["set_const"] is False


def test_the_descriptor_carries_exactly_what_the_browser_declares():
    """Wire parity with `ModelFieldDrConfig` in `core/event/modelFieldDr.ts`.

    A field added on one side only is invisible until a randomization silently does
    nothing, so the two declarations are compared directly.
    """
    source = (TEMPLATE / "src" / "core" / "event" / "modelFieldDr.ts").read_text()
    block = re.search(
        r"export interface ModelFieldDrConfig \{(.*?)^\}", source, re.S | re.M
    )
    assert block is not None, "ModelFieldDrConfig is not declared where expected"
    declared = set(re.findall(r"^\s{2}(\w+)[?]?:", block.group(1), re.M))

    descriptor = model_field_dr_descriptor(
        _TermCfg(geom_friction, {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)}),
        _Env(),
    )
    # `name` and `mode` are attached by `serialize_event`, not by the descriptor.
    assert declared == set(descriptor) | {"name"}


def test_serialize_event_emits_the_descriptor_with_its_name_and_mode(tmp_path):
    """End to end through the path the Builder actually takes."""
    pytest.importorskip("mjlab")
    from mjswan._onnx_build import serialize_event

    entry = serialize_event(
        "fingertip_friction_slide",
        _TermCfg(
            geom_friction,
            {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5), "axes": [0]},
        ),
        _Env(),
        tmp_path,
    )
    assert entry["name"] == "fingertip_friction_slide"
    assert entry["mode"] == "startup"
    assert entry["kind"] == "model_field"
    assert entry["entity_names"] == [
        "robot/lf_tip_collision",
        "robot/rf_tip_collision",
    ]
    # No graph: the browser draws these itself from the seeded stream.
    assert "onnx" not in entry
    assert not list(tmp_path.rglob("*.onnx"))


class TestAnUntraceableEventFailsTheBuild:
    """What happens when a term traces to nothing and is not a model-field DR.

    This used to be emitted as ``{"native": True, "reason": ...}``, which the runtime
    skips silently — so a reset randomization the task is configured to apply just did
    not happen, and nothing said so. Only the cases below, where there is provably
    nothing to write, stay native; anything else fails the build.
    """

    @staticmethod
    def _serialize(term_cfg, tmp_path, env=None):
        pytest.importorskip("mjlab")
        from mjswan._onnx_build import serialize_event

        return serialize_event("ev", term_cfg, env or _Env(), tmp_path)

    def test_an_unexplained_no_write_raises_and_names_both_escape_hatches(
        self, tmp_path
    ):
        def push_robot(env, env_ids, velocity_range=None):
            return None

        with pytest.raises(ValueError, match="register_event") as excinfo:
            self._serialize(_TermCfg(push_robot, {}, mode="interval"), tmp_path)
        assert "ts_src" in str(excinfo.value)

    def test_randomize_terrain_stays_native_with_its_reason(self, tmp_path):
        def randomize_terrain(env, env_ids):
            return None

        entry = self._serialize(_TermCfg(randomize_terrain, {}, mode="reset"), tmp_path)
        assert entry["native"] is True
        assert "one baked terrain" in entry["reason"]

    def test_encoder_bias_stays_native_with_its_reason(self, tmp_path):
        def encoder_bias(env, env_ids, bias_range=(0.0, 0.0), asset_cfg=None):
            return None

        entry = self._serialize(
            _TermCfg(encoder_bias, {"bias_range": (-0.01, 0.01)}, mode="reset"),
            tmp_path,
        )
        assert entry["native"] is True
        assert "policy config" in entry["reason"]

    def test_a_root_write_onto_a_fixed_base_entity_stays_native(self, tmp_path):
        """mjlab's manipulation tasks configure `reset_base` on their fixed arms."""

        def reset_root_state_uniform(
            env, env_ids, pose_range=None, velocity_range=None, asset_cfg=None
        ):
            return None

        env = _Env()
        env.scene["robot"].is_fixed_base = True
        entry = self._serialize(
            _TermCfg(
                reset_root_state_uniform,
                {"asset_cfg": _fingertips(), "pose_range": {}, "velocity_range": {}},
                mode="reset",
            ),
            tmp_path,
            env,
        )
        assert entry["native"] is True
        assert "fixed-base" in entry["reason"]

    def test_the_same_root_write_onto_a_floating_base_entity_raises(self, tmp_path):
        """The check is the entity, not the term name: a mobile base must trace."""

        def reset_root_state_uniform(
            env, env_ids, pose_range=None, velocity_range=None, asset_cfg=None
        ):
            return None

        with pytest.raises(ValueError, match="could not be traced"):
            self._serialize(
                _TermCfg(
                    reset_root_state_uniform,
                    {"asset_cfg": _fingertips(), "pose_range": {}},
                    mode="reset",
                ),
                tmp_path,
            )


class TestFlatPatchSpawnTraces:
    """A terrain scene spawns on a flat patch as a traced term.

    It was a `ts_src` class drawing from `Math.random()`, so it could neither replay
    from the seeded stream nor be checked numerically. As a traced body the patch table
    bakes in and the two draws become the graph's `rand` input — which is only worth
    anything if the draw actually reaches the Gather, hence the runtime assertions.
    """

    PATCHES = [[-4.0, -4.0, 0.1], [0.0, 0.0, 0.0], [4.0, 4.0, -0.2]]

    @staticmethod
    def _trace(patches, yaw_range=(-3.14, 3.14)):
        pytest.importorskip("mjlab")
        torch = pytest.importorskip("torch")
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        from mjswan.compile import trace_event_term
        from mjswan.envs.mdp.events import reset_root_state_on_flat_patch

        class _Data:
            def __init__(self):
                # (N, 13): pos, quat, lin/ang vel — standing height 0.8, identity yaw.
                self.default_root_state = torch.tensor(
                    [[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6]
                )

        class _Entity:
            def __init__(self):
                self.data = _Data()

            def write_root_link_pose_to_sim(self, pose, env_ids=None):
                self.written = pose

        class _Scene(dict):
            def __getitem__(self, name):
                return self.setdefault(name, _Entity())

        class _Env:
            def __init__(self):
                self.scene = _Scene()
                self.num_envs = 1
                self.device = "cpu"

        return trace_event_term(
            reset_root_state_on_flat_patch,
            {
                "asset_cfg": SceneEntityCfg("robot"),
                "patches": patches,
                "yaw_range": yaw_range,
            },
            _Env(),
            name="reset_base",
            mode="reset",
        )

    def test_it_traces_to_one_root_pose_write_with_two_draws(self):
        export = self._trace(self.PATCHES)
        assert [w["kind"] for w in export.write_targets] == ["root_pose"]
        # One draw picks the patch (scaled to an index), one picks the yaw.
        assert export.rand_dim == 2
        flat = [bound for pair in export.rand_ranges for bound in pair]
        assert flat == pytest.approx([0.0, 1.0, -3.14, 3.14])

    def _run(self, export, rand0, rand1=0.0):
        import numpy as np

        ort = pytest.importorskip("onnxruntime")
        sess = ort.InferenceSession(export.onnx_bytes)
        feeds = {}
        for spec in sess.get_inputs():
            if spec.name == "rand":
                feeds[spec.name] = np.array([rand0, rand1], dtype=np.float32)
                continue
            shape = [1 if not isinstance(d, int) else d for d in spec.shape]
            feeds[spec.name] = np.zeros(shape, dtype=np.float32)
        return sess.run(None, feeds)[0].reshape(-1)

    def test_the_draw_reaches_the_gather(self):
        """A baked index would spawn on the same patch forever — the silent failure."""
        export = self._trace(self.PATCHES)
        picked = [tuple(self._run(export, r)[:2].round(4)) for r in (0.0, 0.5, 0.99)]
        assert len(set(picked)) == 3
        assert picked == [(-4.0, -4.0), (0.0, 0.0), (4.0, 4.0)]

    def test_a_draw_of_one_clamps_to_the_last_patch(self):
        export = self._trace(self.PATCHES)
        assert tuple(self._run(export, 1.0)[:2].round(4)) == (4.0, 4.0)

    def test_the_standing_height_is_baked_from_the_default_root_state(self):
        """Read live instead and the browser spawns the robot inside the terrain: its
        keyframe restore zeroes `mjData.xpos` before the reset events run."""
        export = self._trace(self.PATCHES)
        assert export.input_slots == []
        assert self._run(export, 0.0)[2] == pytest.approx(0.1 + 0.8)

    def test_the_yaw_draw_rotates_the_root(self):
        export = self._trace(self.PATCHES)
        assert self._run(export, 0.0, -3.0)[3:].tolist() != pytest.approx(
            self._run(export, 0.0, 3.0)[3:].tolist()
        )


class TestWriteTargetEntity:
    """Which entity a traced write lands on, and how many it may land on.

    The browser resolves a root write through this name, and a scene with two floating
    bodies — a robot and a thrown ball — has a free joint each. Reading the name off an
    `asset_cfg` param only is what left it `null`: mjlab's own terms take that param, but
    a task's term is free to take a plain `ball_name`, and then the write went to
    whichever free joint came first in the model. The robot got launched; the ball never
    moved. mjlab writes per entity, so the tracer keys captures the same way.
    """

    @staticmethod
    def _env():
        torch = pytest.importorskip("torch")

        class _Data:
            def __init__(self):
                self.root_link_pos_w = torch.zeros(1, 3)
                self.default_root_state = torch.zeros(1, 13)

        class _Entity:
            def __init__(self):
                self.data = _Data()
                self.written = 0

            def write_root_link_pose_to_sim(self, pose, env_ids=None):
                self.written += 1

            def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
                self.written += 1

            def write_root_state_to_sim(self, root_state, env_ids=None):
                self.written += 1

        class _Scene(dict):
            def __getitem__(self, name):
                return self.setdefault(name, _Entity())

            @property
            def entities(self):
                return {name: self[name] for name in ("robot", "ball")}

        class _Env:
            def __init__(self):
                self.scene = _Scene()
                self.num_envs = 1
                self.device = "cpu"

        return _Env()

    @staticmethod
    def _trace(func, params, env=None):
        pytest.importorskip("mjlab")
        pytest.importorskip("torch")
        from mjswan.compile import trace_event_term

        return trace_event_term(
            func,
            params,
            env if env is not None else TestWriteTargetEntity._env(),
            name="throw",
            mode="interval",
        )

    def test_the_write_names_the_entity_it_was_made_on(self):
        torch = pytest.importorskip("torch")

        def throw(env, env_ids, ball_name="ball"):
            ball = env.scene[ball_name]
            ball.write_root_link_pose_to_sim(torch.zeros(1, 7), env_ids=env_ids)
            ball.write_root_link_velocity_to_sim(torch.zeros(1, 6), env_ids=env_ids)

        export = self._trace(throw, {"ball_name": "ball"})
        assert [(w["kind"], w["entity"]) for w in export.write_targets] == [
            ("root_pose", "ball"),
            ("root_velocity", "ball"),
        ]
        # The graph output carries the entity too, so a second one cannot collide.
        assert [w["outputs"] for w in export.write_targets] == [
            ["ball__root_pose__pose"],
            ["ball__root_velocity__velocity"],
        ]
        assert export.output_names == [
            "ball__root_pose__pose",
            "ball__root_velocity__velocity",
        ]

    def test_an_asset_cfg_still_names_it(self):
        """mjlab's own convention keeps working, and stays the fallback."""
        torch = pytest.importorskip("torch")
        pytest.importorskip("mjlab")
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        def reset(env, env_ids, asset_cfg=None):
            env.scene[asset_cfg.name].write_root_link_pose_to_sim(
                torch.zeros(1, 7), env_ids=env_ids
            )

        export = self._trace(reset, {"asset_cfg": SceneEntityCfg("robot")})
        assert [w["entity"] for w in export.write_targets] == ["robot"]

    def test_two_entities_each_get_their_own_target(self):
        """One write per entity reaches the browser, as in mjlab — the robot's root and
        the ball's are different addresses."""
        torch = pytest.importorskip("torch")

        def throw_both(env, env_ids):
            for name in ("ball", "robot"):
                env.scene[name].write_root_link_pose_to_sim(
                    torch.zeros(1, 7), env_ids=env_ids
                )

        export = self._trace(throw_both, {})
        assert [(w["kind"], w["entity"]) for w in export.write_targets] == [
            ("root_pose", "ball"),
            ("root_pose", "robot"),
        ]
        assert export.output_names == [
            "ball__root_pose__pose",
            "robot__root_pose__pose",
        ]

    def test_a_root_state_write_splits_into_pose_and_velocity(self):
        """mjlab's own split of the 13-wide state, so a term using it traces as well as
        one calling the two writes itself."""
        torch = pytest.importorskip("torch")

        def reset(env, env_ids):
            env.scene["ball"].write_root_state_to_sim(
                torch.zeros(1, 13), env_ids=env_ids
            )

        export = self._trace(reset, {})
        assert [(w["kind"], w["entity"]) for w in export.write_targets] == [
            ("root_pose", "ball"),
            ("root_velocity", "ball"),
        ]

    def test_iterating_the_scene_never_touches_the_live_entities(self):
        """A term walking `scene.entities` gets recording stand-ins: writing through the
        live ones would move the tracing env under every term traced after it."""
        torch = pytest.importorskip("torch")
        env = self._env()

        def reset_all(env, env_ids):
            for entity in env.scene.entities.values():
                entity.write_root_link_pose_to_sim(torch.zeros(1, 7), env_ids=env_ids)

        export = self._trace(reset_all, {}, env=env)
        assert {w["entity"] for w in export.write_targets} == {"robot", "ball"}
        assert [e.written for e in env.scene.entities.values()] == [0, 0]

    def test_an_uncaptured_write_is_refused_not_forwarded(self):
        """Forwarding it would mutate the live env and emit no graph output for it."""
        torch = pytest.importorskip("torch")

        def spin(env, env_ids):
            env.scene["robot"].write_root_link_velocity_b_to_sim(torch.zeros(1, 6))

        with pytest.raises(ValueError, match="does not capture"):
            self._trace(spin, {})


def test_reset_scene_to_default_needs_no_graph():
    """The runtime's own reset already restores every entity's default state, so mjlab's
    default event has nothing left to write."""
    from mjswan._onnx_build import _EVENTS_WITH_NOTHING_TO_WRITE

    assert "reset_scene_to_default" in _EVENTS_WITH_NOTHING_TO_WRITE


class TestApplyTerrainSpawn:
    """`add_scene_mjlab` swaps mjlab's root-spawn reset once the terrain has patches."""

    @staticmethod
    def _scene(terrain_data, events):
        from mjswan.scene import SceneConfig

        scene = SceneConfig(name="s")
        scene.terrain_data = terrain_data
        scene.events = events
        return scene

    @staticmethod
    def _uniform_event():
        pytest.importorskip("mjlab")
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        from mjswan.managers.event_manager import EventTermCfg

        def reset_root_state_uniform(env, env_ids, **kwargs) -> None: ...

        return EventTermCfg(
            func=reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("go1"),
                "pose_range": {"yaw": (-1.0, 1.0)},
            },
        )

    def test_it_replaces_the_uniform_reset_and_keeps_entity_and_yaw(self):
        from mjswan.envs.mdp.events import (
            apply_terrain_spawn,
            reset_root_state_on_flat_patch,
        )

        patches = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.5]]
        scene = self._scene(
            {"flat_patches": {"spawn": patches}}, {"reset_base": self._uniform_event()}
        )
        apply_terrain_spawn(scene)

        term = scene.events["reset_base"]
        assert term.func is reset_root_state_on_flat_patch
        assert term.mode == "reset"
        assert term.params["patches"] == patches
        assert term.params["asset_cfg"].name == "go1"
        assert term.params["yaw_range"] == (-1.0, 1.0)

    def test_it_leaves_a_scene_without_terrain_alone(self):
        from mjswan.envs.mdp.events import apply_terrain_spawn

        event = self._uniform_event()
        scene = self._scene(None, {"reset_base": event})
        apply_terrain_spawn(scene)
        assert scene.events["reset_base"] is event


def test_serialize_events_reports_each_term_it_traces(monkeypatch):
    """The build's progress line names the event it is on; keep the hook wired."""
    from mjswan import _onnx_build

    monkeypatch.setattr(
        _onnx_build, "serialize_event", lambda name, cfg, env, out: {"name": name}
    )
    seen: list[str] = []
    _onnx_build.serialize_events(
        {"reset_slider": object(), "reset_hinge": object()},
        env=None,
        out_dir=None,
        on_term=seen.append,
    )
    assert seen == ["reset_slider", "reset_hinge"]
