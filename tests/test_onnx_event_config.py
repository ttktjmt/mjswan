"""Event terms through the Builder: model-field randomization, write targets, and the
native and refusal paths.

A model-field event (mjlab's `dr.*`) writes `env.sim.model`, so its writes become graph
outputs and the fields it reads become slots, each naming its elements. The tests below
run mjlab's own bodies against a tiny real model, and pin two ways to get it wrong:

* **Scope.** An unresolved `SceneEntityCfg` has `geom_ids=slice(None)`, so tracing the
  raw params would randomize *every* geom (Lift: 56 instead of its 12 fingertip geoms).
* **Defaults.** mjlab's wrappers do not share an `operation` default (`geom_friction`
  is `abs`, `body_com_offset` `add`, `body_mass` `scale`), and `abs` would replace a
  body's mass with the range value rather than scale it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "src" / "mjswan" / "template"

# The world owns the floor; `robot` the rest, its names prefixed as mjlab attaches them.
MJCF = """
<mujoco>
  <asset><mesh name="shell" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot/torso" pos="0 0 1">
      <freejoint name="robot/root"/>
      <geom name="robot/torso_collision" type="box" size="0.1 0.1 0.1" mass="2"/>
      <geom name="robot/shell_visual" type="mesh" mesh="shell" contype="0"
            conaffinity="0" mass="0"/>
      <body name="robot/hand" pos="0 0 -0.3">
        <joint name="robot/wrist" type="hinge" axis="0 1 0" damping="0.5"
               armature="0.01"/>
        <geom name="robot/lf_tip_collision" type="sphere" size="0.05" mass="0.3"/>
        <geom name="robot/rf_tip_collision" type="sphere" size="0.04" mass="0.2"/>
        <body name="robot/finger" pos="0 0 -0.1">
          <joint name="robot/knuckle" type="hinge" axis="1 0 0" damping="0.2"
                 armature="0.02"/>
          <geom name="robot/finger_collision" type="capsule" size="0.01 0.03"
                mass="0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
TIPS = ["robot/lf_tip_collision", "robot/rf_tip_collision"]


class _Entity:
    """What `dr.*` and `SceneEntityCfg.resolve` read off an mjlab `Entity`: its own
    unprefixed names, and `indexing` into the model, free joint excluded."""

    def __init__(self, mj_model, torch):
        import mujoco

        def owned(obj, count):
            names = [mujoco.mj_id2name(mj_model, obj, i) or "" for i in range(count)]
            return [
                (i, n.removeprefix("robot/")) for i, n in enumerate(names) if "/" in n
            ]

        geoms = owned(mujoco.mjtObj.mjOBJ_GEOM, mj_model.ngeom)
        bodies = owned(mujoco.mjtObj.mjOBJ_BODY, mj_model.nbody)
        joints = [
            (j, n)
            for j, n in owned(mujoco.mjtObj.mjOBJ_JOINT, mj_model.njnt)
            if mj_model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
        ]
        self.geom_names = [n for _, n in geoms]
        self.body_names = [n for _, n in bodies]
        self.joint_names = [n for _, n in joints]
        self.num_geoms, self.num_bodies = len(geoms), len(bodies)
        self.num_joints = len(joints)
        self.is_fixed_base = False
        self.data = SimpleNamespace()
        ids = lambda values: torch.tensor(list(values), dtype=torch.int)  # noqa: E731
        self.indexing = SimpleNamespace(
            geom_ids=ids(i for i, _ in geoms),
            body_ids=ids(i for i, _ in bodies),
            joint_ids=ids(j for j, _ in joints),
            joint_v_adr=ids(mj_model.jnt_dofadr[j] for j, _ in joints),
            joint_q_adr=ids(mj_model.jnt_qposadr[j] for j, _ in joints),
        )

    def _find(self, keys, names, preserve_order):
        from mjlab.utils.lab_api.string import resolve_matching_names

        return resolve_matching_names(keys, names, preserve_order)

    def find_geoms(self, keys, geom_subset=None, preserve_order=False):
        return self._find(keys, geom_subset or self.geom_names, preserve_order)

    def find_bodies(self, keys, preserve_order=False):
        return self._find(keys, self.body_names, preserve_order)

    def find_joints(self, keys, joint_subset=None, preserve_order=False):
        return self._find(keys, joint_subset or self.joint_names, preserve_order)


class _Model:
    """`env.sim.model` as mujoco_warp holds it: float fields per world, `(1, n, ...)`."""

    def __init__(self, mj_model, torch):
        self._mj, self._torch = mj_model, torch

    def __getattr__(self, name):
        value = np.array(getattr(self._mj, name))
        if name == "geom_aabb":
            value = value.reshape(-1, 2, 3)
        tensor = self._torch.as_tensor(value)
        if tensor.is_floating_point():
            tensor = tensor.float()[None]
        setattr(self, name, tensor)
        return tensor


class _TinyEnv:
    num_envs = 1
    device = "cpu"

    def __init__(self):
        import mujoco

        torch = pytest.importorskip("torch")
        mj_model = mujoco.MjModel.from_xml_string(MJCF)
        model = _Model(mj_model, torch)
        self.scene = {"robot": _Entity(mj_model, torch)}
        self.sim = SimpleNamespace(
            mj_model=mj_model,
            model=model,
            per_world_default_fields=set(),
            get_default_field=lambda field: torch.as_tensor(
                np.array(getattr(mj_model, field)), dtype=getattr(model, field).dtype
            ),
        )


def _term(func, params, mode="startup"):
    from mjswan.managers.event_manager import EventTermCfg

    return EventTermCfg(func=func, mode=mode, params=params)


def _robot(**names):
    """An unresolved `SceneEntityCfg`, as a task config holds it."""
    pytest.importorskip("mjlab")
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    return SceneEntityCfg("robot", **names)


def _fingertips():
    return _robot(geom_names=(".*_tip_collision",))


def _serialize(func, params, tmp_path, *, mode="startup", env=None):
    pytest.importorskip("mjlab")
    from mjswan.build.mdp import serialize_event

    return serialize_event("dr", _term(func, params, mode), env or _TinyEnv(), tmp_path)


def _dr():
    pytest.importorskip("mjlab")
    from mjlab.envs.mdp import dr

    return dr


def _rows(mj_model, slot):
    """The model rows a slot or write target names: what the browser resolves."""
    if slot["element"] in ("dof", "qpos"):
        adr = mj_model.jnt_dofadr if slot["element"] == "dof" else mj_model.jnt_qposadr
        return [
            int(adr[mj_model.joint(name).id]) + offset
            for name, offset in zip(slot["names"], slot["offsets"])
        ]
    return [getattr(mj_model, slot["element"])(name).id for name in slot["names"]]


def _run(entry, tmp_path, rand):
    """Run the event's graph on the compiled model's values; its outputs by name."""
    ort = pytest.importorskip("onnxruntime")
    mj_model = _TinyEnv().sim.mj_model
    feeds = {"rand": np.asarray(rand, dtype=np.float32)}
    for slot in entry["input_slots"]:
        field = np.array(getattr(mj_model, slot["model"]), dtype=np.float32)
        rows = field.reshape(field.shape[0], -1)[_rows(mj_model, slot)]
        feeds[slot["input"]] = rows.reshape(slot["shape"])
    session = ort.InferenceSession(str(tmp_path / entry["onnx"]))
    declared = {i.name for i in session.get_inputs()}
    names = [o.name for o in session.get_outputs()]
    outputs = session.run(None, {k: v for k, v in feeds.items() if k in declared})
    return {name: out.reshape(-1) for name, out in zip(names, outputs)}


def _written(entry, tmp_path, rand, field):
    """``{(element name, offset): value}`` the event writes into *field*."""
    outputs = _run(entry, tmp_path, rand)
    out = {}
    for target in entry["write_targets"]:
        if target.get("field") != field:
            continue
        values = outputs[target["outputs"][0]]
        for (element, offset), value in zip(target["cells"], values):
            out[(target["names"][element], offset)] = float(value)
    return out


class TestScope:
    """Only the elements the event's cfg names, never the world's."""

    def test_a_scoped_event_reads_and_writes_only_its_own_elements(self, tmp_path):
        entry = _serialize(
            _dr().geom_friction,
            {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)},
            tmp_path,
        )
        (target,) = entry["write_targets"]
        assert target["kind"] == "model"
        assert target["names"] == TIPS
        assert {tuple(s["names"]) for s in entry["input_slots"]} == {tuple(TIPS)}

    def test_an_unscoped_cfg_means_the_whole_entity_never_the_world(self, tmp_path):
        entry = _serialize(
            _dr().geom_friction, {"asset_cfg": _robot(), "ranges": (0.3, 1.5)}, tmp_path
        )
        names = entry["write_targets"][0]["names"]
        assert "floor" not in names
        assert names == [
            "robot/torso_collision",
            "robot/shell_visual",
            *TIPS,
            "robot/finger_collision",
        ]

    def test_a_dof_is_named_by_its_joint_and_offset(self, tmp_path):
        # The browser compiles its own model, so ids would not survive the trip.
        entry = _serialize(
            _dr().joint_damping,
            {"asset_cfg": _robot(joint_names=("knuckle",)), "ranges": (0.1, 0.2)},
            tmp_path,
        )
        (target,) = entry["write_targets"]
        assert (target["element"], target["names"], target["offsets"]) == (
            "dof",
            ["robot/knuckle"],
            [0],
        )


class TestOperations:
    """Each mjlab wrapper's own `operation`, as its body runs it."""

    def test_an_omitted_operation_is_the_wrappers_own(self, tmp_path):
        # `scale`, against the compiled mass: 2 kg x 1.1, never 1.1 kg.
        entry = _serialize(
            _dr().body_mass,
            {"asset_cfg": _robot(body_names=("torso",)), "ranges": (0.8, 1.2)},
            tmp_path,
        )
        assert any(s.get("default") for s in entry["input_slots"])
        assert _written(entry, tmp_path, [1.1], "body_mass") == {
            ("robot/torso", 0): pytest.approx(2.2)
        }

    def test_add_offsets_the_compiled_value(self, tmp_path):
        entry = _serialize(
            _dr().body_com_offset,
            {"asset_cfg": _robot(body_names=("torso",)), "ranges": (-0.1, 0.1)},
            tmp_path,
        )
        ipos = _TinyEnv().sim.mj_model.body("robot/torso").ipos
        written = _written(entry, tmp_path, [0.01, 0.02, 0.03], "body_ipos")
        assert written == {
            ("robot/torso", axis): pytest.approx(ipos[axis] + delta)
            for axis, delta in enumerate([0.01, 0.02, 0.03])
        }

    def test_abs_reads_no_default_and_writes_the_draw(self, tmp_path):
        entry = _serialize(
            _dr().geom_friction,
            {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)},
            tmp_path,
        )
        assert not any(s.get("default") for s in entry["input_slots"])
        assert _written(entry, tmp_path, [0.4, 0.9], "geom_friction") == {
            (TIPS[0], 0): pytest.approx(0.4),
            (TIPS[1], 0): pytest.approx(0.9),
        }

    def test_only_the_targeted_axes_are_written(self, tmp_path):
        # Lift's three friction events each own one axis, so they compose.
        entry = _serialize(
            _dr().geom_friction,
            {"asset_cfg": _fingertips(), "ranges": (1e-4, 2e-2), "axes": [1]},
            tmp_path,
        )
        assert [cell[1] for t in entry["write_targets"] for cell in t["cells"]] == [
            1,
            1,
        ]

    def test_string_keyed_ranges_resolve_by_name_pattern(self, tmp_path):
        entry = _serialize(
            _dr().geom_friction,
            {
                "asset_cfg": _fingertips(),
                "ranges": {"lf_.*": (0.4, 0.4), "rf_.*": (0.7, 0.7)},
            },
            tmp_path,
        )
        assert _written(entry, tmp_path, [0.4, 0.7], "geom_friction") == {
            (TIPS[0], 0): pytest.approx(0.4),
            (TIPS[1], 0): pytest.approx(0.7),
        }

    def test_set_const_follows_mjlabs_recompute_level(self, tmp_path):
        mass = _serialize(
            _dr().body_mass,
            {"asset_cfg": _robot(body_names=("torso",)), "ranges": (0.8, 1.2)},
            tmp_path / "mass",
        )
        friction = _serialize(
            _dr().geom_friction,
            {"asset_cfg": _fingertips(), "ranges": (0.3, 1.5)},
            tmp_path / "friction",
        )
        assert mass["set_const"] is True
        assert "set_const" not in friction

    def test_the_build_leaves_the_live_model_alone(self, tmp_path):
        env = _TinyEnv()
        before = env.sim.model.body_mass.clone()
        _serialize(
            _dr().body_mass,
            {"asset_cfg": _robot(), "ranges": (0.8, 1.2)},
            tmp_path,
            env=env,
        )
        assert env.sim.model.body_mass.equal(before)

    def test_a_class_term_mjlab_built_traces_through_its_own_instance(self, tmp_path):
        from dataclasses import replace

        from mjswan.build.mdp.provenance import resolved_params

        class Mass:
            built = 0

            def __init__(self, cfg, env):
                type(self).built += 1
                self.cfg = cfg
                self._env = env

            def __call__(self, env, env_ids, asset_cfg, ranges):
                _dr().body_mass(self._env, env_ids, asset_cfg=asset_cfg, ranges=ranges)

        env = _TinyEnv()
        cfg = _term(Mass, {"asset_cfg": _robot(), "ranges": (0.8, 1.2)})
        # mjlab's manager resolves a copy and sets its func to the instance, so the
        # instance's cfg leads back to it.
        built = replace(cfg, params=resolved_params(cfg.params, env))
        built.func = Mass(built, env)
        env.event_manager = SimpleNamespace(get_term_cfg=lambda name: built)
        entry = _serialize(Mass, cfg.params, tmp_path, env=env)
        assert Mass.built == 1
        assert entry["set_const"] is True


class TestDistributions:
    """`rand` carries each sampler as the uniforms behind it; the graph maps them."""

    def test_a_log_uniform_draw_rides_as_its_log(self, tmp_path):
        entry = _serialize(
            _dr().joint_damping,
            {
                "asset_cfg": _robot(joint_names=("wrist",)),
                "ranges": (0.5, 2.0),
                "operation": "scale",
                "distribution": "log_uniform",
            },
            tmp_path,
        )
        assert entry["rand_ranges"] == [pytest.approx([math.log(0.5), math.log(2.0)])]
        written = _written(entry, tmp_path, [math.log(1.5)], "dof_damping")
        assert written == {("robot/wrist", 0): pytest.approx(0.5 * 1.5)}

    def test_a_gaussian_draw_rides_as_two_uniforms(self, tmp_path):
        entry = _serialize(
            _dr().joint_armature,
            {
                "asset_cfg": _robot(joint_names=("wrist",)),
                "ranges": (1.0, 0.1),
                "operation": "scale",
                "distribution": "gaussian",
            },
            tmp_path,
        )
        assert entry["rand_ranges"] == [[0.0, 1.0], [0.0, 1.0]]
        u1, u2 = 0.3, 0.6
        z = math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)
        written = _written(entry, tmp_path, [u1, u2], "dof_armature")
        assert written == {("robot/wrist", 0): pytest.approx(0.01 * (1.0 + 0.1 * z))}


class TestGeomSize:
    """`dr.geom_size` recomputes the broadphase bounds in the same call."""

    def test_the_bounds_follow_the_new_size(self, tmp_path):
        entry = _serialize(
            _dr().geom_size,
            {"asset_cfg": _robot(geom_names=("lf_tip_collision",)), "ranges": (2, 2)},
            tmp_path,
        )
        assert [t["field"] for t in entry["write_targets"]] == [
            "geom_size",
            "geom_size",
            "geom_size",
            "geom_rbound",
            "geom_aabb",
        ]
        rand = [2.0, 2.0, 2.0]
        assert _written(entry, tmp_path, rand, "geom_rbound") == {
            (TIPS[0], 0): pytest.approx(0.1)
        }
        # The half-size row of `(2, 3)`: cells 3..5.
        assert _written(entry, tmp_path, rand, "geom_aabb") == {
            (TIPS[0], offset): pytest.approx(0.1) for offset in (3, 4, 5)
        }

    def test_a_geom_whose_bounds_do_not_follow_from_its_size_fails_the_build(
        self, tmp_path
    ):
        with pytest.raises(ValueError, match="could not be traced") as excinfo:
            _serialize(
                _dr().geom_size,
                {
                    "asset_cfg": _robot(geom_names=("shell_visual",)),
                    "ranges": (0.9, 1.1),
                },
                tmp_path,
            )
        assert "MESH" in str(excinfo.value)


def test_an_authors_own_model_write_traces_too(tmp_path):
    """Any body writing `env.sim.model` traces, mjlab's or not."""
    pytest.importorskip("mjlab")

    def stiffen_pads(env, env_ids, scale=1.5):
        solref = env.sim.model.geom_solref
        solref[0, 3] = solref[0, 3] * scale

    entry = _serialize(stiffen_pads, {}, tmp_path, mode="reset")
    assert _written(entry, tmp_path, [], "geom_solref") == {
        (TIPS[0], 0): pytest.approx(0.02 * 1.5),
        (TIPS[0], 1): pytest.approx(1.0 * 1.5),
    }


def test_an_event_reading_an_unserved_sim_attribute_fails_and_names_it(tmp_path):
    """`env.sim` serves the model only: the rest would act on the trace env."""
    pytest.importorskip("mjlab")

    def scale_model(env, env_ids):
        env.sim.mj_model.nq  # noqa: B018 - the read is the point

    with pytest.raises(ValueError, match="could not be traced") as excinfo:
        _serialize(scale_model, {}, tmp_path, mode="reset")
    assert "env.sim.mj_model" in str(excinfo.value)


def _declared(source: Path, interface: str) -> set[str]:
    block = re.search(
        rf"export interface {interface}(?: extends \w+)? \{{(.*?)^\}}",
        source.read_text(),
        re.S | re.M,
    )
    assert block is not None, f"{interface} is not declared where expected"
    return set(re.findall(r"^\s{2}(\w+)[?]?:", block.group(1), re.M))


def test_a_model_write_and_slot_carry_what_the_browser_declares(tmp_path):
    """Wire parity with `modelWrite.ts` and `session.ts`: a key on one side only is a
    randomization that silently does nothing."""
    core = TEMPLATE / "src" / "core"
    elements = _declared(core / "onnx" / "slotReader" / "model.ts", "ModelElements")
    target_keys = _declared(core / "event" / "modelWrite.ts", "ModelWriteTarget")
    slot_keys = _declared(core / "onnx" / "session.ts", "OnnxInputSlot") | elements

    entry = _serialize(
        _dr().joint_damping,
        {"asset_cfg": _robot(joint_names=("wrist",)), "ranges": (0.5, 2.0)},
        tmp_path,
    )
    assert set(entry["write_targets"][0]) == target_keys | elements
    for slot in entry["input_slots"]:
        assert set(slot) <= slot_keys


class TestManualEvents:
    """`mode="manual"`: no schedule at all, fired from the control panel's button."""

    def test_it_serializes_with_its_label_and_no_schedule(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("mjlab")
        from mjswan.build.mdp import serialize_event
        from mjswan.managers.event_manager import EventTermCfg

        def throw(env, env_ids, ball_name="ball"):
            env.scene[ball_name].write_root_link_pose_to_sim(
                torch.zeros(1, 7), env_ids=env_ids
            )

        entry = serialize_event(
            "throw_overhead",
            EventTermCfg(
                func=throw,
                mode="manual",
                params={"ball_name": "ball"},
                label="Throw overhead",
            ),
            TestWriteTargetEntity._env(),
            tmp_path,
        )
        assert entry["mode"] == "manual"
        assert entry["label"] == "Throw overhead"
        assert "interval_range_s" not in entry
        assert "is_global_time" not in entry
        # Still a traced graph writing the entity it was made on.
        assert entry["write_targets"][0]["entity"] == "ball"

    def test_disabled_when_travels_to_the_browser(self, tmp_path):
        torch = pytest.importorskip("torch")
        pytest.importorskip("mjlab")
        from mjswan.build.mdp import serialize_event
        from mjswan.managers.event_manager import EventTermCfg

        def throw(env, env_ids, ball_name="ball"):
            env.scene[ball_name].write_root_link_pose_to_sim(
                torch.zeros(1, 7), env_ids=env_ids
            )

        entry = serialize_event(
            "throw_overhead",
            EventTermCfg(
                func=throw,
                mode="manual",
                params={"ball_name": "ball"},
                disabled_when="throw_ball",
            ),
            TestWriteTargetEntity._env(),
            tmp_path,
        )
        assert entry["disabled_when"] == "throw_ball"

    def test_a_gate_that_names_no_interval_term_is_refused(self):
        """A dead gate greys the button out forever, or never — and says nothing."""
        from mjswan.build.mdp.event import _check_disabled_when
        from mjswan.managers.event_manager import EventTermCfg

        def throw(env, env_ids):
            raise AssertionError("never traced")

        auto = EventTermCfg(func=throw, mode="interval", interval_range_s=(1.0, 4.0))
        gated = EventTermCfg(func=throw, mode="manual", disabled_when="throw_ball")
        _check_disabled_when({"throw_overhead": gated, "throw_ball": auto})

        with pytest.raises(ValueError, match="throw_ball"):
            _check_disabled_when({"throw_overhead": gated})
        with pytest.raises(ValueError, match="interval"):
            # The gate must be the schedule, not another button.
            _check_disabled_when(
                {
                    "throw_overhead": gated,
                    "throw_ball": EventTermCfg(func=throw, mode="manual"),
                }
            )
        with pytest.raises(ValueError, match="manual"):
            _check_disabled_when(
                {
                    "throw_ball": EventTermCfg(
                        func=throw,
                        mode="interval",
                        interval_range_s=(1.0, 4.0),
                        disabled_when="throw_ball",
                    )
                }
            )

    def test_a_manual_term_carrying_an_interval_is_refused(self, tmp_path):
        from mjswan.build.mdp import serialize_event
        from mjswan.managers.event_manager import EventTermCfg

        def throw(env, env_ids):
            raise AssertionError("never traced")

        with pytest.raises(ValueError, match="manual") as excinfo:
            serialize_event(
                "throw",
                EventTermCfg(func=throw, mode="manual", interval_range_s=(1.0, 4.0)),
                None,
                tmp_path,
            )
        assert "interval" in str(excinfo.value)


class TestAnUntraceableEventFailsTheBuild:
    """What happens when a term traces to nothing.

    The runtime skips a native event silently, so only the cases below, where there is
    provably nothing to write, stay native; anything else fails the build.
    """

    @staticmethod
    def _serialize(term_cfg, tmp_path, env=None):
        pytest.importorskip("mjlab")
        from mjswan.build.mdp import serialize_event

        return serialize_event("ev", term_cfg, env or _TinyEnv(), tmp_path)

    def test_an_unexplained_no_write_raises_and_names_both_escape_hatches(
        self, tmp_path
    ):
        def push_robot(env, env_ids, velocity_range=None):
            return None

        with pytest.raises(ValueError, match="register_event") as excinfo:
            self._serialize(_term(push_robot, {}, mode="interval"), tmp_path)
        assert "ts_src" in str(excinfo.value)

    def test_randomize_terrain_stays_native_with_its_reason(self, tmp_path):
        def randomize_terrain(env, env_ids):
            return None

        entry = self._serialize(_term(randomize_terrain, {}, mode="reset"), tmp_path)
        assert entry["native"] is True
        assert "one baked terrain" in entry["reason"]

    def test_encoder_bias_stays_native_with_its_reason(self, tmp_path):
        def encoder_bias(env, env_ids, bias_range=(0.0, 0.0), asset_cfg=None):
            return None

        entry = self._serialize(
            _term(encoder_bias, {"bias_range": (-0.01, 0.01)}, mode="reset"),
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

        env = _TinyEnv()
        env.scene["robot"].is_fixed_base = True
        entry = self._serialize(
            _term(
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
                _term(
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

    Reading the name off an `asset_cfg` param alone left it `null` for a term taking a
    plain `ball_name`, and the write then went to the model's *first* free joint: the
    robot got launched, the ball never moved. mjlab writes per entity, so does the key.
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
        """One write per entity, as in mjlab: the two roots are different addresses."""
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
        """mjlab's own split of the 13-wide state, so such a term traces too."""
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
        """Stand-ins: writing through the live ones would move the sim under later terms."""
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
            env.scene["robot"].write_root_com_velocity_to_sim(torch.zeros(1, 6))

        with pytest.raises(ValueError, match="does not capture"):
            self._trace(spin, {})


def test_reset_scene_to_default_needs_no_graph():
    """The runtime's reset already restores every default, so nothing is left to write."""
    pytest.importorskip("torch")
    from mjswan.compile.event import EVENTS_WITH_NOTHING_TO_WRITE

    assert "reset_scene_to_default" in EVENTS_WITH_NOTHING_TO_WRITE


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
        from mjswan.envs.mdp.events import reset_root_state_on_flat_patch
        from mjswan.mjlab.event import apply_terrain_spawn

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
        from mjswan.mjlab.event import apply_terrain_spawn

        event = self._uniform_event()
        scene = self._scene(None, {"reset_base": event})
        apply_terrain_spawn(scene)
        assert scene.events["reset_base"] is event


def test_serialize_events_reports_each_term_it_traces(monkeypatch):
    """The build's progress line names the event it is on; keep the hook wired."""
    from mjswan.build.mdp import event

    monkeypatch.setattr(
        event,
        "serialize_event",
        lambda name, cfg, env, out, **kw: {"name": name},
    )
    seen: list[str] = []
    event.serialize_events(
        {"reset_slider": object(), "reset_hinge": object()},
        env=None,
        out_dir=None,
        on_term=seen.append,
    )
    assert seen == ["reset_slider", "reset_hinge"]
