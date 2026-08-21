"""Tests for mjswan.adapters.mjlab_adapter — mjlab type conversion.

Layer: L1 (pure Python, no MuJoCo/ONNX/mjlab required).

These tests simulate mjlab types by creating lightweight mock classes
with the same attributes that the adapter inspects, placed in a fake
``mjlab.*`` module path.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mjswan.adapters.mjlab_adapter import (
    adapt_actions,
    adapt_commands,
    adapt_observations,
    adapt_terminations,
    resolve_action_scales,
    resolve_pd_gains,
    resolve_runner_defaults,
)
from mjswan.envs.mdp.observations import ObservationBinding
from mjswan.envs.mdp.terminations import TerminationBinding
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg

# ---------------------------------------------------------------------------
# Fake mjlab types — classes whose __module__ starts with "mjlab"
# ---------------------------------------------------------------------------


def _make_mjlab_class(class_name: str, **defaults: Any) -> type:
    """Create a simple dataclass-like class that appears to come from mjlab."""

    class Cls:
        def __init__(self, **kwargs: Any):
            for k, v in {**defaults, **kwargs}.items():
                setattr(self, k, v)

    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    Cls.__module__ = "mjlab.fake"
    return Cls


# Fake mjlab observation functions (callables with __name__ and __module__)
def _make_mjlab_obs_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.observations"
    return fn


def _make_mjlab_term_func(name: str):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = "mjlab.envs.mdp.terminations"
    return fn


# Fake mjlab config classes
FakeMjlabObsTermCfg = _make_mjlab_class(
    "ObservationTermCfg",
    func=None,
    params={},
    scale=None,
    clip=None,
    history_length=0,
    noise=None,
)

FakeMjlabObsGroupCfg = _make_mjlab_class(
    "ObservationGroupCfg",
    terms={},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=None,
)

FakeMjlabTermTermCfg = _make_mjlab_class(
    "TerminationTermCfg",
    func=None,
    params={},
    time_out=False,
)

FakeMjlabJointPositionActionCfg = _make_mjlab_class(
    "JointPositionActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    use_default_offset=True,
    stiffness=None,
    damping=None,
)

FakeMjlabJointEffortActionCfg = _make_mjlab_class(
    "JointEffortActionCfg",
    entity_name="robot",
    clip=None,
    actuator_names=(".*",),
    scale=1.0,
    offset=0.0,
    stiffness=None,
    damping=None,
)

# `MyoMuscleActivationActionCfg` is standalone (no dataclass, no BaseActionCfg),
# so mirror only the fields the adapter inspects.
FakeMyoMuscleActivationActionCfg = _make_mjlab_class(
    "MyoMuscleActivationActionCfg",
    entity_name="robot",
    actuator_names=("m1", "m2"),
)

FakeMjlabSceneEntityCfg = _make_mjlab_class(
    "SceneEntityCfg",
    **{
        "name": "robot",
        "joint_names": None,
        "site_names": None,
    },
)

FakeMjlabMotionCommandCfg = _make_mjlab_class(
    "MotionCommandCfg",
    anchor_body_name="torso_link",
    body_names=("pelvis", "torso_link"),
    entity_name="robot",
)


# ===================================================================
# Tests: Observations
# ===================================================================


class TestAdaptObservations:
    def test_none_passthrough(self):
        assert adapt_observations(None) is None

    def test_mjswan_types_unchanged(self):
        obs_func = ObservationBinding("BaseLinearVelocity")
        group = ObservationGroupCfg(terms={"vel": ObservationTermCfg(func=obs_func)})
        result = adapt_observations({"policy": group})
        assert result is not None
        assert result["policy"] is group

    def test_mjlab_obs_term_converted(self):
        # `base_lin_vel` is a DSL term (ADR 0003) — adapter resolves to a callable.
        mjlab_func = _make_mjlab_obs_func("base_lin_vel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, params={"world_frame": True})
        mjlab_group = FakeMjlabObsGroupCfg(
            terms={"base_vel": mjlab_term},
            enable_corruption=True,
        )

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        group = result["policy"]
        assert isinstance(group, ObservationGroupCfg)
        assert "base_vel" in group.terms
        term = group.terms["base_vel"]
        assert isinstance(term, ObservationTermCfg)
        assert callable(term.func)
        assert term.params == {"world_frame": True}

    def test_mjlab_obs_scale_and_history(self):
        """mjlab's dense stack arrives as its offsets: oldest frame first.

        mjlab flattens the term's buffer chronologically, so a bare `history_length=3`
        — which mjswan reads newest-first — would hand the policy its history reversed:
        the same width, the same numbers, time running backwards.
        """
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func, scale=0.5, history_length=3)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.scale == 0.5
        assert term.history_steps == (2, 1, 0)

    def test_mjlab_group_history_reaches_every_term(self):
        """A group-level count overrides the terms', as mjlab's manager does — and the
        adapted group keeps no count of its own, so nothing can apply it twice."""
        mjlab_group = FakeMjlabObsGroupCfg(
            terms={
                "jp": FakeMjlabObsTermCfg(func=_make_mjlab_obs_func("joint_pos_rel")),
                "jv": FakeMjlabObsTermCfg(
                    func=_make_mjlab_obs_func("joint_vel_rel"), history_length=2
                ),
            },
            history_length=4,
        )

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        group = result["policy"]
        assert group.history_length is None
        assert group.terms["jp"].history_steps == (3, 2, 1, 0)
        assert group.terms["jv"].history_steps == (3, 2, 1, 0)

    def test_mjlab_single_frame_history_is_no_history(self):
        """`history_length=1` is a one-frame buffer: the term's own width, unstacked."""
        mjlab_group = FakeMjlabObsGroupCfg(
            terms={
                "jp": FakeMjlabObsTermCfg(func=_make_mjlab_obs_func("joint_pos_rel"))
            },
            history_length=1,
        )

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.history_steps is None
        assert term.history_length == 0

    def test_mjlab_cfg_subclass_is_still_adapted(self):
        """A task subclassing an mjlab config to add a field of its own is still mjlab.

        The subclass reports its own module, so a check on the class alone passes it
        through unadapted — and the mjlab *terms* inside it then reach the serializer,
        which fails on the first mjswan-only field.
        """

        class TaskObsGroupCfg(FakeMjlabObsGroupCfg):  # defined here, not in mjlab
            pass

        group = TaskObsGroupCfg(
            terms={
                "jp": FakeMjlabObsTermCfg(func=_make_mjlab_obs_func("joint_pos_rel"))
            },
            history_length=4,
        )

        result = adapt_observations({"policy": group})
        assert result is not None
        assert isinstance(result["policy"], ObservationGroupCfg)
        assert isinstance(result["policy"].terms["jp"], ObservationTermCfg)
        assert result["policy"].terms["jp"].history_steps == (3, 2, 1, 0)

    def test_mjlab_asset_cfg_kept_intact_for_tracing(self):
        # A plain (non-Binding) func is traced to ONNX at build time (ADR 0005) via `func(env,
        # **params)` — params must reach the tracer unchanged, including the real `asset_cfg`
        # object mjlab's own function expects, not a flattened entity_name/joint_names stand-in.
        mjlab_func = _make_mjlab_obs_func("joint_pos_rel")
        asset_cfg = FakeMjlabSceneEntityCfg(
            name="robot", joint_names=("joint1", "joint2")
        )
        mjlab_term = FakeMjlabObsTermCfg(
            func=mjlab_func, params={"asset_cfg": asset_cfg}
        )
        mjlab_group = FakeMjlabObsGroupCfg(terms={"jp": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        term = result["policy"].terms["jp"]
        assert term.params["asset_cfg"] is asset_cfg

    def test_any_mjlab_obs_func_passes_through_for_tracing(self):
        # ADR 0005: there is no mjswan-side mirror to resolve by name — any mjlab function
        # (however unfamiliar) is passed straight through and traced to ONNX at build time.
        mjlab_func = _make_mjlab_obs_func("nonexistent_function")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"x": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        assert result["policy"].terms["x"].func is mjlab_func

    def test_multiple_groups(self):
        # The adapter resolves these to callables.
        f1 = _make_mjlab_obs_func("base_ang_vel")
        f2 = _make_mjlab_obs_func("projected_gravity")
        g1 = FakeMjlabObsGroupCfg(terms={"ang": FakeMjlabObsTermCfg(func=f1)})
        g2 = FakeMjlabObsGroupCfg(terms={"grav": FakeMjlabObsTermCfg(func=f2)})

        # Two keys a multi-input policy could actually consume; a group named for a
        # training-only mjlab network is a different case, covered below.
        result = adapt_observations({"policy": g1, "adapt_hx": g2})
        assert result is not None
        assert callable(result["policy"].terms["ang"].func)
        assert callable(result["adapt_hx"].terms["grav"].func)


class TestObservationGroupKey:
    """The dict key is the ONNX input name, so the adapter — not the caller — owns it.

    `OnnxModule` defaults `in_keys` to `["policy"]` and warns-and-returns on an input it
    cannot find, so a group under mjlab's own name (`"actor"`) yields a policy that never
    acts, with no build-time error. Hence: hand in the group, not a key for it.
    """

    def test_single_mjlab_group_lands_under_policy_key(self):
        func = _make_mjlab_obs_func("base_ang_vel")
        group = FakeMjlabObsGroupCfg(terms={"ang": FakeMjlabObsTermCfg(func=func)})

        result = adapt_observations(group)

        assert result is not None
        assert list(result) == ["policy"]
        assert isinstance(result["policy"], ObservationGroupCfg)
        assert callable(result["policy"].terms["ang"].func)

    def test_single_mjswan_group_lands_under_policy_key(self):
        group = ObservationGroupCfg(
            terms={"ang": ObservationTermCfg(func=ObservationBinding(ts_name="X"))}
        )

        result = adapt_observations(group)

        assert result is not None
        # The same object: an mjswan group needs no conversion, only a key.
        assert result == {"policy": group}
        assert result["policy"] is group

    def test_dict_form_still_passes_keys_through(self):
        group = ObservationGroupCfg(terms={})
        assert adapt_observations({"custom_input": group}) == {"custom_input": group}

    def test_training_only_group_is_dropped_with_warning(self):
        actor = ObservationGroupCfg(terms={})
        critic = ObservationGroupCfg(terms={})

        with pytest.warns(RuntimeWarning, match="critic"):
            result = adapt_observations({"policy": actor, "critic": critic})

        # mjlab exports only the actor, so a critic group has no input to feed — leaving
        # it in would trace it, bundle it, and evaluate it every control step for nothing.
        assert result == {"policy": actor}

    def test_a_group_of_only_training_terms_leaves_nothing(self):
        with pytest.warns(RuntimeWarning, match="critic"):
            assert adapt_observations({"critic": ObservationGroupCfg(terms={})}) == {}

    def test_tracking_observation_functions_are_mapped(self):
        motion_anchor = _make_mjlab_obs_func("motion_anchor_pos_b")
        body_pos = _make_mjlab_obs_func("robot_body_pos_b")
        result = adapt_observations(
            {
                "policy": FakeMjlabObsGroupCfg(
                    terms={
                        "anchor": FakeMjlabObsTermCfg(func=motion_anchor),
                        "body": FakeMjlabObsTermCfg(func=body_pos),
                    }
                )
            }
        )
        # The adapter resolves these to callables, not ObservationBinding sentinels.
        assert result is not None
        assert callable(result["policy"].terms["anchor"].func)
        assert callable(result["policy"].terms["body"].func)


class TestMjlabGroupDictSelection:
    """An mjlab `env_cfg.observations` is keyed by *network*; mjswan's by *ONNX input*.

    Two namespaces that look alike, so the adapter remaps the one it can recognise and
    keeps its hands off the one it cannot. Getting that boundary wrong in either
    direction is silent at build time: an unrecognised key means no ONNX input to feed,
    and a wrongly-remapped one means the wrong vector on the right input.
    """

    def test_whole_mjlab_dict_reduces_to_the_actor_group(self):
        actor = ObservationGroupCfg(terms={})
        critic = ObservationGroupCfg(terms={})

        result = adapt_observations({"actor": actor, "critic": critic})

        assert result == {"policy": actor}

    def test_actor_only_dict_is_renamed(self):
        actor = ObservationGroupCfg(terms={})
        assert adapt_observations({"actor": actor}) == {"policy": actor}

    def test_a_dict_without_an_actor_is_left_alone(self):
        # `examples/demo` relies on this: `balance.json` declares `in_keys: ["observation"]`,
        # `decap.json` `["obs_history"]`, ANYmal's `["obs"]`. Remapping any of them to
        # "policy" would leave the runtime looking for an input nothing supplies.
        for key in ("observation", "obs", "obs_history"):
            group = ObservationGroupCfg(terms={})
            assert adapt_observations({key: group}) == {key: group}

    def test_a_multi_input_dict_is_left_alone(self):
        # Facet: `in_keys: ["command", "policy", ...]` — two real inputs, both ours.
        policy = ObservationGroupCfg(terms={})
        command = ObservationGroupCfg(terms={})
        result = adapt_observations({"policy": policy, "command": command})
        assert result == {"policy": policy, "command": command}

    def test_runner_obs_groups_win_over_the_actor_name(self):
        # A task free to name its groups anything; `obs_groups["actor"]` is the only thing
        # that actually knows which one the exported network reads.
        proprio = ObservationGroupCfg(terms={})
        privileged = ObservationGroupCfg(terms={})

        result = adapt_observations(
            {"proprio": proprio, "privileged": privileged},
            policy_groups=("proprio",),
        )

        assert result == {"policy": proprio}

    def test_runner_obs_groups_beat_a_literal_actor_key(self):
        actor = ObservationGroupCfg(terms={})
        other = ObservationGroupCfg(terms={})
        result = adapt_observations(
            {"actor": actor, "other": other}, policy_groups=("other",)
        )
        assert result == {"policy": other}

    def test_concatenated_groups_are_refused_not_truncated(self):
        # rsl-rl lets one network read several groups concatenated. mjswan feeds one vector
        # per input, so silently taking the first would mean a short observation and a
        # policy fed garbage — the one case that has to be loud.
        with pytest.raises(ValueError, match="cannot concatenate"):
            adapt_observations(
                {
                    "a": ObservationGroupCfg(terms={}),
                    "b": ObservationGroupCfg(terms={}),
                },
                policy_groups=("a", "b"),
            )

    def test_a_dict_sharing_no_key_with_the_task_is_left_alone(self):
        # A task id is not evidence that *this* dict is the task's. On an mjlab scene a
        # policy may still carry a config declaring `in_keys: ["observation"]`, and
        # remapping it because the task calls its group "proprio" would break it.
        group = ObservationGroupCfg(terms={})
        assert adapt_observations(
            {"observation": group}, policy_groups=("proprio",)
        ) == {"observation": group}

    def test_a_literal_actor_key_still_wins_when_the_task_names_another(self):
        actor = ObservationGroupCfg(terms={})
        assert adapt_observations({"actor": actor}, policy_groups=("proprio",)) == {
            "policy": actor
        }

    def test_an_empty_dict_selects_nothing_rather_than_failing(self):
        # `observations={}` is how a policy says it has none; a task id must not turn
        # that into an error.
        assert adapt_observations({}, policy_groups=("actor",)) == {}

    def test_concatenated_groups_only_raise_for_the_tasks_own_dict(self):
        # No overlap with the task's group names, so this is somebody else's dict and the
        # concatenation the task does is none of its business.
        group = ObservationGroupCfg(terms={})
        assert adapt_observations({"policy": group}, policy_groups=("a", "b")) == {
            "policy": group
        }

    def test_a_single_group_ignores_policy_groups(self):
        # Already unambiguous: there is one group, and it is the policy's.
        group = ObservationGroupCfg(terms={})
        assert adapt_observations(group, policy_groups=("anything",)) == {
            "policy": group
        }

    def test_mjlab_groups_in_the_dict_are_still_converted(self):
        func = _make_mjlab_obs_func("base_ang_vel")
        actor = FakeMjlabObsGroupCfg(terms={"ang": FakeMjlabObsTermCfg(func=func)})

        result = adapt_observations({"actor": actor, "critic": actor})

        assert result is not None
        assert list(result) == ["policy"]
        assert isinstance(result["policy"], ObservationGroupCfg)


# ===================================================================
# Tests: Terminations
# ===================================================================


class TestAdaptTerminations:
    def test_none_passthrough(self):
        assert adapt_terminations(None) is None

    def test_mjswan_types_unchanged(self):
        term_func = TerminationBinding("TimeOut")
        cfg = TerminationTermCfg(func=term_func, time_out=True)
        result = adapt_terminations({"time_out": cfg})
        assert result is not None
        assert result["time_out"] is cfg

    def test_mjlab_term_converted(self):
        # The adapter resolves this to a callable, not a TerminationBinding sentinel.
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert isinstance(term, TerminationTermCfg)
        assert callable(term.func)
        assert term.params == {"limit_angle": 1.0}
        assert term.time_out is False

    def test_mjlab_time_out_flag(self):
        # `time_out` is a DSL term (ADR 0003) — resolved to a callable.
        mjlab_func = _make_mjlab_term_func("time_out")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func, time_out=True)

        result = adapt_terminations({"timeout": mjlab_cfg})
        assert result is not None
        assert result["timeout"].time_out is True
        assert callable(result["timeout"].func)

    def test_mjlab_term_keeps_asset_cfg_intact_for_tracing(self):
        # As with observations, a plain func's params reach the tracer unflattened.
        mjlab_func = _make_mjlab_term_func("bad_orientation")
        asset_cfg = FakeMjlabSceneEntityCfg(name="robot", body_names=("torso_link",))
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"limit_angle": 1.0, "asset_cfg": asset_cfg},
            time_out=False,
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        term = result["fallen"]
        assert term.params == {"limit_angle": 1.0, "asset_cfg": asset_cfg}

    def test_any_mjlab_term_func_passes_through_for_tracing(self):
        # No mjswan-side mirror: any mjlab function passes straight through to the tracer.
        mjlab_func = _make_mjlab_term_func("nonexistent_term")
        mjlab_cfg = FakeMjlabTermTermCfg(func=mjlab_func)

        result = adapt_terminations({"x": mjlab_cfg})
        assert result is not None
        assert result["x"].func is mjlab_func


# ===================================================================
# Tests: Actions
# ===================================================================


class TestAdaptActions:
    def test_none_passthrough(self):
        assert adapt_actions(None) is None


class TestAdaptCommands:
    def test_none_passthrough(self):
        assert adapt_commands(None) is None

    def test_motion_command_cfg_converts_to_tracking_command(self):
        result = adapt_commands({"motion": FakeMjlabMotionCommandCfg()})
        assert result is not None
        command = result["motion"]
        assert command.term_name == "TrackingCommand"
        assert command.params["anchor_body_name"] == "torso_link"
        assert command.params["body_names"] == ["pelvis", "torso_link"]

    def test_a_traced_command_gets_mjlabs_debug_drawing_without_being_asked(self):
        """The binding declares no `viz`; the cfg class is mjlab's, so one is derived.

        Otherwise a `debug_vis=True` task the author forgot is silently blank.
        """
        from mjswan.command import CommandBinding, _custom_registry, register_command

        cfg_cls = _make_mjlab_class(
            "LiftingCommandCfg",
            entity_name="cube",
            debug_vis=True,
            viz=SimpleNamespace(target_color=(1.0, 0.5, 0.0, 0.3)),
        )
        register_command(
            "LiftingCommandCfg",
            CommandBinding(state_fields=["target_pos"], command_field="target_pos"),
        )
        try:
            result = adapt_commands({"lift_height": cfg_cls()})
        finally:
            _custom_registry.pop("LiftingCommandCfg", None)

        assert result is not None
        viz = result["lift_height"].pending_trace.viz
        assert viz == [
            {
                "shape": "sphere",
                "radius": 0.03,
                "color": [1.0, 0.5, 0.0, 0.3],
                "origin": {"state": "target_pos"},
            }
        ]

    def test_a_registered_cfg_adapts_from_outside_the_mjlab_package(self):
        """The registry decides, not the defining module."""
        from mjswan.command import CommandBinding, _custom_registry, register_command

        class SkateCommandCfg:
            resampling_time_range = (20.0, 20.0)
            debug_vis = False

        assert not SkateCommandCfg.__module__.startswith("mjlab")
        register_command(
            "SkateCommandCfg",
            CommandBinding(state_fields=["command_b"], command_field="command_b"),
        )
        try:
            result = adapt_commands({"skate": SkateCommandCfg()})
        finally:
            _custom_registry.pop("SkateCommandCfg", None)

        assert result is not None
        pending = result["skate"].pending_trace
        assert pending is not None
        assert pending.state_fields == ["command_b"]

    def test_a_task_owned_action_subclass_adapts_by_name(self):
        """A tracking task's own `ActionTermCfg` subclass is not in the `mjlab` package."""
        from mjswan.envs.mdp.actions import ReferenceJointPositionActionCfg

        cfg_cls = _make_mjlab_class(
            "ReferenceJointPositionActionCfg",
            entity_name="robot",
            actuator_names=(".*",),
            scale={".*_knee_joint": 0.5},
            offset=0.0,
            clip=None,
            command_name="motion",
        )
        cfg_cls.__module__ = "some_task.env.mdp.actions"
        cfg = cfg_cls()

        result = adapt_actions({"joint_pos": cfg})
        assert result is not None
        adapted = result["joint_pos"]
        assert isinstance(adapted, ReferenceJointPositionActionCfg)
        assert adapted is not cfg
        assert adapted.command_name == "motion"

    def test_an_unknown_foreign_action_is_copied_not_shared(self):
        """`resolve_action_scales` rewrites what it is given; the caller keeps its own."""

        class ForeignActionCfg:
            def __init__(self) -> None:
                self.scale = {".*": 0.5}

        cfg = ForeignActionCfg()
        result = adapt_actions({"joint_pos": cfg})
        assert result is not None
        assert result["joint_pos"] is not cfg

        resolve_action_scales(result, ["robot/left_knee_joint"])  # type: ignore[arg-type]
        assert cfg.scale == {".*": 0.5}

    def test_mjswan_types_unchanged(self):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        cfg = JointPositionActionCfg(scale=0.5)
        result = adapt_actions({"joint_pos": cfg})
        assert result is not None
        assert result["joint_pos"] is cfg

    def test_mjlab_joint_position_converted(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale=0.25,
            offset=0.1,
            use_default_offset=True,
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        action = result["jp"]
        assert isinstance(action, JointPositionActionCfg)
        assert action.scale == 0.25
        assert action.offset == 0.1
        assert action.use_default_offset is True
        assert action.stiffness == 40.0
        assert action.damping == 2.5

    def test_mjlab_joint_effort_converted(self):
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=2.0)

        result = adapt_actions({"torque": mjlab_cfg})
        assert result is not None
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        action = result["torque"]
        assert isinstance(action, JointEffortActionCfg)
        assert action.scale == 2.0

    def test_mjlab_unknown_action_warns(self):
        FakeUnknown = _make_mjlab_class(
            "SomeWeirdActionCfg",
            entity_name="robot",
            clip=None,
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
        )
        mjlab_cfg = FakeUnknown()

        with pytest.warns(RuntimeWarning, match="no mjswan equivalent"):
            result = adapt_actions({"weird": mjlab_cfg})

        assert result is not None
        assert "weird" not in result

    def test_mixed_mjswan_and_mjlab(self):
        """Both mjswan-native and mjlab types in the same dict."""
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        mjswan_cfg = JointPositionActionCfg(scale=0.5)
        mjlab_cfg = FakeMjlabJointEffortActionCfg(scale=3.0)

        result = adapt_actions({"jp": mjswan_cfg, "torque": mjlab_cfg})
        assert result is not None
        assert result["jp"] is mjswan_cfg
        from mjswan.envs.mdp.actions import JointEffortActionCfg

        assert isinstance(result["torque"], JointEffortActionCfg)


# ===================================================================
# Tests: PD gains from the entity's actuator configs
# ===================================================================


def _fake_actuator_cls(class_name: str) -> type:
    """An actuator cfg the adapter recognizes by the class name in its MRO."""

    class Cls:
        def __init__(self, target_names_expr, stiffness, damping):
            self.target_names_expr = target_names_expr
            self.stiffness = stiffness
            self.damping = damping

    Cls.__name__ = class_name
    return Cls


#: The browser owes the ideal-PD family's gains; the builtin position one contributes none.
_FakeIdealPdActuatorCfg = _fake_actuator_cls("IdealPdActuatorCfg")
_FakeBuiltinPositionActuatorCfg = _fake_actuator_cls("BuiltinPositionActuatorCfg")


def _env_cfg_with(*actuators: Any) -> Any:
    entity = SimpleNamespace(articulation=SimpleNamespace(actuators=actuators))
    return SimpleNamespace(scene=SimpleNamespace(entities={"robot": entity}))


class TestResolvePdGains:
    """A motor actuator's PD runs in the browser, off gains mjlab keeps on the entity."""

    JOINTS = [
        "robot/left_knee_joint",
        "robot/right_knee_joint",
        "robot/waist_yaw_joint",
    ]

    def test_gains_expand_to_the_joints_the_patterns_match(self):
        from mjswan.envs.mdp.actions import ReferenceJointPositionActionCfg

        actions = {"joint_pos": ReferenceJointPositionActionCfg()}
        resolve_pd_gains(
            actions,
            self.JOINTS,
            _env_cfg_with(
                _FakeIdealPdActuatorCfg((".*_knee_joint",), 99.0, 6.3),
                _FakeIdealPdActuatorCfg(("waist_yaw_joint",), 40.0, 2.5),
            ),
        )

        term = actions["joint_pos"]
        assert term.stiffness == {
            "robot/left_knee_joint": 99.0,
            "robot/right_knee_joint": 99.0,
            "robot/waist_yaw_joint": 40.0,
        }
        assert term.damping == {
            "robot/left_knee_joint": 6.3,
            "robot/right_knee_joint": 6.3,
            "robot/waist_yaw_joint": 2.5,
        }

    def test_a_builtin_position_actuator_contributes_nothing(self):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        actions = {"joint_pos": JointPositionActionCfg()}
        resolve_pd_gains(
            actions,
            self.JOINTS,
            _env_cfg_with(_FakeBuiltinPositionActuatorCfg((".*",), 99.0, 6.3)),
        )

        assert actions["joint_pos"].stiffness is None
        assert actions["joint_pos"].damping is None

    def test_an_explicit_gain_wins(self):
        from mjswan.envs.mdp.actions import JointPositionActionCfg

        actions = {"joint_pos": JointPositionActionCfg(stiffness=1.0)}
        resolve_pd_gains(
            actions,
            self.JOINTS,
            _env_cfg_with(_FakeIdealPdActuatorCfg((".*",), 99.0, 6.3)),
        )

        assert actions["joint_pos"].stiffness == 1.0
        assert actions["joint_pos"].damping is None


# ===================================================================
# Tests: End-to-end serialization after adaptation
# ===================================================================


class TestAdaptedSerialization:
    """Ensure adapted plain-callable terms defer to ONNX tracing (ADR 0005)."""

    def test_adapted_obs_to_dict_requires_tracing(self):
        # A plain-callable func (mjlab's own, resolved by the adapter with no mirror lookup)
        # cannot be serialized via to_dict()/to_list() directly — it must be traced to ONNX
        # against a live env at build time (mjswan._onnx_build.serialize_observation_group).
        mjlab_func = _make_mjlab_obs_func("last_action")
        mjlab_term = FakeMjlabObsTermCfg(func=mjlab_func)
        mjlab_group = FakeMjlabObsGroupCfg(terms={"la": mjlab_term})

        result = adapt_observations({"policy": mjlab_group})
        assert result is not None
        with pytest.raises(TypeError, match="serialize_observation_group"):
            result["policy"].to_list()

    def test_adapted_term_to_dict_requires_tracing(self):
        mjlab_func = _make_mjlab_term_func("root_height_below_minimum")
        mjlab_cfg = FakeMjlabTermTermCfg(
            func=mjlab_func,
            params={"minimum_height": 0.2},
        )

        result = adapt_terminations({"fallen": mjlab_cfg})
        assert result is not None
        with pytest.raises(TypeError, match="serialize_termination"):
            result["fallen"].to_dict()

    def test_adapted_action_serializes(self):
        mjlab_cfg = FakeMjlabJointPositionActionCfg(
            scale={"hip": 0.5, "knee": 0.3},
            stiffness=40.0,
            damping=2.5,
        )

        result = adapt_actions({"jp": mjlab_cfg})
        assert result is not None
        d = result["jp"].to_dict()
        assert d["type"] == "joint_position"
        assert d["scale"] == {"hip": 0.5, "knee": 0.3}
        assert d["stiffness"] == 40.0
        assert d["damping"] == 2.5


# ---------------------------------------------------------------------------
# Muscle action adaptation: MyoMuscleActivationActionCfg → MuscleActivationActionCfg
# ---------------------------------------------------------------------------


class TestMuscleActionAdaptation:
    """Adapt myosuite4's ``MyoMuscleActivationActionCfg`` to mjswan."""

    def test_adapted_to_muscle_activation_action_cfg(self):
        from mjswan.envs.mdp.actions import MuscleActivationActionCfg

        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        assert isinstance(result["muscles"], MuscleActivationActionCfg)

    def test_adapted_action_serializes_to_muscle_activation_type(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert d["type"] == "muscle_activation"

    def test_actuator_names_prefixed_with_entity_name(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1", "m2"),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert d["actuator_names"] == ["robot/m1", "robot/m2"]

    def test_normalize_defaults_to_true_when_source_lacks_field(self):
        # MyoMuscleActivationActionCfg has no `normalize` field and always applies the sigmoid
        # mapping in upstream; the adapted cfg must keep the mjswan default (normalize=True),
        # which serializes as the key being omitted (default-suppression).
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1",),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert "normalize" not in d
        assert result["muscles"].normalize is True

    def test_default_scale_offset_preserved_when_source_omits_them(self):
        mjlab_cfg = FakeMyoMuscleActivationActionCfg(
            entity_name="robot",
            actuator_names=("m1",),
        )

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        d = result["muscles"].to_dict()
        assert "scale" not in d
        assert "offset" not in d
        assert result["muscles"].scale == 1.0
        assert result["muscles"].offset == 0.0

    def test_class_name_alias_dispatch(self):
        # The adapter looks up the source class name in _ACTION_CLASS_ALIASES. If the source
        # class is renamed upstream, this test would catch the break by failing dispatch.
        mjlab_cfg = FakeMyoMuscleActivationActionCfg()
        assert type(mjlab_cfg).__name__ == "MyoMuscleActivationActionCfg"

        result = adapt_actions({"muscles": mjlab_cfg})
        assert result is not None
        assert "muscles" in result


# ===================================================================
# Tests: mjlab runner config (rl_cfg) — the two fields playback needs
# ===================================================================


class TestResolveRunnerDefaults:
    """`obs_groups` and `clip_actions` live on the *runner* config, not the env config.

    Everything else on it is training-only or already inside the exported ONNX — rsl-rl
    bakes the observation normalizer into the graph ahead of the MLP, so playback does
    not have to reproduce it.
    """

    @staticmethod
    def _install(monkeypatch, rl_cfg):
        module = ModuleType("mjlab.tasks.registry")
        setattr(module, "load_rl_cfg", lambda task_id: rl_cfg)
        monkeypatch.setitem(sys.modules, "mjlab", ModuleType("mjlab"))
        monkeypatch.setitem(sys.modules, "mjlab.tasks", ModuleType("mjlab.tasks"))
        monkeypatch.setitem(sys.modules, "mjlab.tasks.registry", module)

    def test_reads_obs_groups_and_clip_actions(self, monkeypatch):
        rl_cfg = SimpleNamespace(
            obs_groups={"actor": ("actor",), "critic": ("critic",)}, clip_actions=100.0
        )
        self._install(monkeypatch, rl_cfg)

        result = resolve_runner_defaults("some-task")

        assert result.policy_obs_groups == ("actor",)
        assert result.clip_actions == 100.0

    def test_clip_actions_zero_survives(self, monkeypatch):
        # A real bound, and the one a truthiness check would eat.
        self._install(
            monkeypatch,
            SimpleNamespace(obs_groups={"actor": ("actor",)}, clip_actions=0.0),
        )
        assert resolve_runner_defaults("t").clip_actions == 0.0

    def test_a_task_may_name_its_actor_group_anything(self, monkeypatch):
        self._install(
            monkeypatch,
            SimpleNamespace(obs_groups={"actor": ("proprio",)}, clip_actions=None),
        )
        assert resolve_runner_defaults("t").policy_obs_groups == ("proprio",)

    def test_no_task_id_means_no_defaults(self):
        result = resolve_runner_defaults(None)
        assert result.policy_obs_groups is None
        assert result.clip_actions is None

    def test_an_unknown_task_is_not_fatal(self, monkeypatch):
        module = ModuleType("mjlab.tasks.registry")

        def _raise(task_id):
            raise KeyError(task_id)

        setattr(module, "load_rl_cfg", _raise)
        monkeypatch.setitem(sys.modules, "mjlab", ModuleType("mjlab"))
        monkeypatch.setitem(sys.modules, "mjlab.tasks", ModuleType("mjlab.tasks"))
        monkeypatch.setitem(sys.modules, "mjlab.tasks.registry", module)

        # A hand-built env_cfg has no registered task; that is a fallback, not an error.
        assert resolve_runner_defaults("nope").policy_obs_groups is None

    @pytest.mark.slow
    @pytest.mark.mjlab
    def test_against_a_real_mjlab_task(self):
        """Pin the shape against mjlab itself, since this reads its config directly."""
        pytest.importorskip("mjlab")
        import mjlab.tasks  # noqa: F401 — populates the registry

        result = resolve_runner_defaults("Mjlab-Velocity-Flat-Unitree-G1")
        # mjlab's default is `{"actor": ("actor",), "critic": ("critic",)}`; if upstream
        # renames or restructures it, this fails and says so.
        assert result.policy_obs_groups == ("actor",)
