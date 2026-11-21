"""`MdpConfig`: the unit a policy runs against.

Layer: L1. Pins ownership and identity: the five term sets travel as one object, two
policies handed the same object share one MDP, and an MDP's id comes from its name, from
the policy it was built for, or from first-use order on its scene, never from its content.
"""

from __future__ import annotations

import pytest

import mjswan
from mjswan.builder import Builder
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.event_manager import EventTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg


def _never(env):  # a stand-in term body; nothing traces it here
    raise AssertionError("not called")


def _plain_scene(minimal_model):
    return (
        Builder()
        .add_project(name="P")
        .add_scene(name="S", model=minimal_model, control_dt=0.02)
    )


class TestSugarBuildsAnAnonymousMdp:
    def test_the_term_sets_land_on_the_policys_mdp(self, minimal_model, minimal_onnx):
        scene = _plain_scene(minimal_model)
        actions = {"joint_pos": JointPositionActionCfg(actuator_names=(".*",))}
        terminations = {"fallen": TerminationTermCfg(func=_never)}

        cfg = scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions=actions,
            terminations=terminations,
            commands={"velocity": mjswan.velocity_command()},
        )._config

        assert isinstance(cfg.mdp, mjswan.MdpConfig)
        assert list(cfg.mdp.actions) == ["joint_pos"]
        assert list(cfg.mdp.terminations) == ["fallen"]
        assert list(cfg.mdp.commands) == ["velocity"]
        # The properties are views onto the same object, not copies.
        assert cfg.actions is cfg.mdp.actions
        assert cfg.terminations is cfg.mdp.terminations
        assert cfg.commands is cfg.mdp.commands

    def test_each_sugar_call_is_its_own_mdp(self, minimal_model, minimal_onnx):
        # Equal content, separate objects: two MDPs. Identity is by object.
        scene = _plain_scene(minimal_model)
        a = scene.add_policy(name="A", policy=minimal_onnx, commands={})._config
        b = scene.add_policy(name="B", policy=minimal_onnx, commands={})._config
        assert a.mdp is not b.mdp
        # Each belongs to one policy, so each is named after it rather than numbered.
        assert scene._config.mdp_ids == ["a", "b"]

    def test_mdp_and_term_sets_together_are_refused(self, minimal_model, minimal_onnx):
        scene = _plain_scene(minimal_model)
        with pytest.raises(ValueError, match="not both"):
            scene.add_policy(
                name="Policy",
                policy=minimal_onnx,
                mdp=mjswan.MdpConfig(),
                terminations={"fallen": TerminationTermCfg(func=_never)},
            )


class TestASharedMdpIsOneMdp:
    def test_two_policies_handed_one_object_share_it(self, minimal_model, minimal_onnx):
        scene = _plain_scene(minimal_model)
        mdp = mjswan.MdpConfig(
            actions={"joint_pos": JointPositionActionCfg(actuator_names=(".*",))},
            commands={},
        )
        a = scene.add_policy(name="model_1000", policy=minimal_onnx, mdp=mdp)._config
        b = scene.add_policy(name="model_2000", policy=minimal_onnx, mdp=mdp)._config

        assert a.mdp is mdp and b.mdp is mdp
        assert scene._config.mdps == [mdp]
        assert scene._config.mdp_id(mdp) == "mdp_0"

    def test_the_first_user_adapts_it_in_place_and_the_second_reuses_that(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        mdp = mjswan.MdpConfig(commands=None)
        scene.add_policy(name="A", policy=minimal_onnx, mdp=mdp)
        # `None` commands on a plain scene derive nothing, and adaptation normalizes
        # that to `{}` so the build never sees a `None` term set.
        assert mdp.commands == {}
        assert mdp._adapted is True
        scene.add_policy(name="B", policy=minimal_onnx, mdp=mdp)
        assert mdp.commands == {}


class TestMdpIds:
    def test_a_sugar_built_mdp_takes_its_policys_id(self, minimal_model, minimal_onnx):
        # `mdp/locomotion/` beside `policy/locomotion.onnx`, rather than `mdp/mdp_0/`.
        scene = _plain_scene(minimal_model)
        policy = scene.add_policy(
            name="Locomotion", policy=minimal_onnx, commands={}
        )._config
        assert policy.id == "locomotion"
        assert scene._config.mdp_id(policy.mdp) == "locomotion"

    def test_a_sugar_built_mdp_follows_a_renamed_policy(
        self, minimal_model, minimal_onnx
    ):
        # The policy's id is the unique one, so the MDP derived from it is unique too.
        scene = _plain_scene(minimal_model)
        scene.add_policy(name="model_2000", policy=minimal_onnx, commands={})
        with pytest.warns(RuntimeWarning, match="policy"):
            second = scene.add_policy(
                name="model_2000", policy=minimal_onnx, commands={}
            )._config
        assert second.id == "model_2000_1"
        assert scene._config.mdp_ids == ["model_2000", "model_2000_1"]

    def test_unnamed_shared_mdps_number_in_first_use_order(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        first = mjswan.MdpConfig(commands={})
        second = mjswan.MdpConfig(commands={})
        scene.add_policy(name="A", policy=minimal_onnx, mdp=first)
        scene.add_policy(name="B", policy=minimal_onnx, mdp=second)
        scene.add_policy(name="C", policy=minimal_onnx, mdp=first)  # reuse, no new id
        assert scene._config.mdp_ids == ["mdp_0", "mdp_1"]

    def test_a_named_mdp_takes_its_sanitized_name(self, minimal_model, minimal_onnx):
        scene = _plain_scene(minimal_model)
        mdp = mjswan.MdpConfig(name="Velocity Rough", commands={})
        scene.add_policy(name="A", policy=minimal_onnx, mdp=mdp)
        assert scene._config.mdp_id(mdp) == "velocity_rough"

    def test_numbering_is_per_scene(self, minimal_model, minimal_onnx):
        # Adding a scene in front of another must not renumber the other's MDPs.
        project = Builder().add_project(name="P")
        s1 = project.add_scene(name="S1", model=minimal_model, control_dt=0.02)
        s2 = project.add_scene(name="S2", model=minimal_model, control_dt=0.02)
        s1.add_policy(name="A", policy=minimal_onnx, mdp=mjswan.MdpConfig(commands={}))
        s2.add_policy(name="B", policy=minimal_onnx, mdp=mjswan.MdpConfig(commands={}))
        assert s1._config.mdp_ids == ["mdp_0"]
        assert s2._config.mdp_ids == ["mdp_0"]

    def test_a_named_mdp_colliding_with_a_number_is_renamed(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        # Shared by hand, so numbered: it takes `mdp_0`.
        scene.add_policy(
            name="A", policy=minimal_onnx, mdp=mjswan.MdpConfig(commands={})
        )
        with pytest.warns(RuntimeWarning, match="mdp"):
            scene.add_policy(
                name="B",
                policy=minimal_onnx,
                mdp=mjswan.MdpConfig(name="mdp 0", commands={}),
            )
        assert scene._config.mdp_ids == ["mdp_0", "mdp_0_1"]


class TestEventsAreTheFifthField:
    def test_a_policy_that_says_nothing_takes_the_scenes_events(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        scene.set_events({"push": EventTermCfg(func=_never, mode="interval")})
        cfg = scene.add_policy(name="A", policy=minimal_onnx, commands={})._config
        assert cfg.events is not None and list(cfg.events) == ["push"]
        # Adapted (copied) from the scene's, so equal rather than the same dict.
        assert cfg.mdp.events == scene._config.events

    def test_an_explicit_events_field_overrides_the_scenes(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        scene.set_events({"push": EventTermCfg(func=_never, mode="interval")})
        mine = {"dr": EventTermCfg(func=_never, mode="startup")}
        cfg = scene.add_policy(
            name="A", policy=minimal_onnx, commands={}, events=mine
        )._config
        assert list(cfg.events) == ["dr"]

    def test_an_empty_dict_means_no_events_not_derive(
        self, minimal_model, minimal_onnx
    ):
        scene = _plain_scene(minimal_model)
        scene.set_events({"push": EventTermCfg(func=_never, mode="interval")})
        cfg = scene.add_policy(
            name="A", policy=minimal_onnx, commands={}, events={}
        )._config
        assert not cfg.events

    def test_derive_term_sets_returns_five(self, minimal_model):
        scene = _plain_scene(minimal_model)
        result = scene._derive_term_sets(None, None, None, None, None)
        assert len(result) == 5
