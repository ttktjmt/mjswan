"""A scene's term defaults: what its terms do when the viewer opens, written as
``term_defaults`` beside its MDPs (format 10)."""

from __future__ import annotations

import warnings

import pytest

import mjswan
from mjswan.build.manifest import term_defaults_entry
from mjswan.builder import Builder
from mjswan.mdp import TerminationDefault


class TestTerminationDefault:
    def test_writes_the_engines_on_fire(self):
        assert TerminationDefault().to_dict() == {"enabled": True, "on_fire": "reset"}
        assert TerminationDefault("delay", seconds=0.2).to_dict() == {
            "enabled": True,
            "on_fire": {"delay": 0.2},
        }
        assert TerminationDefault("freeze", seconds=3).to_dict() == {
            "enabled": True,
            "on_fire": {"freeze": 3.0},
        }
        assert TerminationDefault("freeze").to_dict()["on_fire"] == "freeze"
        assert TerminationDefault(enabled=False).to_dict() == {
            "enabled": False,
            "on_fire": "reset",
        }

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"on_fire": "delay"}, ValueError),
            ({"on_fire": "reset", "seconds": 1.0}, ValueError),
            ({"on_fire": "delay", "seconds": 0}, ValueError),
            ({"on_fire": "freeze", "seconds": 61}, ValueError),
            ({"on_fire": "delay", "seconds": float("nan")}, ValueError),
            ({"on_fire": "delay", "seconds": True}, TypeError),
            ({"on_fire": "stop"}, ValueError),
            ({"enabled": "no"}, TypeError),
        ],
    )
    def test_refuses_what_the_viewer_could_not_keep(self, kwargs, error):
        with pytest.raises(error):
            TerminationDefault(**kwargs)

    def test_is_public(self):
        assert mjswan.TerminationDefault is TerminationDefault


class TestSetTermDefaults:
    def _scene(self, minimal_model):
        return Builder().add_project(name="P").add_scene(name="S", model=minimal_model)

    def test_keeps_terminations_and_events_by_name(self, minimal_model):
        scene = self._scene(minimal_model)
        scene.set_term_defaults(
            terminations={
                "bad_orientation": TerminationDefault("delay", seconds=0.2),
                "time_out": False,
            },
            events={"push_robot": False},
        )
        assert scene._config.term_defaults == {
            "terminations": {
                "bad_orientation": {"enabled": True, "on_fire": {"delay": 0.2}},
                "time_out": {"enabled": False, "on_fire": "reset"},
            },
            "events": {"push_robot": {"enabled": False}},
        }

    def test_a_second_call_replaces_the_first(self, minimal_model):
        scene = self._scene(minimal_model)
        scene.set_term_defaults(events={"push_robot": False})
        scene.set_term_defaults(terminations={"time_out": False})
        assert list(scene._config.term_defaults) == ["terminations"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"terminations": {"": False}},
            {"terminations": {"fell": "delay"}},
            {"events": {"push_robot": "off"}},
        ],
    )
    def test_refuses_a_bad_name_or_value(self, minimal_model, kwargs):
        with pytest.raises((TypeError, ValueError)):
            self._scene(minimal_model).set_term_defaults(**kwargs)


class TestManifestEntry:
    def test_sits_on_the_scene_and_warns_for_a_name_no_mdp_has(
        self, tmp_path, minimal_model, build_manifest
    ):
        project = Builder().add_project(name="P")
        project.add_scene(name="S", model=minimal_model).set_term_defaults(
            terminations={"bad_orientation": TerminationDefault("freeze")}
        )
        with pytest.warns(RuntimeWarning, match="bad_orientation"):
            manifest = build_manifest(project._builder, tmp_path / "out")
        assert manifest["projects"][0]["scenes"][0]["term_defaults"] == {
            "terminations": {"bad_orientation": {"enabled": True, "on_fire": "freeze"}}
        }

    def test_a_scene_without_defaults_writes_none(
        self, tmp_path, minimal_model, build_manifest
    ):
        project = Builder().add_project(name="P")
        project.add_scene(name="S", model=minimal_model)
        manifest = build_manifest(project._builder, tmp_path / "out")
        assert "term_defaults" not in manifest["projects"][0]["scenes"][0]

    def test_names_any_of_the_scenes_mdps_has_pass_silently(self, minimal_model):
        scene = (
            Builder()
            .add_project(name="P")
            .add_scene(name="S", model=minimal_model)
            .set_term_defaults(
                terminations={"bad_orientation": False}, events={"push_robot": False}
            )
        )
        mdps = [
            {"terminations": {"bad_orientation": {"name": "bad_orientation"}}},
            {"events": [{"name": "push_robot", "mode": "interval"}]},
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            entry = term_defaults_entry(scene._config, mdps)
        assert entry["events"] == {"push_robot": {"enabled": False}}
