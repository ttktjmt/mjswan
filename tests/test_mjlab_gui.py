"""Recording an mjlab command term's viser GUI as a UI descriptor.

Layer: L1 (the recorder is dataclasses only) plus one ``mjlab``-marked check
running mjlab's real ``create_gui``, so a viewer change surfaces here rather
than in a built ``config.json``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mjswan.mjlab.gui import record_gui


class _JoystickTerm:
    """Shaped like mjlab's velocity joystick, labelled unlike it: the recorder
    must key off structure (range sign, order, kind), never label text."""

    def create_gui(
        self, name, server, get_env_idx, on_change=None, request_action=None
    ):
        with server.gui.add_folder(name.capitalize()):
            server.gui.add_checkbox("Turn On", initial_value=False)
            for label, limit in [("axis_a", 2.0), ("axis_b", 0.5)]:
                server.gui.add_slider(
                    f"Max {label}", initial_value=limit, step=0.1, min=0.1, max=10.0
                )
                server.gui.add_slider(
                    label, min=-limit, max=limit, step=0.05, initial_value=0.0
                )
            server.gui.add_button("Zero", icon="square-x")


class _SilentTerm:
    """Overrides nothing — ``CommandTerm.create_gui`` is a no-op base hook."""

    def create_gui(
        self, name, server, get_env_idx, on_change=None, request_action=None
    ):
        pass


def _inputs(term: object) -> list[dict]:
    ui = record_gui(term, "twist")
    assert ui is not None
    return ui["inputs"]


def test_records_controls_in_declaration_order():
    assert [(i["type"], i["name"]) for i in _inputs(_JoystickTerm())] == [
        ("checkbox", "enabled"),
        ("slider", "axis_a"),
        ("slider", "axis_b"),
        ("button", "zero"),
    ]


def test_first_checkbox_takes_the_reserved_enabled_name():
    """`OnnxCommand.isUiEnabled` looks for exactly `enabled`; the label is free."""
    checkbox = _inputs(_JoystickTerm())[0]
    assert checkbox["name"] == "enabled"
    assert checkbox["label"] == "Turn On"
    assert checkbox["default"] is False


def test_one_sided_slider_folds_into_the_next_axis_range():
    """A "Max <label>" companion is never a command axis, and axis sliders map
    onto the command vector positionally."""
    axis_a, axis_b = _inputs(_JoystickTerm())[1:3]
    assert axis_a["adjustable_range"] == {
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "default": 2.0,
    }
    assert axis_b["adjustable_range"]["default"] == 0.5
    # No label: the browser synthesizes `Max <label>` from the axis it belongs to.
    assert "label" not in axis_a["adjustable_range"]


def test_axis_sliders_carry_range_verbatim_and_the_enable_gate():
    assert _inputs(_JoystickTerm())[1] | {"adjustable_range": None} == {
        "type": "slider",
        "name": "axis_a",
        "label": "axis_a",
        "min": -2.0,
        "max": 2.0,
        "step": 0.05,
        "default": 0.0,
        "enabled_when": "enabled",
        "adjustable_range": None,
    }


def test_a_term_declaring_nothing_records_nothing():
    """Every term has `create_gui`, so emptiness is the only fallback signal."""
    assert record_gui(_SilentTerm(), "twist") is None


def test_an_unknown_viser_control_raises_rather_than_dropping_it():
    class _Term:
        def create_gui(self, name, server, *_args, **_kwargs):
            server.gui.add_dropdown("Mode", ("a", "b"))

    with pytest.raises(AttributeError):
        record_gui(_Term(), "twist")


def test_a_buttons_icon_is_recorded_for_the_panel_to_draw():
    inputs = _inputs(_JoystickTerm())
    assert inputs[-1] == {
        "type": "button",
        "name": "zero",
        "label": "Zero",
        "icon": "square-x",
    }


def test_a_button_without_an_icon_says_nothing_about_one():
    class _Term:
        def create_gui(
            self, name, server, get_env_idx, on_change=None, request_action=None
        ):
            server.gui.add_button("Start Here")

    assert _inputs(_Term()) == [
        {"type": "button", "name": "start_here", "label": "Start Here"}
    ]


@pytest.mark.mjlab
def test_records_mjlabs_real_velocity_joystick():
    """`UniformVelocityCommand.create_gui` reads only `self.cfg.ranges`, so no
    built term or live env is needed — just bind it to a stand-in `self`."""
    velocity_command = pytest.importorskip("mjlab.tasks.velocity.mdp.velocity_command")
    cfg = velocity_command.UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=velocity_command.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )
    term = SimpleNamespace(cfg=cfg)
    term.create_gui = velocity_command.UniformVelocityCommand.create_gui.__get__(term)

    inputs = _inputs(term)

    # mjlab's viewer labels, verbatim.
    assert inputs[-1]["icon"] == "square-x"
    assert [(i["type"], i["name"], i["label"]) for i in inputs] == [
        ("checkbox", "enabled", "Enable"),
        ("slider", "lin_vel_x", "lin_vel_x"),
        ("slider", "lin_vel_y", "lin_vel_y"),
        ("slider", "ang_vel_z", "ang_vel_z"),
        ("button", "zero", "Zero"),
    ]
    # Symmetric around mjlab's own upper limit, defaulting to zero.
    assert [(i["min"], i["max"], i["default"]) for i in inputs[1:4]] == [
        (-1.0, 1.0, 0.0),
        (-0.5, 0.5, 0.0),
        (-1.0, 1.0, 0.0),
    ]
    # viser's hard 0.1..10.0 companion bounds, initialized to the task's limit.
    assert inputs[1]["adjustable_range"] == {
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "default": 1.0,
    }
