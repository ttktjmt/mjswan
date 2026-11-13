"""An unrelated command config name cannot select the builtin tracking contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from mjswan.managers.command_manager import (
    CommandBinding,
    _custom_registry,
    register_command,
)
from mjswan.mjlab.command import adapt_commands


@dataclass
class MotionCommandCfg:
    """MyoSuite-shaped config: model plus a multi-clip task, not mjlab's clip."""

    model: str = "myofullbody"
    task: object = field(
        default_factory=lambda: SimpleNamespace(
            motion=SimpleNamespace(clips="kit_train", root="clips")
        )
    )
    entity_name: str = "robot"


def test_default_tracking_binding_refuses_a_different_same_named_contract():
    with pytest.raises(TypeError, match="register_command") as exc:
        adapt_commands({"motion": MotionCommandCfg()})
    assert "motion_file" in str(exc.value)
    assert "anchor_body_name" in str(exc.value)
    assert "body_names" in str(exc.value)
    assert f"{__name__}.MotionCommandCfg" in str(exc.value)


def test_real_mjlab_builtin_contract_still_adapts():
    pytest.importorskip("mjlab")
    from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

    cfg = unitree_g1_flat_tracking_env_cfg(play=True).commands["motion"]
    result = adapt_commands({"motion": cfg})["motion"]
    assert result.term_name == "TrackingCommand"
    assert result.params["anchor_body_name"] == cfg.anchor_body_name
    assert result.params["body_names"] == list(cfg.body_names)


@pytest.mark.parametrize("kind", ["traced", "custom_serializer", "shared_serializer"])
def test_explicit_registration_can_supply_the_custom_contract(kind):
    from mjswan.envs.mdp.commands import serialize_motion_command

    original = _custom_registry["MotionCommandCfg"]
    binding = {
        "traced": CommandBinding(state_fields=["reference"], command_field="reference"),
        "custom_serializer": CommandBinding(
            ts_name="CustomCommand", serializer=lambda cfg: {"model": cfg.model}
        ),
        "shared_serializer": CommandBinding(
            ts_name="ExplicitCustom", serializer=serialize_motion_command
        ),
    }[kind]
    try:
        register_command("MotionCommandCfg", binding)
        cfg = MotionCommandCfg()
        result = adapt_commands({"motion": cfg})["motion"]
        if kind == "traced":
            assert result.pending_trace.mjlab_cfg is cfg
            assert result.pending_trace.command_field == "reference"
        elif kind == "custom_serializer":
            assert result.term_name == "CustomCommand"
            assert result.params == {"model": "myofullbody"}
        else:
            assert result.term_name == "ExplicitCustom"
    finally:
        register_command("MotionCommandCfg", original)
