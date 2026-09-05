"""Tests for ``JointPositionActionCfg``'s ``ema_alpha`` / ``warmup_time_s``.

Both mirror mjlab action terms that smooth the processed joint-position target across
control steps and hold the default pose for the first moments of an episode. The
filter itself runs browser-side; what is checked here is that the config validates its
inputs and reaches ``policy.json`` for the TS runtime to read.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mjswan.builder import Builder
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.utils import name2id


class TestSerialization:
    def test_defaults_omit_both_fields(self):
        assert (
            JointPositionActionCfg()
            .to_dict()
            .keys()
            .isdisjoint({"ema_alpha", "warmup_time_s"})
        )

    def test_values_emitted(self):
        entry = JointPositionActionCfg(ema_alpha=0.5, warmup_time_s=0.4).to_dict()
        assert entry["ema_alpha"] == 0.5
        assert entry["warmup_time_s"] == 0.4

    def test_no_op_values_omitted(self):
        entry = JointPositionActionCfg(ema_alpha=1.0, warmup_time_s=0.0).to_dict()
        assert "ema_alpha" not in entry
        assert "warmup_time_s" not in entry

    @pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
    def test_alpha_outside_unit_interval_raises(self, alpha):
        with pytest.raises(ValueError, match="ema_alpha"):
            JointPositionActionCfg(ema_alpha=alpha).to_dict()

    def test_negative_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup_time_s"):
            JointPositionActionCfg(warmup_time_s=-1.0).to_dict()


class TestBuilderRoundTrip:
    @pytest.fixture(autouse=True)
    def _no_frontend(self, monkeypatch):
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.shutil.copytree", MagicMock())

    def test_fields_reach_policy_json(self, tmp_path, minimal_model, minimal_onnx):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            control_dt=0.05, name="S", model=minimal_model
        )
        scene.add_policy(
            name="Policy",
            policy=minimal_onnx,
            actions={
                "joint_pos": JointPositionActionCfg(ema_alpha=0.5, warmup_time_s=0.4)
            },
        )
        builder._save_web(tmp_path / "out")

        data = json.loads(
            (
                tmp_path / "out" / "main" / "assets" / name2id("S") / "policy.json"
            ).read_text()
        )
        term = data["actions"]["joint_pos"]
        assert term["type"] == "joint_position"
        assert term["ema_alpha"] == 0.5
        assert term["warmup_time_s"] == 0.4
