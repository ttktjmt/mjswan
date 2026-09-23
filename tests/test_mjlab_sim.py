from __future__ import annotations

import sys
from types import ModuleType

import mujoco

from mjswan.mjlab import apply_mjlab_sim_options, ensure_mjlab_extensions


class TestMjlabCompat:
    def test_ensure_mjlab_extensions_adds_apply_to_spec(
        self, monkeypatch, minimal_spec
    ):
        mjlab_module = ModuleType("mjlab")
        mjlab_sim_pkg = ModuleType("mjlab.sim")
        mjlab_sim_module = ModuleType("mjlab.sim.sim")

        class FakeMujocoCfg:
            def __init__(self):
                self.disableflags = ("contact",)
                self.enableflags = ()

        mjlab_sim_module.MujocoCfg = FakeMujocoCfg

        monkeypatch.setitem(sys.modules, "mjlab", mjlab_module)
        monkeypatch.setitem(sys.modules, "mjlab.sim", mjlab_sim_pkg)
        monkeypatch.setitem(sys.modules, "mjlab.sim.sim", mjlab_sim_module)

        ensure_mjlab_extensions()

        cfg = FakeMujocoCfg()
        cfg.apply_to_spec(minimal_spec)
        assert (
            minimal_spec.option.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        ) == int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_apply_mjlab_sim_options_uses_apply_to_spec_when_available(
        self, minimal_spec
    ):
        class FakeMujocoCfg:
            def __init__(self):
                self.called = False

            def apply_to_spec(self, spec):
                self.called = True
                spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

        sim_cfg = type("FakeSimCfg", (), {"mujoco": FakeMujocoCfg()})()

        apply_mjlab_sim_options(minimal_spec, sim_cfg)

        assert sim_cfg.mujoco.called is True
        assert (
            minimal_spec.option.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        ) == int(mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_apply_mjlab_sim_options_falls_back_to_flag_copy(self, minimal_spec):
        mujoco_cfg = type(
            "FakeMujocoCfg",
            (),
            {"disableflags": ("contact",), "enableflags": ()},
        )()
        sim_cfg = type("FakeSimCfg", (), {"mujoco": mujoco_cfg})()

        apply_mjlab_sim_options(minimal_spec, sim_cfg)

        assert (
            minimal_spec.option.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        ) == int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
