"""mjlab's simulation options, applied to the ``MjSpec`` a scene is built from.

mjlab keeps its MuJoCo flags on ``SimulationCfg.mujoco`` and applies them to the compiled
model; the browser compiles the spec itself, so the same flags are written into the spec
here.
"""

from __future__ import annotations

from typing import Any

import mujoco


def _apply_mujoco_cfg_to_option(mujoco_cfg: Any, option: Any) -> None:
    """Apply mjlab ``MujocoCfg`` option flags to a MuJoCo option object."""
    for field_name, enum_type, prefix in (
        ("disableflags", mujoco.mjtDisableBit, "mjDSBL_"),
        ("enableflags", mujoco.mjtEnableBit, "mjENBL_"),
    ):
        for flag_name in getattr(mujoco_cfg, field_name, None) or ():
            enum_name = f"{prefix}{str(flag_name).upper()}"
            flag = getattr(enum_type, enum_name, None)
            if flag is not None:
                setattr(
                    option,
                    field_name,
                    int(getattr(option, field_name)) | int(flag),
                )


def ensure_mjlab_extensions() -> None:
    """Install mjswan compatibility extensions onto mjlab classes when available."""
    try:
        from mjlab.sim.sim import MujocoCfg
    except ImportError:
        return

    if hasattr(MujocoCfg, "apply_to_spec"):
        return

    def apply_to_spec(self: Any, spec: mujoco.MjSpec) -> None:
        _apply_mujoco_cfg_to_option(self, spec.option)

    setattr(MujocoCfg, "apply_to_spec", apply_to_spec)


def apply_mjlab_sim_options(spec: mujoco.MjSpec, sim_cfg: Any | None) -> None:
    """Apply mjlab simulation flags to an exported MuJoCo spec."""
    mujoco_cfg = getattr(sim_cfg, "mujoco", None)
    if mujoco_cfg is None:
        return

    apply_to_spec = getattr(mujoco_cfg, "apply_to_spec", None)
    if callable(apply_to_spec):
        apply_to_spec(spec)
        return

    _apply_mujoco_cfg_to_option(mujoco_cfg, spec.option)


__all__ = [
    "apply_mjlab_sim_options",
    "ensure_mjlab_extensions",
]
