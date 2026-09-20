"""`UniformVelocityCommandCfg`'s trace-friendly rewrite, against the real mjlab term.

Layer: L1 (no env build, no trace — the term is constructed against a stand-in env).

The rewrite in `mjswan.envs.mdp.commands` is a second copy of mjlab's math, and the
parity harness cannot check it: `run_command_parity` traces the *overridden* term and
compares the graph against that same term, so it only establishes "graph == override".
"override == mjlab" is what this file pins. Getting it wrong is silent — a well-formed
command of the right width that mjlab would not have issued.
"""

from __future__ import annotations

import itertools
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg  # noqa: E402

from mjswan.envs.mdp.commands import bind_velocity_override  # noqa: E402
from mjswan.managers.command_manager import _custom_registry  # noqa: E402

HEADING_W = 0.7
"""The stand-in robot's yaw. Non-zero so a world-frame rotation is not the identity."""


class _FakeEnv:
    """Enough env for `CommandTerm.__init__`: `num_envs`, `device`, and the entity. No
    scene is compiled, so these tests stay out of the `slow` tier."""

    def __init__(self, heading: float = HEADING_W):
        robot = type(
            "_Robot",
            (),
            {"data": type("_Data", (), {"heading_w": torch.tensor([heading])})()},
        )()
        self.scene = {"robot": robot}
        self.num_envs = 1
        self.device = "cpu"


def _cfg(**overrides) -> UniformVelocityCommandCfg:
    """mjlab's own velocity-task shape: heading tracking, standing and forward envs."""
    params = dict(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.1,
        rel_heading_envs=0.3,
        rel_forward_envs=0.2,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
    )
    params.update(overrides)
    return UniformVelocityCommandCfg(**params)


#: A command mid-episode. `ang_vel_z` is deliberately *not* what heading tracking would
#: produce, or the heading cases pass whether the rewrite tracks heading or not.
_SEED = {
    "vel_command_b": [[0.4, -0.3, -0.45]],
    "vel_command_w": [[0.9, 0.2, -0.1]],
    "heading_target": [1.2],
}


def _seed(term, *, heading: bool, world: bool, standing: bool) -> None:
    for name, value in _SEED.items():
        setattr(term, name, torch.tensor(value))
    term.is_heading_env = torch.tensor([heading])
    term.is_world_env = torch.tensor([world])
    term.is_standing_env = torch.tensor([standing])
    term.is_forward_env = torch.tensor([False])


def _pair(cfg):
    """A live mjlab term and an identically-configured overridden one."""
    live = cfg.build(_FakeEnv())
    rewritten = cfg.build(_FakeEnv())
    bind_velocity_override(rewritten)
    return live, rewritten


def test_the_binding_ships_with_mjswan():
    """It used to live in `examples/`, out of reach of any project outside this repo."""
    binding = _custom_registry["UniformVelocityCommandCfg"]
    assert binding.is_onnx_traced
    assert binding.command_field == "vel_command_b"


@pytest.mark.parametrize(
    "heading, world, standing", list(itertools.product([False, True], repeat=3))
)
def test_update_command_matches_mjlab(heading, world, standing):
    """Every combination of the three per-env modes, from the same state."""
    live, rewritten = _pair(_cfg())
    _seed(live, heading=heading, world=world, standing=standing)
    _seed(rewritten, heading=heading, world=world, standing=standing)

    live._update_command()
    rewritten._update_command()

    assert torch.allclose(live.vel_command_b, rewritten.vel_command_b, atol=1e-6)
    assert torch.allclose(live.vel_command_w, rewritten.vel_command_w, atol=1e-6)


def test_heading_tracking_actually_moves_the_yaw():
    """Guards the test above: a rewrite ignoring `is_heading_env` would still pass it."""
    live, _ = _pair(_cfg())
    _seed(live, heading=True, world=False, standing=False)
    live._update_command()
    assert not math.isclose(
        live.vel_command_b[0, 2].item(), _SEED["vel_command_b"][0][2]
    )


def test_heading_command_off_is_respected():
    """With the cfg default `False`, mjlab never touches yaw, even with `is_heading_env`."""
    cfg = _cfg(
        heading_command=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.5, 0.5)
        ),
    )
    live, rewritten = _pair(cfg)
    _seed(live, heading=True, world=False, standing=False)
    _seed(rewritten, heading=True, world=False, standing=False)

    live._update_command()
    rewritten._update_command()

    assert torch.allclose(live.vel_command_b, rewritten.vel_command_b, atol=1e-6)
    assert rewritten.vel_command_b[0, 2].item() == pytest.approx(-0.45)


def test_a_forward_only_env_gets_mjlabs_clamp():
    """mjlab's velocity tasks set `rel_forward_envs=0.2` and `play=True` keeps it, so
    without this the rewrite differs one resample in five."""
    _, rewritten = _pair(_cfg(rel_forward_envs=1.0))
    torch.manual_seed(0)
    rewritten._resample_command(torch.arange(1))

    assert rewritten.is_forward_env.tolist() == [True]
    assert rewritten.vel_command_b[0, 0].item() >= 0.3
    assert rewritten.vel_command_b[0, 1].item() == 0.0
    assert rewritten.vel_command_b[0, 2].item() == 0.0


def test_the_world_frame_reference_keeps_the_unclamped_sample():
    """mjlab copies `vel_command_w` *before* the forward clamp, so a world+forward env
    rotates the raw sample rather than the clamped one."""
    _, rewritten = _pair(_cfg(rel_forward_envs=1.0))
    torch.manual_seed(0)
    rewritten._resample_command(torch.arange(1))

    assert rewritten.vel_command_w[0, 1].item() != 0.0
    assert rewritten.vel_command_w[0, 2].item() != 0.0


def test_the_bodies_write_nothing_the_binding_does_not_carry():
    """A tensor the rewrite assigns but `state_fields` omits is dropped between browser
    steps — the term would silently restart from its build-time value each frame."""
    _, rewritten = _pair(_cfg())
    before = {
        name: value.clone()
        for name, value in vars(rewritten).items()
        if isinstance(value, torch.Tensor)
    }
    torch.manual_seed(0)
    rewritten._resample_command(torch.arange(1))
    rewritten._update_command()

    changed = {
        name
        for name, value in vars(rewritten).items()
        if isinstance(value, torch.Tensor)
        and (name not in before or not torch.equal(value, before[name]))
    }
    declared = set(_custom_registry["UniformVelocityCommandCfg"].state_fields or [])
    assert changed <= declared, f"undeclared state: {sorted(changed - declared)}"


def test_an_unmodelled_cfg_field_is_refused():
    """`init_velocity_prob` writes the robot's root state during resampling, gated on
    that same draw. Tracing it as if absent would be a silent difference."""
    term = _cfg(init_velocity_prob=0.5).build(_FakeEnv())
    with pytest.raises(ValueError, match="init_velocity_prob"):
        bind_velocity_override(term)
