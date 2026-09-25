"""`last_action(action_name=...)` slice resolution (ADR 0005 §Decision).

Layer: L1 (pure Python — a stub action manager, no mjlab env build).

mjlab's `last_action(action_name=...)` is `get_term(name).raw_action`: that one
term's slice of the policy output, not the whole vector
(`mjlab/envs/mdp/observations.py`). The browser is fed the vector whole — one policy
output, one inference — so it needs the slice's offset to reproduce the read, and
`ActionManager.process_action` defines that offset by accumulating `action_term_dim`
in config order. `action_term_offset` mirrors exactly that walk.

Every reference task declares a single action term, where the slice and the whole
vector coincide — which is why the runtime reading the vector's head went unnoticed.
These cases are therefore the coverage for the two-term shape; no buildable task
exercises it yet.
"""

from __future__ import annotations

import pytest

# Pure-Python walk, but `mjswan.compile` imports torch at load time.
torch = pytest.importorskip("torch")

from mjswan.compile.native import action_term_offset  # noqa: E402


class _StubActionManager:
    """Only the two members the offset walk reads."""

    def __init__(self, terms: dict[str, int]) -> None:
        self.active_terms = list(terms)
        self.action_term_dim = list(terms.values())


class _StubEnv:
    def __init__(self, terms: dict[str, int]) -> None:
        self.action_manager = _StubActionManager(terms)


def test_offset_accumulates_in_config_order() -> None:
    env = _StubEnv({"arm": 7, "gripper": 1, "torso": 3})
    assert action_term_offset(env, "arm") == 0
    assert action_term_offset(env, "gripper") == 7
    assert action_term_offset(env, "torso") == 8


def test_single_term_resolves_to_zero() -> None:
    """The shape every reference task has, and the reason the bug stayed latent."""
    env = _StubEnv({"joint_pos": 29})
    assert action_term_offset(env, "joint_pos") == 0


def test_unknown_term_raises_and_names_the_available_ones() -> None:
    """Degrading would hand the runtime the vector's head at the right width.

    That is the silently-wrong observation the offset exists to prevent, so an
    unresolvable name fails the build instead — as mjlab's own `get_term` does.
    """
    env = _StubEnv({"arm": 7, "gripper": 1})
    with pytest.raises(ValueError, match=r"does not define.*arm, gripper"):
        action_term_offset(env, "grippr")
