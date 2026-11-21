"""`last_action(action_name=...)` slice resolution.

Layer: L1 (pure Python: a stub action manager, no mjlab env build).

mjlab's `last_action(action_name=...)` is `get_term(name).raw_action`: that one
term's slice of the policy output, not the whole vector
(`mjlab/envs/mdp/observations.py`). The graph is fed the vector whole, as the runtime
holds it, and slices it at the offset `ActionManager.process_action` gives the term by
accumulating `action_term_dim` in config order. `action_term_window` mirrors that walk.

Every reference task declares a single action term, where the slice and the whole
vector coincide, so none of them exercises the two-term shape these cases cover.
"""

from __future__ import annotations

import pytest

# Pure-Python walk, but `mjswan.compile` imports torch at load time.
torch = pytest.importorskip("torch")

from mjswan.compile.proxy import action_term_window  # noqa: E402


class _StubActionManager:
    """Only the two members the offset walk reads."""

    def __init__(self, terms: dict[str, int]) -> None:
        self.active_terms = list(terms)
        self.action_term_dim = list(terms.values())


def test_offset_accumulates_in_config_order() -> None:
    manager = _StubActionManager({"arm": 7, "gripper": 1, "torso": 3})
    assert action_term_window(manager, "arm") == (0, 7)
    assert action_term_window(manager, "gripper") == (7, 1)
    assert action_term_window(manager, "torso") == (8, 3)


def test_single_term_spans_the_whole_vector() -> None:
    """The shape every reference task has, and the reason the bug stayed latent."""
    manager = _StubActionManager({"joint_pos": 29})
    assert action_term_window(manager, "joint_pos") == (0, 29)


def test_unknown_term_raises_and_names_the_available_ones() -> None:
    """Degrading would hand the runtime the vector's head at the right width.

    That is the silently-wrong observation the offset exists to prevent, so an
    unresolvable name fails the build instead — as mjlab's own `get_term` does.
    """
    manager = _StubActionManager({"arm": 7, "gripper": 1})
    with pytest.raises(ValueError, match=r"does not define.*arm, gripper"):
        action_term_window(manager, "grippr")
