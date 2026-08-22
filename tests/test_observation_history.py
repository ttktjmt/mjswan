"""Per-term observation history, dense and sparse.

Layer: L1 (config serialization only — no env, no trace).

mjlab stacks each term's frames *before* concatenating the group, so history is a
per-term property. Two things here are silent when wrong. A history the build
declares but the runtime cannot read leaves the policy reading one frame where it
was trained on many — same length only by accident, and no error either side. And a
group carrying per-term history must not fuse: a fused graph emits one
concatenation, which the runtime could only stack whole, giving step-major order
where mjlab gives term-major.

The runtime half of the contract (offset → frame, order, priming, layout) is pinned in
`core/observation/__tests__/HistoryObservation.test.ts`.
"""

from __future__ import annotations

import pytest

# `mjswan.compile` imports torch, so the package import below waits for the skip.
torch = pytest.importorskip("torch")

from mjswan._onnx_build import (  # noqa: E402
    _apply_observation_pipeline,
    _group_is_fusable,
)
from mjswan.managers.observation_manager import (  # noqa: E402
    ObservationGroupCfg,
    ObservationTermCfg,
)


def _term(**kwargs) -> ObservationTermCfg:
    return ObservationTermCfg(func=lambda env: torch.zeros(1, 3), **kwargs)


def _entry(term_cfg: ObservationTermCfg, group_history: int | None = None) -> dict:
    return _apply_observation_pipeline({"name": "x"}, term_cfg, group_history)


def test_a_dense_length_ships_as_history_length():
    assert _entry(_term(history_length=4))["history_length"] == 4


def test_no_history_ships_no_history_keys():
    entry = _entry(_term())
    assert "history_length" not in entry
    assert "history_offsets" not in entry


def test_sparse_offsets_ship_verbatim():
    """The offsets are the policy's, not a count — 21 frames back, 6 of them read."""
    entry = _entry(_term(history_steps=(0, 1, 2, 4, 8, 20)))
    assert entry["history_offsets"] == [0, 1, 2, 4, 8, 20]
    # A length alongside them would be a second, conflicting answer.
    assert "history_length" not in entry


def test_sparse_offsets_win_over_both_lengths():
    """Sparse offsets are per-term by construction; a group count cannot override."""
    entry = _entry(_term(history_steps=(0, 3), history_length=9), group_history=7)
    assert entry["history_offsets"] == [0, 3]
    assert "history_length" not in entry


def test_group_history_overrides_a_terms_own_length():
    assert _entry(_term(history_length=2), group_history=5)["history_length"] == 5


def test_a_group_zero_switches_history_off():
    """mjlab assigns the group's count whenever it is set, so an explicit 0 is an
    instruction, not an absent value."""
    entry = _entry(_term(history_length=4), group_history=0)
    assert "history_length" not in entry
    assert "history_offsets" not in entry


def test_interleaved_only_ships_alongside_history():
    """It describes a stack's layout, so without a stack it says nothing."""
    assert "history_interleaved" not in _entry(_term(history_interleaved=True))
    assert _entry(_term(history_steps=(0, 2), history_interleaved=True))[
        "history_interleaved"
    ]
    assert _entry(_term(history_length=3, history_interleaved=True))[
        "history_interleaved"
    ]
    # A group-supplied stack is still a stack for the term to lay out.
    assert _entry(_term(history_interleaved=True), group_history=4)[
        "history_interleaved"
    ]


@pytest.mark.parametrize(
    "term_cfg, fusable",
    [
        (_term(), True),
        (_term(history_length=1), True),
        (_term(history_length=2), False),
        # A single non-zero offset is still a delayed frame the runtime has to hold.
        (_term(history_steps=(4,)), False),
        (_term(history_steps=(0, 8)), False),
    ],
    ids=["none", "depth-1", "dense", "one-sparse-offset", "sparse"],
)
def test_only_a_stackless_group_fuses(term_cfg, fusable):
    group = ObservationGroupCfg(terms={"a": _term(), "b": term_cfg})
    assert _group_is_fusable(group) is fusable


def test_group_level_history_also_blocks_fusion():
    group = ObservationGroupCfg(terms={"a": _term()}, history_length=3)
    assert _group_is_fusable(group) is False


def test_a_group_zero_lets_a_stacking_term_fuse():
    group = ObservationGroupCfg(terms={"a": _term(history_length=4)}, history_length=0)
    assert _group_is_fusable(group) is True
