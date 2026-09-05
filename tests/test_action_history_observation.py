"""`action_history`: mjlab's older action buffers, served natively (ADR 0005 §Decision).

Layer: L1 (pure Python — a stub action manager, no mjlab env build).

mjlab's `ActionManager` keeps a three-deep window of raw policy actions — `action`,
`prev_action`, `prev_prev_action` — but its own `last_action` observation only ever
reads the newest. A task observing action history reads the older two directly, which
left them with no browser counterpart at all: not traceable (the action manager is not
scene state) and not native (only `last_action` was).

`action_history(age=...)` is that read, and is classified native so the runtime serves
it from the same window rather than tracing a graph for it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mjswan.compile.tracer import (  # noqa: E402
    NATIVE_OBSERVATION_FUNCS,
    native_observation_entry,
)
from mjswan.envs.mdp.observations import action_history  # noqa: E402


class _StubActionManager:
    def __init__(self, terms: dict[str, int], window: list[list[float]]) -> None:
        self.active_terms = list(terms)
        self.action_term_dim = list(terms.values())
        self.action, self.prev_action, self.prev_prev_action = (
            torch.tensor([row]) for row in window
        )


class _StubEnv:
    def __init__(self, terms: dict[str, int], window: list[list[float]]) -> None:
        self.action_manager = _StubActionManager(terms, window)


WINDOW = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


@pytest.fixture
def env() -> _StubEnv:
    return _StubEnv({"joint_pos": 3}, WINDOW)


class TestRead:
    @pytest.mark.parametrize("age,expected", list(enumerate(WINDOW)))
    def test_age_walks_back_through_the_window(self, env, age, expected):
        assert action_history(env, age=age).tolist() == [expected]

    def test_default_is_one_step_back(self, env):
        """`age=0` is mjlab's own `last_action`; this function exists for the rest."""
        assert action_history(env).tolist() == [WINDOW[1]]

    def test_age_past_the_window_raises(self, env):
        with pytest.raises(ValueError, match="age"):
            action_history(env, age=3)

    def test_action_name_narrows_to_that_terms_slice(self):
        env = _StubEnv({"arm": 2, "gripper": 1}, WINDOW)
        assert action_history(env, age=1, action_name="gripper").tolist() == [[6.0]]

    def test_unknown_action_name_raises_and_names_the_available_ones(self, env):
        with pytest.raises(ValueError, match="joint_pos"):
            action_history(env, action_name="nope")


class TestNativeClassification:
    def test_classified_as_prev_action(self):
        assert NATIVE_OBSERVATION_FUNCS[action_history.__name__] == "prev_action"

    def test_entry_carries_the_age(self, env):
        entry = native_observation_entry("hist", action_history, {"age": 2}, env)
        assert entry == {"name": "hist", "native": "prev_action", "age": 2}

    def test_age_zero_is_omitted_so_it_reads_as_last_action(self, env):
        entry = native_observation_entry("hist", action_history, {"age": 0}, env)
        assert "age" not in entry

    def test_entry_carries_the_slice_offset_alongside_the_age(self):
        env = _StubEnv({"arm": 2, "gripper": 1}, WINDOW)
        entry = native_observation_entry(
            "hist", action_history, {"age": 1, "action_name": "gripper"}, env
        )
        assert entry["age"] == 1
        assert entry["action_offset"] == 2
