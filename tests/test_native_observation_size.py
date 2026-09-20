"""Native observation widths when the trace env cannot supply them.

`last_action` and `generated_commands` read env-level state, not `entity.data`.
A trace env built by `build_single_entity_trace_env` (a plain `add_scene()`
scene) has no action terms and no command manager, so the first answers with an
empty vector and the second raises — yet a fused graph has to fix both widths at
export time. They come from the policy config instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _trace_env():
    """A minimal trace env: entity data, but no actions and no commands."""

    class _Data:
        def __init__(self):
            self.root_link_ang_vel_b = torch.tensor([[0.0, 0.1, 0.2]])

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": type("E", (), {"data": _Data()})()}

        def __getitem__(self, name):
            return self._entities[name]

    class _ActionManager:
        # mjlab sizes this by `total_action_dim`, which is 0 with no action terms.
        action = torch.zeros((1, 0))

    class _CommandManager:
        def get_command(self, name):
            return None  # mjlab's NullCommandManager.

    class _Env:
        def __init__(self):
            self.scene = _Scene()
            self.action_manager = _ActionManager()
            self.command_manager = _CommandManager()

    return _Env()


def _specs(native_size_actions=None, native_size_command=None):
    from mjlab.envs.mdp import observations as obs_fns

    from mjswan.compile.group import GroupTermSpec

    return [
        GroupTermSpec("base_ang_vel", obs_fns.base_ang_vel, {}),
        GroupTermSpec(
            "last_action", obs_fns.last_action, {}, native_size=native_size_actions
        ),
        GroupTermSpec(
            "velocity_cmd",
            obs_fns.generated_commands,
            {"command_name": "velocity"},
            native_size=native_size_command,
        ),
    ]


def test_declared_widths_fix_the_graph_inputs():
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group

    export = trace_observation_group(
        _specs(native_size_actions=29, native_size_command=3),
        _trace_env(),
        name="policy",
    )

    assert export.layout == [
        {"name": "base_ang_vel", "size": 3},
        {"name": "last_action", "size": 29},
        {"name": "velocity_cmd", "size": 3},
    ]
    assert export.reference_output.reshape(1, -1).shape[-1] == 35


def test_a_width_neither_side_knows_fails_the_build():
    """Not silently zero-wide: that shortens the policy's input vector."""
    pytest.importorskip("mjlab")
    from mjswan.compile.group import trace_observation_group

    with pytest.raises(ValueError, match="last_action"):
        trace_observation_group(
            _specs(native_size_command=3), _trace_env(), name="policy"
        )


def test_policy_native_sizes_reads_the_policy_config():
    from mjswan.build.mdp import policy_native_sizes
    from mjswan.command import Button, ui_command, velocity_command

    sizes = policy_native_sizes(
        {"policy_joint_names": ["j"] * 29},
        {
            "velocity": velocity_command(),
            # Buttons carry no value, so this command contributes no width.
            "buttons": ui_command([Button(name="go", label="Go")]),
        },
    )

    assert sizes == {"prev_action": 29, "command:velocity": 3}


def test_policy_num_actions_wins_over_joint_names():
    from mjswan.build.mdp import policy_native_sizes

    sizes = policy_native_sizes(
        {"policy_joint_names": ["j"] * 29, "policy_num_actions": 80}, None
    )
    assert sizes == {"prev_action": 80}
