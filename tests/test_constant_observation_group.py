"""A group whose every term is constant has no graph to fuse.

A policy can declare an input the browser has no live value for — a padding slot
the trained network still expects. Such a term reads nothing off the env, so the
group it sits in has no time-varying input and cannot become a graph. The build
must fall back to serializing each term on its own (where a constant bakes into
`native: constant`) instead of failing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _trace_env():
    class _Data:
        def __init__(self):
            self.root_link_ang_vel_b = torch.tensor([[0.0, 0.1, 0.2]])

    class _Scene:
        def __init__(self):
            self.sensors = {}
            self._entities = {"robot": type("E", (), {"data": _Data()})()}

        def __getitem__(self, name):
            return self._entities[name]

    class _Env:
        def __init__(self):
            self.scene = _Scene()

    return _Env()


def _padding(env, **_):
    """Reads nothing off the env: a fixed-width padding term."""
    del env
    return torch.zeros(1, 5)


def _other_padding(env, **_):
    del env
    return torch.full((1, 2), 0.25)


def test_all_constant_group_serializes_per_term(tmp_path):
    from mjswan.build.mdp import serialize_observation_group
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    group = ObservationGroupCfg(
        terms={
            "pad": ObservationTermCfg(func=_padding),
            "pad2": ObservationTermCfg(func=_other_padding),
        }
    )
    entry = serialize_observation_group(group, _trace_env(), tmp_path, "command")

    # Per-term list, not a fused dict — there is no graph for a fused entry to name.
    assert isinstance(entry, list)
    assert [e["name"] for e in entry] == ["pad", "pad2"]
    assert [e["native"] for e in entry] == ["constant", "constant"]
    assert [e["size"] for e in entry] == [5, 2]
    assert entry[0]["value"] == [0.0] * 5
    assert entry[1]["value"] == [0.25, 0.25]
    # Nothing exported: a baked constant needs no .onnx file.
    assert list(tmp_path.rglob("*.onnx")) == []


def test_group_with_one_dynamic_term_still_fuses(tmp_path):
    """The fallback must not swallow groups that do have a graph."""
    from mjlab.envs.mdp import observations as obs_fns

    from mjswan.build.mdp import serialize_observation_group
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    group = ObservationGroupCfg(
        terms={
            "pad": ObservationTermCfg(func=_padding),
            "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
        }
    )
    entry = serialize_observation_group(group, _trace_env(), tmp_path, "policy")

    assert isinstance(entry, dict)
    assert entry["fused"].endswith(".onnx")
    assert entry["size"] == 8  # 5 padding + 3 ang vel
    assert [e["name"] for e in entry["layout"]] == ["pad", "base_ang_vel"]


def test_a_baked_term_is_named_in_a_warning(tmp_path):
    """Both bake sites: the lone term and the fused group."""
    from mjlab.envs.mdp import observations as obs_fns

    from mjswan.build.mdp import serialize_observation_group
    from mjswan.compile.group import GroupTermSpec, trace_observation_group
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    group = ObservationGroupCfg(terms={"pad": ObservationTermCfg(func=_padding)})
    with pytest.warns(RuntimeWarning, match="'pad' reads no simulation state"):
        serialize_observation_group(group, _trace_env(), tmp_path, "command")

    specs = [
        GroupTermSpec("pad", _padding, {}),
        GroupTermSpec("base_ang_vel", obs_fns.base_ang_vel, {}),
    ]
    with pytest.warns(RuntimeWarning, match="'pad' reads no simulation state"):
        trace_observation_group(specs, _trace_env(), name="policy")


def test_constant_group_raises_from_the_fused_path(tmp_path):
    """The fallback exists because tracing is the only way to know: assert the signal."""
    from mjswan.compile.group import (
        ConstantGroup,
        GroupTermSpec,
        trace_observation_group,
    )

    specs = [GroupTermSpec("pad", _padding, {})]
    with pytest.raises(ConstantGroup, match="no time-varying state"):
        trace_observation_group(specs, _trace_env(), name="command")
