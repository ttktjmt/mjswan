"""What a document says about *which function* a term is (issue #118).

Layer: L1 (pure Python) for the manifest keys and the ONNX stamp, plus one
torch-gated test through the real observation serializer.
"""

from __future__ import annotations

import pytest

from mjswan._graph_io import stamp_provenance, write_onnx
from mjswan._onnx_build import _param_json, _provenance


def base_lin_vel(env, asset_cfg):
    """Root linear velocity in the asset's root frame.

    Second paragraph, which must not travel.
    """
    del env, asset_cfg


class _EntityCfg:
    """Duck-typed SceneEntityCfg: a name plus the patterns it was declared with."""

    def __init__(self):
        self.name = "robot"
        self.joint_names = [".*_hip_.*", ".*_knee_joint"]
        self.body_names = None
        self.joint_ids = [3, 4, 5]  # resolved indices: never serialized


class TestProvenance:
    def test_names_the_function_and_its_first_doc_line(self):
        out = _provenance(base_lin_vel, {"asset_cfg": _EntityCfg()})
        assert out["func"] == f"{__name__}:base_lin_vel"
        assert out["doc"] == "Root linear velocity in the asset's root frame."
        assert out["params"] == {
            "asset_cfg": {
                "entity": "robot",
                "joint_names": [".*_hip_.*", ".*_knee_joint"],
            }
        }

    def test_a_class_is_named_like_a_function(self):
        out = _provenance(_EntityCfg)
        assert out["func"] == f"{__name__}:_EntityCfg"
        assert "params" not in out

    def test_undocumented_lambda_still_gets_a_func(self):
        out = _provenance(lambda env: env)
        assert out["func"].endswith(".<lambda>")
        assert "doc" not in out

    def test_params_never_carry_objects(self):
        assert _param_json(0.5) == 0.5
        assert _param_json((1, 2)) == [1, 2]
        assert _param_json(object()) == "object"
        assert _param_json(None) is None


def _tiny_model() -> bytes:
    pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y])
    return helper.make_model(graph).SerializeToString()


class TestStamp:
    def test_stamps_producer_doc_and_metadata(self):
        onnx = pytest.importorskip("onnx")
        stamped = stamp_provenance(
            _tiny_model(), {"kind": "obs", "term": "actor", "func": "m:f"}
        )
        model = onnx.load_from_string(stamped)
        assert model.producer_name == "mjswan"
        assert model.doc_string == "mjswan obs graph 'actor', traced from m:f"
        assert {p.key: p.value for p in model.metadata_props} == {
            "mjswan.kind": "obs",
            "mjswan.term": "actor",
            "mjswan.func": "m:f",
        }

    def test_write_onnx_compares_what_lands_on_disk(self, tmp_path):
        pytest.importorskip("onnx")
        meta = {"kind": "obs", "term": "actor"}
        write_onnx(tmp_path, "obs/actor.onnx", _tiny_model(), meta=meta)
        # The same term traced twice, stamped twice, is still one graph.
        write_onnx(tmp_path, "obs/actor.onnx", _tiny_model(), meta=meta)
        with pytest.raises(ValueError, match="obs/actor.onnx"):
            write_onnx(tmp_path, "obs/actor.onnx", _tiny_model(), meta={"kind": "term"})


def test_constant_observation_entry_carries_provenance(tmp_path):
    torch = pytest.importorskip("torch")
    from mjswan._onnx_build import serialize_observation_term
    from mjswan.managers.observation_manager import ObservationTermCfg

    def padding(env, **_):
        """A fixed-width padding term."""
        del env
        return torch.zeros(1, 5)

    class _Env:
        class scene:  # noqa: N801 — mirrors env.scene
            sensors: dict = {}

    entry = serialize_observation_term(
        "pad", ObservationTermCfg(func=padding), _Env(), tmp_path, None
    )
    assert entry["native"] == "constant"
    assert entry["func"].endswith(
        ":test_constant_observation_entry_carries_provenance.<locals>.padding"
    )
    assert entry["doc"] == "A fixed-width padding term."
