"""Shared pytest fixtures for the mjswan test suite."""

import gzip
import struct
from pathlib import Path

import mujoco
import onnx
import pytest
from onnx import TensorProto, helper


@pytest.fixture
def minimal_spec(tmp_path: Path) -> mujoco.MjSpec:
    """Minimal MuJoCo spec — single sphere, no external assets."""
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(
        '<mujoco model="simple">'
        '<worldbody><geom type="sphere" size="0.1"/></worldbody>'
        "</mujoco>"
    )
    return mujoco.MjSpec.from_file(str(xml_path))


@pytest.fixture
def minimal_model() -> mujoco.MjModel:
    """Minimal MuJoCo MjModel — single sphere, loaded from XML string."""
    return mujoco.MjModel.from_xml_string(
        '<mujoco model="simple">'
        '<worldbody><geom type="sphere" size="0.1"/></worldbody>'
        "</mujoco>"
    )


@pytest.fixture
def minimal_spz(tmp_path: Path) -> Path:
    """Minimal .spz file — valid 16-byte header (NGSp v2), zero Gaussian points."""
    header = struct.pack(
        "<IIIBBBB",
        0x5053474E,  # magic "NGSP"
        2,  # version
        0,  # num_points = 0
        0,  # sh_degree
        0,  # fractional_bits
        0,  # flags
        0,  # reserved
    )
    path = tmp_path / "background.spz"
    path.write_bytes(gzip.compress(header))
    return path


@pytest.fixture
def minimal_onnx() -> onnx.ModelProto:
    """Minimal valid ONNX model — single Identity node (X → Y)."""
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "minimal", [X], [Y])
    return helper.make_model(graph)
