from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from mjswan.source.wandb import fetch_onnx, resolve_artifact_path


class TestResolveWandbArtifactPath:
    def test_resolves_artifact_url_with_file_path(self):
        artifact_name, artifact_type, file_path = resolve_artifact_path(
            "https://wandb.ai/ttktjmt-org/csv_to_npz/artifacts/motions/"
            "mimickit_spinkick_safe/v0/files/motion.npz"
        )

        assert artifact_name == "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        assert artifact_type == "motions"
        assert file_path == "motion.npz"

    def test_resolves_fully_qualified_artifact_name(self):
        artifact_name, artifact_type, file_path = resolve_artifact_path(
            "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        )

        assert artifact_name == "ttktjmt-org/csv_to_npz/mimickit_spinkick_safe:v0"
        assert artifact_type == "motions"
        assert file_path == "motion.npz"


class _RunFile:
    """A run file that downloads the way wandb does: the name joined onto the root."""

    def __init__(self, name: str, model: onnx.ModelProto, **save) -> None:
        self.name, self.updated_at, self.model, self.save = name, "1", model, save
        self.downloaded = False

    def download(self, root: str, replace: bool = False) -> None:
        self.downloaded = True
        path = Path(root) / self.name
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(self.model, str(path), **self.save)


def _serve(monkeypatch, run_file: _RunFile) -> None:
    run = SimpleNamespace(files=lambda: [run_file])
    api = SimpleNamespace(run=lambda path: run)
    monkeypatch.setattr(
        "mjswan.source.wandb._wandb", lambda: SimpleNamespace(Api=lambda: api)
    )


class TestFetchOnnx:
    @pytest.mark.parametrize(
        "name",
        [
            "../walk.onnx",
            "/tmp/walk.onnx",
            r"..\..\walk.onnx",
            r"C:\walk.onnx",
            r"\walk.onnx",
            "C:walk.onnx",
        ],
    )
    def test_refuses_a_name_outside_the_download_dir(
        self, monkeypatch, minimal_onnx, name
    ):
        run_file = _RunFile(name, minimal_onnx)
        _serve(monkeypatch, run_file)

        with pytest.raises(ValueError, match="outside the download directory"):
            fetch_onnx("org/proj/run")
        assert not run_file.downloaded

    def test_loads_a_file_in_a_folder(self, monkeypatch, minimal_onnx):
        _serve(monkeypatch, _RunFile("exported/walk.onnx", minimal_onnx))

        name, model = fetch_onnx("org/proj/run")

        assert name == "walk"
        assert model.graph.node[0].op_type == "Identity"

    def test_reads_no_file_beside_it(self, monkeypatch):
        """External data names files the download never fetched, such as ~/.netrc."""
        weight = numpy_helper.from_array(np.zeros(1024, np.float32), "w")
        graph = helper.make_graph(
            [helper.make_node("Identity", ["w"], ["y"])],
            "g",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])],
            [weight],
        )
        _serve(
            monkeypatch,
            _RunFile(
                "walk.onnx",
                helper.make_model(graph),
                save_as_external_data=True,
                location="w.bin",
            ),
        )

        _, loaded = fetch_onnx("org/proj/run")

        assert loaded.graph.initializer[0].data_location == TensorProto.EXTERNAL
        assert not loaded.graph.initializer[0].raw_data
