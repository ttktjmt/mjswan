"""Where a traced graph lands, and what happens when two of them want one path."""

from __future__ import annotations

import pytest

from mjswan.build.mdp.graph import onnx_ref, write_onnx


class TestOnnxRef:
    def test_unscoped_ref_is_kind_and_name(self):
        assert onnx_ref("event", "push_robot") == "event/push_robot.onnx"

    def test_a_scope_puts_the_graph_under_its_owner(self):
        assert onnx_ref("obs", "policy", "walk") == "walk/obs/policy.onnx"

    def test_two_policies_naming_one_group_get_two_paths(self):
        # Both networks read a group of the same name, so the name alone cannot separate
        # them.
        assert onnx_ref("obs", "policy", "walk") != onnx_ref("obs", "policy", "crawl")

    @pytest.mark.parametrize(
        "name", ["a/b", "..", ".hidden", "a b", "a\\b", "%2e%2e", "", "jöint"]
    )
    def test_a_name_the_viewer_cannot_read_as_a_file_is_refused(self, name):
        """The viewer refuses a document naming such a file, so the build does first."""
        with pytest.raises(ValueError, match="Rename it"):
            onnx_ref("obs", name, "mdp/mdp_0")

    @pytest.mark.parametrize("name", ["base_lin_vel", "Track-Pose", "joint_pos.v2"])
    def test_a_term_name_is_its_files(self, name):
        assert onnx_ref("obs", name) == f"obs/{name}.onnx"


class TestWriteOnnx:
    def test_writes_the_bytes_at_the_ref(self, tmp_path):
        write_onnx(tmp_path, "walk/obs/policy.onnx", b"graph")
        assert (tmp_path / "walk" / "obs" / "policy.onnx").read_bytes() == b"graph"

    def test_rewriting_the_same_graph_passes_through(self, tmp_path):
        # One term traced twice within a build is not a collision.
        write_onnx(tmp_path, "obs/policy.onnx", b"graph")
        write_onnx(tmp_path, "obs/policy.onnx", b"graph")
        assert (tmp_path / "obs" / "policy.onnx").read_bytes() == b"graph"

    def test_a_different_graph_at_one_path_fails_the_build(self, tmp_path):
        write_onnx(tmp_path, "obs/policy.onnx", b"walk graph")
        with pytest.raises(ValueError, match="obs/policy.onnx"):
            write_onnx(tmp_path, "obs/policy.onnx", b"crawl graph")
        # The first writer's graph survives; the build stops rather than shipping either.
        assert (tmp_path / "obs" / "policy.onnx").read_bytes() == b"walk graph"
