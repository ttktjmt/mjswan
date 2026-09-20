import pytest

from mjswan.source.hf import choose_policy_filename, policy_name_for


class TestChoosePolicyFilename:
    def test_prefers_policy_onnx(self):
        assert (
            choose_policy_filename(["final.onnx", "other.onnx", "policy.onnx"])
            == "policy.onnx"
        )

    def test_falls_back_to_final_onnx(self):
        assert choose_policy_filename(["final.onnx", "lineage/step_10.onnx"]) == (
            "final.onnx"
        )

    def test_takes_the_only_candidate(self):
        assert choose_policy_filename(["exports/walk.onnx"]) == "exports/walk.onnx"

    def test_refuses_to_guess_between_several(self):
        with pytest.raises(ValueError, match="Pass filename="):
            choose_policy_filename(["a.onnx", "b.onnx"])

    def test_names_the_ambiguous_candidates(self):
        with pytest.raises(ValueError) as excinfo:
            choose_policy_filename(["a.onnx", "b.onnx"], repo_id="org/repo")
        assert "a.onnx" in str(excinfo.value)
        assert "org/repo" in str(excinfo.value)

    def test_empty_repo_raises(self):
        with pytest.raises(ValueError, match="No .onnx file"):
            choose_policy_filename([])


class TestPolicyNameFor:
    def test_generic_stem_falls_back_to_the_repo_name(self):
        assert policy_name_for("HannesVonEssen/microduck-running", "policy.onnx") == (
            "microduck-running"
        )

    def test_final_onnx_is_generic_too(self):
        assert policy_name_for("org/g1-walk", "final.onnx") == "g1-walk"

    def test_specific_stem_wins(self):
        assert policy_name_for("org/microduck", "policies/roulade.onnx") == "roulade"

    def test_repo_id_without_owner(self):
        assert policy_name_for("microduck", "policy.onnx") == "microduck"
