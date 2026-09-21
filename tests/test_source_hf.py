import pytest

from mjswan.source.hf import (
    choose_policy_filename,
    policy_name_for,
    scene_name_for,
    splat_name_for,
)


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


class TestSceneNameFor:
    """A scene is several files, so its directory is what identifies it."""

    def test_generic_stem_falls_back_to_the_directory(self):
        assert scene_name_for("org/assets", "scenes/unitree_g1/scene.xml") == (
            "unitree_g1"
        )

    def test_model_xml_is_generic_too(self):
        assert scene_name_for("org/assets", "robots/go2/model.xml") == "go2"

    def test_a_specific_stem_wins(self):
        assert scene_name_for("org/assets", "scenes/g1/g1_with_hands.xml") == (
            "g1_with_hands"
        )

    def test_a_generic_stem_at_the_root_falls_back_to_the_repo(self):
        assert scene_name_for("org/unitree-g1", "scene.xml") == "unitree-g1"


class TestSplatNameFor:
    """A splat is one file, so there is no directory to fall back on."""

    def test_a_specific_stem_names_the_capture(self):
        assert splat_name_for("org/assets", "splats/street.spz") == "street"

    def test_a_generic_stem_falls_back_to_the_repo(self):
        assert splat_name_for("org/unitree-g1", "background.spz") == "unitree-g1"

    def test_the_fallback_is_case_insensitive(self):
        assert splat_name_for("org/lab", "Splat.spz") == "lab"
