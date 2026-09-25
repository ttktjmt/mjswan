from mjswan.source.wandb import resolve_artifact_path


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
