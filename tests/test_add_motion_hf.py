"""``PolicyHandle.add_motion_hf`` with the Hub itself stubbed.

Only ``source.hf._hub`` is replaced, so the download arguments, the name and the
payload all go through the real call.
"""

from __future__ import annotations

import pytest

import mjswan


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A Hub stub serving one clip; returns what the last download asked for."""
    clip = tmp_path / "spinkick.npz"
    clip.write_bytes(b"clip-bytes")
    calls: dict[str, object] = {}

    class _Hub:
        @staticmethod
        def hf_hub_download(
            repo_id, filename, revision=None, repo_type=None, token=None
        ):
            calls.update(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=token,
            )
            return str(clip)

    monkeypatch.setattr("mjswan.source.hf._hub", lambda: _Hub)
    return calls


@pytest.fixture
def policy(minimal_model, minimal_onnx):
    return (
        mjswan.Builder()
        .add_project(name="P")
        .add_scene(name="S", model=minimal_model, control_dt=0.02)
        .add_policy(name="track", policy=minimal_onnx, policy_joint_names=["a", "b"])
    )


def _add(policy, **kwargs):
    return policy.add_motion_hf(
        "org/motions",
        "clips/spinkick.npz",
        anchor_body_name="torso_link",
        body_names=("pelvis", "torso_link"),
        **kwargs,
    )


class TestAddMotionHf:
    def test_a_clip_is_fetched_from_a_dataset_repository_by_default(
        self, policy, fake_hub
    ):
        """Retargeted motion sets are published as datasets, not models."""
        _add(policy)

        assert fake_hub["repo_type"] == "dataset"
        assert (fake_hub["repo_id"], fake_hub["filename"]) == (
            "org/motions",
            "clips/spinkick.npz",
        )

    def test_the_clip_is_named_after_its_file_and_carries_its_bytes(
        self, policy, fake_hub
    ):
        motion = _add(policy)._config

        assert motion.name == "spinkick"
        assert motion.data == b"clip-bytes"
        assert motion.body_names == ("pelvis", "torso_link")

    def test_revision_repo_type_and_token_reach_the_download(self, policy, fake_hub):
        _add(policy, revision="9a1c2f0", repo_type="model", token="hf_x")

        assert (fake_hub["revision"], fake_hub["repo_type"], fake_hub["token"]) == (
            "9a1c2f0",
            "model",
            "hf_x",
        )

    def test_the_joint_order_defaults_to_the_policys(self, policy, fake_hub):
        assert _add(policy)._config.dataset_joint_names == ["a", "b"]
        given = _add(policy, name="reversed", dataset_joint_names=["b", "a"])
        assert given._config.dataset_joint_names == ["b", "a"]

    def test_an_explicit_name_and_default_are_kept(self, policy, fake_hub):
        motion = _add(policy, name="Spin Kick", default=True)._config

        assert (motion.name, motion.default) == ("Spin Kick", True)
