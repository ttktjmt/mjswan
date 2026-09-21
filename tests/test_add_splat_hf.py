"""``SceneHandle.add_splat_hf`` end to end, with the Hub itself stubbed.

A splat is one opaque file, so unlike a scene there is nothing to compile and nothing
to resolve beside it. What these cover is that the downloaded file reaches ``source=``
— so the ``.spz`` is bundled and the deployed app needs no network — and that the
placement arguments arrive unchanged.
"""

from __future__ import annotations

import pytest

import mjswan

SPZ_BYTES = b"not really a splat"


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A Hub stub backed by a directory; returns the calls it saw."""
    cache = tmp_path / "cache"
    cache.mkdir()
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
            local = cache / filename.replace("/", "__")
            local.write_bytes(SPZ_BYTES)
            return str(local)

    monkeypatch.setattr("mjswan.source.hf._hub", lambda: _Hub)
    return calls


@pytest.fixture
def builder():
    return mjswan.Builder()


@pytest.fixture
def scene(builder, minimal_spec):
    return builder.add_project(name="Test").add_scene(spec=minimal_spec, name="Robot")


class TestAddSplatHf:
    def test_the_downloaded_file_becomes_the_bundled_source(self, scene, fake_hub):
        handle = scene.add_splat_hf("org/assets", "splats/street.spz")

        assert handle._config.url is None
        assert handle._config.source is not None
        from pathlib import Path

        assert Path(handle._config.source).read_bytes() == SPZ_BYTES

    def test_names_the_splat_after_its_file(self, scene, fake_hub):
        handle = scene.add_splat_hf("org/assets", "splats/street.spz")

        assert handle._config.name == "street"

    def test_a_generic_stem_falls_back_to_the_repository(self, scene, fake_hub):
        handle = scene.add_splat_hf("org/unitree-g1", "background.spz")

        assert handle._config.name == "unitree-g1"

    def test_an_explicit_name_wins(self, scene, fake_hub):
        handle = scene.add_splat_hf("org/assets", "splats/street.spz", name="Street")

        assert handle._config.name == "Street"

    def test_placement_arguments_are_forwarded(self, scene, fake_hub):
        handle = scene.add_splat_hf(
            "org/assets",
            "splats/street.spz",
            scale=3.275,
            x_offset=0.1,
            y_offset=0.2,
            z_offset=0.708,
            roll=1.0,
            pitch=2.0,
            yaw=40.0,
            control=True,
        )

        cfg = handle._config
        assert (cfg.scale, cfg.x_offset, cfg.y_offset, cfg.z_offset) == (
            3.275,
            0.1,
            0.2,
            0.708,
        )
        assert (cfg.roll, cfg.pitch, cfg.yaw) == (1.0, 2.0, 40.0)
        assert cfg.control is True

    def test_a_collider_url_passes_through(self, scene, fake_hub):
        """A collider is not bundled, so a Hub-hosted one is named by its URL."""
        url = "https://huggingface.co/org/assets/resolve/main/splats/street.glb"
        handle = scene.add_splat_hf("org/assets", "splats/street.spz", collider_url=url)

        assert handle._config.collider_url == url

    def test_the_revision_and_repo_type_reach_the_hub(self, scene, fake_hub):
        scene.add_splat_hf(
            "org/assets",
            "splats/street.spz",
            revision="abc123",
            repo_type="dataset",
            token="hf_x",
        )

        assert fake_hub["repo_id"] == "org/assets"
        assert fake_hub["filename"] == "splats/street.spz"
        assert fake_hub["revision"] == "abc123"
        assert fake_hub["repo_type"] == "dataset"
        assert fake_hub["token"] == "hf_x"

    def test_a_model_repository_is_the_default(self, scene, fake_hub):
        scene.add_splat_hf("org/assets", "splats/street.spz")

        assert fake_hub["repo_type"] == "model"

    def test_several_splats_share_one_scene(self, scene, fake_hub):
        scene.add_splat_hf("org/assets", "splats/street.spz")
        scene.add_splat_hf("org/assets", "splats/lab.spz")

        assert [s.id for s in scene._config.splats] == ["street", "lab"]


class TestBuildRoundTrip:
    def test_a_hub_splat_is_bundled(
        self, builder, scene, fake_hub, build_manifest, tmp_path
    ):
        scene.add_splat_hf("org/assets", "splats/street.spz", scale=3.275)

        manifest = build_manifest(builder, tmp_path / "dist")

        entry = manifest["projects"][0]["scenes"][0]["splats"][0]
        assert entry["path"] == "assets/street.spz"
        assert "url" not in entry
        bundled = tmp_path / "dist" / "test" / "robot" / "assets" / "street.spz"
        assert bundled.read_bytes() == SPZ_BYTES
