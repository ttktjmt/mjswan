"""Tests for the Gaussian Splat feature.

Layer breakdown:
  L1 (pure logic / lightweight I/O): TestSplatConfigToDict, TestAddSplat,
                                     TestSplatHandle, TestBuildSplatConfigDict,
                                     TestSaveConfigJsonSplats
  L3 slow (triggers frontend build): TestFullBuildSplat

Run only L1 tests (pre-commit):  pytest -m "not slow"
Run all tests (CI):               pytest
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from mjswan.builder import Builder
from mjswan.scene import SceneConfig
from mjswan.splat import SplatConfig
from mjswan.utils import name2id


# ===========================================================================
# L1 — SplatConfig.to_dict()
# ===========================================================================
class TestSplatConfigToDict:
    def test_required_keys_always_present(self):
        d = SplatConfig(name="Outdoor").to_dict()
        assert "name" in d
        assert "scale" in d
        assert "xOffset" in d
        assert "yOffset" in d
        assert "zOffset" in d

    def test_default_values(self):
        d = SplatConfig(name="Outdoor").to_dict()
        assert d["name"] == "Outdoor"
        assert d["scale"] == 1.0
        assert d["xOffset"] == 0.0
        assert d["yOffset"] == 0.0
        assert d["zOffset"] == 0.0

    def test_url_included_when_set(self):
        d = SplatConfig(name="S", url="https://example.com/bg.spz").to_dict()
        assert d["url"] == "https://example.com/bg.spz"

    def test_url_absent_when_none(self):
        d = SplatConfig(name="S").to_dict()
        assert "url" not in d

    def test_roll_omitted_when_zero(self):
        d = SplatConfig(name="S", roll=0.0).to_dict()
        assert "roll" not in d

    def test_roll_included_when_nonzero(self):
        d = SplatConfig(name="S", roll=45.0).to_dict()
        assert d["roll"] == 45.0

    def test_pitch_omitted_when_zero(self):
        assert "pitch" not in SplatConfig(name="S", pitch=0.0).to_dict()

    def test_pitch_included_when_nonzero(self):
        assert SplatConfig(name="S", pitch=-10.0).to_dict()["pitch"] == -10.0

    def test_yaw_omitted_when_zero(self):
        assert "yaw" not in SplatConfig(name="S", yaw=0.0).to_dict()

    def test_yaw_included_when_nonzero(self):
        assert SplatConfig(name="S", yaw=90.0).to_dict()["yaw"] == 90.0

    def test_collider_url_included_when_set(self):
        d = SplatConfig(name="S", collider_url="https://example.com/col.glb").to_dict()
        assert d["colliderUrl"] == "https://example.com/col.glb"

    def test_collider_url_absent_when_none(self):
        assert "colliderUrl" not in SplatConfig(name="S").to_dict()

    def test_control_included_when_true(self):
        assert SplatConfig(name="S", control=True).to_dict()["control"] is True

    def test_control_absent_when_false(self):
        assert "control" not in SplatConfig(name="S", control=False).to_dict()

    def test_path_not_in_dict(self):
        # path is injected externally by Builder._build_splat_config_dict
        d = SplatConfig(name="S", source="bg.spz").to_dict()
        assert "path" not in d


# ===========================================================================
# L1 — SceneHandle.add_splat() validation and wiring
# ===========================================================================
class TestAddSplat:
    def _make_scene(self, minimal_model):
        return Builder().add_project(name="P").add_scene(name="S", model=minimal_model)

    def test_raises_when_neither_source_nor_url(self, minimal_model):
        scene = self._make_scene(minimal_model)
        with pytest.raises(ValueError, match="source"):
            scene.add_splat("Outdoor")

    def test_raises_when_both_source_and_url(self, minimal_model):
        scene = self._make_scene(minimal_model)
        with pytest.raises(ValueError, match="not both"):
            scene.add_splat(
                "Outdoor", source="bg.spz", url="https://example.com/bg.spz"
            )

    def test_returns_splat_handle_with_url(self, minimal_model):
        from mjswan.splat import SplatHandle

        scene = self._make_scene(minimal_model)
        handle = scene.add_splat("Outdoor", url="https://example.com/bg.spz")
        assert isinstance(handle, SplatHandle)

    def test_returns_splat_handle_with_source(self, minimal_model):
        from mjswan.splat import SplatHandle

        scene = self._make_scene(minimal_model)
        handle = scene.add_splat("Outdoor", source="bg.spz")
        assert isinstance(handle, SplatHandle)

    def test_splat_appended_to_scene(self, minimal_model):
        scene = self._make_scene(minimal_model)
        scene.add_splat("A", url="https://example.com/a.spz")
        scene.add_splat("B", url="https://example.com/b.spz")
        assert len(scene._config.splats) == 2

    def test_handle_properties_reflect_args(self, minimal_model):
        scene = self._make_scene(minimal_model)
        handle = scene.add_splat(
            "Outdoor",
            url="https://example.com/bg.spz",
            scale=2.5,
            x_offset=0.1,
            y_offset=0.2,
            z_offset=0.3,
            roll=10.0,
            pitch=20.0,
            yaw=30.0,
        )
        assert handle.url == "https://example.com/bg.spz"
        assert handle.scale == 2.5
        assert handle.x_offset == 0.1
        assert handle.y_offset == 0.2
        assert handle.z_offset == 0.3
        assert handle.roll == 10.0
        assert handle.pitch == 20.0
        assert handle.yaw == 30.0


# ===========================================================================
# L1 — SplatHandle
# ===========================================================================
class TestSplatHandle:
    def _make_handle(self, minimal_model, **kwargs):
        scene = Builder().add_project(name="P").add_scene(name="S", model=minimal_model)
        return scene.add_splat("Outdoor", **kwargs)

    def test_source_property(self, minimal_model):
        handle = self._make_handle(minimal_model, source="bg.spz")
        assert handle.source == "bg.spz"

    def test_url_property(self, minimal_model):
        handle = self._make_handle(minimal_model, url="https://example.com/bg.spz")
        assert handle.url == "https://example.com/bg.spz"

    def test_set_metadata_stores_value(self, minimal_model):
        handle = self._make_handle(minimal_model, url="https://example.com/bg.spz")
        handle.set_metadata("capture_date", "2024-01-01")
        assert handle._config.metadata["capture_date"] == "2024-01-01"

    def test_set_metadata_returns_self(self, minimal_model):
        handle = self._make_handle(minimal_model, url="https://example.com/bg.spz")
        result = handle.set_metadata("key", "value")
        assert result is handle


# ===========================================================================
# L1 — Builder._build_splat_config_dict()
# ===========================================================================
class TestBuildSplatConfigDict:
    def _builder_and_scene(self, minimal_model):
        builder = Builder()
        scene_config = SceneConfig(name="My Scene", model=minimal_model)
        return builder, scene_config

    def test_source_splat_adds_path_key(self, minimal_model):
        builder, scene = self._builder_and_scene(minimal_model)
        splat = SplatConfig(name="My Splat", source="bg.spz")
        d = builder._build_splat_config_dict(scene, splat)
        assert "path" in d

    def test_source_splat_path_uses_name2id(self, minimal_model):
        builder, scene = self._builder_and_scene(minimal_model)
        splat = SplatConfig(name="My Splat", source="bg.spz")
        d = builder._build_splat_config_dict(scene, splat)
        expected = f"{name2id('My Scene')}/{name2id('My Splat')}.spz"
        assert d["path"] == expected

    def test_url_splat_has_no_path_key(self, minimal_model):
        builder, scene = self._builder_and_scene(minimal_model)
        splat = SplatConfig(name="My Splat", url="https://example.com/bg.spz")
        d = builder._build_splat_config_dict(scene, splat)
        assert "path" not in d

    def test_url_splat_has_url_key(self, minimal_model):
        builder, scene = self._builder_and_scene(minimal_model)
        splat = SplatConfig(name="My Splat", url="https://example.com/bg.spz")
        d = builder._build_splat_config_dict(scene, splat)
        assert d["url"] == "https://example.com/bg.spz"


# ===========================================================================
# L1 — splat entries in _save_config_json()
# ===========================================================================
class TestSaveConfigJsonSplats:
    def _read_scene(self, tmp_path: Path) -> dict:
        config = json.loads((tmp_path / "assets" / "config.json").read_text())
        return config["projects"][0]["scenes"][0]

    def test_no_splats_key_when_scene_has_no_splats(self, tmp_path, minimal_model):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        builder._save_config_json(tmp_path)
        assert "splats" not in self._read_scene(tmp_path)

    def test_url_splat_appears_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_splat("Outdoor", url="https://example.com/bg.spz")
        builder._save_config_json(tmp_path)
        splats = self._read_scene(tmp_path)["splats"]
        assert len(splats) == 1
        assert splats[0]["url"] == "https://example.com/bg.spz"

    def test_url_splat_has_no_path_key(self, tmp_path, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_splat("Outdoor", url="https://example.com/bg.spz")
        builder._save_config_json(tmp_path)
        assert "path" not in self._read_scene(tmp_path)["splats"][0]

    def test_source_splat_has_path_key(self, tmp_path, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_splat("Outdoor", source="bg.spz")
        builder._save_config_json(tmp_path)
        splat = self._read_scene(tmp_path)["splats"][0]
        assert splat["path"] == f"{name2id('S')}/{name2id('Outdoor')}.spz"

    def test_multiple_splats_all_in_config(self, tmp_path, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_splat("A", url="https://example.com/a.spz")
        scene.add_splat("B", url="https://example.com/b.spz")
        builder._save_config_json(tmp_path)
        splats = self._read_scene(tmp_path)["splats"]
        assert len(splats) == 2
        assert splats[0]["name"] == "A"
        assert splats[1]["name"] == "B"


# ===========================================================================
# L3 slow — full build pipeline
# Run with: pytest -m slow
# ===========================================================================
@pytest.mark.slow
class TestFullBuildSplat:
    def test_source_spz_copied_to_scene_dir(self, tmp_path, minimal_model, minimal_spz):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="My Scene", model=minimal_model
        )
        scene.add_splat("My Splat", source=str(minimal_spz))
        builder.build(tmp_path / "out")
        expected = (
            tmp_path
            / "out"
            / "main"
            / "assets"
            / name2id("My Scene")
            / f"{name2id('My Splat')}.spz"
        )
        assert expected.exists()

    def test_copied_spz_path_matches_config(self, tmp_path, minimal_model, minimal_spz):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(
            name="My Scene", model=minimal_model
        )
        scene.add_splat("My Splat", source=str(minimal_spz))
        builder.build(tmp_path / "out")

        config = json.loads((tmp_path / "out" / "assets" / "config.json").read_text())
        config_path = config["projects"][0]["scenes"][0]["splats"][0]["path"]
        spz_file = tmp_path / "out" / "main" / "assets" / config_path
        assert spz_file.exists()

    def test_missing_source_emits_runtime_warning(self, tmp_path, minimal_model):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=minimal_model)
        scene.add_splat("Outdoor", source="/nonexistent/path/bg.spz")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            builder.build(tmp_path / "out")
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert any("bg.spz" in str(w.message) for w in runtime_warnings)
