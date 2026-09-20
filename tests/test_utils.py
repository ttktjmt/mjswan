"""Tests for mjswan.utils — path normalization, XML rewriting, PNG encoding, ZIP generation.

Layer breakdown:
  L1 (pure Python, no MuJoCo): TestName2Id, TestStripLeadingDotdot,
                                TestMakeZipSafePath, TestRewriteXmlPaths,
                                TestBufferTextureToPng
  L2 (synthetic MuJoCo spec):  TestToZipDeflated, TestCollectSpecAssets
"""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest

from mjswan.document.ids import name2id, unique_id
from mjswan.utils import (
    _buffer_texture_to_png,
    _make_zip_safe_path,
    _rewrite_xml_paths,
    _strip_leading_dotdot,
    collect_spec_assets,
    to_zip_deflated,
)

# The frontend's `sanitizeName` reads the same table (src/manifest/index.test.ts): a URL
# is resolved by that function and a directory named by this one, so a case they disagree
# on is a link that opens the wrong scene (ADR 0006 §4).
NAME2ID_CASES = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "src/mjswan/template/src/manifest/name2id_cases.json"
    ).read_text()
)


# ===========================================================================
# L1 — name2id
# ===========================================================================
class TestName2Id:
    def test_spaces_become_underscores(self):
        assert name2id("My Project") == "my_project"

    def test_hyphens_become_underscores(self):
        assert name2id("Test-Scene") == "test_scene"

    def test_uppercase_becomes_lowercase(self):
        assert name2id("FooBar") == "foobar"

    def test_mixed_separators(self):
        assert name2id("Complex Name-With Spaces") == "complex_name_with_spaces"

    def test_already_clean_unchanged(self):
        assert name2id("simple") == "simple"

    @pytest.mark.parametrize(("name", "expected"), NAME2ID_CASES)
    def test_agrees_with_the_frontend_on_the_shared_table(self, name, expected):
        assert name2id(name) == expected


# ===========================================================================
# L1 — unique_id
# ===========================================================================
class TestUniqueId:
    def test_free_base_is_returned_as_is(self):
        assert unique_id("flat_terrain", set()) == "flat_terrain"

    def test_a_taken_base_gets_the_first_free_suffix(self):
        assert unique_id("flat_terrain", {"flat_terrain"}) == "flat_terrain_1"
        assert (
            unique_id("flat_terrain", {"flat_terrain", "flat_terrain_1"})
            == "flat_terrain_2"
        )

    def test_a_gap_in_the_suffixes_is_filled_first(self):
        assert (
            unique_id("s", {"s", "s_2"}) == "s_1"
        )  # `_1` is free, so it is taken before `_3`


# ===========================================================================
# L1 — _strip_leading_dotdot
# ===========================================================================
class TestStripLeadingDotdot:
    def test_no_dotdot_unchanged(self):
        assert _strip_leading_dotdot("meshes/foo.stl") == "meshes/foo.stl"

    def test_single_dotdot_stripped(self):
        assert _strip_leading_dotdot("../meshes/foo.stl") == "meshes/foo.stl"

    def test_multiple_dotdot_stripped(self):
        assert (
            _strip_leading_dotdot("../../myo_sim/meshes/clavicle.stl")
            == "myo_sim/meshes/clavicle.stl"
        )

    def test_dotdot_within_path_normalized(self):
        # a/b/../../c.stl  →  c.stl  (no leading ..)
        assert _strip_leading_dotdot("a/b/../../c.stl") == "c.stl"

    def test_only_dotdot_returns_empty(self):
        assert _strip_leading_dotdot("../..") == ""

    def test_empty_string_unchanged(self):
        assert _strip_leading_dotdot("") == ""

    def test_current_dir_prefix_stripped(self):
        assert _strip_leading_dotdot("./meshes/foo.stl") == "meshes/foo.stl"

    def test_deep_dotdot_chain_stripped(self):
        assert (
            _strip_leading_dotdot("../../../../simhive/myo_sim/meshes/a.stl")
            == "simhive/myo_sim/meshes/a.stl"
        )


# ===========================================================================
# L1 — _make_zip_safe_path
# ===========================================================================
class TestMakeZipSafePath:
    def test_clean_relative_path_unchanged(self):
        assert _make_zip_safe_path("meshes/foo.stl") == "meshes/foo.stl"

    def test_relative_with_dotdot_stripped(self):
        assert _make_zip_safe_path("../meshes/foo.stl") == "meshes/foo.stl"

    def test_absolute_unix_path_becomes_basename(self):
        assert _make_zip_safe_path("/Users/alice/models/skybox.png") == "skybox.png"

    def test_absolute_deep_path_becomes_basename(self):
        assert (
            _make_zip_safe_path(
                "/home/user/.venv/lib/python3.12/site-packages/pkg/model/tex.png"
            )
            == "tex.png"
        )

    def test_windows_absolute_path_becomes_basename(self):
        assert _make_zip_safe_path("C:/Users/bob/models/mesh.stl") == "mesh.stl"

    def test_empty_string_unchanged(self):
        assert _make_zip_safe_path("") == ""

    def test_only_dotdot_returns_empty(self):
        assert _make_zip_safe_path("../..") == ""


# ===========================================================================
# L1 — _rewrite_xml_paths
# ===========================================================================
class TestRewriteXmlPaths:
    def test_meshdir_removed_from_compiler(self):
        xml = '<mujoco><compiler meshdir="../" angle="radian"/></mujoco>'
        result = _rewrite_xml_paths(xml, mesh_dir="../", texture_dir="")
        compiler = ET.fromstring(result).find("compiler")
        assert "meshdir" not in compiler.attrib
        assert compiler.get("angle") == "radian"  # other attrs preserved

    def test_texturedir_removed_from_compiler(self):
        xml = '<mujoco><compiler texturedir="../"/></mujoco>'
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="../")
        assert "texturedir" not in ET.fromstring(result).find("compiler").attrib

    def test_mesh_file_path_prepended_and_dotdot_stripped(self):
        xml = (
            '<mujoco><compiler meshdir="../"/>'
            '<asset><mesh name="m" file="myo_sim/meshes/foo.stl"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml, mesh_dir="../", texture_dir="")
        mesh = ET.fromstring(result).find(".//mesh")
        assert not mesh.get("file").startswith("..")

    def test_texture_file_dotdot_stripped(self):
        xml = (
            '<mujoco><compiler texturedir="../"/>'
            '<asset><texture name="t" file="../scene/tex.png"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="../")
        tex = ET.fromstring(result).find(".//texture")
        assert tex.get("file") == "scene/tex.png"

    def test_absolute_mesh_path_becomes_basename(self):
        abs_path = "/home/user/.venv/lib/site-packages/pkg/meshes/robot.stl"
        xml = (
            f'<mujoco><compiler meshdir=".."/>'
            f'<asset><mesh name="r" file="{abs_path}"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml, mesh_dir="..", texture_dir="")
        assert ET.fromstring(result).find(".//mesh").get("file") == "robot.stl"

    def test_absolute_texture_path_becomes_basename(self):
        abs_path = "/Users/alice/.venv/lib/site-packages/pkg/scene/skybox.png"
        xml = (
            f'<mujoco><compiler texturedir=".."/>'
            f'<asset><texture name="sky" file="{abs_path}"/></asset></mujoco>'
        )
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="..")
        assert ET.fromstring(result).find(".//texture").get("file") == "skybox.png"

    def test_hfield_file_dotdot_stripped(self):
        xml = '<mujoco><asset><hfield name="h" file="../terrain.png"/></asset></mujoco>'
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="")
        assert ET.fromstring(result).find(".//hfield").get("file") == "terrain.png"

    def test_mesh_without_file_attr_left_alone(self):
        xml = '<mujoco><asset><mesh name="m"/></asset></mujoco>'
        result = _rewrite_xml_paths(xml, mesh_dir="../", texture_dir="")
        assert ET.fromstring(result).find(".//mesh").get("file") is None

    def test_classless_nested_default_removed(self):
        xml = (
            "<mujoco><default>"
            "<default/>"
            '<default class="robot"><geom density="100"/></default>'
            "</default></mujoco>"
        )
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="")
        outer = ET.fromstring(result).find("default")
        for child in outer:
            if child.tag == "default":
                assert child.get("class"), "Classless nested <default> must be removed"

    def test_root_default_without_class_preserved(self):
        xml = (
            "<mujoco><default>"
            '<default class="robot"><geom density="100"/></default>'
            "</default></mujoco>"
        )
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="")
        assert ET.fromstring(result).find("default") is not None

    def test_repeated_default_class_merged(self):
        xml = (
            "<mujoco><default>"
            '<default class="robot">'
            '<default class="robot"><geom density="100"/></default>'
            "</default>"
            "</default></mujoco>"
        )
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="")
        outer = ET.fromstring(result).find("default")
        robot = next(c for c in outer if c.tag == "default")
        assert robot.get("class") == "robot"
        for child in robot:
            if child.tag == "default":
                assert child.get("class") != "robot", "Duplicate class must be merged"


# ===========================================================================
# L1 — _buffer_texture_to_png
# ===========================================================================
class TestBufferTextureToPng:
    def test_rgb_output_has_png_signature(self):
        png = _buffer_texture_to_png(bytes(4 * 4 * 3), 4, 4)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_rgba_produces_bytes(self):
        png = _buffer_texture_to_png(bytes(4 * 4 * 4), 4, 4)
        assert len(png) > 0

    def test_zero_area_raises(self):
        with pytest.raises(ValueError, match="zero-area"):
            _buffer_texture_to_png(b"", 0, 0)

    def test_unsupported_channels_raises(self):
        with pytest.raises(ValueError, match="unsupported channel count"):
            _buffer_texture_to_png(bytes(5), 1, 1)  # 5 channels = unsupported

    def test_mismatched_data_length_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            _buffer_texture_to_png(bytes(7), 2, 2)  # 7 / 4 pixels is not integer


# ===========================================================================
# L2 — to_zip_deflated (synthetic MjSpec, no external asset files)
# ===========================================================================
class TestToZipDeflated:
    def test_compression_type_is_deflate(self, minimal_spec):
        buf = io.BytesIO()
        to_zip_deflated(minimal_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            assert all(
                info.compress_type == zipfile.ZIP_DEFLATED for info in zf.infolist()
            )

    def test_zip_contains_xml_entry(self, minimal_spec):
        buf = io.BytesIO()
        to_zip_deflated(minimal_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            assert "simple.xml" in zf.namelist()

    def test_xml_has_no_meshdir_attribute(self, minimal_spec):
        buf = io.BytesIO()
        to_zip_deflated(minimal_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            xml = zf.read("simple.xml").decode()
            for compiler in ET.fromstring(xml).iter("compiler"):
                assert "meshdir" not in compiler.attrib

    def test_no_dotdot_in_zip_entry_names(self, minimal_spec):
        buf = io.BytesIO()
        to_zip_deflated(minimal_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                assert ".." not in name, f"ZIP entry contains '..': {name}"

    def test_no_absolute_paths_in_zip_entry_names(self, minimal_spec):
        buf = io.BytesIO()
        to_zip_deflated(minimal_spec, buf)
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                assert not name.startswith("/"), f"ZIP entry is absolute: {name}"

    def test_write_to_file_path_creates_file(self, minimal_spec, tmp_path: Path):
        out = str(tmp_path / "sub" / "model.mjz")
        to_zip_deflated(minimal_spec, out)
        assert Path(out).is_file()

    def test_spec_not_mutated_after_call(self, minimal_spec):
        spec = minimal_spec
        spec.to_xml()  # trigger any MuJoCo-internal normalisation first
        orig_meshdir = spec.meshdir
        orig_texturedir = spec.texturedir
        orig_mesh_files = [(m.name, m.file) for m in spec.meshes]

        to_zip_deflated(spec, io.BytesIO())

        assert spec.meshdir == orig_meshdir
        assert spec.texturedir == orig_texturedir
        assert [(m.name, m.file) for m in spec.meshes] == orig_mesh_files

    def test_falls_back_to_basename_for_spec_assets(self):
        class FakeMesh:
            def __init__(self, file: str):
                self.file = file

        class FakeSpec:
            modelname = "fake"
            modelfiledir = ""
            meshdir = ""
            texturedir = ""
            textures = []
            hfields = []
            skins = []
            assets = {"clavicle.stl": b"mesh-bytes"}

            def __init__(self):
                self.meshes = [FakeMesh("../myo_sim/meshes/clavicle.stl")]

            def to_xml(self) -> str:
                return (
                    '<mujoco model="fake"><asset>'
                    '<mesh name="clavicle" file="../myo_sim/meshes/clavicle.stl"/>'
                    "</asset></mujoco>"
                )

        buf = io.BytesIO()
        to_zip_deflated(FakeSpec(), buf)
        buf.seek(0)

        with zipfile.ZipFile(buf) as zf:
            assert "myo_sim/meshes/clavicle.stl" in zf.namelist()
            assert zf.read("myo_sim/meshes/clavicle.stl") == b"mesh-bytes"
            xml = zf.read("fake.xml").decode()
            mesh = ET.fromstring(xml).find(".//mesh")
            assert mesh.get("file") == "myo_sim/meshes/clavicle.stl"


# ===========================================================================
# L2 — collect_spec_assets (synthetic MjSpec)
# ===========================================================================
class TestCollectSpecAssets:
    def test_returns_empty_dict_for_spec_with_no_assets(self, minimal_spec):
        assets = collect_spec_assets(minimal_spec)
        assert assets == {}

    def test_return_type_is_dict_of_bytes(self, minimal_spec):
        assets = collect_spec_assets(minimal_spec)
        assert isinstance(assets, dict)
        assert all(isinstance(v, bytes) for v in assets.values())
