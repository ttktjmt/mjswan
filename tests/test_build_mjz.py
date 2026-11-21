"""Tests for mjswan.build.mjz: path normalization, XML rewriting, PNG encoding, ZIP generation.

Layer breakdown:
  L1 (pure Python, no MuJoCo): TestStripLeadingDotdot, TestMakeZipSafePath,
                                TestRewriteXmlPaths, TestDefaultsUnreadable,
                                TestBufferTextureToPng
  L2 (synthetic MuJoCo spec):  TestToZipDeflated, TestAttachedModels,
                                TestDefaultClasses, TestCollectSpecAssets
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjswan.build.mjz import (
    _TENDON_FIELDS,
    _absolute_entry_names,
    _buffer_texture_to_png,
    _defaults_unreadable,
    _drop_empty_nameless_defaults,
    _make_zip_safe_path,
    _rewrite_xml_paths,
    _strip_leading_dotdot,
    collect_spec_assets,
    to_zip_deflated,
)


# ===========================================================================
# L1: _strip_leading_dotdot
# ===========================================================================
class TestStripLeadingDotdot:
    def test_no_dotdot_unchanged(self):
        assert _strip_leading_dotdot("meshes/foo.stl") == "meshes/foo.stl"

    def test_single_dotdot_stripped(self):
        assert _strip_leading_dotdot("../meshes/foo.stl") == "meshes/foo.stl"

    def test_multiple_dotdot_stripped(self):
        assert (
            _strip_leading_dotdot("../../models/meshes/arm.stl")
            == "models/meshes/arm.stl"
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
            _strip_leading_dotdot("../../../../shared/models/meshes/a.stl")
            == "shared/models/meshes/a.stl"
        )


# ===========================================================================
# L1: _make_zip_safe_path
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
# L1: _absolute_entry_names
# ===========================================================================
class TestAbsoluteEntryNames:
    def test_a_unique_basename_stays_a_basename(self):
        assert _absolute_entry_names(["/models/bowl/textured.obj"]) == {
            "/models/bowl/textured.obj": "textured.obj"
        }

    def test_shared_basenames_take_the_shortest_tail_that_differs(self):
        names = _absolute_entry_names(
            [
                "/ycb/024_bowl/google_64k/meshes/textured.obj",
                "/ycb/011_banana/google_64k/meshes/textured.obj",
            ]
        )
        assert names == {
            "/ycb/024_bowl/google_64k/meshes/textured.obj": (
                "024_bowl/google_64k/meshes/textured.obj"
            ),
            "/ycb/011_banana/google_64k/meshes/textured.obj": (
                "011_banana/google_64k/meshes/textured.obj"
            ),
        }

    def test_one_file_named_twice_is_one_entry(self):
        names = _absolute_entry_names(["/a/mesh.stl", "/a/./mesh.stl"])
        assert names == {"/a/mesh.stl": "mesh.stl"}

    def test_a_path_that_is_another_ones_tail_stays_relative(self):
        names = _absolute_entry_names(["/a/x.stl", "/b/a/x.stl"])
        assert names == {"/a/x.stl": "a/x.stl", "/b/a/x.stl": "b/a/x.stl"}

    def test_relative_paths_are_left_out(self):
        assert _absolute_entry_names(["meshes/foo.stl", "../bar.png"]) == {}


# ===========================================================================
# L1: _rewrite_xml_paths
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
            '<asset><mesh name="m" file="models/meshes/foo.stl"/></asset></mujoco>'
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

    def test_recovered_entry_replaces_the_path(self):
        xml = (
            '<mujoco><compiler meshdir="meshes"/><asset>'
            '<mesh name="a" file="bone.stl"/><mesh name="b" file="bone.stl"/>'
            "</asset></mujoco>"
        )
        result = _rewrite_xml_paths(
            xml,
            mesh_dir="meshes",
            texture_dir="",
            recovered={(1, "file"): "bone_0.stl"},
        )
        files = [m.get("file") for m in ET.fromstring(result).iter("mesh")]
        assert files == ["meshes/bone.stl", "bone_0.stl"]

    def test_defaults_are_left_as_written(self):
        xml = '<mujoco><default><default class="robot"/><default/></default></mujoco>'
        result = _rewrite_xml_paths(xml, mesh_dir="", texture_dir="")
        nested = ET.fromstring(result).find("default").findall("default")
        assert [d.get("class") for d in nested] == ["robot", None]


class TestDefaultsUnreadable:
    def test_named_classes_read_back(self):
        xml = (
            "<mujoco><default>"
            '<default class="arm"><default class="hand"/></default>'
            "</default></mujoco>"
        )
        assert not _defaults_unreadable(ET.fromstring(xml))

    def test_a_nested_class_without_a_name_does_not(self):
        xml = "<mujoco><default><default/></default></mujoco>"
        assert _defaults_unreadable(ET.fromstring(xml))

    def test_a_class_named_twice_does_not(self):
        xml = (
            "<mujoco><default>"
            '<default class="arm"/><default class="body"><default class="arm"/></default>'
            "</default></mujoco>"
        )
        assert _defaults_unreadable(ET.fromstring(xml))

    def test_empty_nameless_classes_go_however_deep(self):
        xml = (
            "<mujoco><default>"
            '<default class="arm"/><default><default><default/></default></default>'
            "</default></mujoco>"
        )
        root = ET.fromstring(xml)

        _drop_empty_nameless_defaults(root)

        assert [d.get("class") for d in root.iter("default")] == [None, "arm"]
        assert not _defaults_unreadable(root)

    def test_a_nameless_class_that_sets_something_stays(self):
        xml = '<mujoco><default><default><geom group="2"/></default></default></mujoco>'
        root = ET.fromstring(xml)

        _drop_empty_nameless_defaults(root)

        assert _defaults_unreadable(root)


# ===========================================================================
# L1: _buffer_texture_to_png
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
# L2: to_zip_deflated (synthetic MjSpec, no external asset files)
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

    def test_attached_meshes_sharing_a_basename_keep_their_own_files(
        self, tmp_path: Path
    ):
        """Two attached models that each ship a ``part.obj`` (YCB objects do)."""
        tetra = "v 0 0 0\nv {s} 0 0\nv 0 {s} 0\nv 0 0 {s}\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
        scene = mujoco.MjSpec()
        for name, size in (("small", 0.1), ("large", 0.3)):
            mesh_dir = tmp_path / name / "meshes"
            mesh_dir.mkdir(parents=True)
            (mesh_dir / "part.obj").write_text(tetra.format(s=size))
            child = mujoco.MjSpec()
            child.add_mesh(name="part", file=str(mesh_dir / "part.obj"))
            body = child.worldbody.add_body(name="body")
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="part")
            scene.attach(child, prefix=f"{name}/", frame=scene.worldbody.add_frame())

        out = tmp_path / "scene.mjz"
        to_zip_deflated(scene, str(out))

        with zipfile.ZipFile(out) as zf:
            assert {"small/meshes/part.obj", "large/meshes/part.obj"} <= set(
                zf.namelist()
            )
        model = mujoco.MjSpec.from_zip(str(out)).compile()
        extents = {
            model.mesh(i).name: float(
                model.mesh_vert[
                    model.mesh_vertadr[i] : model.mesh_vertadr[i]
                    + model.mesh_vertnum[i]
                ].max()
            )
            for i in range(model.nmesh)
        }
        assert extents["large/part"] > 2 * extents["small/part"]

    def test_falls_back_to_basename_for_spec_assets(
        self, monkeypatch: pytest.MonkeyPatch
    ):
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
            bodies = []
            frames = []
            assets = {"arm.stl": b"mesh-bytes"}

            def __init__(self):
                self.meshes = [FakeMesh("../models/meshes/arm.stl")]

            def to_xml(self) -> str:
                return (
                    '<mujoco model="fake"><asset>'
                    '<mesh name="arm" file="../models/meshes/arm.stl"/>'
                    "</asset></mujoco>"
                )

        packed = '<mujoco><asset><mesh name="arm" file="arm.stl"/></asset></mujoco>'
        monkeypatch.setattr(
            "mjswan.build.mjz._encode",
            lambda spec, extra: (ET.fromstring(packed), {"arm.stl": b"mesh-bytes"}),
        )
        buf = io.BytesIO()
        to_zip_deflated(FakeSpec(), buf)
        buf.seek(0)

        with zipfile.ZipFile(buf) as zf:
            assert "models/meshes/arm.stl" in zf.namelist()
            assert zf.read("models/meshes/arm.stl") == b"mesh-bytes"
            xml = zf.read("fake.xml").decode()
            mesh = ET.fromstring(xml).find(".//mesh")
            assert mesh.get("file") == "models/meshes/arm.stl"


# ===========================================================================
# L2: to_zip_deflated on attached models (synthetic MjSpec, files on disk)
# ===========================================================================
TETRA = "v 0 0 0\nv {s} 0 0\nv 0 {s} 0\nv 0 0 {s}\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def _arm_on_disk(folder: Path, size: float) -> mujoco.MjSpec:
    """A model in its own folder whose mesh is found through a relative meshdir."""
    (folder / "meshes").mkdir(parents=True)
    (folder / "meshes" / "bone.obj").write_text(TETRA.format(s=size))
    (folder / "arm.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/>'
        '<asset><mesh name="bone" file="bone.obj"/></asset>'
        '<worldbody><body name="upper"><geom type="mesh" mesh="bone"/></body></worldbody>'
        "</mujoco>"
    )
    return mujoco.MjSpec.from_file(str(folder / "arm.xml"))


def _round_trip(spec: mujoco.MjSpec) -> mujoco.MjModel:
    buf = io.BytesIO()
    to_zip_deflated(spec, buf)
    buf.seek(0)
    return mujoco.MjSpec.from_zip(buf).compile()


def _mesh_extent(model: mujoco.MjModel, name: str) -> float:
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
    start = model.mesh_vertadr[i]
    return float(model.mesh_vert[start : start + model.mesh_vertnum[i]].max())


class TestAttachedModels:
    def test_meshes_of_a_model_attached_from_another_folder_are_packed(
        self, tmp_path: Path
    ):
        scene = mujoco.MjSpec()
        arm = _arm_on_disk(tmp_path / "arm", 0.2)
        scene.attach(arm, prefix="a/", frame=scene.worldbody.add_frame())
        reference = scene.compile()

        model = _round_trip(scene)

        assert _mesh_extent(model, "a/bone") == pytest.approx(
            _mesh_extent(reference, "a/bone")
        )

    def test_meshes_with_the_same_relative_path_keep_their_own_files(
        self, tmp_path: Path
    ):
        scene = mujoco.MjSpec()
        for name, size in (("left", 0.1), ("right", 0.3)):
            arm = _arm_on_disk(tmp_path / name, size)
            scene.attach(arm, prefix=f"{name}/", frame=scene.worldbody.add_frame())
        reference = scene.compile()

        model = _round_trip(scene)

        for name in ("left/bone", "right/bone"):
            assert _mesh_extent(model, name) == pytest.approx(
                _mesh_extent(reference, name)
            )
        assert _mesh_extent(model, "right/bone") > 2 * _mesh_extent(model, "left/bone")

    def test_a_buffer_texture_and_an_attached_mesh_are_both_packed(
        self, tmp_path: Path
    ):
        scene = mujoco.MjSpec()
        texture = scene.add_texture(
            name="checker", type=mujoco.mjtTexture.mjTEXTURE_2D, width=2, height=2
        )
        texture.nchannel = 3
        texture.data = bytes(range(12))
        scene.add_material(name="paint", textures=["", "checker"])
        scene.worldbody.add_geom(size=[0.1, 0, 0], material="paint")
        arm = _arm_on_disk(tmp_path / "arm", 0.2)
        scene.attach(arm, prefix="a/", frame=scene.worldbody.add_frame())
        reference = scene.compile()
        assets = dict(scene.assets)

        model = _round_trip(scene)

        np.testing.assert_array_equal(model.tex_data, reference.tex_data)
        assert _mesh_extent(model, "a/bone") == pytest.approx(
            _mesh_extent(reference, "a/bone")
        )
        assert dict(scene.assets) == assets

    def test_a_file_under_the_same_path_in_the_scenes_folder_is_not_taken(
        self, tmp_path: Path
    ):
        scene = _arm_on_disk(tmp_path / "scene", 0.1)
        arm = _arm_on_disk(tmp_path / "arm", 0.3)
        scene.attach(arm, prefix="a/", frame=scene.worldbody.add_frame())
        reference = scene.compile()

        model = _round_trip(scene)

        for name in ("bone", "a/bone"):
            assert _mesh_extent(model, name) == pytest.approx(
                _mesh_extent(reference, name)
            )

    def test_a_stray_file_under_the_same_path_is_left_out(self, tmp_path: Path):
        stray = TETRA.format(s=0.1)
        (tmp_path / "scene" / "meshes").mkdir(parents=True)
        (tmp_path / "scene" / "meshes" / "bone.obj").write_text(stray)
        (tmp_path / "scene" / "scene.xml").write_text(
            '<mujoco><compiler meshdir="meshes"/><worldbody/></mujoco>'
        )
        scene = mujoco.MjSpec.from_file(str(tmp_path / "scene" / "scene.xml"))
        arm = _arm_on_disk(tmp_path / "arm", 0.3)
        scene.attach(arm, prefix="a/", frame=scene.worldbody.add_frame())
        reference = scene.compile()
        buf = io.BytesIO()

        to_zip_deflated(scene, buf)

        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            assert stray.encode() not in [zf.read(n) for n in zf.namelist()]
        buf.seek(0)
        model = mujoco.MjSpec.from_zip(buf).compile()
        assert _mesh_extent(model, "a/bone") == pytest.approx(
            _mesh_extent(reference, "a/bone")
        )

    def test_files_the_encoder_cannot_check_are_packed_with_a_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        def refuse(spec: mujoco.MjSpec, extra: dict[str, bytes]):
            raise RuntimeError("no encoder")

        monkeypatch.setattr("mjswan.build.mjz._encode", refuse)

        with pytest.warns(RuntimeWarning, match="not checked"):
            model = _round_trip(_arm_on_disk(tmp_path / "arm", 0.2))

        assert model.nmesh == 1

    def test_a_file_nothing_can_find_raises(self, monkeypatch: pytest.MonkeyPatch):
        class FakeMesh:
            file = "gone.stl"

        class FakeSpec:
            modelname = "fake"
            modelfiledir = meshdir = texturedir = ""
            meshes = [FakeMesh()]
            textures = hfields = skins = bodies = frames = []
            assets: dict[str, bytes] = {}

            def to_xml(self) -> str:
                return (
                    '<mujoco><asset><mesh name="m" file="gone.stl"/></asset></mujoco>'
                )

        empty = (ET.fromstring("<mujoco/>"), {})
        monkeypatch.setattr("mjswan.build.mjz._encode", lambda spec, extra: empty)

        with pytest.raises(FileNotFoundError, match="gone.stl"):
            to_zip_deflated(FakeSpec(), io.BytesIO())


ARM_WITH_CLASS = """
<mujoco>
  <default><default class="arm"><geom group="{group}" rgba="{rgba}"/>
    <tendon stiffness="{stiffness}" damping="{damping}" frictionloss="0.1" limited="true"
      range="-1 1" margin="0.02" springlength="0.3"/></default></default>
  <worldbody><body name="upper_{side}" childclass="arm"><joint name="j_{side}"/>
    <geom name="g_{side}" size="0.1"/></body></worldbody>
  <tendon><fixed name="t_{side}" class="arm"><joint joint="j_{side}" coef="1"/></fixed></tendon>
</mujoco>
"""


def _arm(side: str, group=1, rgba="1 0 0 1", stiffness=5, damping=0.5) -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        ARM_WITH_CLASS.format(
            side=side, group=group, rgba=rgba, stiffness=stiffness, damping=damping
        )
    )


class TestDefaultClasses:
    def _two_arms_without_a_prefix(self) -> mujoco.MjSpec:
        """Two arms attached without a prefix, whose classes share names but not values."""
        scene = mujoco.MjSpec()
        for arm in (_arm("r"), _arm("l", 2, "0 0 1 1", 7, 0.9)):
            scene.attach(arm, prefix="", suffix="", frame=scene.worldbody.add_frame())
        return scene

    def test_classes_named_twice_load_back_with_their_values(self):
        scene = self._two_arms_without_a_prefix()
        reference = scene.compile()

        model = _round_trip(scene)

        for side in ("r", "l"):
            ref, got = reference.geom(f"g_{side}"), model.geom(f"g_{side}")
            assert got.group == ref.group
            np.testing.assert_allclose(got.rgba, ref.rgba)
            got_t, ref_t = (
                model.tendon(f"t_{side}").id,
                reference.tendon(f"t_{side}").id,
            )
            for field in (f for f in dir(reference) if f.startswith("tendon_")):
                np.testing.assert_allclose(
                    getattr(model, field)[got_t],
                    getattr(reference, field)[ref_t],
                    err_msg=field,
                )
        assert reference.geom("g_r").group != reference.geom("g_l").group
        assert reference.tendon_damping[0] != reference.tendon_damping[1]

    def test_every_field_of_a_tendon_is_written_out(self):
        """A field MuJoCo adds to MjsTendon fails here until _TENDON_FIELDS has it."""
        not_from_a_class = {
            "compiler",
            "default",
            "id",
            "info",
            "name",
            "path",
            "signature",
            "wrap_geom",
            "wrap_joint",
            "wrap_pulley",
            "wrap_site",
        }
        fields = {f for f in dir(mujoco.MjsTendon) if not f.startswith("_")}

        assert fields - not_from_a_class == set(_TENDON_FIELDS.values())

    def test_writing_them_out_leaves_the_spec_as_it_was(self):
        scene = self._two_arms_without_a_prefix()

        to_zip_deflated(scene, io.BytesIO())

        assert scene.to_xml() == self._two_arms_without_a_prefix().to_xml()

    def test_a_root_class_inside_another_childclass_keeps_its_values(self):
        scene = mujoco.MjSpec.from_string(
            '<mujoco><default><default class="torso"><site group="3"/></default></default>'
            '<worldbody><body name="pelvis" childclass="torso">'
            '<body name="arm" childclass="main"><site name="shoulder"/></body>'
            "</body></worldbody></mujoco>"
        )

        model = _round_trip(scene)

        assert model.site("shoulder").group == 0

    def test_classes_that_load_back_are_kept(self, tmp_path: Path):
        scene = _arm("r")
        out = tmp_path / "arm.mjz"

        to_zip_deflated(scene, str(out))

        with zipfile.ZipFile(out) as zf:
            root = ET.fromstring(zf.read(f"{scene.modelname}.xml"))
        assert [d.get("class") for d in root.iter("default")] == [None, "arm"]

    def test_an_empty_class_left_by_an_unprefixed_attach_is_dropped(
        self, tmp_path: Path
    ):
        """mjlab attaches its terrain so; the classes that do load back stay."""
        scene = _arm("r")
        floor = mujoco.MjSpec.from_string(
            '<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/>'
            "</worldbody></mujoco>"
        )
        scene.attach(floor, prefix="", suffix="", frame=scene.worldbody.add_frame())
        reference = scene.compile()
        out = tmp_path / "scene.mjz"

        to_zip_deflated(scene, str(out))

        with zipfile.ZipFile(out) as zf:
            root = ET.fromstring(zf.read(f"{scene.modelname}.xml"))
        assert [d.get("class") for d in root.iter("default")] == [None, "arm"]
        model = mujoco.MjSpec.from_zip(str(out)).compile()
        assert model.geom("g_r").group == reference.geom("g_r").group


# ===========================================================================
# L2: collect_spec_assets (synthetic MjSpec)
# ===========================================================================
class TestCollectSpecAssets:
    def test_returns_empty_dict_for_spec_with_no_assets(self, minimal_spec):
        assets = collect_spec_assets(minimal_spec)
        assert assets == {}

    def test_return_type_is_dict_of_bytes(self, minimal_spec):
        assets = collect_spec_assets(minimal_spec)
        assert isinstance(assets, dict)
        assert all(isinstance(v, bytes) for v in assets.values())
