"""Skins: the config, how ``add_skin`` resolves it, and what the build writes.

Layer: L1 (no Node build, no network; a URL source is faked).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjswan import Builder, SkinConfig
from mjswan.cloud.publish import PublishError, plan_publish
from mjswan.skin import resolve_body, resolve_skin

BONES = {"hips": "pelvis", "chest": "torso_link"}

ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="robot/pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
      <body name="robot/torso_link" pos="0 0 0.2">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.1"/>
        <body name="left/hand"><geom type="sphere" size="0.01"/></body>
        <body name="right/hand"><geom type="sphere" size="0.01"/></body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def robot() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ROBOT_XML)


@pytest.fixture
def vrm(tmp_path: Path) -> Path:
    """Enough of a VRM for what mjswan reads of it: the binary glTF magic."""
    path = tmp_path / "avatar.vrm"
    path.write_bytes(b"glTF" + bytes(range(64)))
    return path


def _rotate(q: list[float], v: tuple[float, float, float]) -> np.ndarray:
    """Rotate *v* by the ``[x, y, z, w]`` quaternion *q*."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


class TestSkinConfig:
    def test_rejects_a_name_that_is_not_a_vrm_bone(self):
        with pytest.raises(ValueError, match="leftElbow"):
            SkinConfig(bones={"hips": "pelvis", "leftElbow": "left_elbow_link"})

    def test_requires_hips(self):
        with pytest.raises(ValueError, match="hips"):
            SkinConfig(bones={"chest": "torso_link"})

    def test_rejects_an_offset_for_an_unmapped_bone(self):
        with pytest.raises(ValueError, match="leftUpperArm"):
            SkinConfig(bones=BONES, rest_offsets={"leftUpperArm": (0, 0, -90)})

    def test_rejects_a_non_positive_height_ratio(self):
        with pytest.raises(ValueError, match="height_ratio"):
            SkinConfig(bones=BONES, height_ratio=0.0)

    def test_rejects_a_non_positive_root_height(self):
        with pytest.raises(ValueError, match="root_height"):
            SkinConfig(bones=BONES, root_height=-0.3)

    def test_rejects_an_unknown_retarget(self):
        with pytest.raises(ValueError, match="retarget"):
            SkinConfig(bones=BONES, retarget="ik")  # type: ignore[arg-type]

    def test_rejects_root_height_beside_height_ratio(self):
        with pytest.raises(ValueError, match="not both"):
            SkinConfig(bones=BONES, height_ratio=1.1, root_height=0.278)


class TestToDict:
    def test_entry_names_the_bundled_file(self):
        entry = SkinConfig(bones=BONES, name="Sample A", id="sample_a").to_dict()
        assert (entry["id"], entry["name"]) == ("sample_a", "Sample A")
        assert entry["path"] == "assets/sample_a.vrm"
        assert entry["bones"] == BONES
        assert entry["offset"] == [0.0, 1.0, 0.0]
        assert "height_ratio" not in entry

    def test_hide_robot_only_when_set(self):
        assert "hide_robot" not in SkinConfig(bones=BONES).to_dict()
        assert SkinConfig(bones=BONES, hide_robot=True).to_dict()["hide_robot"] is True

    def test_brightness_and_springs_only_when_not_the_default(self):
        entry = SkinConfig(bones=BONES).to_dict()
        assert "brightness" not in entry and "mujoco_springs" not in entry
        entry = SkinConfig(bones=BONES, brightness=1.4, mujoco_springs=False).to_dict()
        assert entry["brightness"] == 1.4
        assert entry["mujoco_springs"] is False

    def test_hips_offset_only_when_set(self):
        assert "hips_offset" not in SkinConfig(bones=BONES).to_dict()
        entry = SkinConfig(bones=BONES, hips_offset=(-0.16, 0.0, 0.0)).to_dict()
        assert entry["hips_offset"] == [-0.16, 0.0, 0.0]

    def test_rejects_a_non_positive_brightness(self):
        with pytest.raises(ValueError, match="brightness"):
            SkinConfig(bones=BONES, brightness=0)

    def test_height_ratio_when_set(self):
        assert (
            SkinConfig(bones=BONES, height_ratio=1.1).to_dict()["height_ratio"] == 1.1
        )

    def test_retarget_only_when_not_rotation(self):
        assert "retarget" not in SkinConfig(bones=BONES).to_dict()
        for mode in ("position", "hybrid"):
            entry = SkinConfig(bones=BONES, retarget=mode).to_dict()
            assert entry["retarget"] == mode

    def test_root_height_only_when_set(self):
        assert "root_height" not in SkinConfig(bones=BONES).to_dict()
        entry = SkinConfig(bones=BONES, root_height=0.278).to_dict()
        assert entry["root_height"] == 0.278

    def test_single_axis_offset_is_the_quaternion_about_that_axis(self):
        entry = SkinConfig(bones=BONES, rest_offsets={"chest": (0, 0, 90)}).to_dict()
        s = math.sqrt(0.5)
        np.testing.assert_allclose(
            entry["rest_offsets"]["chest"], [0, 0, s, s], atol=1e-12
        )

    def test_offsets_turn_about_x_then_y_then_z(self):
        """(0, -90, -90): a T-pose left forearm (+x) points forward (+z), palm (-y) in (-x)."""
        q = SkinConfig(bones=BONES, rest_offsets={"chest": (0, -90, -90)}).to_dict()[
            "rest_offsets"
        ]["chest"]
        np.testing.assert_allclose(_rotate(q, (1, 0, 0)), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(_rotate(q, (0, -1, 0)), [-1, 0, 0], atol=1e-12)


class TestResolve:
    def test_body_found_without_its_entity_prefix(self, robot):
        assert resolve_body(robot, "pelvis") == "robot/pelvis"
        assert resolve_body(robot, "robot/pelvis") == "robot/pelvis"

    def test_ambiguous_suffix_finds_nothing(self, robot):
        assert resolve_body(robot, "hand") is None

    def test_source_becomes_an_absolute_path(self, robot, vrm, monkeypatch):
        monkeypatch.chdir(vrm.parent)
        config = resolve_skin(SkinConfig(bones=BONES, source=vrm.name), robot)
        assert config.source == str(vrm)

    def test_missing_file(self, robot, tmp_path):
        missing = tmp_path / "nope.vrm"
        with pytest.raises(FileNotFoundError, match="nope.vrm"):
            resolve_skin(SkinConfig(bones=BONES, source=str(missing)), robot)

    def test_unset_source(self, robot):
        with pytest.raises(ValueError, match="source"):
            resolve_skin(SkinConfig(bones=BONES), robot)

    def test_a_file_that_is_not_binary_gltf(self, robot, tmp_path):
        path = tmp_path / "avatar.vrm"
        path.write_text("{}")
        with pytest.raises(ValueError, match="not a VRM"):
            resolve_skin(SkinConfig(bones=BONES, source=str(path)), robot)

    def test_a_body_the_model_lacks(self, robot, vrm):
        config = SkinConfig(
            bones={"hips": "pelvis", "head": "head_link"}, source=str(vrm)
        )
        with pytest.raises(ValueError, match="head_link"):
            resolve_skin(config, robot)

    def test_url_is_downloaded(self, robot, vrm, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            "mjswan.source.url.fetch_file", lambda url: fetched.append(url) or vrm
        )
        url = "https://example.com/avatar.vrm"
        config = resolve_skin(SkinConfig(bones=BONES, source=url), robot)
        assert fetched == [url]
        assert config.source == str(vrm)


class TestBuild:
    def _scene(self, out: Path) -> dict:
        return json.loads((out / "manifest.json").read_text())["projects"][0]["scenes"][
            0
        ]

    def test_bundles_each_skin_unmodified_in_order(
        self, tmp_path, robot, vrm, build_manifest
    ):
        other = tmp_path / "other.vrm"
        other.write_bytes(b"glTF" + bytes(range(64, 0, -1)))
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=robot)
        first = scene.add_skin("Sample A", SkinConfig(bones=BONES, source=str(vrm)))
        scene.add_skin("Shino", SkinConfig(bones=BONES, source=str(other)))
        out = tmp_path / "out"
        build_manifest(builder, out)

        entries = self._scene(out)["skins"]
        assert [e["name"] for e in entries] == ["Sample A", "Shino"]
        assert first.id == "sample_a" and entries[0]["path"] == "assets/sample_a.vrm"
        for entry, src in zip(entries, (vrm, other)):
            assert (out / "p" / "s" / entry["path"]).read_bytes() == src.read_bytes()

    def test_a_repeated_name_is_made_unique(self, robot, vrm):
        scene = Builder().add_project(name="P").add_scene(name="S", model=robot)
        scene.add_skin("A", SkinConfig(bones=BONES, source=str(vrm)))
        with pytest.warns(RuntimeWarning):
            second = scene.add_skin("A", SkinConfig(bones=BONES, source=str(vrm)))
        assert second.id == "a_1"

    def test_no_skins_key_without_one(self, tmp_path, robot, build_manifest):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=robot)
        out = tmp_path / "out"
        build_manifest(builder, out)
        assert "skins" not in self._scene(out)


def test_publish_refuses_a_build_with_a_skin(tmp_path, robot, vrm, build_manifest):
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        name="S", spec=mujoco.MjSpec.from_string(ROBOT_XML)
    )
    scene.add_skin("A", SkinConfig(bones=BONES, source=str(vrm)))
    out = tmp_path / "out"
    build_manifest(builder, out)
    with pytest.raises(PublishError, match="VRM skins"):
        plan_publish(out)
