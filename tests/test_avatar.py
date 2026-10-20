"""The VRM avatar: its config, how ``set_avatar`` resolves it, and what the build writes.

Layer: L1 (no Node build, no network; a URL source is faked).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjswan import AvatarConfig, Builder
from mjswan.avatar import resolve_avatar, resolve_body
from mjswan.cloud.publish import PublishError, plan_publish

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


class TestAvatarConfig:
    def test_rejects_a_name_that_is_not_a_vrm_bone(self):
        with pytest.raises(ValueError, match="leftElbow"):
            AvatarConfig(bones={"hips": "pelvis", "leftElbow": "left_elbow_link"})

    def test_requires_hips(self):
        with pytest.raises(ValueError, match="hips"):
            AvatarConfig(bones={"chest": "torso_link"})

    def test_rejects_an_offset_for_an_unmapped_bone(self):
        with pytest.raises(ValueError, match="leftUpperArm"):
            AvatarConfig(bones=BONES, rest_offsets={"leftUpperArm": (0, 0, -90)})

    def test_rejects_a_non_positive_height_ratio(self):
        with pytest.raises(ValueError, match="height_ratio"):
            AvatarConfig(bones=BONES, height_ratio=0.0)


class TestToDict:
    def test_entry_names_the_bundled_file(self):
        entry = AvatarConfig(bones=BONES).to_dict()
        assert entry["path"] == "assets/avatar.vrm"
        assert entry["bones"] == BONES
        assert entry["offset"] == [0.0, 1.0, 0.0]
        assert "height_ratio" not in entry

    def test_height_ratio_when_set(self):
        assert (
            AvatarConfig(bones=BONES, height_ratio=1.1).to_dict()["height_ratio"] == 1.1
        )

    def test_single_axis_offset_is_the_quaternion_about_that_axis(self):
        entry = AvatarConfig(bones=BONES, rest_offsets={"chest": (0, 0, 90)}).to_dict()
        s = math.sqrt(0.5)
        np.testing.assert_allclose(
            entry["rest_offsets"]["chest"], [0, 0, s, s], atol=1e-12
        )

    def test_offsets_turn_about_x_then_y_then_z(self):
        """(0, -90, -90): a T-pose left forearm (+x) points forward (+z), palm (-y) in (-x)."""
        q = AvatarConfig(bones=BONES, rest_offsets={"chest": (0, -90, -90)}).to_dict()[
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
        config = resolve_avatar(AvatarConfig(bones=BONES, source=vrm.name), robot)
        assert config.source == str(vrm)

    def test_missing_file(self, robot, tmp_path):
        missing = tmp_path / "nope.vrm"
        with pytest.raises(FileNotFoundError, match="nope.vrm"):
            resolve_avatar(AvatarConfig(bones=BONES, source=str(missing)), robot)

    def test_unset_source(self, robot):
        with pytest.raises(ValueError, match="source"):
            resolve_avatar(AvatarConfig(bones=BONES), robot)

    def test_a_file_that_is_not_binary_gltf(self, robot, tmp_path):
        path = tmp_path / "avatar.vrm"
        path.write_text("{}")
        with pytest.raises(ValueError, match="not a VRM"):
            resolve_avatar(AvatarConfig(bones=BONES, source=str(path)), robot)

    def test_a_body_the_model_lacks(self, robot, vrm):
        config = AvatarConfig(
            bones={"hips": "pelvis", "head": "head_link"}, source=str(vrm)
        )
        with pytest.raises(ValueError, match="head_link"):
            resolve_avatar(config, robot)

    def test_url_is_downloaded(self, robot, vrm, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            "mjswan.source.url.fetch_file", lambda url: fetched.append(url) or vrm
        )
        url = "https://example.com/avatar.vrm"
        config = resolve_avatar(AvatarConfig(bones=BONES, source=url), robot)
        assert fetched == [url]
        assert config.source == str(vrm)


class TestBuild:
    def _scene(self, out: Path) -> dict:
        return json.loads((out / "manifest.json").read_text())["projects"][0]["scenes"][
            0
        ]

    def test_bundles_the_file_unmodified(self, tmp_path, robot, vrm, build_manifest):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=robot)
        scene.set_avatar(AvatarConfig(bones=BONES, source=str(vrm)))
        out = tmp_path / "out"
        build_manifest(builder, out)

        entry = self._scene(out)["avatar"]
        assert entry["bones"] == BONES
        assert (out / "p" / "s" / entry["path"]).read_bytes() == vrm.read_bytes()

    def test_no_avatar_key_without_one(self, tmp_path, robot, build_manifest):
        builder = Builder()
        builder.add_project(name="P").add_scene(name="S", model=robot)
        out = tmp_path / "out"
        build_manifest(builder, out)
        assert "avatar" not in self._scene(out)


def test_publish_refuses_a_build_with_an_avatar(tmp_path, robot, vrm, build_manifest):
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        name="S", spec=mujoco.MjSpec.from_string(ROBOT_XML)
    )
    scene.set_avatar(AvatarConfig(bones=BONES, source=str(vrm)))
    out = tmp_path / "out"
    build_manifest(builder, out)
    with pytest.raises(PublishError, match="VRM avatar"):
        plan_publish(out)
