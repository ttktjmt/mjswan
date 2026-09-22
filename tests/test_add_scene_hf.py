"""``ProjectHandle.add_scene_hf`` end to end, with the Hub itself stubbed.

A MuJoCo model is several files (the XML names meshes MuJoCo resolves relative to it),
so these cover the part `fetch_file` cannot do: bringing the whole directory down and
compiling the spec where it lands.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

import mjswan

SCENE_XML = """
<mujoco model="two_joint">
  <asset><mesh name="link" file="meshes/link.stl"/></asset>
  <worldbody>
    <body><joint name="hip" type="hinge" axis="0 1 0"/>
      <geom type="mesh" mesh="link"/>
      <body pos="0 0 -0.3"><joint name="knee" type="hinge" axis="0 1 0"/>
        <geom type="mesh" mesh="link"/></body></body>
  </worldbody>
  <actuator>
    <position name="hip_act" joint="hip"/><position name="knee_act" joint="knee"/>
  </actuator>
</mujoco>
"""


def _tetrahedron_stl() -> bytes:
    """A valid binary STL: MuJoCo refuses a mesh with fewer than four vertices."""
    import struct

    v = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    out = b"\0" * 80 + struct.pack("<I", len(faces))
    for a, b, c in faces:
        out += struct.pack("<12fH", 0, 0, 1, *v[a], *v[b], *v[c], 0)
    return out


TETRAHEDRON_STL = _tetrahedron_stl()


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A Hub stub whose repository is a directory tree; returns its root."""
    repo = tmp_path / "repo"
    (repo / "scenes/unitree_g1/meshes").mkdir(parents=True)
    (repo / "scenes/unitree_g1/scene.xml").write_text(SCENE_XML)
    (repo / "scenes/unitree_g1/meshes/link.stl").write_bytes(TETRAHEDRON_STL)
    (repo / "scenes/unitree_g1/LICENSE").write_text("BSD-3-Clause\n\nExample text.\n")
    (repo / "policies").mkdir()
    (repo / "policies/walk.onnx").write_bytes(b"not really onnx")

    calls: dict[str, object] = {}

    class _Hub:
        @staticmethod
        def snapshot_download(
            repo_id, revision=None, repo_type=None, token=None, allow_patterns=None
        ):
            calls["allow_patterns"] = allow_patterns
            calls["repo_id"] = repo_id
            return str(repo)

    monkeypatch.setattr("mjswan.source.hf._hub", lambda: _Hub)
    return calls


@pytest.fixture
def project():
    return mjswan.Builder().add_project(name="Test")


class TestAddSceneHf:
    def test_compiles_the_model_with_its_meshes(self, project, fake_hub):
        scene = project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        assert scene._config.spec is not None
        assert scene._config.spec.modelname == "two_joint"

    def test_names_the_scene_after_its_directory(self, project, fake_hub):
        scene = project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        assert scene._config.name == "unitree_g1"

    def test_an_explicit_name_wins(self, project, fake_hub):
        scene = project.add_scene_hf(
            "org/assets", "scenes/unitree_g1/scene.xml", name="G1"
        )

        assert scene._config.name == "G1"

    def test_downloads_only_the_model_s_own_directory(self, project, fake_hub):
        """The repository holds other assets; a scene should not drag them along."""
        project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        assert fake_hub["allow_patterns"] == ["scenes/unitree_g1/*"]

    def test_allow_patterns_overrides_the_default(self, project, fake_hub):
        project.add_scene_hf(
            "org/assets",
            "scenes/unitree_g1/scene.xml",
            allow_patterns=["scenes/*", "shared/meshes/*"],
        )

        assert fake_hub["allow_patterns"] == ["scenes/*", "shared/meshes/*"]

    def test_a_license_beside_the_model_is_detected(self, project, fake_hub):
        """`add_scene(spec=)`'s ADR 0007 detection works on a downloaded tree too."""
        scene = project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        assert [a.component for a in scene._config.attributions] == ["unitree_g1"]

    def test_control_dt_and_metadata_are_forwarded(self, project, fake_hub):
        scene = project.add_scene_hf(
            "org/assets",
            "scenes/unitree_g1/scene.xml",
            control_dt=0.02,
            metadata={"note": "x"},
        )

        assert scene._config.control_dt == 0.02
        assert scene._config.metadata == {"note": "x"}

    def test_a_directory_instead_of_an_xml_raises(self, project, fake_hub):
        with pytest.raises(ValueError, match="not its directory"):
            project.add_scene_hf("org/assets", "scenes/unitree_g1")


class TestFetchDir:
    def test_returns_the_named_subdirectory(self, fake_hub):
        from mjswan.source.hf import fetch_dir

        local = fetch_dir("org/assets", "scenes/unitree_g1")

        assert local.is_dir()
        assert (local / "scene.xml").exists()

    def test_the_whole_repo_needs_no_pattern(self, fake_hub):
        from mjswan.source.hf import fetch_dir

        local = fetch_dir("org/assets")

        assert fake_hub["allow_patterns"] is None
        assert (local / "policies" / "walk.onnx").exists()

    def test_a_missing_directory_says_so(self, fake_hub):
        from mjswan.source.hf import fetch_dir

        with pytest.raises(ValueError, match="not a directory"):
            fetch_dir("org/assets", "scenes/nope")

    def test_a_trailing_slash_does_not_double(self, fake_hub):
        from mjswan.source.hf import fetch_dir

        fetch_dir("org/assets", "scenes/unitree_g1/")

        assert fake_hub["allow_patterns"] == ["scenes/unitree_g1/*"]


class TestBuildRoundTrip:
    def test_a_hub_scene_builds(self, project, fake_hub, build_manifest, tmp_path):
        project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        manifest = build_manifest(project._builder, tmp_path / "dist")

        scene = manifest["projects"][0]["scenes"][0]
        assert scene["id"] == "unitree_g1"
        assert Path(tmp_path / "dist" / "test" / "unitree_g1" / "scene.mjz").exists()


class TestActuatedJointNames:
    """The order a policy's actions come out in, which `policy_joint_names` wants."""

    def test_actuator_order_not_joint_order(self, project, fake_hub):
        scene = project.add_scene_hf("org/assets", "scenes/unitree_g1/scene.xml")

        assert scene.actuated_joint_names() == ["hip", "knee"]

    def test_none_when_the_model_does_not_say_unambiguously(self, project):
        """No actuators: "the i-th action drives this joint" has no answer."""
        passive = project.add_scene(
            name="Passive",
            spec=mujoco.MjSpec.from_string(
                '<mujoco><worldbody><body><joint name="hip" type="hinge"/>'
                '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
            ),
        )

        assert passive.actuated_joint_names() is None
