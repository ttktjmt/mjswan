"""Skins: reading the file, naming its bones, measuring it against the robot, and what the
build writes.

Layer: L1 (no Node build, no network; a URL source is faked).
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import skin_fixture as fx

from mjswan import Builder, SkinConfig
from mjswan.document.skin import read_skin_file
from mjswan.license.vrm import describe, vrm_license
from mjswan.rig.humanoid import VRM_PARENTS, resolve_humanoid
from mjswan.rig.pose import (
    euler_to_quat,
    pose_at_rest,
    resolve_body,
    resolve_joint,
    standing_height,
)
from mjswan.skin import SpringChain, SpringCollider, ToonShading, resolve_skin

BONES = fx.BONES


@pytest.fixture
def robot() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(fx.robot_xml())


@pytest.fixture
def vrm(tmp_path: Path) -> Path:
    return fx.write_vrm(tmp_path / "avatar.vrm")


@pytest.fixture
def glb(tmp_path: Path) -> Path:
    return fx.write_glb(tmp_path / "avatar.glb", fx.skeleton_gltf())


def _rotate(q: list[float], v: tuple[float, float, float]) -> np.ndarray:
    """Rotate *v* by the ``[x, y, z, w]`` quaternion *q*."""
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


class TestSkinFile:
    def test_a_vrm_1_names_its_bones(self, vrm):
        file = read_skin_file(vrm)
        assert (file.format, file.vrm_version) == ("vrm", "1")
        assert file.names[file.human_bones["leftHand"]] == "leftHand"
        assert file.meta["name"] == "Sample"

    def test_a_vrm_0_is_turned_about_and_its_thumbs_renamed(self, tmp_path):
        gltf = fx.skeleton_gltf(mirror=True)
        gltf["nodes"].append({"name": "thumb"})
        gltf["extensions"] = {
            "VRM": {
                "meta": {"title": "Old"},
                "humanoid": {
                    "humanBones": [
                        {"bone": "hips", "node": 0},
                        {"bone": "leftThumbProximal", "node": len(gltf["nodes"]) - 1},
                    ]
                },
            }
        }
        file = read_skin_file(fx.write_glb(tmp_path / "old.vrm", gltf))
        assert file.vrm_version == "0"
        assert file.human_bones["leftThumbMetacarpal"] == len(gltf["nodes"]) - 1
        # Its left hand at -x in the file, at +x in VRM 1.0's frame.
        hand = [bone for bone, _, _ in fx.SKELETON].index("leftHand")
        assert file.rest[hand][0] < 0
        np.testing.assert_allclose(file.positions()[hand], [0.7, 1.3, 0], atol=1e-9)

    def test_a_glb_is_scaled_and_turned_into_the_avatar_frame(self, tmp_path):
        gltf = fx.skeleton_gltf(units=100, facing="+x")
        file = read_skin_file(fx.write_glb(tmp_path / "cm.glb", gltf))
        assert (file.format, file.human_bones) == ("gltf", {})
        hand = file.node("leftHand")
        np.testing.assert_allclose(
            file.positions(scale=0.01, facing="+x")[hand], [0.7, 1.3, 0], atol=1e-9
        )

    def test_refuses_a_gltf_with_files_beside_it(self, tmp_path):
        path = tmp_path / "avatar.gltf"
        path.write_text(json.dumps(fx.skeleton_gltf()))
        with pytest.raises(ValueError, match=r"export it as \.glb"):
            read_skin_file(path)

    def test_refuses_what_is_not_glb(self, tmp_path):
        path = tmp_path / "avatar.vrm"
        path.write_bytes(b"PK\x03\x04" + bytes(32))
        with pytest.raises(ValueError, match=r"not a binary glTF"):
            read_skin_file(path)

    def test_refuses_a_buffer_outside_the_file(self, tmp_path):
        gltf = {**fx.skeleton_gltf(), "buffers": [{"uri": "mesh.bin", "byteLength": 4}]}
        with pytest.raises(ValueError, match=r"buffers\[0\].*everything embedded"):
            read_skin_file(fx.write_glb(tmp_path / "a.glb", gltf))

    def test_refuses_compression_and_says_how_to_export_without(self, tmp_path):
        gltf = {
            **fx.skeleton_gltf(),
            "extensionsUsed": ["KHR_draco_mesh_compression"],
            "extensionsRequired": ["KHR_draco_mesh_compression"],
        }
        with pytest.raises(ValueError, match="KHR_draco.*without compression"):
            read_skin_file(fx.write_glb(tmp_path / "a.glb", gltf))

    def test_refuses_a_truncated_file(self, tmp_path, vrm):
        short = tmp_path / "short.glb"
        short.write_bytes(b"glTF" + bytes(4))
        with pytest.raises(ValueError, match="not a binary glTF"):
            read_skin_file(short)
        cut = tmp_path / "cut.vrm"
        cut.write_bytes(vrm.read_bytes()[:-8])
        with pytest.raises(ValueError, match="truncated"):
            read_skin_file(cut)

    def test_a_humanoid_off_the_file_is_a_value_error(self, tmp_path):
        def vrm_with(bones: dict) -> Path:
            gltf = {
                **fx.skeleton_gltf(),
                "extensions": {"VRMC_vrm": {"humanoid": {"humanBones": bones}}},
            }
            return fx.write_glb(tmp_path / "a.vrm", gltf)

        with pytest.raises(
            ValueError, match=r"not a valid glTF \(KeyError\('node'\)\)"
        ):
            read_skin_file(vrm_with({"hips": {}}))
        with pytest.raises(ValueError, match="puts hips on nodes it does not have"):
            read_skin_file(vrm_with({"hips": {"node": 99}}))

    def test_a_data_uri_is_embedded_in_any_case(self, tmp_path):
        uri = "DATA:application/octet-stream;base64,AAAA"
        gltf = {**fx.skeleton_gltf(), "buffers": [{"uri": uri, "byteLength": 3}]}
        assert read_skin_file(fx.write_glb(tmp_path / "a.glb", gltf)).format == "gltf"

    def test_a_missing_node_names_the_closest(self, glb):
        with pytest.raises(ValueError, match="no node 'leftHnd'.*leftHand"):
            read_skin_file(glb).node("leftHnd")


class TestHumanoid:
    def test_vrm_nests_its_bones_as_three_vrm_does(self):
        assert len(VRM_PARENTS) == 55
        assert VRM_PARENTS["leftUpperArm"] == "leftShoulder"
        assert VRM_PARENTS["leftShoulder"] == "upperChest"
        assert VRM_PARENTS["leftThumbMetacarpal"] == "leftHand"
        assert VRM_PARENTS["rightLittleDistal"] == "rightLittleIntermediate"

    @pytest.mark.parametrize("prefix", ["mixamorig:", "mixamorig1:", ""])
    def test_the_mixamo_preset_matches_with_or_without_its_prefix(
        self, tmp_path, prefix
    ):
        names = {"hips": "Hips", "leftUpperArm": "LeftArm", "leftHand": "LeftHand"}
        gltf = fx.skeleton_gltf(names={b: prefix + n for b, n in names.items()})
        file = read_skin_file(fx.write_glb(tmp_path / "a.glb", gltf))
        nodes = resolve_humanoid("mixamo", file)
        assert {bone: file.names[n] for bone, n in nodes.items()} == {
            b: prefix + n for b, n in names.items()
        }

    def test_refuses_an_unknown_preset_a_non_vrm_bone_and_no_hips(self, glb):
        file = read_skin_file(glb)
        with pytest.raises(ValueError, match="No humanoid preset 'unreal'"):
            resolve_humanoid("unreal", file)
        with pytest.raises(ValueError, match="leftElbow"):
            resolve_humanoid({"hips": "hips", "leftElbow": "leftLowerArm"}, file)
        with pytest.raises(ValueError, match="no hips"):
            resolve_humanoid({"spine": "spine"}, file)


class TestPose:
    def test_a_body_or_joint_found_without_its_entity_prefix(self, robot):
        assert resolve_body(robot, "pelvis") == "robot/pelvis"
        assert resolve_body(robot, "robot/pelvis") == "robot/pelvis"
        assert resolve_joint(robot, "waist") == 1
        assert resolve_body(robot, "l_hand") == "robot/l_hand"
        assert resolve_body(robot, "hand") is None

    def test_rest_pose_is_a_keyframe_or_joints(self, robot):
        assert pose_at_rest(robot, "lifted").qpos[2] == pytest.approx(1.0)
        data = pose_at_rest(robot, {"l_elbow": 0.5})
        assert data.qpos[robot.jnt_qposadr[resolve_joint(robot, "l_elbow")]] == 0.5
        with pytest.raises(ValueError, match=r"'stand' is not a keyframe.*'lifted'"):
            pose_at_rest(robot, "stand")
        with pytest.raises(ValueError, match="'elbow', which is not a joint"):
            pose_at_rest(robot, {"elbow": 0.5})
        with pytest.raises(ValueError, match="'root' is neither"):
            pose_at_rest(robot, {"root": 0.5})

    def test_standing_height_is_over_the_soles_not_the_floor(self, robot):
        assert standing_height(robot, 1, pose_at_rest(robot, None)) == pytest.approx(
            0.9
        )
        lifted = pose_at_rest(robot, "lifted")
        assert lifted.xpos[1][2] == pytest.approx(1.0)
        assert standing_height(robot, 1, lifted) == pytest.approx(0.9)

    def test_single_axis_turn(self):
        s = math.sqrt(0.5)
        np.testing.assert_allclose(euler_to_quat((0, 0, 90)), [0, 0, s, s], atol=1e-12)

    def test_turns_about_x_then_y_then_z(self):
        """(0, -90, -90): a T-pose left forearm (+x) points forward (+z), palm (-y) in (-x)."""
        q = euler_to_quat((0, -90, -90))
        np.testing.assert_allclose(_rotate(q, (1, 0, 0)), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(_rotate(q, (0, -1, 0)), [-1, 0, 0], atol=1e-12)


class TestVrmLicense:
    def test_vrm_1_defaults_are_the_strictest(self):
        terms = vrm_license({"name": "A", "authors": ["B"]}, "1")
        assert not terms.redistribution and not terms.commercial
        assert terms.avatar_use == "author" and terms.credit_required
        assert terms.needs_attestation

    def test_vrm_1_open_terms(self):
        terms = vrm_license(fx.OPEN_META, "1")
        assert terms.redistribution and terms.commercial
        assert terms.avatar_use == "everyone" and not terms.credit_required
        assert not terms.needs_attestation
        assert describe(terms) == "Sample by mjswan (https://vrm.dev/licenses/1.0/)"

    def test_vrm_1_personal_use_and_licensed_persons(self):
        terms = vrm_license(
            {
                **fx.OPEN_META,
                "commercialUsage": "personalNonProfit",
                "avatarPermission": "onlySeparatelyLicensedPerson",
            },
            "1",
        )
        assert not terms.commercial and terms.avatar_use == "licensed"

    def test_reads_its_strings_as_mjswan_cloud_does(self):
        # An empty or non-string value is unset: "" is not a commercial use.
        meta = {
            **fx.OPEN_META,
            "commercialUsage": "",
            "name": "",
            "authors": ["A", "", 3],
        }
        terms = vrm_license(meta, "1")
        assert not terms.commercial and terms.name is None and terms.authors == ("A",)

    @pytest.mark.parametrize(
        ("meta", "redistribution", "commercial", "avatar_use", "credit"),
        [
            (
                {
                    "licenseName": "CC0",
                    "allowedUserName": "Everyone",
                    "commercialUssageName": "Allow",
                },
                True,
                True,
                "everyone",
                False,
            ),
            (
                {
                    "licenseName": "CC_BY_NC",
                    "allowedUserName": "Everyone",
                    "commercialUssageName": "Allow",
                },
                True,
                False,
                "everyone",
                True,
            ),
            (
                {
                    "licenseName": "Redistribution_Prohibited",
                    "allowedUserName": "OnlyAuthor",
                },
                False,
                False,
                "author",
                False,
            ),
            (
                {"licenseName": "Other", "allowedUserName": "ExplicitlyLicensedPerson"},
                True,
                False,
                "licensed",
                True,
            ),
            ({}, True, False, "author", False),
        ],
    )
    def test_vrm_0(self, meta, redistribution, commercial, avatar_use, credit):
        terms = vrm_license(meta, "0")
        assert (
            terms.redistribution,
            terms.commercial,
            terms.avatar_use,
            terms.credit_required,
        ) == (
            redistribution,
            commercial,
            avatar_use,
            credit,
        )


class TestSkinConfig:
    def test_needs_a_bone(self):
        with pytest.raises(ValueError, match="empty"):
            SkinConfig(bones={})

    def test_rejects_an_offset_for_an_unmapped_bone(self):
        with pytest.raises(ValueError, match="neck"):
            SkinConfig(bones=BONES, rest_offsets={"neck": (0, 0, -90)})

    def test_a_limb_is_three_mapped_bones(self):
        with pytest.raises(ValueError, match="three bones"):
            SkinConfig(bones=BONES, limbs=[("hips", "spine")])  # type: ignore[list-item]
        with pytest.raises(ValueError, match="neck"):
            SkinConfig(bones=BONES, limbs=[("hips", "spine", "neck")])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("scale", 0.0),
            ("brightness", 0.0),
            ("height_ratio", -1.0),
            ("facing", "up"),
            ("retarget", "ik"),
            ("reparent", {"hips": "hips"}),
        ],
    )
    def test_rejects(self, field, value):
        with pytest.raises(ValueError, match=field):
            SkinConfig(bones=BONES, **{field: value})

    def test_defaults_are_hybrid_over_the_robot(self):
        config = SkinConfig(bones=BONES)
        assert config.retarget == "hybrid" and config.hide_robot
        assert config.offset == (0.0, 0.0, 0.0) and config.hips_offset is None
        assert not config.mujoco_springs


class TestResolve:
    def test_source_becomes_an_absolute_path(self, robot, vrm, monkeypatch):
        monkeypatch.chdir(vrm.parent)
        skin = resolve_skin(SkinConfig(bones=BONES, source=vrm.name), robot)
        assert skin.config.source == str(vrm)

    def test_missing_or_unset_source(self, robot, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.vrm"):
            resolve_skin(
                SkinConfig(bones=BONES, source=str(tmp_path / "nope.vrm")), robot
            )
        with pytest.raises(ValueError, match="source"):
            resolve_skin(SkinConfig(bones=BONES), robot)

    def test_url_is_downloaded(self, robot, vrm, monkeypatch):
        fetched = []
        monkeypatch.setattr(
            "mjswan.source.url.fetch_file", lambda url: fetched.append(url) or vrm
        )
        url = "https://example.com/avatar.vrm"
        skin = resolve_skin(SkinConfig(bones=BONES, source=url), robot)
        assert fetched == [url]
        assert skin.config.source == str(vrm)

    def test_a_download_times_out_and_goes_at_exit(self, monkeypatch):
        from mjswan.source import url

        timeouts, at_exit = [], []
        monkeypatch.setattr(
            url.urllib.request,
            "urlopen",
            lambda _, timeout: timeouts.append(timeout) or io.BytesIO(b"glTF"),
        )
        monkeypatch.setattr(
            url.atexit, "register", lambda *call, **kw: at_exit.append((call, kw))
        )
        path = url.fetch_file("https://example.com/skins/fox.glb")
        assert (path.name, path.read_bytes(), timeouts) == ("fox.glb", b"glTF", [60])
        (remove, *args), kwargs = at_exit[0]
        remove(*args, **kwargs)
        assert not path.parent.exists()

    def test_a_body_the_model_lacks(self, robot, vrm):
        with pytest.raises(ValueError, match="head_link"):
            resolve_skin(
                SkinConfig(bones={**BONES, "head": "head_link"}, source=str(vrm)), robot
            )

    def test_a_vrm_brings_what_a_glb_is_given(self, robot, vrm):
        for field, value in (
            ("scale", 0.01),
            ("material", ToonShading()),
            ("license", "CC0-1.0"),
        ):
            with pytest.raises(ValueError, match=f"is a VRM.*Remove {field}"):
                resolve_skin(
                    SkinConfig(bones=BONES, source=str(vrm), **{field: value}), robot
                )
        with pytest.raises(ValueError, match="is a VRM.*Remove reparent"):
            resolve_skin(
                SkinConfig(bones=BONES, source=str(vrm), reparent={"leftHand": "hips"}),
                robot,
            )

    def test_a_vrm_is_mapped_by_vrm_bone_from_its_hips(self, robot, vrm):
        with pytest.raises(ValueError, match="leftElbow"):
            resolve_skin(
                SkinConfig(bones={**BONES, "leftElbow": "l_fore"}, source=str(vrm)),
                robot,
            )
        with pytest.raises(ValueError, match="'hips'"):
            resolve_skin(SkinConfig(bones={"spine": "torso"}, source=str(vrm)), robot)

    def test_a_bone_the_vrm_lacks_stays_at_rest(self, robot, vrm):
        with pytest.warns(UserWarning, match="no jaw; it stays at rest"):
            resolve_skin(
                SkinConfig(bones={**BONES, "jaw": "torso"}, source=str(vrm)), robot
            )

    def test_a_vrm_bends_its_four_limbs(self, robot, vrm):
        skin = resolve_skin(SkinConfig(bones=BONES, source=str(vrm)), robot)
        assert (skin.rig, skin.root, skin.nodes) == ("vrm", "hips", {})
        assert len(skin.limbs) == 4

    def test_a_glb_humanoid_by_table_or_preset(self, robot, tmp_path):
        names = {
            "hips": "Hips",
            "spine": "Spine",
            "leftUpperArm": "LeftArm",
            "leftLowerArm": "LeftForeArm",
            "leftHand": "LeftHand",
        }
        path = fx.write_glb(tmp_path / "a.glb", fx.skeleton_gltf(names=names))
        bones = {bone: BONES[bone] for bone in names}
        by_table = resolve_skin(
            SkinConfig(bones=bones, source=str(path), humanoid=names), robot
        )
        by_preset = resolve_skin(
            SkinConfig(bones=bones, source=str(path), humanoid="mixamo"), robot
        )
        assert by_table.rig == "humanoid" and by_table.root == "hips"
        assert by_table.nodes == by_preset.nodes
        assert by_table.limbs == [("leftUpperArm", "leftLowerArm", "leftHand")]
        with pytest.raises(ValueError, match="'neck'.*which humanoid does not name"):
            resolve_skin(
                SkinConfig(
                    bones={**bones, "neck": "torso"}, source=str(path), humanoid=names
                ),
                robot,
            )

    def test_a_glb_humanoid_is_nested_as_the_file_nests_it(self, robot, tmp_path):
        # A pug's front legs hang from its body, not from the shoulders VRM would have.
        gltf = fx.skeleton_gltf()
        index = {node["name"]: i for i, node in enumerate(gltf["nodes"])}
        gltf["nodes"][index["chest"]]["children"].remove(index["leftUpperArm"])
        gltf["nodes"][index["hips"]]["children"].append(index["leftUpperArm"])
        path = fx.write_glb(tmp_path / "a.glb", gltf)
        humanoid = {bone: bone for bone in BONES}
        skin = resolve_skin(
            SkinConfig(bones=BONES, source=str(path), humanoid=humanoid), robot
        )
        assert ("leftUpperArm", "leftLowerArm", "leftHand") in skin.limbs

    def test_reparent_hangs_a_node_from_another_where_it_rests(self, robot, tmp_path):
        from mjswan.build.skin import skin_entry

        # A foot an IK control carries, as in Quaternius's animals: here the hand, hung
        # from the hips where it rests, which the arm would pose without it.
        gltf = fx.skeleton_gltf()
        nodes = gltf["nodes"]
        index = {node["name"]: i for i, node in enumerate(nodes)}
        hand, forearm, hips = index["leftHand"], index["leftLowerArm"], index["hips"]
        rest = read_skin_file(fx.write_glb(tmp_path / "a.glb", gltf)).rest
        nodes[forearm]["children"].remove(hand)
        nodes[hips]["children"].append(hand)
        nodes[hand]["translation"] = list(rest[hand] - rest[hips])
        path = fx.write_glb(tmp_path / "ik.glb", gltf)
        humanoid = {bone: bone for bone in BONES}
        with pytest.raises(ValueError, match="not a chain"):
            resolve_skin(
                SkinConfig(bones=BONES, source=str(path), humanoid=humanoid), robot
            )

        config = SkinConfig(
            bones=BONES,
            source=str(path),
            humanoid=humanoid,
            reparent={"leftHand": "leftLowerArm"},
        )
        skin = resolve_skin(config, robot)
        assert skin.file.parents[hand] == forearm
        assert np.allclose(skin.file.rest, rest)
        assert ("leftUpperArm", "leftLowerArm", "leftHand") in skin.limbs
        assert skin_entry(skin)["reparent"] == [[hand, forearm]]
        with pytest.raises(ValueError, match="'hips' from 'leftHand', which hangs"):
            resolve_skin(replace(config, reparent={"hips": "leftHand"}), robot)
        with pytest.raises(ValueError, match="no node 'leftHnd'.*leftHand"):
            resolve_skin(replace(config, reparent={"leftHnd": "hips"}), robot)

    def test_a_glb_humanoid_is_carried_by_its_hips(self, robot, tmp_path):
        gltf = fx.skeleton_gltf()
        path = fx.write_glb(tmp_path / "a.glb", gltf)
        with pytest.raises(ValueError, match="one topmost node, hips"):
            resolve_skin(
                SkinConfig(
                    bones={"hips": "pelvis", "head": "torso"},
                    source=str(path),
                    humanoid={"hips": "chest", "head": "spine"},
                ),
                robot,
            )

    def test_a_node_map_keys_bones_by_node_name(self, robot, glb):
        bones = {
            "hips": "pelvis",
            "leftUpperArm": "l_upper",
            "leftLowerArm": "l_fore",
            "leftHand": "l_hand",
        }
        renamed = {f"{bone}_node": body for bone, body in bones.items()}
        path = fx.write_glb(
            glb.with_name("b.glb"),
            fx.skeleton_gltf(names={b: f"{b}_node" for b in bones}),
        )
        limb = ("leftUpperArm_node", "leftLowerArm_node", "leftHand_node")
        skin = resolve_skin(
            SkinConfig(bones=renamed, source=str(path), limbs=[limb]), robot
        )
        assert (skin.rig, skin.root, skin.limbs) == ("nodes", "hips_node", [limb])
        with pytest.raises(ValueError, match="Give humanoid"):
            resolve_skin(
                SkinConfig(
                    bones={"hips": "pelvis", "spine": "torso"}, source=str(path)
                ),
                robot,
            )

    def test_a_node_map_has_one_top_and_limbs_down_its_chains(self, robot, glb):
        with pytest.raises(
            ValueError, match="one topmost node.*'leftHand', 'rightHand'"
        ):
            resolve_skin(
                SkinConfig(
                    bones={"leftHand": "l_hand", "rightHand": "r_hand"}, source=str(glb)
                ),
                robot,
            )
        bones = {
            "hips": "pelvis",
            "leftUpperArm": "l_upper",
            "leftLowerArm": "l_fore",
            "leftHand": "l_hand",
        }
        with pytest.raises(ValueError, match="not a chain"):
            resolve_skin(
                SkinConfig(
                    bones=bones,
                    source=str(glb),
                    limbs=[("hips", "leftUpperArm", "leftHand")],
                ),
                robot,
            )
        with pytest.raises(ValueError, match="humanoid's come from its bones"):
            resolve_skin(
                SkinConfig(
                    bones=bones,
                    source=str(glb),
                    humanoid={b: b for b in bones},
                    limbs=[("hips", "leftUpperArm", "leftLowerArm")],
                ),
                robot,
            )

    def test_springs_hang_from_nodes_by_name(self, robot, tmp_path):
        gltf = fx.skeleton_gltf()
        head = [n["name"] for n in gltf["nodes"]].index("head")
        for i, name in enumerate(("tail0", "tail1", "tail2")):
            gltf["nodes"].append(
                {"name": name, "translation": [0, 0, -0.1], "children": []}
            )
            parent = head if i == 0 else len(gltf["nodes"]) - 2
            gltf["nodes"][parent]["children"].append(len(gltf["nodes"]) - 1)
        path = fx.write_glb(tmp_path / "a.glb", gltf)
        chain = SpringChain(
            root="tail0",
            tip=0.05,
            colliders=(SpringCollider("head", radius=0.1, tail=(0, 0.1, 0)),),
        )
        skin = resolve_skin(
            SkinConfig(
                bones={"hips": "pelvis"},
                source=str(path),
                humanoid={"hips": "hips"},
                springs=[chain],
            ),
            robot,
        )
        (spring,) = skin.springs
        assert [gltf["nodes"][n]["name"] for n in spring["joints"]] == [
            "tail0",
            "tail1",
            "tail2",
        ]
        assert spring["tip"] == 0.05
        assert spring["colliders"] == [
            {
                "node": head,
                "radius": 0.1,
                "offset": [0.0, 0.0, 0.0],
                "tail": [0, 0.1, 0],
            }
        ]
        with pytest.raises(ValueError, match="no child to swing toward"):
            resolve_skin(
                SkinConfig(
                    bones={"hips": "pelvis"},
                    source=str(path),
                    humanoid={"hips": "hips"},
                    springs=[SpringChain(root="tail2")],
                ),
                robot,
            )
        with pytest.raises(
            ValueError, match=r"swing \['head'\], which bones already poses"
        ):
            resolve_skin(
                SkinConfig(
                    bones={"hips": "pelvis", "head": "torso"},
                    source=str(path),
                    humanoid={"hips": "hips", "head": "head"},
                    springs=[SpringChain(root="head")],
                ),
                robot,
            )


class TestMeasure:
    def test_rest_pose_turns_each_bone_to_the_robot_at_it(self, robot, vrm):
        skin = resolve_skin(
            SkinConfig(bones=BONES, source=str(vrm), rest_pose=fx.T_POSE), robot
        )
        s = math.sqrt(0.5)
        # The robot's arm hangs at the zero pose: the avatar's turns down 90 degrees there.
        np.testing.assert_allclose(
            skin.rest_offsets["leftUpperArm"], [0, 0, -s, s], atol=1e-9
        )
        np.testing.assert_allclose(
            _rotate(skin.rest_offsets["leftUpperArm"], (1, 0, 0)), [0, -1, 0], atol=1e-9
        )
        np.testing.assert_allclose(
            skin.rest_offsets["rightUpperArm"], [0, 0, s, s], atol=1e-9
        )
        assert (
            "hips" not in skin.rest_offsets and "leftUpperLeg" not in skin.rest_offsets
        )

    def test_rest_offsets_turn_on_top_of_the_rest_pose(self, robot, vrm):
        config = SkinConfig(
            bones=BONES,
            source=str(vrm),
            rest_pose=fx.T_POSE,
            rest_offsets={"leftUpperArm": (0, 0, 90), "spine": (10, 0, 0)},
        )
        skin = resolve_skin(config, robot)
        np.testing.assert_allclose(
            skin.rest_offsets["leftUpperArm"], [0, 0, 0, 1], atol=1e-9
        )
        np.testing.assert_allclose(
            skin.rest_offsets["spine"], euler_to_quat((10, 0, 0)), atol=1e-12
        )

    def test_a_keyframe_tilting_the_root_is_not_carried_in(self, robot, vrm):
        skin = resolve_skin(
            SkinConfig(bones=BONES, source=str(vrm), rest_pose="bow"), robot
        )
        assert skin.rest_offsets == {}

    def test_root_height_is_over_the_soles_at_rest(self, robot, vrm):
        for rest_pose in (None, "lifted", fx.T_POSE):
            skin = resolve_skin(
                SkinConfig(bones=BONES, source=str(vrm), rest_pose=rest_pose), robot
            )
            assert skin.root_height == pytest.approx(0.9)
        assert (
            resolve_skin(SkinConfig(bones=BONES, source=str(vrm)), None).root_height
            is None
        )

    def test_hips_centred_over_the_robots_limbs(self, vrm):
        robot = mujoco.MjModel.from_xml_string(fx.robot_xml(leg_x=-0.1))
        skin = resolve_skin(SkinConfig(bones=BONES, source=str(vrm)), robot)
        # The arms' roots at 0, the legs' 0.1 m back: the hips go 0.05 m back.
        assert skin.hips_offset == pytest.approx((-0.05, 0.0, 0.0))
        given = resolve_skin(
            SkinConfig(bones=BONES, source=str(vrm), hips_offset=(0.1, 0, 0)), robot
        )
        assert given.hips_offset == (0.1, 0, 0)

    def test_position_warns_when_the_limbs_are_not_the_robots(
        self, robot, tmp_path, recwarn
    ):
        fitted = fx.write_vrm(tmp_path / "fitted.vrm")
        resolve_skin(
            SkinConfig(bones=BONES, source=str(fitted), retarget="position"), robot
        )
        assert not [w for w in recwarn if "stretch" in str(w.message)]
        long = fx.write_vrm(tmp_path / "long.vrm", arms=1.5)
        with pytest.warns(
            UserWarning, match=r"long\.vrm: retarget='position' will stretch it.*x1\.50"
        ):
            resolve_skin(
                SkinConfig(bones=BONES, source=str(long), retarget="position"), robot
            )
        resolve_skin(SkinConfig(bones=BONES, source=str(long)), robot)


class TestBuild:
    def _scene(self, out: Path) -> dict:
        return json.loads((out / "manifest.json").read_text())["projects"][0]["scenes"][
            0
        ]

    def test_bundles_each_file_unmodified_in_order(
        self, tmp_path, robot, vrm, glb, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=robot)
        first = scene.add_skin("Sample A", SkinConfig(bones=BONES, source=str(vrm)))
        scene.add_skin(
            "Doll",
            SkinConfig(
                bones={"hips": "pelvis"},
                source=str(glb),
                humanoid={"hips": "hips"},
                scale=0.5,
            ),
        )
        out = tmp_path / "out"
        build_manifest(builder, out)

        entries = self._scene(out)["skins"]
        assert [e["name"] for e in entries] == ["Sample A", "Doll"]
        assert first.id == "sample_a" and entries[0]["path"] == "assets/sample_a.vrm"
        assert (entries[0]["format"], entries[1]["format"]) == ("vrm", "gltf")
        assert entries[1]["path"] == "assets/doll.glb"
        for entry, src in zip(entries, (vrm, glb)):
            assert (out / "p" / "s" / entry["path"]).read_bytes() == src.read_bytes()

    def test_an_entry_names_nodes_and_leaves_out_defaults(self, robot, glb):
        from mjswan.build.skin import skin_entry

        config = SkinConfig(
            bones={"hips": "pelvis"},
            source=str(glb),
            humanoid={"hips": "hips"},
            id="d",
            name="D",
        )
        entry = skin_entry(resolve_skin(config, robot))
        assert entry == {
            "id": "d",
            "name": "D",
            "format": "gltf",
            "path": "assets/d.glb",
            "bones": {"hips": "pelvis"},
            "nodes": {"hips": 0},
            "root_height": 0.9,
        }
        styled = SkinConfig(
            bones={"hips": "pelvis"},
            source=str(glb),
            humanoid={"hips": "hips"},
            facing="-x",
            material=ToonShading(shade=0.5),
            retarget="rotation",
            hide_robot=False,
            mujoco_springs=True,
            id="d",
            name="D",
        )
        entry = skin_entry(resolve_skin(styled, robot))
        assert entry["facing"] == "-x" and entry["retarget"] == "rotation"
        assert entry["material"] == {"toon": {"shade": 0.5, "toony": 0.5, "shift": 0.0}}
        assert entry["hide_robot"] is False and entry["mujoco_springs"] is True

    def test_a_glb_carries_its_license_beside_it(
        self, tmp_path, robot, glb, build_manifest
    ):
        builder = Builder()
        scene = builder.add_project(name="P").add_scene(name="S", model=robot)
        scene.add_skin(
            "Doll",
            SkinConfig(
                bones={"hips": "pelvis"},
                source=str(glb),
                humanoid={"hips": "hips"},
                license="CC0-1.0",
                notice="Doll by mjswan.",
            ),
        )
        out = tmp_path / "out"
        build_manifest(builder, out)
        assert "CC0" in (out / "p" / "s" / "LICENSE.skin-doll").read_text()
        assert (
            out / "p" / "s" / "NOTICE.skin-doll"
        ).read_text().strip() == "Doll by mjswan."

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
