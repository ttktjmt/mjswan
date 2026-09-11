"""License files in the build (ADR 0007).

L1 — the naming rule, the identifier and the tier table (kept in step with mjswan
Cloud's ``@mjswan/licenses``), detection beside a model on disk, the author API, and
the files ``_save_web`` writes. No network; the Node build and the SPA copy are mocked
by the shared ``build_manifest`` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import mujoco
import pytest

from mjswan.builder import Builder
from mjswan.document import document_files, unpack_document, write_document
from mjswan.licenses import (
    BLOCKED,
    CUSTOM,
    GENERATABLE_LICENSES,
    LICENSE_CONTENT_TYPE,
    LICENSE_FILE_MAX_BYTES,
    Attribution,
    blocked_refusal,
    component_id,
    declare,
    detect_attributions,
    generate_license_text,
    identify_license,
    is_license_file_path,
    known_asset,
    known_attribution,
    license_path_problem,
    license_template,
    parse_license_path,
    resolve_license,
    resolve_notice,
    restricted_warning,
    restriction_label,
    spec_asset_directories,
    tier_of,
)
from mjswan.publish import plan_publish

# ── The naming rule ───────────────────────────────────────────────────────────


class TestNamingRule:
    def test_project_level_license_and_notice(self):
        loc = parse_license_path("demo/LICENSE")
        assert loc is not None
        assert (loc.kind, loc.scope, loc.project) == ("LICENSE", "project", "demo")
        assert parse_license_path("demo/NOTICE").kind == "NOTICE"

    def test_scene_level_component_files(self):
        loc = parse_license_path("demo/go2/LICENSE.unitree_go2")
        assert loc is not None
        assert loc.scope == "scene"
        assert (loc.project, loc.scene, loc.component) == ("demo", "go2", "unitree_go2")

    def test_a_bare_scene_level_file_is_labelled_with_the_scene(self):
        loc = parse_license_path("demo/go2/NOTICE")
        assert loc is not None and loc.component == "go2"

    @pytest.mark.parametrize(
        "path",
        [
            "LICENSE",  # the build root: the engine's own
            "demo/LICENSE.x",  # a suffix at project level
            "demo/go2/mdp/LICENSE",  # a third level
            "demo/go2/license",  # case matters
            "demo/go2/LICENSE.has space",
            "demo/go2/scene.mjz",
        ],
    )
    def test_everything_else_is_not_a_license_location(self, path):
        assert parse_license_path(path) is None
        assert not is_license_file_path(path)

    def test_the_problem_names_the_reason(self):
        assert "build root" in (license_path_problem("LICENSE") or "")
        assert "suffix" in (license_path_problem("demo/LICENSE.x") or "")
        assert "not deeper" in (license_path_problem("demo/go2/mdp/LICENSE") or "")
        assert license_path_problem("demo/LICENSE") is None
        assert license_path_problem("demo/go2/scene.mjz") is None

    def test_component_id_fits_the_suffix_rule(self):
        assert component_id("unitree_go2") == "unitree_go2"
        assert component_id("ANYmal C (v2)") == "ANYmal_C_v2"
        assert component_id("...") == "component"
        assert len(component_id("x" * 100)) == 64


# ── Identification and tiers ──────────────────────────────────────────────────


class TestIdentification:
    @pytest.mark.parametrize("spdx", GENERATABLE_LICENSES)
    def test_the_bundled_texts_are_recognised_without_their_tag(self, spdx):
        found = identify_license(license_template(spdx))
        assert found.spdx == spdx
        assert found.tier == "notice"

    def test_the_spdx_tag_on_the_first_line_wins(self):
        text = "SPDX-License-Identifier: CC-BY-NC-ND-4.0\n\nwhatever follows"
        assert identify_license(text).spdx == "CC-BY-NC-ND-4.0"
        # Only the first line: a tag quoted later in a text is not a declaration.
        assert (
            identify_license("MIT License\nSPDX-License-Identifier: GPL-3.0-only").spdx
            != "GPL-3.0-only"
        )

    def test_a_tag_is_at_most_64_characters(self):
        ok = "LicenseRef-" + "a" * 53
        assert identify_license(f"SPDX-License-Identifier: {ok}\n").spdx == ok
        # A longer run is not a tag, and is not truncated into one.
        too_long = identify_license(f"SPDX-License-Identifier: {'a' * 65}\n")
        assert too_long.spdx == CUSTOM

    def test_creative_commons_variants(self):
        cc = "Attribution-NonCommercial-NoDerivatives 4.0 International Public License"
        assert identify_license(cc).spdx == "CC-BY-NC-ND-4.0"
        assert (
            identify_license("Attribution-ShareAlike 4.0 International").spdx
            == "CC-BY-SA-4.0"
        )
        assert (
            identify_license("Creative Commons Attribution 4.0 International").spdx
            == "CC-BY-4.0"
        )

    def test_gnu_family(self):
        assert (
            identify_license(
                "GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007"
            ).spdx
            == "GPL-3.0-only"
        )
        assert (
            identify_license("GNU LESSER GENERAL PUBLIC LICENSE Version 2.1").spdx
            == "LGPL-2.1-only"
        )
        assert (
            identify_license("GNU AFFERO GENERAL PUBLIC LICENSE").spdx
            == "AGPL-3.0-only"
        )

    def test_the_non_redistributable_licenses_are_blocked(self):
        mpg = (
            "Software Copyright License for non-commercial scientific research purposes"
        )
        assert identify_license(mpg) == identify_license(mpg)
        assert identify_license(mpg).tier == "blocked"
        ur = "Universal Robots A/S\nTerms and Conditions for Graphical Documentation"
        assert identify_license(ur).tier == "blocked"
        for tag in ("LicenseRef-AMASS", "LicenseRef-SMPL"):
            assert (
                identify_license(f"SPDX-License-Identifier: {tag}\n").tier == "blocked"
            )
            assert tag in BLOCKED

    def test_an_unknown_text_is_custom_and_only_shown(self):
        found = identify_license("You may use this model for anything you like.")
        assert found.spdx == CUSTOM
        assert found.tier == "notice"

    def test_tiers(self):
        assert tier_of("MIT") == "notice"
        assert tier_of("CC-BY-4.0") == "notice"
        assert tier_of("CC-BY-NC-4.0") == "restricted"
        assert tier_of("GPL-2.0-only") == "restricted"
        assert tier_of("LicenseRef-MPG-NonCommercial") == "blocked"

    def test_restriction_labels_and_warning_clauses(self):
        assert restriction_label("CC-BY-NC-ND-4.0") == "Non-commercial"
        assert restriction_label("CC-BY-SA-4.0") == "Share-alike"
        assert restriction_label("GPL-3.0-only") == "Copyleft"
        assert restriction_label("MIT") is None
        warning = restricted_warning("demo/go2/LICENSE.lafan1", "CC-BY-NC-ND-4.0")
        assert warning.startswith("demo/go2/LICENSE.lafan1 is CC-BY-NC-ND-4.0 (")
        assert "non-commercial use only" in warning
        assert "no sharing of adapted material" in warning

    def test_blocked_refusal_names_the_file_and_the_license(self):
        message = blocked_refusal("demo/go2/LICENSE.amass", "LicenseRef-AMASS")
        assert message.startswith("demo/go2/LICENSE.amass is the AMASS license")
        assert "cannot be published" in message

    def test_declare_reads_a_build_relative_path(self):
        d = declare("demo/LICENSE", license_template("Apache-2.0").encode())
        assert d is not None and d.spdx == "Apache-2.0" and d.tier == "notice"
        assert d.describe() == "demo/LICENSE (Apache-2.0)"
        notice = declare("demo/go2/NOTICE.go2", b"Copyright 2016 Unitree\n")
        assert notice is not None and notice.spdx is None
        assert notice.describe() == "demo/go2/NOTICE.go2"
        assert declare("LICENSE", b"the engine's") is None


# ── Generation ────────────────────────────────────────────────────────────────


class TestGeneration:
    def test_a_generated_text_is_tagged_and_carries_the_holder(self):
        text = generate_license_text("BSD-3-Clause", "2026 Example Lab")
        first, blank, holder = text.split("\n")[:3]
        assert first == "SPDX-License-Identifier: BSD-3-Clause"
        assert blank == ""
        assert holder == "Copyright (c) 2026 Example Lab."
        assert identify_license(text).spdx == "BSD-3-Clause"

    def test_texts_without_their_own_copyright_line_get_one_prepended(self):
        text = generate_license_text("Apache-2.0", "Example Lab", year=2026)
        assert text.split("\n")[2] == "Copyright (c) 2026 Example Lab"
        assert "Apache License" in text
        # No holder, no line to prepend.
        bare = generate_license_text("Apache-2.0", "", year=2026)
        assert bare.split("\n")[2].startswith("Apache License")

    def test_the_holder_is_the_whole_line_unless_a_year_is_passed(self):
        assert "Copyright (c) Someone\n" in generate_license_text("MIT", "Someone")
        assert "Copyright (c) 2026 Someone\n" in generate_license_text(
            "MIT", "Someone", year=2026
        )

    def test_only_bundled_ids_generate(self):
        with pytest.raises(
            ValueError, match="No standard text is bundled for 'CC-BY-SA-4.0'"
        ):
            generate_license_text("CC-BY-SA-4.0", "x")

    def test_resolve_license_copies_a_file_verbatim(self, tmp_path):
        src = tmp_path / "LICENSE.txt"
        src.write_bytes(b"custom terms\r\n")
        assert resolve_license(src) == b"custom terms\r\n"
        assert resolve_license(str(src)) == b"custom terms\r\n"

    def test_resolve_license_refuses_what_is_neither(self):
        with pytest.raises(ValueError, match="neither an SPDX identifier nor a file"):
            resolve_license("some words that are not a license id")

    def test_resolve_notice_takes_a_path_or_the_text(self, tmp_path):
        src = tmp_path / "NOTICE"
        src.write_bytes(b"from a file\n")
        assert resolve_notice(src) == b"from a file\n"
        assert (
            resolve_notice("Includes the Go2 model by Unitree")
            == b"Includes the Go2 model by Unitree\n"
        )


# ── Known assets ──────────────────────────────────────────────────────────────


class TestKnownAssets:
    def test_matches_menagerie_directories_model_names_and_task_ids(self):
        assert known_asset("unitree_g1").component == "unitree_g1"
        assert known_asset("Mjlab-Velocity-Flat-Unitree-Go1").component == "unitree_go1"
        assert known_asset("g1_29dof_rev_1_0").component == "unitree_g1"
        assert known_asset("anybotics_anymal_c").component == "anybotics_anymal_c"
        assert (
            known_asset("Mjlab-Velocity-Flat-Anymal-C").component
            == "anybotics_anymal_c"
        )

    def test_the_first_name_that_matches_wins_and_none_is_fine(self):
        assert known_asset(None, "simple", "unitree go2").component == "unitree_go2"
        assert known_asset("simple") is None
        assert known_asset() is None

    def test_a_known_attribution_is_a_tagged_generated_file(self):
        attribution = known_attribution("Mjlab-Velocity-Flat-Unitree-G1")
        assert attribution is not None
        assert attribution.component == "unitree_g1"
        assert attribution.license is not None
        assert (
            attribution.license.split(b"\n")[0]
            == b"SPDX-License-Identifier: BSD-3-Clause"
        )
        assert b"Unitree Robotics" in attribution.license
        assert attribution.spdx == "BSD-3-Clause"
        assert attribution.origin.startswith("known asset")
        assert attribution.files() == {"LICENSE.unitree_g1": attribution.license}


# ── Detection beside a model ──────────────────────────────────────────────────

_BSD = license_template("BSD-3-Clause").encode()
_APACHE = license_template("Apache-2.0").encode()


def _write_model(
    directory: Path, name: str = "scene.xml", *, mesh: bool = True, **compiler: str
) -> Path:
    """An MJCF that names a mesh it does not ship (``mesh=False`` for one that builds)."""
    directory.mkdir(parents=True, exist_ok=True)
    attrs = " ".join(f'{k}="{v}"' for k, v in compiler.items())
    asset = '<asset><mesh name="trunk" file="trunk.stl"/></asset>' if mesh else ""
    (directory / name).write_text(
        f'<mujoco model="m"><compiler {attrs}/>{asset}'
        '<worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
    )
    return directory / name


class TestDetection:
    def test_a_menagerie_layout_yields_the_models_file_byte_identical(self, tmp_path):
        model_dir = tmp_path / "mujoco_menagerie" / "unitree_go2"
        (tmp_path / "mujoco_menagerie").mkdir()
        (tmp_path / "mujoco_menagerie" / "LICENSE").write_bytes(
            _APACHE
        )  # the repository's
        xml = _write_model(model_dir, meshdir="assets")
        (model_dir / "LICENSE").write_bytes(_BSD)
        spec = mujoco.MjSpec.from_file(str(xml))

        found = detect_attributions(spec_asset_directories(spec))

        assert [a.component for a in found] == ["unitree_go2"]
        assert found[0].license == _BSD
        assert found[0].notice is None
        assert found[0].spdx == "BSD-3-Clause"
        assert found[0].files() == {"LICENSE.unitree_go2": _BSD}

    def test_a_playground_layout_reaches_the_model_through_its_meshes(self, tmp_path):
        # The XML lives in one checkout, its meshes resolve into another one level
        # below the model's LICENSE; the repository root above that must not be taken.
        menagerie = tmp_path / "external_deps" / "mujoco_menagerie"
        (menagerie / "unitree_go1" / "assets").mkdir(parents=True)
        (menagerie / "LICENSE").write_bytes(_APACHE)
        (menagerie / "unitree_go1" / "LICENSE").write_bytes(_BSD)
        xml = _write_model(
            tmp_path / "playground" / "locomotion" / "go1" / "xmls",
            meshdir="../../../../external_deps/mujoco_menagerie/unitree_go1/assets",
        )
        spec = mujoco.MjSpec.from_file(str(xml))

        found = detect_attributions(spec_asset_directories(spec))

        assert [(a.component, a.license) for a in found] == [("unitree_go1", _BSD)]

    def test_two_parents_not_more(self, tmp_path):
        (tmp_path / "LICENSE").write_bytes(_APACHE)
        xml = _write_model(tmp_path / "a" / "b" / "c")
        spec = mujoco.MjSpec.from_file(str(xml))
        assert detect_attributions(spec_asset_directories(spec)) == []
        # One level nearer and it is in range.
        xml = _write_model(tmp_path / "a" / "b")
        spec = mujoco.MjSpec.from_file(str(xml))
        assert [
            a.license for a in detect_attributions(spec_asset_directories(spec))
        ] == [_APACHE]

    def test_the_nearest_level_wins_and_content_is_de_duplicated(self, tmp_path):
        model_dir = tmp_path / "robot"
        xml = _write_model(model_dir, meshdir="assets")
        (model_dir / "LICENSE").write_bytes(_BSD)
        (model_dir / "NOTICE").write_bytes(b"notice\n")
        (model_dir / "assets").mkdir()
        # The mesh directory itself carries a copy of the same text.
        (model_dir / "assets" / "LICENSE.txt").write_bytes(_BSD)
        spec = mujoco.MjSpec.from_file(str(xml))

        found = detect_attributions(spec_asset_directories(spec))

        assert len(found) == 1
        assert found[0].component == "robot"
        assert (found[0].license, found[0].notice) == (_BSD, b"notice\n")

    def test_a_spec_from_a_string_has_nowhere_to_look(self):
        spec = mujoco.MjSpec.from_string(
            '<mujoco model="m"><asset><mesh name="t" file="trunk.stl"/></asset>'
            '<worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>'
        )
        assert spec_asset_directories(spec) == []

    def test_only_license_shaped_names_count(self, tmp_path):
        xml = _write_model(tmp_path / "m")
        (tmp_path / "m" / "LICENSING_FAQ.md").write_text("no")
        (tmp_path / "m" / "COPYING.LESSER").write_bytes(_APACHE)
        spec = mujoco.MjSpec.from_file(str(xml))
        found = detect_attributions(spec_asset_directories(spec))
        assert [a.license for a in found] == [_APACHE]


# ── The build ─────────────────────────────────────────────────────────────────


def _project_with_scene(builder: Builder, minimal_model, name="Demo"):
    project = builder.add_project(name=name)
    scene = project.add_scene(name="Humanoid", model=minimal_model)
    return project, scene


class TestBuildOutput:
    def test_the_builders_license_reaches_every_project(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder(license="Apache-2.0", copyright="2026 Example")
        _project_with_scene(builder, minimal_model, "Demo")
        _project_with_scene(builder, minimal_model, "Other")
        build_manifest(builder, tmp_path / "dist")

        for project in ("demo", "other"):
            text = (tmp_path / "dist" / project / "LICENSE").read_text()
            assert text.split("\n")[0] == "SPDX-License-Identifier: Apache-2.0"
            assert "Copyright (c) 2026 Example" in text
        assert not (tmp_path / "dist" / "demo" / "NOTICE").exists()

    def test_a_project_can_override_it_with_a_file_copied_verbatim(
        self, tmp_path, minimal_model, build_manifest
    ):
        src = tmp_path / "MY-LICENSE"
        src.write_bytes(b"custom terms\n")
        builder = Builder(license="MIT", copyright="x")
        builder.add_project(name="Demo", license=src).add_scene(
            name="S", model=minimal_model
        )
        build_manifest(builder, tmp_path / "dist")
        assert (
            tmp_path / "dist" / "demo" / "LICENSE"
        ).read_bytes() == b"custom terms\n"

    def test_set_license_and_set_notice(self, tmp_path, minimal_model, build_manifest):
        builder = Builder()
        project, _ = _project_with_scene(builder, minimal_model)
        assert project.set_license("MIT", copyright="2026 Someone") is project
        assert project.set_notice("Built with mjswan.") is project
        build_manifest(builder, tmp_path / "dist")
        assert (
            "Copyright (c) 2026 Someone"
            in (tmp_path / "dist" / "demo" / "LICENSE").read_text()
        )
        assert (
            tmp_path / "dist" / "demo" / "NOTICE"
        ).read_bytes() == b"Built with mjswan.\n"

    def test_a_bad_license_argument_fails_at_the_call(self, minimal_model):
        with pytest.raises(ValueError, match="No standard text is bundled"):
            Builder(license="CC-BY-SA-4.0")
        with pytest.raises(ValueError, match="neither an SPDX identifier nor a file"):
            Builder().add_project(name="x", license="not a thing")

    def test_attributions_are_written_beside_the_scene(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder()
        _, scene = _project_with_scene(builder, minimal_model)
        scene.add_attribution("lafan1", license="CC-BY-4.0", copyright="Ubisoft")
        scene.add_attribution("clip", notice="Retargeted by us.")
        build_manifest(builder, tmp_path / "dist")

        scene_dir = tmp_path / "dist" / "demo" / "humanoid"
        lafan = (scene_dir / "LICENSE.lafan1").read_text()
        assert lafan.split("\n")[0] == "SPDX-License-Identifier: CC-BY-4.0"
        assert "Copyright (c)" in lafan and "Ubisoft" in lafan
        assert not (scene_dir / "NOTICE.lafan1").exists()
        assert (scene_dir / "NOTICE.clip").read_bytes() == b"Retargeted by us.\n"
        assert not (scene_dir / "LICENSE.clip").exists()

    def test_add_attribution_replaces_and_clear_removes(self, minimal_model):
        _, scene = _project_with_scene(Builder(), minimal_model)
        scene.add_attribution("go2", license="BSD-3-Clause", copyright="a")
        scene.add_attribution("go2", license="MIT", copyright="b")
        assert [a.spdx for a in scene._config.attributions] == ["MIT"]
        assert scene.clear_attributions() is scene
        assert scene._config.attributions == []

    def test_add_attribution_refuses_nothing_and_bad_names(self, minimal_model):
        _, scene = _project_with_scene(Builder(), minimal_model)
        with pytest.raises(ValueError, match="needs a license, a notice, or both"):
            scene.add_attribution("x")
        with pytest.raises(ValueError, match="Component 'has space'"):
            scene.add_attribution("has space", notice="n")

    def test_detected_files_are_written_and_the_manifest_does_not_change(
        self, tmp_path, minimal_model, build_manifest
    ):
        model_dir = tmp_path / "unitree_go2"
        xml = _write_model(model_dir, mesh=False)
        (model_dir / "LICENSE").write_bytes(_BSD)

        plain = Builder()
        plain.add_project(name="Demo").add_scene(name="S", model=minimal_model)
        plain_manifest = build_manifest(plain, tmp_path / "plain")

        licensed = Builder(license="Apache-2.0", copyright="x")
        licensed.add_project(name="Demo").add_scene(
            name="S", spec=mujoco.MjSpec.from_file(str(xml))
        )
        licensed_manifest = build_manifest(licensed, tmp_path / "licensed")

        written = tmp_path / "licensed" / "demo" / "s" / "LICENSE.unitree_go2"
        assert written.read_bytes() == _BSD
        # The scene asset differs (.mjb vs .mjz); everything else in the manifest is
        # the same with and without license files — they are not manifest keys.
        for m in (plain_manifest, licensed_manifest):
            m["projects"][0]["scenes"][0].pop("scene")
        assert plain_manifest == licensed_manifest
        assert "license" not in json.dumps(licensed_manifest).lower()

    def test_the_document_carries_the_files_and_publish_takes_them(
        self, tmp_path, minimal_model, build_manifest
    ):
        builder = Builder(license="Apache-2.0", copyright="x")
        _, scene = _project_with_scene(builder, minimal_model)
        scene.add_attribution("go2", license="BSD-3-Clause", copyright="Unitree")
        out = tmp_path / "dist"
        build_manifest(builder, out)
        (out / "LICENSE").write_text(
            "the engine's Apache-2.0"
        )  # what the SPA copy puts there

        files = {p.as_posix() for p in document_files(out)}
        assert {"demo/LICENSE", "demo/humanoid/LICENSE.go2"} <= files
        assert "LICENSE" not in files

        unpacked = unpack_document(write_document(out), tmp_path / "unpacked")
        assert (unpacked / "demo" / "LICENSE").read_bytes() == (
            out / "demo" / "LICENSE"
        ).read_bytes()
        assert (unpacked / "demo" / "humanoid" / "LICENSE.go2").read_bytes() == (
            out / "demo" / "humanoid" / "LICENSE.go2"
        ).read_bytes()

        plan = plan_publish(out)
        paths = {f.upload_path for f in plan.files}
        assert {"demo/LICENSE", "demo/humanoid/LICENSE.go2"} <= paths
        assert "LICENSE" not in paths
        by_path = {e["path"]: e for e in plan.manifest()}
        assert by_path["demo/LICENSE"]["contentType"] == LICENSE_CONTENT_TYPE
        assert [d.describe() for d in plan.licenses] == [
            "demo/LICENSE (Apache-2.0)",
            "demo/humanoid/LICENSE.go2 (BSD-3-Clause)",
        ]


class TestInfoCli:
    def test_info_lists_the_license_files_with_their_identifiers(
        self, tmp_path, minimal_model, build_manifest
    ):
        from typer.testing import CliRunner

        from mjswan._cli import app

        builder = Builder(license="Apache-2.0", copyright="x")
        _, scene = _project_with_scene(builder, minimal_model)
        scene.add_attribution("go2", license="BSD-3-Clause", copyright="Unitree")
        scene.add_attribution("clip", notice="n")
        out = tmp_path / "dist"
        build_manifest(builder, out)

        result = CliRunner().invoke(app, ["info", str(out)])
        assert result.exit_code == 0, result.output
        assert "License: Apache-2.0" in result.output
        assert "demo/LICENSE" in result.output
        assert "License: BSD-3-Clause" in result.output
        assert "NOTICE.clip" in result.output


def test_the_size_cap_matches_the_platform():
    assert LICENSE_FILE_MAX_BYTES == 64 * 1024
    assert Attribution("x", license=b"a").files() == {"LICENSE.x": b"a"}


@pytest.fixture
def build_manifest(monkeypatch):
    """A local copy of conftest's, so this module reads on its own: run `_save_web`
    with the Node build and SPA copy mocked and return the manifest written."""

    def build(builder, out: Path) -> dict:
        monkeypatch.setattr("mjswan.builder.ClientBuilder", MagicMock())
        monkeypatch.setattr("mjswan.builder.install_spa", MagicMock(return_value=True))
        builder._save_web(out)
        return json.loads((out / "manifest.json").read_text())

    return build
