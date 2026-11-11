"""Tests for mjswan Cloud publishing (mjswan.cloud.publish + app.publish + CLI).

Layer: L1 (pure Python, no MuJoCo/ONNX/network required; safe for pre-commit).
The HTTP transport is faked, so no real requests are made.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from mjswan.cloud.publish import (
    DATA_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_PIECES,
    MAX_TOTAL_BYTES,
    HttpResponse,
    HttpTransport,
    PublishError,
    plan_publish,
    publish_dist,
    resolve_api_base,
    resolve_token,
    resolve_web_base,
    simulation_url,
)
from mjswan.cloud.transport import USER_AGENT


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path, monkeypatch):
    """Keep token resolution from picking up a developer's real `mjswan login`
    session (~/.config/mjswan) — point credential storage at an empty tmp dir."""
    monkeypatch.setenv("MJSWAN_CONFIG_HOME", str(tmp_path / "mjswan-cfg"))


# ── Fakes ──────────────────────────────────────────────────────────────────


def _hashes(file_map: dict) -> list[str]:
    """Every piece and text the map names, once each, the texts first."""
    out = dict.fromkeys(file_map["licenses"].values())
    for file in file_map["files"].values():
        for e in file.get("entries", [file]):
            for sha, _ in e.get("chunks", [[e["sha256"], e["size"]]]):
                out[sha] = None
    return list(out)


class FakeTransport(HttpTransport):
    """Records calls; asks for every piece the file map names but those ``held``, and
    answers verify by counting them down."""

    def __init__(
        self,
        *,
        sim_id: str = "abc1234",
        upload_id: str = "11111111-1111-1111-1111-111111111111",
        commit_id: str = "abc1234",
        session_status: int = 200,
        commit_status: int = 200,
        commit_body: dict | None = None,
        put_status: int = 200,
        session_error: dict | None = None,
        held: frozenset[str] = frozenset(),
        released: list[str] | None = None,
    ) -> None:
        self.posts: list[tuple[str, dict, str]] = []
        self.puts: list[tuple[str, int, str]] = []
        self.put_headers: list[tuple[str, str, str | None]] = []
        self.put_bodies: dict[str, bytes] = {}
        self._sim_id = sim_id
        self._upload_id = upload_id
        self._commit_id = commit_id
        self._session_status = session_status
        self._commit_status = commit_status
        self._commit_body = commit_body
        self._put_status = put_status
        self._session_error = session_error
        self._held = held
        self._released = released
        self._pending = 0

    def _presigned(self, sha: str) -> dict:
        # Shaped like the real `PresignedPiece`: the exact header values the server signed.
        return {
            "sha256": sha,
            "url": f"https://r2.example.com/uploads/{self._upload_id}/{sha}",
            "contentType": "application/octet-stream",
            "cacheControl": "no-store",
        }

    def post_json(self, url: str, body: dict, token: str) -> HttpResponse:
        self.posts.append((url, body, token))
        if url.endswith("/upload-session"):
            if self._session_error is not None:
                return HttpResponse(
                    self._session_status, json.dumps(self._session_error).encode()
                )
            uploads = [
                self._presigned(sha)
                for sha in _hashes(body["files"])
                if sha not in self._held
            ]
            self._pending = len(uploads)
            payload = {
                "upload_id": self._upload_id,
                "sim_id": self._sim_id,
                "uploads": uploads,
                "expires_in": 900,
            }
            return HttpResponse(self._session_status, json.dumps(payload).encode())
        if url.endswith("/verify"):
            self._pending = max(0, self._pending - body["limit"])
            return HttpResponse(200, json.dumps({"remaining": self._pending}).encode())
        if url.endswith("/commit"):
            if self._pending:
                return HttpResponse(409, b'{"code": "not_verified"}')
            if self._released:
                uploads = [self._presigned(sha) for sha in self._released]
                self._released = None
                self._pending = len(uploads)
                payload = {"code": "files_released", "uploads": uploads}
                return HttpResponse(409, json.dumps(payload).encode())
            body_out = (
                self._commit_body
                if self._commit_body is not None
                else {"id": self._commit_id}
            )
            return HttpResponse(self._commit_status, json.dumps(body_out).encode())
        raise AssertionError(f"unexpected POST {url}")

    def put_bytes(
        self,
        url: str,
        data: bytes,
        content_type: str,
        cache_control: str | None = None,
    ) -> HttpResponse:
        self.puts.append((url, len(data), content_type))
        self.put_headers.append((url, content_type, cache_control))
        self.put_bodies[url.rsplit("/", 1)[-1]] = bytes(data)
        return HttpResponse(self._put_status, b"")


def _make_dist(
    tmp_path: Path, *, uses_custom_js: bool = False, scene: str = "scene.mjz"
) -> Path:
    """A realistic built dist/: data files plus app-shell files to be excluded."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    scene_dir = dist / "demo" / "humanoid"
    (scene_dir / "policy").mkdir(parents=True)
    (scene_dir / "assets").mkdir(parents=True)
    (scene_dir / "mdp" / "mdp_0" / "obs").mkdir(parents=True)

    config = {
        "format": 1,
        "version": "0.7.0",
        "uses_custom_js": uses_custom_js,
        "projects": [
            {
                "id": "demo",
                "name": "Demo",
                "scenes": [
                    {
                        "id": "humanoid",
                        "name": "Humanoid",
                        "scene": scene,
                        "mdps": [{"id": "mdp_0"}],
                        "policies": [
                            {
                                "id": "walk",
                                "name": "Walk",
                                "mdp": "mdp_0",
                                "onnx": "policy/walk.onnx",
                            }
                        ],
                    }
                ],
            }
        ],
    }
    (dist / "manifest.json").write_text(json.dumps(config))

    # Data files (should be uploaded).
    with zipfile.ZipFile(scene_dir / scene, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("scene.xml", "<mujoco/>")
        archive.writestr("assets/box.stl", b"STL" * 10)
    (scene_dir / "policy" / "walk.onnx").write_bytes(b"ONNX" * 10)
    (scene_dir / "mdp" / "mdp_0" / "obs" / "actor.onnx").write_bytes(b"GRAPH" * 2)
    (scene_dir / "assets" / "walk_run.npz").write_bytes(b"NPZ" * 10)

    # App-shell files (must NOT be uploaded).
    (dist / "index.html").write_text("<!doctype html>")
    (dist / "assets" / "index-abc.js").write_text("console.log(1)")
    (dist / "assets" / "index-abc.css").write_text("body{}")
    (dist / "assets" / "mujoco-x.wasm").write_bytes(b"\0asm")
    (dist / "assets" / ".mjswan-build-meta.json").write_text("{}")
    return dist


# ── HttpTransport User-Agent (Cloudflare 1010 guard) ─────────────────────────


class TestHttpTransportUserAgent:
    def test_post_and_put_set_a_non_default_user_agent(self, monkeypatch):
        """Cloudflare 403s the stdlib's Python-urllib agent; we must override it."""
        seen: list = []

        class _Resp:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req):
            seen.append(req.get_header("User-agent"))
            return _Resp()

        monkeypatch.setattr("mjswan.cloud.publish.urllib.request.urlopen", fake_urlopen)
        t = HttpTransport()
        t.post_json("https://api.mjswan.com/x", {}, "tok")
        t.put_bytes("https://r2.example.com/x", b"data", "application/json")

        assert seen == [USER_AGENT, USER_AGENT]
        assert "Python-urllib" not in USER_AGENT
        assert USER_AGENT.startswith("mjswan/")


# ── plan_publish ─────────────────────────────────────────────────────────────


class TestPlanPublish:
    def test_refuses_a_symlink(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        outside = tmp_path / "outside.txt"
        outside.write_text("not part of the build")
        (dist / "demo" / "humanoid" / "assets" / "leak.npz").symlink_to(outside)

        with pytest.raises(PublishError, match="Symbolic links"):
            plan_publish(dist)

    def test_collects_only_data_files(self, tmp_path: Path):
        plan = plan_publish(_make_dist(tmp_path))
        paths = {f.upload_path for f in plan.files}
        assert paths == {
            "manifest.json",
            "demo/humanoid/scene.mjz",
            "demo/humanoid/policy/walk.onnx",
            "demo/humanoid/mdp/mdp_0/obs/actor.onnx",
            "demo/humanoid/assets/walk_run.npz",
        }

    def test_manifest_uploads_from_the_root_it_already_sits_at(self, tmp_path: Path):
        plan = plan_publish(_make_dist(tmp_path))
        manifest = next(f for f in plan.files if f.upload_path == "manifest.json")
        assert manifest.source == tmp_path / "dist" / "manifest.json"

    def test_the_spa_bundles_own_json_is_not_document_data(self, tmp_path: Path):
        # `assets/` at the root is the SPA's directory, so a build-cache key there is
        # engine bookkeeping rather than simulation data.
        plan = plan_publish(_make_dist(tmp_path))
        assert not any(f.upload_path.startswith("assets/") for f in plan.files)

    def test_excludes_html_js_css_wasm(self, tmp_path: Path):
        plan = plan_publish(_make_dist(tmp_path))
        paths = {f.upload_path for f in plan.files}
        assert not any(p.endswith((".html", ".js", ".css", ".wasm")) for p in paths)

    def test_the_file_map_lists_every_file_but_the_descriptor(self, tmp_path: Path):
        from mjswan.cloud.blob import sha256

        plan = plan_publish(_make_dist(tmp_path))
        files = plan.file_map["files"]
        assert plan.file_map["format"] == 1 and plan.file_map["licenses"] == {}
        assert set(files) == {
            "demo/humanoid/scene.mjz",
            "demo/humanoid/policy/walk.onnx",
            "demo/humanoid/mdp/mdp_0/obs/actor.onnx",
            "demo/humanoid/assets/walk_run.npz",
        }
        assert files["demo/humanoid/policy/walk.onnx"] == {
            "sha256": sha256(b"ONNX" * 10),
            "size": 40,
        }
        # An archive is its entries in their order, inflated; the platform zips them again.
        assert files["demo/humanoid/scene.mjz"] == {
            "entries": [
                {"name": "scene.xml", "sha256": sha256(b"<mujoco/>"), "size": 9},
                {"name": "assets/box.stl", "sha256": sha256(b"STL" * 10), "size": 30},
            ]
        }
        assert plan.pieces[sha256(b"<mujoco/>")] == b"<mujoco/>"

    def test_a_cut_file_sends_its_pieces(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        policy = (Path(__file__).parent / "blob_vectors" / "policy.onnx").read_bytes()
        (dist / "demo/humanoid/policy/walk.onnx").write_bytes(policy)
        plan = plan_publish(dist)
        chunks = plan.file_map["files"]["demo/humanoid/policy/walk.onnx"]["chunks"]
        assert len(chunks) == 6
        assert b"".join(plan.pieces[sha] for sha, _ in chunks) == policy

    def test_rejects_custom_js(self, tmp_path: Path):
        with pytest.raises(PublishError) as exc:
            plan_publish(_make_dist(tmp_path, uses_custom_js=True))
        assert exc.value.file == "manifest.json"
        assert "custom-ts" in str(exc.value).lower()

    def test_refuses_an_mjb_scene_rather_than_publishing_it_without_one(
        self, tmp_path: Path
    ):
        with pytest.raises(PublishError) as exc:
            plan_publish(_make_dist(tmp_path, scene="scene.mjb"))
        assert exc.value.file == "demo/humanoid/scene.mjb"
        assert "add_scene(spec=...)" in str(exc.value)

    def test_missing_config(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(PublishError, match="No manifest.json"):
            plan_publish(empty)

    def test_invalid_json(self, tmp_path: Path):
        dist = tmp_path / "dist"
        dist.mkdir(parents=True)
        (dist / "manifest.json").write_text("{not json")
        with pytest.raises(PublishError, match="Invalid manifest.json"):
            plan_publish(dist)

    def test_rejects_oversized_file(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_FILE_BYTES", 5)
        with pytest.raises(PublishError) as exc:
            plan_publish(dist)
        assert exc.value.file is not None

    def test_rejects_too_many_files(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_FILES", 2)
        with pytest.raises(PublishError, match="Too many files"):
            plan_publish(dist)

    def test_rejects_total_size(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_TOTAL_BYTES", 10)
        with pytest.raises(PublishError, match="scenes unpacked"):
            plan_publish(dist)

    def test_a_build_past_the_cap_is_refused_before_it_is_held_whole(
        self, tmp_path: Path, monkeypatch
    ):
        from mjswan.cloud import blob

        dist = _make_dist(tmp_path)
        with zipfile.ZipFile(dist / "demo/humanoid/scene.mjz", "w") as archive:
            for i in range(100):
                archive.writestr(f"m{i}.stl", bytes([i]) * 50)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_TOTAL_BYTES", 1000)
        cuts = []
        monkeypatch.setattr(
            "mjswan.cloud.publish.cut",
            lambda name, data: cuts.append(name) or blob.cut(name, data),
        )
        with pytest.raises(PublishError, match="scenes unpacked"):
            plan_publish(dist)
        # The three other files, then entries only up to the cap, not all 100.
        assert len(cuts) <= 3 + 1000 // 50

    def test_a_piece_two_files_share_counts_once(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        (dist / "demo/humanoid/assets/again.npz").write_bytes(b"NPZ" * 10)
        # scene.xml 9 + box.stl 30 + walk.onnx 40 + actor.onnx 10 + walk_run.npz 30
        monkeypatch.setattr("mjswan.cloud.publish.MAX_TOTAL_BYTES", 119)
        plan_publish(dist)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_PIECES", 4)
        with pytest.raises(PublishError, match="over 4 pieces"):
            plan_publish(dist)

    def test_an_archive_is_held_to_the_cap_entry_by_entry(
        self, tmp_path: Path, monkeypatch
    ):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.cloud.publish.MAX_FILE_BYTES", 400)
        with zipfile.ZipFile(dist / "demo/humanoid/scene.mjz", "w") as archive:
            archive.writestr("scene.xml", b"x" * 300)
            archive.writestr("a.stl", b"y" * 300)
        plan_publish(dist)  # the archive is larger, its entries are not
        with zipfile.ZipFile(dist / "demo/humanoid/scene.mjz", "a") as archive:
            archive.writestr("big.stl", b"x" * 401, zipfile.ZIP_DEFLATED)
        with pytest.raises(PublishError) as exc:
            plan_publish(dist)
        assert exc.value.file == "demo/humanoid/scene.mjz/big.stl"

    def test_a_damaged_archive_is_refused(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        (dist / "demo/humanoid/scene.mjz").write_bytes(b"MJZ" * 10)
        with pytest.raises(PublishError, match="not a readable zip archive") as exc:
            plan_publish(dist)
        assert exc.value.file == "demo/humanoid/scene.mjz"

    def test_default_caps_are_spec_values(self):
        assert MAX_FILE_BYTES == 50 * 1024 * 1024
        assert MAX_TOTAL_BYTES == 128 * 1024 * 1024
        assert MAX_PIECES == 1000
        assert MAX_FILES == 200
        assert DATA_EXTENSIONS == frozenset(
            {".json", ".mjz", ".onnx", ".npz", ".ply", ".spz", ".vrm", ".glb"}
        )


# ── license files (ADR 0007 §3) ──────────────────────────────────────────────


def _with_licenses(dist: Path, files: dict[str, str | bytes]) -> Path:
    for rel, content in files.items():
        target = dist / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content)
        else:
            target.write_bytes(content)
    return dist


class TestLicenseFiles:
    def test_admits_project_and_scene_files_as_text_never_the_root(self, tmp_path):
        from mjswan.license import license_template

        dist = _with_licenses(
            _make_dist(tmp_path),
            {
                "LICENSE": "the engine's Apache-2.0",
                "demo/LICENSE": license_template("Apache-2.0"),
                "demo/NOTICE": "Built with mjswan.\n",
                "demo/humanoid/LICENSE.unitree_go2": license_template("BSD-3-Clause"),
                "demo/humanoid/NOTICE": "a notice for the scene itself\n",
                "demo/humanoid/mdp/LICENSE": "too deep to count\n",
            },
        )
        plan = plan_publish(dist)
        paths = {f.upload_path for f in plan.files}
        assert {
            "demo/LICENSE",
            "demo/NOTICE",
            "demo/humanoid/LICENSE.unitree_go2",
            "demo/humanoid/NOTICE",
        } <= paths
        assert "LICENSE" not in paths
        assert "demo/humanoid/mdp/LICENSE" not in paths
        # Listed by their texts' hashes, never as pieces.
        assert set(plan.file_map["licenses"]) == {
            "demo/LICENSE",
            "demo/NOTICE",
            "demo/humanoid/LICENSE.unitree_go2",
            "demo/humanoid/NOTICE",
        }
        assert not any("LICENSE" in p or "NOTICE" in p for p in plan.file_map["files"])
        assert [d.describe() for d in plan.licenses] == [
            "demo/LICENSE (Apache-2.0)",
            "demo/NOTICE",
            "demo/humanoid/LICENSE.unitree_go2 (BSD-3-Clause)",
            "demo/humanoid/NOTICE",
        ]
        assert plan.license_warnings() == []

    def test_the_declaration_is_printed_before_the_upload(self, tmp_path):
        from mjswan.license import license_template

        dist = _with_licenses(
            _make_dist(tmp_path), {"demo/LICENSE": license_template("MIT")}
        )
        messages: list[str] = []
        transport = FakeTransport()
        publish_dist(
            dist, token="tok", transport=transport, on_progress=messages.append
        )
        assert messages[0] == "License file: demo/LICENSE (MIT)"
        text = (dist / "demo/LICENSE").read_bytes()
        assert text in transport.put_bodies.values()

    def test_a_restricted_license_warns_and_still_publishes(self, tmp_path):
        dist = _with_licenses(
            _make_dist(tmp_path),
            {
                "demo/humanoid/LICENSE.lafan1": "Attribution-NonCommercial-NoDerivatives "
                "4.0 International Public License\n..."
            },
        )
        warnings_seen: list[str] = []
        transport = FakeTransport()
        result = publish_dist(
            dist, token="tok", transport=transport, on_warning=warnings_seen.append
        )
        assert result.id == "abc1234"
        assert len(warnings_seen) == 1
        assert warnings_seen[0].startswith(
            "demo/humanoid/LICENSE.lafan1 is CC-BY-NC-ND-4.0 (non-commercial use only"
        )

    def test_a_restricted_license_warns_through_warnings_by_default(self, tmp_path):
        dist = _with_licenses(
            _make_dist(tmp_path),
            {"demo/LICENSE": "GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007\n"},
        )
        with pytest.warns(RuntimeWarning, match="demo/LICENSE is GPL-3.0-only"):
            publish_dist(dist, token="tok", transport=FakeTransport())

    def test_a_blocked_license_is_refused_before_any_network_call(self, tmp_path):
        dist = _with_licenses(
            _make_dist(tmp_path),
            {
                "demo/humanoid/LICENSE.amass": "SPDX-License-Identifier: LicenseRef-AMASS\n"
            },
        )
        transport = FakeTransport()
        with pytest.raises(PublishError) as exc:
            publish_dist(dist, token="tok", transport=transport)
        assert exc.value.file == "demo/humanoid/LICENSE.amass"
        assert "the AMASS license" in str(exc.value)
        assert "cannot be published" in str(exc.value)
        assert transport.posts == []

    def test_the_max_planck_text_is_recognised_without_a_tag(self, tmp_path):
        dist = _with_licenses(
            _make_dist(tmp_path),
            {
                "demo/humanoid/LICENSE.smpl": "Software Copyright License for "
                "non-commercial scientific research purposes\n"
            },
        )
        with pytest.raises(PublishError, match="Max Planck non-commercial"):
            plan_publish(dist)

    def test_a_custom_text_is_carried_without_comment(self, tmp_path):
        dist = _with_licenses(
            _make_dist(tmp_path), {"demo/LICENSE": "Use it however you like.\n"}
        )
        plan = plan_publish(dist)
        assert [d.describe() for d in plan.licenses] == ["demo/LICENSE (Custom)"]
        assert plan.license_warnings() == []

    def test_an_oversized_license_file_is_refused(self, tmp_path):
        from mjswan.license import LICENSE_FILE_MAX_BYTES

        dist = _with_licenses(
            _make_dist(tmp_path), {"demo/LICENSE": b"x" * (LICENSE_FILE_MAX_BYTES + 1)}
        )
        with pytest.raises(
            PublishError, match="License file exceeds the 64 KiB limit"
        ) as exc:
            plan_publish(dist)
        assert exc.value.file == "demo/LICENSE"

    def test_the_cli_prints_the_warning(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from mjswan.cli import app
        from mjswan.cloud.publish import PublishResult

        def fake_publish_dist(dist_dir, **kwargs):
            kwargs["on_warning"]("demo/LICENSE is GPL-3.0-only (copyleft).")
            return PublishResult(id="s1", sim_id="s1", upload_id="u")

        monkeypatch.setattr("mjswan.cloud.publish.publish_dist", fake_publish_dist)
        result = CliRunner().invoke(
            app, ["publish", str(_make_dist(tmp_path)), "--token", "tok"]
        )
        assert result.exit_code == 0, result.output
        assert "Warning:" in result.output
        assert "GPL-3.0-only" in result.output


# ── token resolution ─────────────────────────────────────────────────────────


class TestResolveToken:
    def test_explicit_token(self, monkeypatch):
        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)
        assert resolve_token("tok-123") == "tok-123"

    def test_env_token(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_TOKEN", "env-tok")
        assert resolve_token(None) == "env-tok"

    def test_missing_token(self, monkeypatch):
        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)
        with pytest.raises(PublishError, match="MJSWAN_TOKEN"):
            resolve_token(None)


class TestResolveApiBase:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_API_BASE", "https://env.example.com")
        assert (
            resolve_api_base("https://flag.example.com/") == "https://flag.example.com"
        )

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_API_BASE", "https://v2-api.example.com/")
        assert resolve_api_base(None) == "https://v2-api.example.com"

    def test_default(self, monkeypatch):
        monkeypatch.delenv("MJSWAN_API_BASE", raising=False)
        assert resolve_api_base(None) == "https://api.mjswan.com"

    def test_publish_dist_uses_env_base(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MJSWAN_API_BASE", "https://v2-api.example.com")
        transport = FakeTransport()
        publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        assert transport.posts[0][0] == (
            "https://v2-api.example.com/api/simulations/upload-session"
        )


class TestResolveWebBase:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_WEB_BASE", "https://env.example.com")
        assert (
            resolve_web_base("https://flag.example.com/") == "https://flag.example.com"
        )

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_WEB_BASE", "https://web.example.com/")
        assert resolve_web_base() == "https://web.example.com"

    def test_default(self, monkeypatch):
        monkeypatch.delenv("MJSWAN_WEB_BASE", raising=False)
        assert resolve_web_base() == "https://mjswan.com"

    def test_simulation_url(self, monkeypatch):
        monkeypatch.delenv("MJSWAN_WEB_BASE", raising=False)
        assert simulation_url("7kuhIq6") == "https://mjswan.com/s/7kuhIq6"


# ── publish_dist (full flow with fake transport) ─────────────────────────────


class TestPublishDist:
    def test_happy_path(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        result = publish_dist(dist, title="My Sim", token="tok", transport=transport)
        assert result.id == "abc1234"
        assert result.upload_id == "11111111-1111-1111-1111-111111111111"

        # upload-session, verify until nothing remains, then commit.
        steps = [url.rsplit("/", 1)[-1] for url, _, _ in transport.posts]
        assert steps == ["upload-session", "verify", "commit"]

        # Every piece of the four data files, and nothing of the descriptor or the shell.
        bodies = transport.put_bodies.values()
        assert len(transport.puts) == 5
        assert b"ONNX" * 10 in bodies and b"<mujoco/>" in bodies
        assert not any(b"projects" in b or b"doctype" in b for b in bodies)

    def test_verify_is_called_until_nothing_remains(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("mjswan.cloud.publish.VERIFY_STEP", 2)
        transport = FakeTransport()
        messages: list[str] = []
        publish_dist(
            _make_dist(tmp_path),
            token="tok",
            transport=transport,
            on_progress=messages.append,
        )
        verifies = [b for u, b, _ in transport.posts if u.endswith("/verify")]
        assert verifies == [{"upload_id": transport._upload_id, "limit": 2}] * 3
        assert [m for m in messages if m.startswith("Checking")] == [
            "Checking 0 of 5 piece(s)…",
            "Checking 2 of 5 piece(s)…",
            "Checking 4 of 5 piece(s)…",
        ]

    @pytest.mark.parametrize(
        "answer",
        [
            lambda left: {"remaining": left},
            lambda left: {},
            lambda left: {"remaining": str(left - 1)},
            lambda left: {"remaining": -1},
        ],
        ids=["stuck", "missing", "a-string", "negative"],
    )
    def test_a_verify_count_that_does_not_fall_stops_the_publish(
        self, tmp_path: Path, answer
    ):
        class StuckTransport(FakeTransport):
            def post_json(self, url, body, token):
                if url.endswith("/verify"):
                    self.posts.append((url, body, token))
                    payload = json.dumps(answer(self._pending)).encode()
                    return HttpResponse(200, payload)
                return super().post_json(url, body, token)

        transport = StuckTransport()
        with pytest.raises(PublishError, match="verify answered"):
            publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        steps = [url.rsplit("/", 1)[-1] for url, _, _ in transport.posts]
        assert steps == ["upload-session", "verify"]

    @pytest.mark.parametrize("key", ["sha256", "url", "contentType", "cacheControl"])
    def test_an_upload_without_a_field_is_refused_before_any_put(
        self, tmp_path: Path, key: str
    ):
        class PartialTransport(FakeTransport):
            def post_json(self, url, body, token):
                resp = super().post_json(url, body, token)
                if url.endswith("/upload-session"):
                    payload = resp.json()
                    del payload["uploads"][-1][key]
                    return HttpResponse(200, json.dumps(payload).encode())
                return resp

        transport = PartialTransport()
        with pytest.raises(PublishError, match=f"without {key}"):
            publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        assert transport.puts == []

    def test_only_what_the_platform_lacks_is_sent(self, tmp_path: Path):
        from mjswan.cloud.blob import sha256

        held = frozenset(sha256(b) for b in (b"<mujoco/>", b"STL" * 10, b"NPZ" * 10))
        transport = FakeTransport(held=held)
        publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        assert sorted(transport.put_bodies.values()) == [b"GRAPH" * 2, b"ONNX" * 10]

        everything = FakeTransport(
            held=held | {sha256(b"GRAPH" * 2), sha256(b"ONNX" * 10)}
        )
        publish_dist(_make_dist(tmp_path / "again"), token="tok", transport=everything)
        steps = [url.rsplit("/", 1)[-1] for url, _, _ in everything.posts]
        assert everything.puts == [] and steps == ["upload-session", "commit"]

    def test_a_piece_a_takedown_released_goes_again(self, tmp_path: Path):
        from mjswan.cloud.blob import sha256

        released = sha256(b"NPZ" * 10)
        transport = FakeTransport(held=frozenset({released}), released=[released])
        result = publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        assert result.id == "abc1234"
        steps = [url.rsplit("/", 1)[-1] for url, _, _ in transport.posts]
        assert steps == ["upload-session", "verify", "commit", "verify", "commit"]
        assert transport.put_bodies[released] == b"NPZ" * 10

    def test_put_sends_the_headers_the_server_signed(self, tmp_path: Path):
        """Both are in `SignedHeaders`, so a value this side invents is a 403."""
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(dist, title="My Sim", token="tok", transport=transport)

        assert transport.put_headers  # the loop ran at all
        assert {(ct, cc) for _, ct, cc in transport.put_headers} == {
            ("application/octet-stream", "no-store")
        }

    def test_commit_re_resolves_the_token(self, tmp_path: Path, monkeypatch):
        """A long upload can outlive the token, and the commit is what pays."""
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        tokens = iter(["token-at-upload", "token-at-verify", "token-at-commit"])
        monkeypatch.setattr(
            "mjswan.cloud.publish.resolve_token", lambda _t: next(tokens)
        )

        publish_dist(dist, title="T", token="ignored", transport=transport)

        used = {url.rsplit("/", 1)[-1]: tok for url, _, tok in transport.posts}
        assert used == {
            "upload-session": "token-at-upload",
            "verify": "token-at-verify",
            "commit": "token-at-commit",
        }

    def test_authorization_header_and_body(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(
            dist,
            title="T",
            description="D",
            tags=["a", "b"],
            token="my-token",
            transport=transport,
        )
        session_url, body, token = transport.posts[0]
        assert token == "my-token"
        assert body["title"] == "T"
        assert body["description"] == "D"
        assert body["tags"] == ["a", "b"]
        assert body["config"]["version"] == "0.7.0"
        assert body["files"]["format"] == 1 and "manifest" not in body

    def test_default_title_from_first_project(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(dist, token="tok", transport=transport)
        _, body, _ = transport.posts[0]
        assert body["title"] == "Demo"

    def test_custom_api_base(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(
            dist,
            token="tok",
            api_base="https://staging.example.com/",
            transport=transport,
        )
        assert transport.posts[0][0] == (
            "https://staging.example.com/api/simulations/upload-session"
        )

    def test_token_from_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MJSWAN_TOKEN", "env-token")
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(dist, transport=transport)
        assert transport.posts[0][2] == "env-token"

    def test_custom_js_rejected_before_network(self, tmp_path: Path):
        dist = _make_dist(tmp_path, uses_custom_js=True)
        transport = FakeTransport()
        with pytest.raises(PublishError):
            publish_dist(dist, token="tok", transport=transport)
        assert transport.posts == []  # no request made

    def test_422_surfaced_verbatim(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport(
            session_status=422,
            session_error={
                "error": "this term uses custom-TS and is not supported",
                "file": "policy.json",
            },
        )
        with pytest.raises(PublishError) as exc:
            publish_dist(dist, token="tok", transport=transport)
        assert "custom-TS" in str(exc.value)
        assert exc.value.file == "policy.json"

    @pytest.mark.parametrize(
        "body", [{}, {"id": 7}, {"id": "[link=https://example.invalid]x"}]
    )
    def test_commit_without_a_valid_id(self, tmp_path: Path, body):
        dist = _make_dist(tmp_path)
        transport = FakeTransport(commit_body=body)
        with pytest.raises(PublishError, match="commit response has no valid id"):
            publish_dist(dist, token="tok", transport=transport)

    def test_failed_put_raises(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport(put_status=500)
        with pytest.raises(PublishError, match="Upload failed"):
            publish_dist(dist, token="tok", transport=transport)

    def test_invalid_presigned_url_surfaced(self, tmp_path: Path):
        """Server with unset R2 creds signs undefined.r2... URLs — fail clearly,
        before attempting a PUT that would surface as an opaque TLS error."""

        class BadUrlTransport(FakeTransport):
            def post_json(self, url, body, token):
                resp = super().post_json(url, body, token)
                if url.endswith("/upload-session"):
                    payload = resp.json()
                    for upload in payload["uploads"]:
                        upload["url"] = (
                            "https://undefined.r2.cloudflarestorage.com/"
                            f"{upload['sha256']}?X-Amz-Credential=undefined"
                        )
                    return HttpResponse(200, json.dumps(payload).encode())
                return resp

        transport = BadUrlTransport()
        with pytest.raises(PublishError, match="server-side misconfiguration"):
            publish_dist(_make_dist(tmp_path), token="tok", transport=transport)
        assert transport.puts == []  # never attempted the doomed PUT

    def test_progress_callback(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        messages: list[str] = []
        publish_dist(
            dist, token="tok", transport=FakeTransport(), on_progress=messages.append
        )
        assert any("Committing" in m for m in messages)
        assert any("Uploading" in m for m in messages)


# ── app.publish() ────────────────────────────────────────────────────────────


class TestAppPublish:
    def test_delegates_to_publish_dist(self, tmp_path: Path, monkeypatch):
        from mjswan.app import MjswanApp

        captured: dict = {}

        def fake_publish_dist(dist_dir, **kwargs):
            captured["dist_dir"] = dist_dir
            captured.update(kwargs)
            from mjswan.cloud.publish import PublishResult

            return PublishResult(id="zzz", sim_id="zzz", upload_id="u")

        monkeypatch.setattr("mjswan.cloud.publish.publish_dist", fake_publish_dist)
        dist = _make_dist(tmp_path)
        result = MjswanApp(dist).publish(title="Hello", tags=["x"])
        assert result.id == "zzz"
        assert captured["title"] == "Hello"
        assert captured["tags"] == ["x"]
        assert Path(captured["dist_dir"]) == dist


# ── `mjswan publish` CLI ─────────────────────────────────────────────────────


class TestPublishCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_publish_success(self, tmp_path: Path, monkeypatch):
        from mjswan.cli import app
        from mjswan.cloud.publish import PublishResult

        captured: dict = {}

        def fake_publish_dist(dist_dir, **kwargs):
            captured["dist_dir"] = dist_dir
            captured.update(kwargs)
            return PublishResult(id="sim777", sim_id="sim777", upload_id="u")

        monkeypatch.setattr("mjswan.cloud.publish.publish_dist", fake_publish_dist)
        dist = _make_dist(tmp_path)

        result = self._runner().invoke(
            app,
            [
                "publish",
                str(dist),
                "--token",
                "tok",  # supplied → no auto-login
                "--title",
                "Demo",
                "--tag",
                "robotics",
                "--tag",
                "humanoid",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "sim777" in result.output
        assert captured["title"] == "Demo"
        assert captured["tags"] == ["robotics", "humanoid"]

    def test_publish_missing_dir(self, tmp_path: Path):
        from mjswan.cli import app

        result = self._runner().invoke(app, ["publish", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_publish_auto_logs_in_when_no_token(self, tmp_path: Path, monkeypatch):
        """No --token, no env, no stored session → publish triggers login first."""
        from mjswan.cli import app
        from mjswan.cloud import auth
        from mjswan.cloud.publish import PublishResult

        monkeypatch.setenv("MJSWAN_CONFIG_HOME", str(tmp_path / "cfg"))
        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)

        login_called = {"n": 0}

        def fake_login(*, open_browser, on_progress=None):
            login_called["n"] += 1
            creds = auth.Credentials(
                "tok", "r", expires_at=__import__("time").time() + 3600, username="ada"
            )
            auth.save_credentials(creds)
            return creds

        monkeypatch.setattr("mjswan.cloud.auth.login", fake_login)
        monkeypatch.setattr(
            "mjswan.cloud.publish.publish_dist",
            lambda dist_dir, **kw: PublishResult(id="s1", sim_id="s1", upload_id="u"),
        )

        result = self._runner().invoke(app, ["publish", str(_make_dist(tmp_path))])
        assert result.exit_code == 0, result.output
        assert login_called["n"] == 1
        assert "ada" in result.output  # signed-in account surfaced

    def test_publish_skips_login_when_token_given(self, tmp_path: Path, monkeypatch):
        from mjswan.cli import app
        from mjswan.cloud.publish import PublishResult

        monkeypatch.setattr(
            "mjswan.cloud.auth.login",
            lambda **k: pytest.fail("login must not run when a token is provided"),
        )
        monkeypatch.setattr(
            "mjswan.cloud.publish.publish_dist",
            lambda dist_dir, **kw: PublishResult(id="s1", sim_id="s1", upload_id="u"),
        )
        result = self._runner().invoke(
            app, ["publish", str(_make_dist(tmp_path)), "--token", "explicit"]
        )
        assert result.exit_code == 0, result.output

    def test_publish_error_surfaced(self, tmp_path: Path, monkeypatch):
        from mjswan.cli import app
        from mjswan.cloud.publish import PublishError

        def fake_publish_dist(dist_dir, **kwargs):
            raise PublishError("nope, custom-TS", file="policy.json")

        monkeypatch.setattr("mjswan.cloud.publish.publish_dist", fake_publish_dist)
        dist = _make_dist(tmp_path)

        result = self._runner().invoke(app, ["publish", str(dist), "--token", "tok"])
        assert result.exit_code == 1
        assert "nope, custom-TS" in result.output
        assert "policy.json" in result.output


# ── skins ────────────────────────────────────────────────────────────────────


class TestSkins:
    """A skin is uploaded as data once mjswan Cloud would take it: drawable, and for a
    VRM, licensed for hosting and, where only its author may use it, confirmed."""

    SKIN = "demo/humanoid/assets/a.vrm"

    def _dist(self, tmp_path: Path, **meta) -> Path:
        import skin_fixture as fx

        dist = _make_dist(tmp_path)
        fx.write_vrm(dist / self.SKIN, meta={**fx.OPEN_META, **meta})
        return dist

    def test_a_vrm_and_a_glb_upload_as_data(self, tmp_path: Path):
        import skin_fixture as fx

        dist = self._dist(tmp_path)
        fx.write_glb(dist / "demo/humanoid/assets/b.glb", fx.skeleton_gltf())
        plan = plan_publish(dist)
        assert {self.SKIN, "demo/humanoid/assets/b.glb"} <= set(plan.file_map["files"])
        assert list(plan.vrms) == [self.SKIN] and plan.to_attest == []
        assert {".vrm", ".glb"} <= DATA_EXTENSIONS

    def test_a_vrm_its_license_forbids_hosting_is_refused(self, tmp_path: Path):
        dist = self._dist(tmp_path, allowRedistribution=False)
        with pytest.raises(PublishError, match="forbids redistribution") as exc:
            plan_publish(dist)
        assert exc.value.file == self.SKIN

    def test_a_compressed_glb_is_refused_with_the_way_out(self, tmp_path: Path):
        import skin_fixture as fx

        dist = _make_dist(tmp_path)
        gltf = {**fx.skeleton_gltf(), "extensionsRequired": ["EXT_meshopt_compression"]}
        fx.write_glb(dist / "demo/humanoid/assets/b.glb", gltf)
        with pytest.raises(PublishError, match="Export it again without compression"):
            plan_publish(dist)

    def test_a_skin_named_as_the_other_format_is_refused(self, tmp_path: Path):
        import skin_fixture as fx

        plain = _make_dist(tmp_path / "plain")
        fx.write_glb(plain / self.SKIN, fx.skeleton_gltf())
        with pytest.raises(PublishError, match="no VRM extension"):
            plan_publish(plain)
        vrm = _make_dist(tmp_path / "vrm")
        fx.write_vrm(vrm / "demo/humanoid/assets/a.glb")
        with pytest.raises(PublishError, match=r"publish it as \.vrm"):
            plan_publish(vrm)

    def test_what_mjswan_cloud_does_not_store_is_refused(self, tmp_path, monkeypatch):
        for case, meta in (
            ("name", {"name": "x" * 257}),
            ("authors", {"authors": [f"a{i}" for i in range(33)]}),
            ("url", {"otherLicenseUrl": "https://example.com/" + "x" * 2048}),
        ):
            with pytest.raises(PublishError, match="longer than mjswan Cloud stores"):
                plan_publish(self._dist(tmp_path / case, **meta))
        monkeypatch.setattr("mjswan.cloud.publish.GLB_JSON_MAX_BYTES", 64)
        with pytest.raises(PublishError, match="glTF JSON chunk; mjswan Cloud"):
            plan_publish(self._dist(tmp_path / "json"))

    def test_a_non_commercial_vrm_warns_and_publishes(self, tmp_path: Path):
        dist = self._dist(tmp_path, commercialUsage="personalNonProfit")
        seen: list[str] = []
        publish_dist(
            dist, token="tok", transport=FakeTransport(), on_warning=seen.append
        )
        assert seen == [
            f"{self.SKIN}: its VRM license allows no commercial use (Sample by mjswan (https://vrm.dev/licenses/1.0/))."
        ]

    def test_a_vrm_s_text_reaches_the_callbacks_without_control_characters(
        self, tmp_path: Path
    ):
        meta = {
            "authors": ["Eve\x1b]0;SPOOFED\x07\x9b"],
            "commercialUsage": "personalNonProfit",
        }
        seen: list[str] = []
        publish_dist(
            self._dist(tmp_path, **meta),
            token="tok",
            transport=FakeTransport(),
            on_progress=seen.append,
            on_warning=seen.append,
        )
        said = "".join(seen)
        assert "Eve]0;SPOOFED" in said and "\x1b" not in said and "\x9b" not in said

    def test_an_author_only_vrm_goes_only_on_the_publishers_word(self, tmp_path: Path):
        dist = self._dist(tmp_path, avatarPermission="onlyAuthor")
        asked = []
        transport = FakeTransport()
        publish_dist(
            dist,
            token="tok",
            transport=transport,
            attest=lambda path, terms: asked.append((path, terms.avatar_use)) or True,
        )
        assert asked == [(self.SKIN, "author")]
        assert transport.posts[-1][1] == {
            "upload_id": "11111111-1111-1111-1111-111111111111",
            "attested_skins": [self.SKIN],
        }
        for attest in (None, lambda path, terms: False):
            transport = FakeTransport()
            with pytest.raises(PublishError, match="did not confirm") as exc:
                publish_dist(dist, token="tok", transport=transport, attest=attest)
            assert exc.value.file == self.SKIN and transport.posts == []

    def test_the_cli_asks_and_answers_no_without_a_terminal(
        self, tmp_path, monkeypatch
    ):
        from typer.testing import CliRunner

        from mjswan.cli import app
        from mjswan.cloud.publish import PublishResult
        from mjswan.license.vrm import vrm_license

        terms = vrm_license({**self._meta(), "avatarPermission": "onlyAuthor"}, "1")
        answers = []

        def fake_publish_dist(dist_dir, **kwargs):
            answers.append(kwargs["attest"](self.SKIN, terms))
            return PublishResult(id="s1", sim_id="s1", upload_id="u")

        monkeypatch.setattr("mjswan.cloud.publish.publish_dist", fake_publish_dist)
        dist = str(_make_dist(tmp_path))
        monkeypatch.setattr("mjswan.cli._interactive", lambda: True)
        result = CliRunner().invoke(
            app, ["publish", dist, "--token", "tok"], input="y\n"
        )
        assert result.exit_code == 0, result.output
        assert "Are you its author, or licensed by them?" in result.output
        assert "only its author use it as an avatar" in result.output
        monkeypatch.setattr("mjswan.cli._interactive", lambda: False)
        CliRunner().invoke(app, ["publish", dist, "--token", "tok"])
        assert answers == [True, False]

    @staticmethod
    def _meta() -> dict:
        import skin_fixture as fx

        return dict(fx.OPEN_META)
