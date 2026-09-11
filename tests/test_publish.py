"""Tests for mjswan Cloud publishing (mjswan.publish + app.publish + CLI).

L1 — pure Python, no MuJoCo/ONNX/network required (safe for pre-commit).
The HTTP transport is faked, so no real requests are made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mjswan.publish import (
    DATA_EXTENSIONS,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    USER_AGENT,
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


@pytest.fixture(autouse=True)
def _isolate_credentials(tmp_path, monkeypatch):
    """Keep token resolution from picking up a developer's real `mjswan login`
    session (~/.config/mjswan) — point credential storage at an empty tmp dir."""
    monkeypatch.setenv("MJSWAN_CONFIG_HOME", str(tmp_path / "mjswan-cfg"))


# ── Fakes ──────────────────────────────────────────────────────────────────


class FakeTransport(HttpTransport):
    """Records calls; echoes presigned PUT URLs derived from the manifest."""

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
    ) -> None:
        self.posts: list[tuple[str, dict, str]] = []
        self.puts: list[tuple[str, int, str]] = []
        self.put_headers: list[tuple[str, str, str | None]] = []
        self._sim_id = sim_id
        self._upload_id = upload_id
        self._commit_id = commit_id
        self._session_status = session_status
        self._commit_status = commit_status
        self._commit_body = commit_body
        self._put_status = put_status
        self._session_error = session_error

    def post_json(self, url: str, body: dict, token: str) -> HttpResponse:
        self.posts.append((url, body, token))
        if url.endswith("/upload-session"):
            if self._session_error is not None:
                return HttpResponse(
                    self._session_status, json.dumps(self._session_error).encode()
                )
            # Shaped like the real `PresignedUpload`: the exact header values the
            # server signed, with `.mjz` octet-stream (the client's advisory says zip).
            uploads = [
                {
                    "path": entry["path"],
                    "url": f"https://r2.example.com/{entry['path']}",
                    "contentType": (
                        "application/json"
                        if entry["path"].endswith(".json")
                        else "application/octet-stream"
                    ),
                    "cacheControl": "public, max-age=31536000, immutable",
                }
                for entry in body["manifest"]
            ]
            payload = {
                "upload_id": self._upload_id,
                "sim_id": self._sim_id,
                "uploads": uploads,
                "expires_in": 900,
            }
            return HttpResponse(self._session_status, json.dumps(payload).encode())
        if url.endswith("/commit"):
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
        return HttpResponse(self._put_status, b"")


def _make_dist(tmp_path: Path, *, uses_custom_js: bool = False) -> Path:
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
                        "scene": "scene.mjz",
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
    (scene_dir / "scene.mjz").write_bytes(b"MJZ" * 10)
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

        monkeypatch.setattr("mjswan.publish.urllib.request.urlopen", fake_urlopen)
        t = HttpTransport()
        t.post_json("https://api.mjswan.com/x", {}, "tok")
        t.put_bytes("https://r2.example.com/x", b"data", "application/json")

        assert seen == [USER_AGENT, USER_AGENT]
        assert "Python-urllib" not in USER_AGENT
        assert USER_AGENT.startswith("mjswan/")


# ── plan_publish ─────────────────────────────────────────────────────────────


class TestPlanPublish:
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

    def test_manifest_has_content_types_and_sizes(self, tmp_path: Path):
        plan = plan_publish(_make_dist(tmp_path))
        manifest = {entry["path"]: entry for entry in plan.manifest()}
        assert manifest["manifest.json"]["contentType"] == "application/json"
        assert manifest["demo/humanoid/scene.mjz"]["contentType"] == "application/zip"
        assert (
            manifest["demo/humanoid/policy/walk.onnx"]["contentType"]
            == "application/octet-stream"
        )
        assert manifest["demo/humanoid/scene.mjz"]["size"] == 30

    def test_rejects_custom_js(self, tmp_path: Path):
        with pytest.raises(PublishError) as exc:
            plan_publish(_make_dist(tmp_path, uses_custom_js=True))
        assert exc.value.file == "manifest.json"
        assert "custom-js" in str(exc.value).lower()

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
        monkeypatch.setattr("mjswan.publish.MAX_FILE_BYTES", 5)
        with pytest.raises(PublishError) as exc:
            plan_publish(dist)
        assert exc.value.file is not None

    def test_rejects_too_many_files(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.publish.MAX_FILES", 2)
        with pytest.raises(PublishError, match="Too many files"):
            plan_publish(dist)

    def test_rejects_total_size(self, tmp_path: Path, monkeypatch):
        dist = _make_dist(tmp_path)
        monkeypatch.setattr("mjswan.publish.MAX_TOTAL_BYTES", 10)
        with pytest.raises(PublishError, match="Total upload size"):
            plan_publish(dist)

    def test_default_caps_are_spec_values(self):
        assert MAX_FILE_BYTES == 50 * 1024 * 1024
        assert MAX_TOTAL_BYTES == 200 * 1024 * 1024
        assert MAX_FILES == 64
        assert DATA_EXTENSIONS == frozenset(
            {".json", ".mjz", ".onnx", ".npz", ".ply", ".spz"}
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
        from mjswan.licenses import license_template

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
        manifest = {e["path"]: e for e in plan.manifest()}
        assert manifest["demo/LICENSE"]["contentType"] == "text/plain; charset=utf-8"
        assert (
            manifest["demo/humanoid/NOTICE"]["contentType"]
            == "text/plain; charset=utf-8"
        )
        assert [d.describe() for d in plan.licenses] == [
            "demo/LICENSE (Apache-2.0)",
            "demo/NOTICE",
            "demo/humanoid/LICENSE.unitree_go2 (BSD-3-Clause)",
            "demo/humanoid/NOTICE",
        ]
        assert plan.license_warnings() == []

    def test_the_declaration_is_printed_before_the_upload(self, tmp_path):
        from mjswan.licenses import license_template

        dist = _with_licenses(
            _make_dist(tmp_path), {"demo/LICENSE": license_template("MIT")}
        )
        messages: list[str] = []
        transport = FakeTransport()
        publish_dist(
            dist, token="tok", transport=transport, on_progress=messages.append
        )
        assert messages[0] == "License file: demo/LICENSE (MIT)"
        assert any(u.endswith("demo/LICENSE") for u, _, _ in transport.puts)

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
        from mjswan.licenses import LICENSE_FILE_MAX_BYTES

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

        from mjswan._cli import app
        from mjswan.publish import PublishResult

        def fake_publish_dist(dist_dir, **kwargs):
            kwargs["on_warning"]("demo/LICENSE is GPL-3.0-only (copyleft).")
            return PublishResult(id="s1", sim_id="s1", upload_id="u")

        monkeypatch.setattr("mjswan.publish.publish_dist", fake_publish_dist)
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

        # One upload-session, then commit.
        urls = [url for url, _, _ in transport.posts]
        assert urls[0].endswith("/api/simulations/upload-session")
        assert urls[-1].endswith("/api/simulations/commit")

        # All five data files were PUT, none of the app-shell files.
        assert len(transport.puts) == 5
        put_urls = {url for url, _, _ in transport.puts}
        assert "https://r2.example.com/manifest.json" in put_urls
        assert all(".html" not in u and ".css" not in u for u in put_urls)

    def test_put_sends_the_headers_the_server_signed(self, tmp_path: Path):
        """Both are in `SignedHeaders`, so a value this side invents is a 403."""
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        publish_dist(dist, title="My Sim", token="tok", transport=transport)

        by_url = {url: (ct, cc) for url, ct, cc in transport.put_headers}
        assert by_url  # the loop ran at all
        for url, (content_type, cache_control) in by_url.items():
            assert cache_control == "public, max-age=31536000, immutable", url
            expected = (
                "application/json"
                if url.endswith(".json")
                else "application/octet-stream"
            )
            assert content_type == expected, url

        # `_content_type_for` calls a `.mjz` application/zip; the server does not.
        mjz = [v for u, v in by_url.items() if u.endswith(".mjz")]
        assert mjz and all(ct == "application/octet-stream" for ct, _ in mjz)

    def test_commit_re_resolves_the_token(self, tmp_path: Path, monkeypatch):
        """A long upload can outlive the token, and the commit is what pays."""
        dist = _make_dist(tmp_path)
        transport = FakeTransport()
        tokens = iter(["token-at-upload", "token-at-commit"])
        monkeypatch.setattr("mjswan.publish.resolve_token", lambda _t: next(tokens))

        publish_dist(dist, title="T", token="ignored", transport=transport)

        used = {url.rsplit("/", 1)[-1]: tok for url, _, tok in transport.posts}
        assert used["upload-session"] == "token-at-upload"
        assert used["commit"] == "token-at-commit"

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
        assert isinstance(body["manifest"], list)

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
                "error": "this term uses custom-JS and is not supported",
                "file": "policy.json",
            },
        )
        with pytest.raises(PublishError) as exc:
            publish_dist(dist, token="tok", transport=transport)
        assert "custom-JS" in str(exc.value)
        assert exc.value.file == "policy.json"

    def test_commit_missing_id(self, tmp_path: Path):
        dist = _make_dist(tmp_path)
        transport = FakeTransport(commit_body={})
        with pytest.raises(PublishError, match="commit response missing id"):
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
                    payload["uploads"] = [
                        {
                            "path": e["path"],
                            "url": "https://undefined.r2.cloudflarestorage.com/"
                            f"{e['path']}?X-Amz-Credential=undefined",
                        }
                        for e in body["manifest"]
                    ]
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
            from mjswan.publish import PublishResult

            return PublishResult(id="zzz", sim_id="zzz", upload_id="u")

        monkeypatch.setattr("mjswan.publish.publish_dist", fake_publish_dist)
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
        from mjswan._cli import app
        from mjswan.publish import PublishResult

        captured: dict = {}

        def fake_publish_dist(dist_dir, **kwargs):
            captured["dist_dir"] = dist_dir
            captured.update(kwargs)
            return PublishResult(id="sim777", sim_id="sim777", upload_id="u")

        monkeypatch.setattr("mjswan.publish.publish_dist", fake_publish_dist)
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
        from mjswan._cli import app

        result = self._runner().invoke(app, ["publish", str(tmp_path / "nope")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_publish_auto_logs_in_when_no_token(self, tmp_path: Path, monkeypatch):
        """No --token, no env, no stored session → publish triggers login first."""
        from mjswan import auth
        from mjswan._cli import app
        from mjswan.publish import PublishResult

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

        monkeypatch.setattr("mjswan.auth.login", fake_login)
        monkeypatch.setattr(
            "mjswan.publish.publish_dist",
            lambda dist_dir, **kw: PublishResult(id="s1", sim_id="s1", upload_id="u"),
        )

        result = self._runner().invoke(app, ["publish", str(_make_dist(tmp_path))])
        assert result.exit_code == 0, result.output
        assert login_called["n"] == 1
        assert "ada" in result.output  # signed-in account surfaced

    def test_publish_skips_login_when_token_given(self, tmp_path: Path, monkeypatch):
        from mjswan._cli import app
        from mjswan.publish import PublishResult

        monkeypatch.setattr(
            "mjswan.auth.login",
            lambda **k: pytest.fail("login must not run when a token is provided"),
        )
        monkeypatch.setattr(
            "mjswan.publish.publish_dist",
            lambda dist_dir, **kw: PublishResult(id="s1", sim_id="s1", upload_id="u"),
        )
        result = self._runner().invoke(
            app, ["publish", str(_make_dist(tmp_path)), "--token", "explicit"]
        )
        assert result.exit_code == 0, result.output

    def test_publish_error_surfaced(self, tmp_path: Path, monkeypatch):
        from mjswan._cli import app
        from mjswan.publish import PublishError

        def fake_publish_dist(dist_dir, **kwargs):
            raise PublishError("nope, custom-JS", file="policy.json")

        monkeypatch.setattr("mjswan.publish.publish_dist", fake_publish_dist)
        dist = _make_dist(tmp_path)

        result = self._runner().invoke(app, ["publish", str(dist), "--token", "tok"])
        assert result.exit_code == 1
        assert "nope, custom-JS" in result.output
        assert "policy.json" in result.output
