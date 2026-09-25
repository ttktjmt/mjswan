"""Tests for mjswan Cloud authentication (mjswan.cloud.auth + login/logout CLI).

Layer: L1 (pure Python, no network). The Supabase token endpoint is faked, and the
interactive browser round-trip is replaced by a fake transport, so no real
requests or browser windows are made.
"""

from __future__ import annotations

import json
import stat
import time

import pytest

from mjswan.cloud import auth
from mjswan.cloud.auth import (
    AuthError,
    AuthTransport,
    Credentials,
    _Response,
    authorize_url,
    clear_credentials,
    credentials_path,
    current_access_token,
    generate_pkce,
    load_credentials,
    refresh_credentials,
    save_credentials,
)

# ── Fakes / fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point credentials storage at a tmp dir for every test."""
    monkeypatch.setenv("MJSWAN_CONFIG_HOME", str(tmp_path / "cfg"))
    # Pin Supabase config to deterministic test values so tests don't depend on
    # the committed defaults (the publishable key default is a placeholder).
    monkeypatch.setenv("MJSWAN_SUPABASE_URL", "https://test-project.supabase.co")
    monkeypatch.setenv("MJSWAN_SUPABASE_PUBLISHABLE_KEY", "test-publishable-key")
    monkeypatch.delenv("MJSWAN_LOGIN_PORT", raising=False)


class FakeAuthTransport(AuthTransport):
    """Returns a canned token response and records requests."""

    def __init__(
        self,
        *,
        status: int = 200,
        access_token: str = "access-1",
        refresh_token: str = "refresh-1",
        expires_in: int = 3600,
        username: str | None = "octocat",
        email: str | None = "octocat@example.com",
        error: dict | None = None,
        user_status: int = 200,
    ) -> None:
        self.calls: list[tuple[str, dict, dict]] = []
        self.gets: list[tuple[str, dict]] = []
        self._status = status
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_in = expires_in
        self._username = username
        self._email = email
        self._error = error
        self._user_status = user_status

    def _user_obj(self) -> dict:
        return {
            "id": "user-uuid",
            "email": self._email,
            "user_metadata": {"user_name": self._username} if self._username else {},
        }

    def post_json(self, url, body, headers) -> _Response:
        self.calls.append((url, body, headers))
        if self._error is not None:
            return _Response(self._status, json.dumps(self._error).encode())
        payload = {
            "access_token": self._access_token,
            "refresh_token": self._refresh_token,
            "expires_in": self._expires_in,
            "user": self._user_obj(),
        }
        return _Response(self._status, json.dumps(payload).encode())

    def get(self, url, headers) -> _Response:
        self.gets.append((url, headers))
        if self._user_status != 200:
            return _Response(self._user_status, b"{}")
        return _Response(200, json.dumps(self._user_obj()).encode())


# ── AuthTransport User-Agent (Cloudflare 1010 guard) ────────────────────────


class TestAuthTransportUserAgent:
    def test_post_and_get_set_non_default_user_agent(self, monkeypatch):
        from mjswan.cloud.transport import USER_AGENT

        seen: list = []

        class _Resp:
            status = 200

            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(
            auth.urllib.request,
            "urlopen",
            lambda req: (seen.append(req.get_header("User-agent")), _Resp())[1],
        )
        t = AuthTransport()
        t.post_json("https://x/auth/v1/token", {}, {"apikey": "k"})
        t.get("https://x/auth/v1/user", {"apikey": "k"})
        assert seen == [USER_AGENT, USER_AGENT]


# ── PKCE ────────────────────────────────────────────────────────────────────


class TestPkce:
    def test_verifier_and_challenge_are_distinct_urlsafe(self):
        verifier, challenge = generate_pkce()
        assert verifier and challenge and verifier != challenge
        # base64url alphabet, no padding
        assert "=" not in verifier and "=" not in challenge
        assert all(c.isalnum() or c in "-_" for c in challenge)

    def test_pairs_are_random(self):
        assert generate_pkce()[0] != generate_pkce()[0]

    def test_authorize_url_shape(self):
        url = authorize_url("chal", "http://127.0.0.1:8765/callback")
        assert url.startswith(auth.supabase_url() + "/auth/v1/authorize?")
        assert "provider=github" in url
        assert "code_challenge=chal" in url
        assert "code_challenge_method=s256" in url
        assert "redirect_to=http%3A%2F%2F127.0.0.1%3A8765%2Fcallback" in url


class TestSupabaseConfig:
    def test_url_and_key_from_env_override(self, monkeypatch):
        monkeypatch.setenv("MJSWAN_SUPABASE_URL", "https://proj.supabase.co/")
        monkeypatch.setenv("MJSWAN_SUPABASE_PUBLISHABLE_KEY", "k123")
        assert auth.supabase_url() == "https://proj.supabase.co"  # trailing / stripped
        assert auth.supabase_publishable_key() == "k123"

    def test_missing_publishable_key_raises(self, monkeypatch):
        # Empty default + no override → actionable error rather than a blank key.
        monkeypatch.delenv("MJSWAN_SUPABASE_PUBLISHABLE_KEY", raising=False)
        monkeypatch.setattr(auth, "DEFAULT_SUPABASE_PUBLISHABLE_KEY", "")
        with pytest.raises(AuthError, match="publishable key"):
            auth.supabase_publishable_key()


# ── Credential storage ──────────────────────────────────────────────────────


class TestCredentialStorage:
    def test_roundtrip(self):
        creds = Credentials("a", "r", expires_at=time.time() + 3600)
        save_credentials(creds)
        loaded = load_credentials()
        assert loaded == creds

    def test_file_is_0600(self):
        save_credentials(Credentials("a", "r", expires_at=time.time() + 1))
        mode = stat.S_IMODE(credentials_path().stat().st_mode)
        assert mode == 0o600

    def test_load_missing_returns_none(self):
        assert load_credentials() is None

    def test_load_corrupt_returns_none(self):
        path = credentials_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert load_credentials() is None

    def test_clear(self):
        save_credentials(Credentials("a", "r", expires_at=time.time() + 1))
        assert clear_credentials() is True
        assert load_credentials() is None
        assert clear_credentials() is False  # idempotent

    def test_config_home_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MJSWAN_CONFIG_HOME", str(tmp_path / "custom"))
        assert credentials_path() == tmp_path / "custom" / "credentials.json"

    def test_xdg_config_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MJSWAN_CONFIG_HOME", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert credentials_path() == tmp_path / "xdg" / "mjswan" / "credentials.json"


# ── Expiry / refresh ──────────────────────────────────────────────────────────


class TestExpiryAndRefresh:
    def test_is_expired_within_skew(self):
        assert Credentials("a", "r", expires_at=time.time() + 10).is_expired()
        assert not Credentials("a", "r", expires_at=time.time() + 3600).is_expired()

    def test_refresh_persists_new_session(self):
        old = Credentials("old-a", "old-r", expires_at=time.time() - 1)
        transport = FakeAuthTransport(access_token="new-a", refresh_token="new-r")
        fresh = refresh_credentials(old, transport)
        assert fresh.access_token == "new-a"
        assert fresh.refresh_token == "new-r"
        # Persisted and uses the refresh grant with the old refresh token.
        assert load_credentials() == fresh
        url, body, headers = transport.calls[0]
        assert url.endswith("grant_type=refresh_token")
        assert body == {"refresh_token": "old-r"}
        assert headers["apikey"] == auth.supabase_publishable_key()

    def test_refresh_failure_raises(self):
        old = Credentials("a", "r", expires_at=time.time() - 1)
        transport = FakeAuthTransport(status=400, error={"error": "bad_grant"})
        with pytest.raises(AuthError, match="bad_grant"):
            refresh_credentials(old, transport)


# ── current_access_token (precedence + auto-refresh) ────────────────────────


class TestCurrentAccessToken:
    def test_none_when_logged_out(self):
        assert current_access_token() is None

    def test_returns_valid_token_without_refresh(self):
        save_credentials(Credentials("valid", "r", expires_at=time.time() + 3600))
        transport = FakeAuthTransport()
        assert current_access_token(transport) == "valid"
        assert transport.calls == []  # no refresh needed

    def test_refreshes_expired_token(self):
        save_credentials(Credentials("stale", "r", expires_at=time.time() - 1))
        transport = FakeAuthTransport(access_token="refreshed")
        assert current_access_token(transport) == "refreshed"
        assert len(transport.calls) == 1


# ── login flow (token exchange; loopback callback faked) ────────────────────


def _fake_server_factory(*, code=None, error=None):
    """An HTTPServer stand-in whose `handle_request` injects a callback result.

    `login()` assigns a fresh `_CallbackResult` to `server.result`, then runs
    `handle_request` on a (real) thread. Our `handle_request` populates that
    object and returns at once, replacing the real loopback round-trip.
    """

    class FakeServer:
        def __init__(self, *_a, **_k):
            self.result = auth._CallbackResult()
            self.timeout = 0

        def handle_request(self):  # the thread target — returns immediately
            self.result.code = code
            self.result.error = error

        def server_close(self):
            pass

    return FakeServer


class TestLoginExchange:
    def test_token_request_pkce_exchange(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(auth, "HTTPServer", _fake_server_factory(code="the-code"))
        monkeypatch.setattr(
            auth.webbrowser, "open", lambda _url: captured.setdefault("opened", True)
        )

        transport = FakeAuthTransport(
            access_token="logged-in", refresh_token="rt", username="ada"
        )
        creds = auth.login(open_browser=True, transport=transport, timeout=1)

        assert creds.access_token == "logged-in"
        assert creds.username == "ada"  # captured from the token response's user
        assert captured.get("opened") is True
        url, body, _ = transport.calls[0]
        assert url.endswith("grant_type=pkce")
        assert body["auth_code"] == "the-code"
        assert "code_verifier" in body
        # Persisted to disk.
        assert load_credentials() == creds

    def test_login_denied_raises(self, monkeypatch):
        monkeypatch.setattr(
            auth, "HTTPServer", _fake_server_factory(error="access_denied")
        )
        monkeypatch.setattr(auth.webbrowser, "open", lambda _u: None)

        with pytest.raises(AuthError, match="denied"):
            auth.login(open_browser=False, transport=FakeAuthTransport(), timeout=1)


# ── resolve_token integration ─────────────────────────────────────────────


class TestResolveTokenWithStoredSession:
    def test_falls_back_to_stored_session(self, monkeypatch):
        from mjswan.cloud.publish import resolve_token

        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)
        save_credentials(Credentials("stored-tok", "r", expires_at=time.time() + 3600))
        assert resolve_token(None) == "stored-tok"

    def test_explicit_token_beats_stored(self, monkeypatch):
        from mjswan.cloud.publish import resolve_token

        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)
        save_credentials(Credentials("stored", "r", expires_at=time.time() + 3600))
        assert resolve_token("explicit") == "explicit"

    def test_error_mentions_login(self, monkeypatch):
        from mjswan.cloud.publish import PublishError, resolve_token

        monkeypatch.delenv("MJSWAN_TOKEN", raising=False)
        with pytest.raises(PublishError, match="mjswan login"):
            resolve_token(None)


# ── login / logout CLI ────────────────────────────────────────────────────


class TestAuthCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_login_success_shows_username(self, monkeypatch):
        from mjswan.cli import app

        def fake_login(**kwargs):
            return Credentials(
                "a", "r", expires_at=time.time() + 3600, username="octocat"
            )

        monkeypatch.setattr("mjswan.cloud.auth.login", fake_login)
        result = self._runner().invoke(app, ["login"])
        assert result.exit_code == 0, result.output
        assert "Logged in" in result.output
        assert "octocat" in result.output

    def test_login_failure(self, monkeypatch):
        from mjswan.cli import app

        def fake_login(**kwargs):
            raise AuthError("port busy")

        monkeypatch.setattr("mjswan.cloud.auth.login", fake_login)
        result = self._runner().invoke(app, ["login"])
        assert result.exit_code == 1
        assert "port busy" in result.output

    def test_logout_when_logged_in(self):
        from mjswan.cli import app

        save_credentials(Credentials("a", "r", expires_at=time.time() + 1))
        result = self._runner().invoke(app, ["logout"])
        assert result.exit_code == 0
        assert "Logged out" in result.output
        assert load_credentials() is None

    def test_logout_when_logged_out(self):
        from mjswan.cli import app

        result = self._runner().invoke(app, ["logout"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output


# ── username extraction + identity (whoami) ───────────────────────────────


class TestIdentity:
    def test_username_from_user_name_key(self):
        creds = Credentials.from_token_response(
            {
                "access_token": "a",
                "refresh_token": "r",
                "expires_in": 3600,
                "user": {
                    "id": "u1",
                    "email": "e@x.com",
                    "user_metadata": {"user_name": "ghlogin"},
                },
            }
        )
        assert creds.username == "ghlogin"
        assert creds.email == "e@x.com"

    def test_username_fallback_chain(self):
        # No user_name → preferred_username → ... → email → id.
        for meta, email, uid, expected in [
            ({"preferred_username": "pref"}, None, "u", "pref"),
            ({"name": "Full Name"}, None, "u", "Full Name"),
            ({}, "only@email", "u", "only@email"),
            ({}, None, "uuid-only", "uuid-only"),
        ]:
            creds = Credentials.from_token_response(
                {
                    "access_token": "a",
                    "refresh_token": "r",
                    "expires_in": 1,
                    "user": {"id": uid, "email": email, "user_metadata": meta},
                }
            )
            assert creds.username == expected

    def test_username_persists_across_save_load(self):
        save_credentials(
            Credentials(
                "a", "r", expires_at=time.time() + 99, username="ada", email="a@b"
            )
        )
        loaded = load_credentials()
        assert loaded.username == "ada" and loaded.email == "a@b"

    def test_fetch_identity_when_logged_out(self):
        from mjswan.cloud.auth import fetch_identity

        assert fetch_identity(FakeAuthTransport()) is None

    def test_fetch_identity_returns_user(self):
        from mjswan.cloud.auth import fetch_identity

        save_credentials(Credentials("a", "r", expires_at=time.time() + 3600))
        transport = FakeAuthTransport(username="octocat", email="o@gh.com")
        identity = fetch_identity(transport)
        assert identity.username == "octocat"
        assert identity.email == "o@gh.com"
        assert transport.gets[0][0].endswith("/auth/v1/user")

    def test_fetch_identity_stale_session_raises(self):
        from mjswan.cloud.auth import fetch_identity

        save_credentials(Credentials("a", "r", expires_at=time.time() + 3600))
        with pytest.raises(AuthError, match="verify session"):
            fetch_identity(FakeAuthTransport(user_status=401))


class TestWhoamiCli:
    def _runner(self):
        from typer.testing import CliRunner

        return CliRunner()

    def test_whoami_logged_in(self, monkeypatch):
        from mjswan.cli import app
        from mjswan.cloud.auth import Identity

        monkeypatch.setattr(
            "mjswan.cloud.auth.fetch_identity",
            lambda *a, **k: Identity("u1", "octocat", "o@gh.com"),
        )
        result = self._runner().invoke(app, ["whoami"])
        assert result.exit_code == 0, result.output
        assert "octocat" in result.output

    def test_whoami_logged_out(self, monkeypatch):
        from mjswan.cli import app

        monkeypatch.setattr("mjswan.cloud.auth.fetch_identity", lambda *a, **k: None)
        result = self._runner().invoke(app, ["whoami"])
        assert result.exit_code == 1
        assert "Not logged in" in result.output

    def test_whoami_stale_session(self, monkeypatch):
        from mjswan.cli import app

        def boom(*a, **k):
            raise AuthError("Could not verify session")

        monkeypatch.setattr("mjswan.cloud.auth.fetch_identity", boom)
        result = self._runner().invoke(app, ["whoami"])
        assert result.exit_code == 1
        assert "verify session" in result.output
