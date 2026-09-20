"""mjswan Cloud authentication (``mjswan login``).

A ``gh``-style **loopback OAuth flow** against Supabase's GitHub OAuth. The CLI
never holds a long-lived secret: it drives the same Supabase GitHub OAuth the
web app uses, captures the resulting session on a transient ``127.0.0.1``
listener, and persists the ``access_token`` + ``refresh_token`` locally. Because
Supabase access tokens are short-lived (~1h) and the refresh token rotates on
use, subsequent commands refresh the access token transparently. See
mjswan-cloud ADR 0001 §6.1.

The flow (PKCE):

1. Generate a PKCE ``code_verifier`` / ``code_challenge`` pair.
2. Start a loopback HTTP listener on ``http://127.0.0.1:<port>/callback``.
3. Open the browser to ``{SUPABASE_URL}/auth/v1/authorize?provider=github&…``
   with the challenge and ``redirect_to`` pointing at the loopback URL.
4. Receive the ``?code=…`` on the callback and exchange it at
   ``POST {SUPABASE_URL}/auth/v1/token?grant_type=pkce`` (header ``apikey``) for
   an ``access_token`` + ``refresh_token``.
5. Persist both (plus ``expires_at``) to ``~/.config/mjswan/credentials.json``
   with ``0600`` permissions.

The Supabase **publishable key** is public (published with the web app), so it
is embedded here; no secret is exposed. The loopback ``redirect_to`` must be in
the Supabase project's Redirect URLs allowlist (mjswan-cloud ADR 0006).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .transport import USER_AGENT

# ── Supabase configuration (public; overridable via env) ──────────────────────

#: Supabase project URL
DEFAULT_SUPABASE_URL: str = "https://rrjgpdagemlmncaugwlx.supabase.co"

#: Public publishable key
DEFAULT_SUPABASE_PUBLISHABLE_KEY: str = "sb_publishable_4vWIuMHT8NZr9zYLvbxjxg_86WfP9_0"

SUPABASE_URL_ENV_VAR: str = "MJSWAN_SUPABASE_URL"
SUPABASE_PUBLISHABLE_KEY_ENV_VAR: str = "MJSWAN_SUPABASE_PUBLISHABLE_KEY"

#: Loopback port
DEFAULT_LOOPBACK_PORT: int = 8765
LOOPBACK_PORT_ENV_VAR: str = "MJSWAN_LOGIN_PORT"
LOOPBACK_HOST: str = "127.0.0.1"
CALLBACK_PATH: str = "/callback"

#: Refresh the access token when it is within this many seconds of expiry.
EXPIRY_SKEW_SECONDS: int = 60

#: How long ``mjswan login`` waits for the browser round-trip before giving up.
LOGIN_TIMEOUT_SECONDS: int = 300


def supabase_url() -> str:
    return os.environ.get(SUPABASE_URL_ENV_VAR, DEFAULT_SUPABASE_URL).rstrip("/")


def supabase_publishable_key() -> str:
    key = os.environ.get(
        SUPABASE_PUBLISHABLE_KEY_ENV_VAR, DEFAULT_SUPABASE_PUBLISHABLE_KEY
    )
    if not key:
        raise AuthError(
            "No Supabase publishable key configured. Set the "
            "DEFAULT_SUPABASE_PUBLISHABLE_KEY constant in mjswan/cloud/auth.py, or set "
            f"${SUPABASE_PUBLISHABLE_KEY_ENV_VAR}, to the publishable key for "
            f"{supabase_url()}."
        )
    return key


def _loopback_port() -> int:
    raw = os.environ.get(LOOPBACK_PORT_ENV_VAR)
    return int(raw) if raw else DEFAULT_LOOPBACK_PORT


class AuthError(Exception):
    """Raised when login, refresh, or credential I/O fails."""


# ── Credential storage ─────────────────────────────────────────────────────────


def credentials_path() -> Path:
    """Resolve the credentials file path.

    Order: ``$MJSWAN_CONFIG_HOME`` → ``$XDG_CONFIG_HOME/mjswan`` →
    ``~/.config/mjswan``. The file holds one JSON object.
    """
    override = os.environ.get("MJSWAN_CONFIG_HOME")
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) / "mjswan" if xdg else Path.home() / ".config" / "mjswan"
    return base / "credentials.json"


def _username_from(user_metadata: dict, *, email=None, user_id=None):
    """Best-effort GitHub username from Supabase user metadata.

    The GitHub provider populates ``user_name`` / ``preferred_username``; fall
    back through other identity hints so ``whoami`` always shows *something*.
    """
    meta = user_metadata or {}
    return (
        meta.get("user_name")
        or meta.get("preferred_username")
        or meta.get("username")
        or meta.get("name")
        or email
        or user_id
    )


@dataclass
class Credentials:
    access_token: str
    refresh_token: str
    expires_at: float  # unix seconds
    username: str | None = None  # GitHub login, for display only
    email: str | None = None

    def is_expired(self, *, skew: int = EXPIRY_SKEW_SECONDS) -> bool:
        return time.time() >= (self.expires_at - skew)

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "username": self.username,
            "email": self.email,
        }

    @classmethod
    def from_token_response(cls, payload: dict) -> Credentials:
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        if not access_token or not refresh_token:
            raise AuthError("Token response missing access_token/refresh_token.")
        # GoTrue returns expires_at (unix seconds); fall back to expires_in.
        expires_at = payload.get("expires_at")
        if expires_at is None:
            expires_at = time.time() + float(payload.get("expires_in", 3600))
        user = payload.get("user") or {}
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at),
            username=_username_from(
                user.get("user_metadata") or {},
                email=user.get("email"),
                user_id=user.get("id"),
            ),
            email=user.get("email"),
        )


def load_credentials() -> Credentials | None:
    """Load stored credentials, or ``None`` if not logged in / unreadable."""
    path = credentials_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        return Credentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=float(data["expires_at"]),
            username=data.get("username"),
            email=data.get("email"),
        )
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        # A corrupt file behaves like "not logged in" rather than crashing the
        # whole command; `mjswan login` will overwrite it.
        return None


def save_credentials(creds: Credentials) -> None:
    """Persist credentials with ``0600`` permissions."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive perms before writing any token bytes.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, json.dumps(creds.to_dict(), indent=2).encode("utf-8"))
    finally:
        os.close(fd)
    # Re-assert perms in case the file pre-existed with looser bits.
    os.chmod(path, 0o600)


def clear_credentials() -> bool:
    """Delete stored credentials. Returns ``True`` if a file was removed."""
    path = credentials_path()
    if path.is_file():
        path.unlink()
        return True
    return False


# ── HTTP transport (injectable so tests need no network) ──────────────────────


@dataclass
class _Response:
    status: int
    body: bytes

    def json(self) -> dict:
        return json.loads(self.body.decode("utf-8")) if self.body else {}


class AuthTransport:
    """Minimal transport for the Supabase auth endpoints. Replaceable in tests."""

    def post_json(self, url: str, body: dict, headers: dict[str, str]) -> _Response:
        data = json.dumps(body).encode("utf-8")
        return self._send(
            urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"User-Agent": USER_AGENT, **headers},
            )
        )

    def get(self, url: str, headers: dict[str, str]) -> _Response:
        return self._send(
            urllib.request.Request(
                url, method="GET", headers={"User-Agent": USER_AGENT, **headers}
            )
        )

    @staticmethod
    def _send(req: urllib.request.Request) -> _Response:
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted base)
                return _Response(status=resp.status, body=resp.read())
        except urllib.error.HTTPError as exc:
            return _Response(status=exc.code, body=exc.read())
        except urllib.error.URLError as exc:
            raise AuthError(f"Network error contacting Supabase: {exc.reason}")


def _token_request(
    grant_type: str, body: dict, transport: AuthTransport
) -> Credentials:
    url = f"{supabase_url()}/auth/v1/token?grant_type={grant_type}"
    headers = {
        "apikey": supabase_publishable_key(),
        "Authorization": f"Bearer {supabase_publishable_key()}",
        "Content-Type": "application/json",
    }
    resp = transport.post_json(url, body, headers)
    if resp.status not in (200, 201):
        try:
            payload = resp.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        msg = (
            payload.get("error_description")
            or payload.get("msg")
            or payload.get("error")
            or f"HTTP {resp.status}"
        )
        raise AuthError(f"Supabase token request ({grant_type}) failed: {msg}")
    return Credentials.from_token_response(resp.json())


# ── PKCE helpers ───────────────────────────────────────────────────────────────


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for the S256 method."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def authorize_url(code_challenge: str, redirect_to: str) -> str:
    query = urllib.parse.urlencode(
        {
            "provider": "github",
            "redirect_to": redirect_to,
            "code_challenge": code_challenge,
            "code_challenge_method": "s256",
        }
    )
    return f"{supabase_url()}/auth/v1/authorize?{query}"


# ── Token refresh / resolution ─────────────────────────────────────────────────


def refresh_credentials(
    creds: Credentials, transport: AuthTransport | None = None
) -> Credentials:
    """Exchange the refresh token for a fresh session and persist it."""
    transport = transport or AuthTransport()
    fresh = _token_request(
        "refresh_token", {"refresh_token": creds.refresh_token}, transport
    )
    save_credentials(fresh)
    return fresh


def current_access_token(transport: AuthTransport | None = None) -> str | None:
    """Return a valid stored access token, refreshing if near expiry.

    Returns ``None`` if the user is not logged in. Raises :class:`AuthError`
    only if a refresh attempt fails.
    """
    creds = load_credentials()
    if creds is None:
        return None
    if creds.is_expired():
        creds = refresh_credentials(creds, transport)
    return creds.access_token


# ── Identity (whoami) ───────────────────────────────────────────────────────────


@dataclass
class Identity:
    user_id: str
    username: str | None
    email: str | None


def fetch_identity(transport: AuthTransport | None = None) -> Identity | None:
    """Fetch the signed-in user from Supabase, refreshing the token if needed.

    Returns ``None`` if not logged in. Raises :class:`AuthError` if the session
    is no longer valid (so ``whoami`` can report a stale login).
    """
    transport = transport or AuthTransport()
    token = current_access_token(transport)
    if token is None:
        return None
    resp = transport.get(
        f"{supabase_url()}/auth/v1/user",
        {"apikey": supabase_publishable_key(), "Authorization": f"Bearer {token}"},
    )
    if resp.status != 200:
        raise AuthError(
            f"Could not verify session (HTTP {resp.status}). Run `mjswan login`."
        )
    user = resp.json()
    return Identity(
        user_id=user.get("id", ""),
        username=_username_from(
            user.get("user_metadata") or {},
            email=user.get("email"),
            user_id=user.get("id"),
        ),
        email=user.get("email"),
    )


# ── The interactive login flow ──────────────────────────────────────────────────


@dataclass
class _CallbackResult:
    code: str | None = None
    error: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures the OAuth ``?code=`` (or ``?error=``) from one request."""

    result: _CallbackResult  # set on the server instance

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        result: _CallbackResult = self.server.result  # type: ignore[attr-defined]
        result.code = params.get("code", [None])[0]
        result.error = params.get("error_description", params.get("error", [None]))[0]
        ok = result.code is not None and result.error is None
        message = (
            "Login successful — you can close this tab and return to the terminal."
            if ok
            else f"Login failed: {result.error or 'no authorization code returned'}."
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            f"<!doctype html><html><body style='font-family:sans-serif;"
            f"padding:2rem'><h2>mjswan</h2><p>{message}</p></body></html>".encode()
        )

    def log_message(self, format, *args) -> None:  # silence default stderr logging
        pass


def login(
    *,
    open_browser: bool = True,
    transport: AuthTransport | None = None,
    timeout: int = LOGIN_TIMEOUT_SECONDS,
    on_progress=None,
) -> Credentials:
    """Run the loopback OAuth flow and persist the resulting credentials.

    Args:
        open_browser: Open the auth URL in the default browser (disable for
            headless contexts; the URL is still reported via ``on_progress``).
        transport: HTTP transport for the token exchange (injectable for tests).
        timeout: Seconds to wait for the browser callback.
        on_progress: Optional callback invoked with human-readable status lines.

    Returns:
        The persisted :class:`Credentials`.

    Raises:
        AuthError: if the loopback port is unavailable, the user denies access,
            the callback times out, or the token exchange fails.
    """
    transport = transport or AuthTransport()
    notify = on_progress or (lambda _msg: None)
    port = _loopback_port()
    redirect_to = f"http://{LOOPBACK_HOST}:{port}{CALLBACK_PATH}"

    verifier, challenge = generate_pkce()
    url = authorize_url(challenge, redirect_to)

    try:
        server = HTTPServer((LOOPBACK_HOST, port), _CallbackHandler)
    except OSError as exc:
        raise AuthError(
            f"Cannot bind loopback port {port} ({exc}). "
            f"Close whatever is using it, or set {LOOPBACK_PORT_ENV_VAR} to a "
            "free port that is also in the Supabase Redirect URLs allowlist."
        )
    server.result = _CallbackResult()  # type: ignore[attr-defined]
    server.timeout = timeout

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    notify("Opening browser to sign in with GitHub…")
    notify(f"If it doesn't open, visit:\n  {url}")
    if open_browser:
        webbrowser.open(url)

    thread.join(timeout=timeout + 1)
    server.server_close()
    result: _CallbackResult = server.result  # type: ignore[attr-defined]

    if thread.is_alive() or (result.code is None and result.error is None):
        raise AuthError(f"Timed out after {timeout}s waiting for the browser sign-in.")
    if result.error is not None:
        raise AuthError(f"Authorization was denied or failed: {result.error}")
    assert result.code is not None

    notify("Exchanging authorization code for a session…")
    creds = _token_request(
        "pkce", {"auth_code": result.code, "code_verifier": verifier}, transport
    )
    save_credentials(creds)
    return creds


__all__ = [
    "DEFAULT_LOOPBACK_PORT",
    "DEFAULT_SUPABASE_PUBLISHABLE_KEY",
    "DEFAULT_SUPABASE_URL",
    "EXPIRY_SKEW_SECONDS",
    "AuthError",
    "AuthTransport",
    "Credentials",
    "Identity",
    "authorize_url",
    "clear_credentials",
    "credentials_path",
    "current_access_token",
    "fetch_identity",
    "generate_pkce",
    "load_credentials",
    "login",
    "refresh_credentials",
    "save_credentials",
    "supabase_publishable_key",
    "supabase_url",
]
