"""Publish a built mjswan ``dist/`` to mjswan Cloud (v2).

This module extracts the **data files** from a local build (manifest.json plus
scene/policy/motion/splat assets and traced MDP graphs; never HTML/JS/WASM) and uploads
them to mjswan Cloud using the three-step presigned-upload protocol described in
mjswan-cloud ADR 0001 §6:

1. ``POST {base}/api/simulations/upload-session`` with a file manifest and the
   parsed ``manifest.json`` → presigned ``PUT`` URLs.
2. ``PUT`` each file's bytes to its presigned URL.
3. ``POST {base}/api/simulations/commit`` with the ``upload_id`` → the sim id.

The platform renders only **declarative** builds (no author-supplied code), so
``publish`` refuses any build whose ``manifest.json`` reports
``uses_custom_js: true`` — a fast, local UX gate that prevents a guaranteed
server-side rejection and the broken render it would otherwise produce. See
mjswan ADR 0003 for the declarative/custom-JS split and the ``uses_custom_js``
marker.

The upload root layout mirrors the build's ``dist/`` tree minus everything but
data files, with ``manifest.json`` at the upload root, where the
engine's ``mount(element, configUrl)`` can resolve every other asset relative to
``configUrl``'s directory.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ── Client-side constraints (mirror the server's; fail fast and locally) ──────

#: File extensions uploaded as simulation *data*. Everything else in ``dist/``
#: (``.html`` / ``.js`` / ``.css`` / ``.wasm`` / fonts / images) is the engine
#: shell and is never sent — the platform loads a pinned engine from its CDN.
DATA_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".mjz", ".onnx", ".npz", ".ply", ".spz"}
)

MAX_FILE_BYTES: int = 50 * 1024 * 1024
MAX_TOTAL_BYTES: int = 200 * 1024 * 1024
MAX_FILES: int = 64

DEFAULT_API_BASE: str = "https://api.mjswan.com"
DEFAULT_WEB_BASE: str = "https://mjswan.com"
TOKEN_ENV_VAR: str = "MJSWAN_TOKEN"
API_BASE_ENV_VAR: str = "MJSWAN_API_BASE"
WEB_BASE_ENV_VAR: str = "MJSWAN_WEB_BASE"


def resolve_api_base(api_base: str | None) -> str:
    """Resolve the Cloud API base URL.

    Order: explicit ``api_base`` → ``$MJSWAN_API_BASE`` → :data:`DEFAULT_API_BASE`
    (the mjswan Cloud Production API). The env var lets a publish target a local
    ``wrangler dev`` (``http://localhost:8787``) or the RC API (``api-rc.mjswan.com``).
    """
    return (api_base or os.environ.get(API_BASE_ENV_VAR) or DEFAULT_API_BASE).rstrip(
        "/"
    )


def resolve_web_base(web_base: str | None = None) -> str:
    """Resolve the web app base URL (the human-facing site, not the API).

    Order: explicit ``web_base`` → ``$MJSWAN_WEB_BASE`` → :data:`DEFAULT_WEB_BASE`.
    Used to turn a published sim id into a viewable page URL.
    """
    return (web_base or os.environ.get(WEB_BASE_ENV_VAR) or DEFAULT_WEB_BASE).rstrip(
        "/"
    )


def simulation_url(sim_id: str, web_base: str | None = None) -> str:
    """The web app page URL for a published simulation."""
    return f"{resolve_web_base(web_base)}/s/{sim_id}"


def _user_agent() -> str:
    """A non-default User-Agent.

    The API is fronted by Cloudflare, which rejects the stdlib's default
    ``Python-urllib/X.Y`` agent with HTTP 403 (error 1010, "banned by browser
    signature"). Any real agent string passes, so identify the CLI explicitly.
    """
    try:
        from importlib.metadata import version

        ver = version("mjswan")
    except Exception:  # pragma: no cover - packaging edge cases
        ver = "0"
    return f"mjswan/{ver} (+https://github.com/ttktjmt/mjswan)"


USER_AGENT: str = _user_agent()

_CONTENT_TYPES: dict[str, str] = {
    ".json": "application/json",
    ".mjz": "application/zip",
    ".onnx": "application/octet-stream",
    ".npz": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".spz": "application/octet-stream",
}


def _content_type_for(path: str) -> str:
    return _CONTENT_TYPES.get(Path(path).suffix.lower(), "application/octet-stream")


class PublishError(Exception):
    """Raised when a publish cannot proceed (validation or server rejection).

    Carries an optional ``file`` so a 422 server response of the shape
    ``{"error", "file"}`` can be surfaced verbatim to the user.
    """

    def __init__(self, message: str, *, file: str | None = None) -> None:
        super().__init__(message)
        self.file = file


# ── HTTP transport (injectable so tests need no network) ──────────────────────


@dataclass
class HttpResponse:
    status: int
    body: bytes

    def json(self) -> dict:
        if not self.body:
            return {}
        return json.loads(self.body.decode("utf-8"))


class HttpTransport:
    """Minimal stdlib HTTP transport. Replaceable in tests.

    Returns an :class:`HttpResponse` for any status code (including 4xx/5xx)
    rather than raising, so callers can read structured error bodies.
    """

    def post_json(self, url: str, body: dict, token: str) -> HttpResponse:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        return self._send(req)

    def put_bytes(
        self,
        url: str,
        data: bytes,
        content_type: str,
        cache_control: str | None = None,
    ) -> HttpResponse:
        """PUT to a presigned URL.

        Both header values are signed into the URL by the Cloud
        (``SignedHeaders = cache-control;content-type;host``), so they are passed in
        rather than derived here: a value this side invents is a 403, not a
        differently-cached object.
        """
        headers = {"Content-Type": content_type, "User-Agent": USER_AGENT}
        if cache_control is not None:
            headers["Cache-Control"] = cache_control
        req = urllib.request.Request(url, data=data, method="PUT", headers=headers)
        return self._send(req)

    @staticmethod
    def _send(req: urllib.request.Request) -> HttpResponse:
        try:
            with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted base URL)
                return HttpResponse(status=resp.status, body=resp.read())
        except urllib.error.HTTPError as exc:
            # 4xx/5xx — read the structured error body instead of raising.
            return HttpResponse(status=exc.code, body=exc.read())
        except urllib.error.URLError as exc:
            raise PublishError(f"Network error contacting mjswan Cloud: {exc.reason}")


# ── File collection ───────────────────────────────────────────────────────────


@dataclass
class _DistFile:
    """A data file selected for upload."""

    upload_path: str  # POSIX path within the upload root (no leading slash)
    source: Path  # absolute path on disk
    size: int


@dataclass
class PublishPlan:
    """Everything resolved locally before any network call is made."""

    dist_dir: Path
    config: dict
    files: list[_DistFile] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def manifest(self) -> list[dict]:
        return [
            {
                "path": f.upload_path,
                "contentType": _content_type_for(f.upload_path),
                "size": f.size,
            }
            for f in self.files
        ]


def _find_config(dist_dir: Path) -> Path:
    """Locate the build's ``manifest.json``, the one descriptor at the document root."""
    candidate = dist_dir / "manifest.json"
    if candidate.is_file():
        return candidate
    raise PublishError(
        f"No manifest.json found in {dist_dir}. Pass the directory produced by "
        "builder.build(). A build with assets/config.json predates document format 1; "
        "rebuild it with this mjswan."
    )


def plan_publish(dist_dir: Path) -> PublishPlan:
    """Validate ``dist_dir`` and build the upload plan without any network I/O.

    Raises :class:`PublishError` on any client-side constraint violation:
    missing/invalid config, custom-JS build, too many/too-large files, or a
    path that escapes the upload root.
    """
    dist_dir = Path(dist_dir).expanduser().resolve()
    if not dist_dir.is_dir():
        raise PublishError(f"Not a directory: {dist_dir}")

    config_path = _find_config(dist_dir)
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise PublishError(f"Invalid manifest.json: {exc}")

    if config.get("uses_custom_js") is True:
        raise PublishError(
            "This build uses custom-JS MDP terms (uses_custom_js: true) and "
            "cannot be published to mjswan Cloud, which renders only declarative "
            "builds. Re-author the custom terms declaratively, or request the "
            "missing capability as an engine built-in. See mjswan ADR 0003.",
            file="manifest.json",
        )

    plan = PublishPlan(dist_dir=dist_dir, config=config)

    # `manifest.json` already sits at the root, where every path under a scene entry
    # resolves against `<project-id>/<scene-id>/`; nothing to hoist (ADR 0006 §8).
    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        if path.parent == dist_dir / "assets":
            continue  # the SPA's own directory: bundle metadata, not the document
        rel = path.relative_to(dist_dir).as_posix()
        _check_safe_path(rel)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise PublishError(
                f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)}MB limit: "
                f"{rel} ({size} bytes)",
                file=rel,
            )
        plan.files.append(_DistFile(upload_path=rel, source=path, size=size))

    _validate_plan(plan)
    return plan


def _check_safe_path(rel: str) -> None:
    if rel.startswith("/") or rel.startswith("\\"):
        raise PublishError(f"Absolute paths are not allowed: {rel}", file=rel)
    parts = Path(rel).parts
    if ".." in parts:
        raise PublishError(f"Path traversal is not allowed: {rel}", file=rel)


def _validate_plan(plan: PublishPlan) -> None:
    if len(plan.files) > MAX_FILES:
        raise PublishError(
            f"Too many files: {len(plan.files)} (max {MAX_FILES}). "
            "Reduce the number of scenes/policies/motions in this build."
        )
    if plan.total_bytes > MAX_TOTAL_BYTES:
        raise PublishError(
            f"Total upload size {plan.total_bytes} bytes exceeds the "
            f"{MAX_TOTAL_BYTES // (1024 * 1024)}MB limit."
        )


# ── Token resolution ───────────────────────────────────────────────────────────


def resolve_token(token: str | None) -> str:
    """Resolve the Supabase access token (GitHub OAuth).

    Order: explicit ``token`` argument → ``MJSWAN_TOKEN`` environment variable →
    credentials stored by ``mjswan login`` (refreshed transparently if near
    expiry). The error explains how to obtain a token.
    """
    resolved = token or os.environ.get(TOKEN_ENV_VAR)
    if resolved:
        return resolved

    # Fall back to a stored `mjswan login` session (imported lazily so publish
    # has no hard dependency on the auth flow).
    from mjswan import auth

    try:
        stored = auth.current_access_token()
    except auth.AuthError as exc:
        raise PublishError(
            f"Stored mjswan Cloud session could not be refreshed: {exc} "
            "Run `mjswan login` to sign in again."
        )
    if stored:
        return stored

    raise PublishError(
        "No mjswan Cloud access token found. Run `mjswan login` to sign in, set "
        f"the {TOKEN_ENV_VAR} environment variable to a Supabase access token "
        "(GitHub OAuth), or pass token=... explicitly."
    )


# ── The publish flow ───────────────────────────────────────────────────────────


@dataclass
class PublishResult:
    id: str
    sim_id: str
    upload_id: str


def publish_dist(
    dist_dir: str | Path,
    *,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    token: str | None = None,
    api_base: str | None = None,
    transport: HttpTransport | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> PublishResult:
    """Publish a built ``dist/`` directory, or a ``.swn`` document, to mjswan Cloud.

    Args:
        dist_dir: A directory produced by ``builder.build()``, or a ``.swn`` written by
            ``MjswanApp.save_document()``, which uploads the same file set.
        title: Simulation title. Defaults to the first project's name.
        description: Optional description.
        tags: Optional list of tags.
        token: Supabase access token. Falls back to ``$MJSWAN_TOKEN``.
        api_base: Cloud API base URL. Falls back to ``$MJSWAN_API_BASE``, then
            ``https://api.mjswan.com``.
        transport: HTTP transport (injectable for tests).
        on_progress: Optional callback invoked with human-readable status lines.

    Returns:
        :class:`PublishResult` with the published simulation id.

    Raises:
        PublishError: on any client-side validation failure or server rejection.
    """
    from .document import DocumentError, as_directory

    transport = transport or HttpTransport()
    base = resolve_api_base(api_base)
    notify = on_progress or (lambda _msg: None)

    # A document uploads exactly the file set its directory would, so the server never
    # needs to know it existed (ADR 0006 §8). The upload runs inside the context: a
    # document's tree is a temporary directory, and `plan.files` point into it.
    try:
        with as_directory(Path(dist_dir)) as tree:
            plan = plan_publish(tree)
            resolved_token = resolve_token(token)

            resolved_title = title or _default_title(plan.config)

            body: dict = {
                "title": resolved_title,
                "manifest": plan.manifest(),
                "config": plan.config,
            }
            if description is not None:
                body["description"] = description
            if tags:
                body["tags"] = tags

            notify(f"Requesting upload session for {len(plan.files)} file(s)…")
            session_resp = transport.post_json(
                f"{base}/api/simulations/upload-session", body, resolved_token
            )
            _raise_for_status(session_resp, "upload-session")
            session = _parse_json(session_resp, "upload-session")

            upload_id = session.get("upload_id")
            if not upload_id:
                raise PublishError("upload-session response missing upload_id.")
            uploads = {u["path"]: u for u in session.get("uploads", [])}

            by_path = {f.upload_path: f for f in plan.files}
            for path, upload in uploads.items():
                f = by_path.get(path)
                if f is None:
                    raise PublishError(
                        f"Server requested upload for unknown file: {path}", file=path
                    )
                url = upload["url"]
                _check_presigned_url(url, path)
                notify(f"Uploading {path} ({f.size} bytes)…")
                # The server's own header values, never this side's: it signs both
                # into the URL and returns them for exactly this reason. The local
                # `_content_type_for` is the advisory sent *with* the manifest, and
                # disagrees for `.mjz` — using it here was half of #119.
                put_resp = transport.put_bytes(
                    url,
                    f.source.read_bytes(),
                    upload.get("contentType") or _content_type_for(path),
                    upload.get("cacheControl"),
                )
                if put_resp.status not in (200, 201, 204):
                    raise PublishError(
                        f"Upload failed for {path}: HTTP {put_resp.status}", file=path
                    )

            notify("Committing…")
            commit_resp = transport.post_json(
                f"{base}/api/simulations/commit",
                {"upload_id": upload_id},
                # Re-resolved, not reused: uploading a real build takes minutes
                # (37 MB took ten), and a token resolved before the first PUT can
                # cross its expiry before the last one. Every byte then lands in R2
                # and the commit 401s, which reads as "not logged in" after a
                # ten-minute wait. Cheap when the token is still good — the resolver
                # only refreshes near expiry (#119).
                resolve_token(token),
            )
            _raise_for_status(commit_resp, "commit")
            commit = _parse_json(commit_resp, "commit")
            sim_id = commit.get("id")
            if not sim_id:
                raise PublishError("commit response missing id.")

            return PublishResult(
                id=sim_id, sim_id=session.get("sim_id", sim_id), upload_id=upload_id
            )
    except DocumentError as exc:
        # Only `unpack_document` raises this; the upload itself speaks PublishError.
        raise PublishError(str(exc), file=Path(dist_dir).name) from exc


def _check_presigned_url(url: str, path: str) -> None:
    """Reject an unusable presigned upload URL with an actionable message.

    A misconfigured server (missing ``R2_ACCOUNT_ID`` / ``R2_ACCESS_KEY_ID`` /
    ``R2_SECRET_ACCESS_KEY``) signs URLs like
    ``https://undefined.r2.cloudflarestorage.com/…?X-Amz-Credential=undefined…``.
    PUTting to that host surfaces as an opaque TLS handshake error; catch it here
    and point at the real cause (server-side storage credentials).
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https") or not host or "undefined" in url:
        raise PublishError(
            f"Server returned an invalid upload URL for {path} ({url!r}). "
            "This is a server-side misconfiguration — the mjswan Cloud API could "
            "not sign an R2 upload URL (its R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / "
            "R2_SECRET_ACCESS_KEY are likely unset). No client change can fix this.",
            file=path,
        )


def _default_title(config: dict) -> str:
    projects = config.get("projects") or []
    if projects and projects[0].get("name"):
        return str(projects[0]["name"])
    return "Untitled simulation"


def _raise_for_status(resp: HttpResponse, step: str) -> None:
    if resp.status in (200, 201):
        return
    # 422 responses are {error, file?}; surface them verbatim.
    try:
        payload = resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}
    error = payload.get("error") or f"{step} failed with HTTP {resp.status}"
    raise PublishError(error, file=payload.get("file"))


def _parse_json(resp: HttpResponse, step: str) -> dict:
    """Parse a successful response body as JSON, or raise an actionable error.

    A 2xx whose body is *not* JSON usually means the request was intercepted
    before it reached the API — most often an access proxy's login/redirect page
    (e.g. Cloudflare Access), or ``$MJSWAN_API_BASE`` pointing at the wrong host.
    Surface that instead of a raw ``JSONDecodeError``.
    """
    try:
        return resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        snippet = resp.body[:160].decode("utf-8", "replace").strip().replace("\n", " ")
        raise PublishError(
            f"{step}: expected a JSON response but got non-JSON (HTTP "
            f"{resp.status}). The request was likely intercepted — e.g. a "
            "login/redirect page from an access proxy (Cloudflare Access), or "
            "$MJSWAN_API_BASE points somewhere that isn't the mjswan Cloud API. "
            f"First bytes: {snippet!r}"
        )


__all__ = [
    "API_BASE_ENV_VAR",
    "DATA_EXTENSIONS",
    "DEFAULT_API_BASE",
    "DEFAULT_WEB_BASE",
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "TOKEN_ENV_VAR",
    "USER_AGENT",
    "WEB_BASE_ENV_VAR",
    "HttpResponse",
    "HttpTransport",
    "PublishError",
    "PublishPlan",
    "PublishResult",
    "plan_publish",
    "publish_dist",
    "resolve_api_base",
    "resolve_token",
    "resolve_web_base",
    "simulation_url",
]
