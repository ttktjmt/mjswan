"""Publish a built mjswan ``dist/`` to mjswan Cloud.

This module extracts the **data files** from a local build (manifest.json plus
scene/policy/motion/splat/skin assets, traced MDP graphs and license files; never
HTML/JS/WASM) and publishes them as mjswan Cloud stores them: each file once, by the
SHA-256 of its bytes, cut into pieces where :mod:`.blob` says (mjswan-cloud ADR 0017):

1. ``POST {base}/api/simulations/upload-session`` with the parsed ``manifest.json`` and
   the file map → a presigned ``PUT`` for each piece or license text the platform does
   not hold yet.
2. ``PUT`` each of those.
3. ``POST {base}/api/simulations/verify`` until nothing remains: the platform checks
   every piece's SHA-256 as it moves it into place.
4. ``POST {base}/api/simulations/commit`` with the ``upload_id`` → the sim id.

The platform renders only **declarative** builds (no author-supplied code), so
``publish`` refuses any build whose ``manifest.json`` reports
``uses_custom_js: true`` — a fast, local UX gate that prevents a guaranteed
server-side rejection and the broken render it would otherwise produce. See
mjswan ADR 0003 for the declarative/custom-TS split and the ``uses_custom_js``
marker.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import urllib.error
import urllib.request
import warnings
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..document.skin import read_skin_file
from ..license import (
    LICENSE_FILE_MAX_BYTES,
    LicenseDeclaration,
    blocked_refusal,
    declare,
    is_license_file_path,
    restricted_warning,
)
from ..license.vrm import VrmLicense, describe, vrm_license
from .blob import cut, sha256
from .transport import USER_AGENT

# ── Client-side constraints (mirror the server's; fail fast and locally) ──────

#: File extensions uploaded as simulation *data*. Everything else in ``dist/``
#: (``.html`` / ``.js`` / ``.css`` / ``.wasm`` / fonts / images) is the engine
#: shell and is never sent — the platform loads a pinned engine from its CDN.
#: ``LICENSE`` / ``NOTICE`` files are admitted by name instead, in the project
#: directory and the directory below it only (ADR 0007 §1).
DATA_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".mjz", ".onnx", ".npz", ".ply", ".spz", ".vrm", ".glb"}
)
SKIN_EXTENSIONS: frozenset[str] = frozenset({".vrm", ".glb"})

#: The most of a skin's glTF JSON mjswan Cloud reads, and of a VRM's meta it stores.
GLB_JSON_MAX_BYTES: int = 8 * 1024 * 1024
VRM_TEXT_MAX: int = 256
VRM_URL_MAX: int = 2048
VRM_AUTHORS_MAX: int = 32

#: Caps on what a build holds, not on what it packs to: a ``.mjz`` entry counts
#: inflated, and a piece two files share counts once (mjswan-cloud ADR 0017 §2.5).
MAX_FILE_BYTES: int = 50 * 1024 * 1024
MAX_TOTAL_BYTES: int = 128 * 1024 * 1024
MAX_FILES: int = 200
MAX_PIECES: int = 1000

#: Pieces the platform checks a call: about five seconds each, so the count moves.
VERIFY_STEP: int = 50
#: ``PUT``s in flight at once.
UPLOAD_CONCURRENCY: int = 6

DEFAULT_API_BASE: str = "https://api.mjswan.com"
DEFAULT_WEB_BASE: str = "https://mjswan.com"
TOKEN_ENV_VAR: str = "MJSWAN_TOKEN"
API_BASE_ENV_VAR: str = "MJSWAN_API_BASE"
WEB_BASE_ENV_VAR: str = "MJSWAN_WEB_BASE"

#: What a published simulation's id may be, before a URL is built from it.
_SIM_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


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


class PublishError(Exception):
    """Raised when a publish cannot proceed (validation or server rejection).

    Carries an optional ``file`` so a 422 server response of the shape
    ``{"error", "file"}`` can be surfaced to the user.
    """

    def __init__(self, message: str, *, file: str | None = None) -> None:
        # Messages carry VRM meta and server text, and the Python API prints them as raised.
        super().__init__(_printable(str(message)))
        self.file = None if file is None else _printable(str(file))


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

        Both headers are in the URL's ``SignedHeaders``, so the caller passes the
        values the Cloud returned; one invented here is a 403.
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
    #: The license files among ``files`` and what each says, in path order.
    licenses: list[LicenseDeclaration] = field(default_factory=list)
    #: Each VRM among ``files`` -> the terms its ``meta`` states, in path order.
    vrms: dict[str, VrmLicense] = field(default_factory=dict)
    #: The file map upload-session takes, and the platform keeps as ``files.json``: every
    #: file but the descriptor as its pieces, a ``.mjz`` entry by entry, and each license
    #: file as the hash of its text (mjswan-cloud ADR 0017 §2.2).
    file_map: dict = field(
        default_factory=lambda: {"format": 1, "files": {}, "licenses": {}}
    )
    #: Each piece's and text's bytes, by hash.
    pieces: dict[str, bytes] = field(default_factory=dict)

    def license_warnings(self) -> list[str]:
        """One line per restricted license or VRM, for the author to read first."""
        return [
            restricted_warning(d.path, d.spdx)
            for d in self.licenses
            if d.tier == "restricted" and d.spdx
        ] + [
            f"{path}: its VRM license allows no commercial use ({describe(terms)})."
            for path, terms in self.vrms.items()
            if not terms.commercial
        ]

    @property
    def to_attest(self) -> list[str]:
        """The VRMs only their author, or someone they licensed, may use as an avatar."""
        return [path for path, terms in self.vrms.items() if terms.needs_attestation]


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
    """Validate ``dist_dir`` and build the upload plan, every file read, cut and hashed,
    without any network I/O.

    Raises :class:`PublishError` on any client-side constraint violation:
    missing/invalid config, custom-TS build, a ``.mjb`` scene, too many/too-large
    files, a path that escapes the upload root, a license file whose terms forbid
    redistribution (ADR 0007 §3), or a skin the engine cannot draw or whose VRM license
    forbids redistribution.
    """
    dist_dir = Path(dist_dir).expanduser().resolve()
    if not dist_dir.is_dir():
        raise PublishError(f"Not a directory: {dist_dir}")

    from ..document import DocumentError, read_manifest

    _find_config(dist_dir)
    try:
        config = read_manifest(dist_dir)
    except json.JSONDecodeError as exc:
        raise PublishError(f"Invalid manifest.json: {exc}")
    except DocumentError as exc:
        raise PublishError(str(exc), file="manifest.json") from None

    if config.get("uses_custom_js") is True:
        raise PublishError(
            "This build uses custom-TS MDP terms (uses_custom_js: true) and "
            "cannot be published to mjswan Cloud, which renders only declarative "
            "builds. Re-author the custom terms declaratively, or request the "
            "missing capability as an engine built-in. See mjswan ADR 0003.",
            file="manifest.json",
        )

    # An .mjb is not uploaded (DATA_EXTENSIONS), so the scene would publish without it.
    for project in config.get("projects", []):
        for scene in project.get("scenes", []):
            if scene.get("scene", "").endswith(".mjb"):
                raise PublishError(
                    f"Scene {scene.get('name')!r} was built with add_scene(model=...), "
                    "which saves it as .mjb, and mjswan Cloud takes .mjz scenes only. "
                    "Build it with add_scene(spec=...) to publish it.",
                    file=f"{project['id']}/{scene['id']}/{scene['scene']}",
                )

    plan = PublishPlan(dist_dir=dist_dir, config=config)

    # `manifest.json` already sits at the root, where every path under a scene entry
    # resolves against `<project-id>/<scene-id>/`; nothing to hoist (ADR 0006 §8).
    for path in sorted(dist_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.parent == dist_dir / "assets":
            continue  # the SPA's own directory: bundle metadata, not the document
        rel = path.relative_to(dist_dir).as_posix()
        # A build writes regular files only; a link would upload whatever it names.
        if path.is_symlink():
            raise PublishError(f"Symbolic links are not published: {rel}", file=rel)
        # The root LICENSE and NOTICE are the engine's, not the work's, and are not admitted.
        if is_license_file_path(rel):
            _check_safe_path(rel)
            size = path.stat().st_size
            if size > LICENSE_FILE_MAX_BYTES:
                raise PublishError(
                    f"License file exceeds the {LICENSE_FILE_MAX_BYTES // 1024} KiB "
                    f"limit: {rel} ({size} bytes)",
                    file=rel,
                )
            declaration = declare(rel, path.read_bytes())
            if declaration is not None:
                if declaration.tier == "blocked" and declaration.spdx:
                    raise PublishError(
                        blocked_refusal(rel, declaration.spdx) + ".", file=rel
                    )
                plan.licenses.append(declaration)
            plan.files.append(_DistFile(upload_path=rel, source=path, size=size))
            continue
        if path.suffix.lower() not in DATA_EXTENSIONS:
            continue
        _check_safe_path(rel)
        size = path.stat().st_size
        # A `.mjz` is held to the cap entry by entry, as the map lists it.
        if path.suffix.lower() != ".mjz":
            _check_size(rel, size)
        if path.suffix.lower() in SKIN_EXTENSIONS:
            _check_skin(plan, rel, path)
        plan.files.append(_DistFile(upload_path=rel, source=path, size=size))

    _validate_plan(plan)
    _map_files(plan)
    return plan


def _check_skin(plan: PublishPlan, rel: str, path: Path) -> None:
    """Refuse a skin mjswan Cloud would: one the engine cannot draw, one named as the
    other format, or a VRM it may not host or store. A VRM's terms go in the plan."""
    try:
        file = read_skin_file(path)
    except ValueError as exc:
        raise PublishError(str(exc), file=rel) from None
    if (path.suffix.lower() == ".vrm") != (file.format == "vrm"):
        raise PublishError(
            f"{rel} is a VRM: publish it as .vrm, whose license is read from its meta."
            if file.format == "vrm"
            else f"{rel} has no VRM extension: a skin that is not a VRM is a .glb.",
            file=rel,
        )
    if file.json_bytes > GLB_JSON_MAX_BYTES:
        raise PublishError(
            f"{rel} has a {file.json_bytes / 2**20:.1f} MB glTF JSON chunk; mjswan Cloud "
            f"reads at most {GLB_JSON_MAX_BYTES // 2**20} MB.",
            file=rel,
        )
    if file.vrm_version is None:
        return
    terms = vrm_license(file.meta, file.vrm_version)
    if _too_long(terms):
        raise PublishError(
            f"{rel} has a VRM meta longer than mjswan Cloud stores (at most "
            f"{VRM_TEXT_MAX} characters per name, {VRM_AUTHORS_MAX} authors and "
            f"{VRM_URL_MAX}-character URLs).",
            file=rel,
        )
    if not terms.redistribution:
        raise PublishError(
            f"{rel}: its VRM license forbids redistribution ({describe(terms)}), and "
            "publishing hosts it for anyone to download.",
            file=rel,
        )
    plan.vrms[rel] = terms


def _too_long(terms: VrmLicense) -> bool:
    # A VRM 0.x names its license, a VRM 1.0 links to it.
    name, url = (terms.license, None) if terms.version == "0" else (None, terms.license)
    return (
        len(terms.authors) > VRM_AUTHORS_MAX
        or any(len(s) > VRM_TEXT_MAX for s in (terms.name, name, *terms.authors) if s)
        or any(len(s) > VRM_URL_MAX for s in (url, terms.other_license_url) if s)
    )


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


def _map_files(plan: PublishPlan) -> None:
    """Fill ``plan.file_map`` and ``plan.pieces``, checking the platform's caps as each
    piece lands so an oversized build is never held whole."""
    sizes: dict[str, int] = {}
    total = 0

    def mapped(label: str, name: str, data: bytes) -> dict:
        nonlocal total
        _check_size(label, len(data))
        entry = cut(name, data)
        at = 0
        for sha, size in entry.get("chunks", [[entry["sha256"], entry["size"]]]):
            if sha not in sizes:
                sizes[sha] = size
                total += size
                plan.pieces[sha] = data[at : at + size]
            at += size
        if len(sizes) > MAX_PIECES:
            raise PublishError(
                f"This build is over {MAX_PIECES} pieces, the most mjswan Cloud takes. "
                "Reduce the number of scenes/policies/motions in it."
            )
        if total > MAX_TOTAL_BYTES:
            raise PublishError(
                f"This build holds over {MAX_TOTAL_BYTES // (1024 * 1024)}MB, scenes "
                "unpacked, the most mjswan Cloud takes."
            )
        return entry

    for f in plan.files:
        path = f.upload_path
        if path == "manifest.json":
            continue  # upload-session takes it as `config`
        if is_license_file_path(path):
            text = f.source.read_bytes()
            plan.file_map["licenses"][path] = digest = sha256(text)
            plan.pieces[digest] = text
        elif path.lower().endswith(".mjz"):
            plan.file_map["files"][path] = {"entries": _entries(path, f.source, mapped)}
        else:
            plan.file_map["files"][path] = mapped(path, path, f.source.read_bytes())


#: The methods build/mjz.py writes; bzip2 and LZMA inflate a whole read first.
_ARCHIVE_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


def _entries(path: str, source: Path, mapped: Callable[[str, str, bytes], dict]):
    """A scene archive's entries in its order, each mapped; the platform zips them again."""
    # The caps mapped() applies count a repeated entry once, so the reads need their own.
    try:
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_PIECES:
                raise PublishError(
                    f"{path} has over {MAX_PIECES} entries, the most mjswan Cloud takes.",
                    file=path,
                )
            out, read = [], 0
            for info in infos:
                label = f"{path}/{info.filename}"
                if info.compress_type not in _ARCHIVE_METHODS:
                    raise PublishError(
                        f"{label} uses a compression no build writes.", file=label
                    )
                _check_size(label, info.file_size)
                read += info.file_size
                if read > MAX_TOTAL_BYTES:
                    raise PublishError(
                        f"{path} unpacks to over {MAX_TOTAL_BYTES} bytes.", file=path
                    )
                # read(n) inflates at most n bytes; read() inflates up to 2**30 first.
                with archive.open(info) as entry:
                    data = entry.read(info.file_size)
                out.append(
                    {"name": info.filename, **mapped(label, info.filename, data)}
                )
            return out
    except (zipfile.BadZipFile, zlib.error) as exc:
        raise PublishError(
            f"{path} is not a readable zip archive ({exc}). Build it again.", file=path
        ) from None


def _check_size(label: str, size: int) -> None:
    if size > MAX_FILE_BYTES:
        raise PublishError(
            f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)}MB limit: {label} "
            f"({size} bytes)",
            file=label,
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

    from . import auth

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
    on_warning: Callable[[str], None] | None = None,
    attest: Callable[[str, VrmLicense], bool] | None = None,
) -> PublishResult:
    """Publish a built ``dist/`` directory, or a ``.swn`` document, to mjswan Cloud.

    Args:
        dist_dir: A directory produced by ``builder.build()``, or a ``.swn`` written by
            ``MjswanApp.save_document()``, which uploads the same file set.
        title: Simulation title. Defaults to the first project's name.
        description: Optional description.
        tags: Optional list of tags.
        token: Supabase access token. Falls back to ``$MJSWAN_TOKEN``, then the
            ``mjswan login`` session.
        api_base: Cloud API base URL. Falls back to ``$MJSWAN_API_BASE``, then
            ``https://api.mjswan.com``.
        transport: HTTP transport (injectable for tests).
        on_progress: Optional callback invoked with human-readable status lines,
            among them one per license file the build declares.
        on_warning: Optional callback for the warning a restricted license or a
            VRM only for non-commercial use earns (ADR 0007 §3); defaults to
            :func:`warnings.warn`. The publish proceeds.
        attest: Asked, for each VRM only its author or someone they licensed may use
            as an avatar, whether the publisher is one of them. The VRM is published
            only on ``True``, and the page shows that it was on their word. ``None``
            refuses every such VRM.

    Returns:
        :class:`PublishResult` with the published simulation id.

    Raises:
        PublishError: on any client-side validation failure or server rejection.
    """
    from ..document import DocumentError, as_directory

    transport = transport or HttpTransport()
    base = resolve_api_base(api_base)
    report = on_progress or (lambda _msg: None)
    caution = on_warning or (
        lambda msg: warnings.warn(msg, category=RuntimeWarning, stacklevel=4)
    )

    # Messages carry names the build's maker chose: file names, VRM authors.
    def notify(msg: str) -> None:
        report(_printable(msg))

    def warn(msg: str) -> None:
        caution(_printable(msg))

    # A document uploads exactly the file set its directory would, so the server never
    # needs to know it existed (ADR 0006 §8). The upload runs inside the context: a
    # document's tree is a temporary directory, and `plan.files` point into it.
    try:
        with as_directory(Path(dist_dir)) as tree:
            plan = plan_publish(tree)
            # Declared before any byte moves: a publish never carries a license file
            # the author did not see.
            for declaration in plan.licenses:
                notify(f"License file: {declaration.describe()}")
            for path, terms in plan.vrms.items():
                notify(f"Skin: {path}: {describe(terms)}")
            for message in plan.license_warnings():
                warn(message)
            attested = [
                path
                for path in plan.to_attest
                if attest is not None and attest(path, plan.vrms[path])
            ]
            for path in plan.to_attest:
                if path not in attested:
                    raise PublishError(
                        f"{path} may be used as an avatar only by its author or someone "
                        "they licensed, and you did not confirm you are.",
                        file=path,
                    )
            resolved_token = resolve_token(token)

            resolved_title = title or _default_title(plan.config)

            body: dict = {
                "title": resolved_title,
                "config": plan.config,
                "files": plan.file_map,
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

            def send(uploads: list[dict]) -> None:
                _send(transport, uploads, plan.pieces, notify)
                # Re-resolved, not reused, here and at commit: an upload runs for
                # minutes and can cross the token's expiry. Near-free, since the
                # resolver only refreshes close to expiry.
                remaining = total = len(uploads)
                while remaining:
                    notify(f"Checking {total - remaining} of {total} piece(s)…")
                    verify_resp = transport.post_json(
                        f"{base}/api/simulations/verify",
                        {"upload_id": upload_id, "limit": VERIFY_STEP},
                        resolve_token(token),
                    )
                    _raise_for_status(verify_resp, "verify")
                    remaining = _remaining(
                        _parse_json(verify_resp, "verify"), remaining
                    )

            def commit_upload() -> HttpResponse:
                notify("Committing…")
                return transport.post_json(
                    f"{base}/api/simulations/commit",
                    {
                        "upload_id": upload_id,
                        **({"attested_skins": attested} if attested else {}),
                    },
                    resolve_token(token),
                )

            send(session.get("uploads", []))
            commit_resp = commit_upload()
            # A piece another simulation's takedown released since upload-session goes
            # again, once (mjswan-cloud ADR 0017 §2.3).
            released = _released(commit_resp)
            if released is not None:
                send(released)
                commit_resp = commit_upload()
            _raise_for_status(commit_resp, "commit")
            commit = _parse_json(commit_resp, "commit")
            sim_id = commit.get("id")
            if not isinstance(sim_id, str) or not _SIM_ID.fullmatch(sim_id):
                raise PublishError("commit response has no valid id.")

            return PublishResult(
                id=sim_id, sim_id=session.get("sim_id", sim_id), upload_id=upload_id
            )
    except DocumentError as exc:
        # Only `unpack_document` raises this; the upload itself speaks PublishError.
        raise PublishError(str(exc), file=Path(dist_dir).name) from exc


def _printable(text: str) -> str:
    """*text* without its control characters, newlines kept."""
    return "".join(c for c in text if c == "\n" or unicodedata.category(c) != "Cc")


def _send(
    transport: HttpTransport,
    uploads: list[dict],
    pieces: dict[str, bytes],
    notify: Callable[[str], None],
) -> None:
    """``PUT`` each piece or text upload-session asked for, a few at a time."""
    for upload in uploads:
        _check_upload(upload)
        if upload["sha256"] not in pieces:
            raise PublishError(f"Server requested an unknown piece: {upload['sha256']}")
        _check_presigned_url(upload["url"], upload["sha256"])
    size = sum(len(pieces[u["sha256"]]) for u in uploads)
    notify(f"Uploading {len(uploads)} piece(s), {size} bytes…")

    def put(upload: dict) -> None:
        # Both headers are the server's signed values, sent as they came.
        resp = transport.put_bytes(
            upload["url"],
            pieces[upload["sha256"]],
            upload["contentType"],
            upload["cacheControl"],
        )
        if resp.status not in (200, 201, 204):
            raise PublishError(f"Upload failed: HTTP {resp.status}")

    with ThreadPoolExecutor(UPLOAD_CONCURRENCY) as pool:
        list(pool.map(put, uploads))


_UPLOAD_KEYS = ("sha256", "url", "contentType", "cacheControl")


def _check_upload(upload: object) -> None:
    """Refuse a presigned ``PUT`` that lacks what sending it takes."""
    missing = [
        key
        for key in _UPLOAD_KEYS
        if not isinstance(upload, dict) or not isinstance(upload.get(key), str)
    ]
    if missing:
        raise PublishError(
            f"mjswan Cloud asked for an upload without {', '.join(missing)}. "
            "Update mjswan; if it is current, report this."
        )


def _remaining(payload: dict, before: int) -> int:
    """What verify says is left. Each call moves at least one piece, so a count that
    does not fall would loop forever."""
    left = payload.get("remaining")
    if type(left) is not int or not 0 <= left < before:
        raise PublishError(
            f"verify answered remaining={left!r} with {before} piece(s) left to "
            "check. Publish again; if it persists, report this."
        )
    return left


def _released(resp: HttpResponse) -> list[dict] | None:
    """The pieces to send again, when commit answers that a takedown released them."""
    if resp.status != 409:
        return None
    try:
        payload = resp.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if payload.get("code") != "files_released":
        return None
    return payload.get("uploads") or None


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
    "MAX_PIECES",
    "MAX_TOTAL_BYTES",
    "TOKEN_ENV_VAR",
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
