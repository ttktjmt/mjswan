"""What both Cloud clients send a request with.

- ``USER_AGENT``: the API is fronted by Cloudflare, which rejects the stdlib's default
  ``Python-urllib/X.Y`` agent with HTTP 403 (error 1010, "banned by browser signature").
  Any real agent string passes.
- ``open_url``: urllib's redirect handler copies a request's headers to the new URL
  whatever its host or scheme, ``Authorization`` included, so no redirect is followed.
- ``require_token_safe``: a URL a token is sent to is https, or http on a loopback host
  for a local server.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

#: The codes urllib's redirect handler would follow.
REDIRECT_CODES: frozenset[int] = frozenset({301, 302, 303, 307, 308})

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _user_agent() -> str:
    try:
        from importlib.metadata import version

        ver = version("mjswan")
    except Exception:  # pragma: no cover - packaging edge cases
        ver = "0"
    return f"mjswan/{ver} (+https://github.com/ttktjmt/mjswan)"


USER_AGENT: str = _user_agent()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: PLR0917
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def open_url(req: urllib.request.Request):
    """Open *req* as ``urlopen`` does, but raise a redirect as its ``HTTPError``."""
    return _OPENER.open(req)  # noqa: S310 (a checked base URL)


def refused_redirect(req: urllib.request.Request, exc: urllib.error.HTTPError) -> str:
    """What to tell the user about the redirect *exc* answered *req* with."""
    host = urllib.parse.urlsplit(req.full_url).netloc
    target = exc.headers.get("Location") or "no location"
    return (
        f"{host} answered with a redirect ({exc.code}) to {target}. mjswan follows no "
        "redirect, which would take the request's credentials along; point it at the "
        "URL the redirect names, if you trust it."
    )


def require_token_safe(url: str, name: str, error: type[Exception]) -> str:
    """*url* if a token may be sent to it: https, or http on a loopback host for a local
    server. Raises *error*, naming the URL as *name*, otherwise."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    if host and (
        parts.scheme == "https" or (parts.scheme == "http" and host in _LOOPBACK_HOSTS)
    ):
        return url
    raise error(
        f"{name} is {url!r}, and requests to it carry your token: use an https URL, "
        "or http on localhost, 127.0.0.1 or ::1 for a local server."
    )


__all__ = [
    "REDIRECT_CODES",
    "USER_AGENT",
    "open_url",
    "refused_redirect",
    "require_token_safe",
]
