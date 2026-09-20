"""The one header both Cloud clients send.

The API is fronted by Cloudflare, which rejects the stdlib's default
``Python-urllib/X.Y`` agent with HTTP 403 (error 1010, "banned by browser signature").
Any real agent string passes, so the CLI identifies itself explicitly. :mod:`.publish`
and :mod:`.auth` both send it, and share it from here rather than one importing the
other for it.
"""

from __future__ import annotations


def _user_agent() -> str:
    try:
        from importlib.metadata import version

        ver = version("mjswan")
    except Exception:  # pragma: no cover - packaging edge cases
        ver = "0"
    return f"mjswan/{ver} (+https://github.com/ttktjmt/mjswan)"


USER_AGENT: str = _user_agent()

__all__ = ["USER_AGENT"]
