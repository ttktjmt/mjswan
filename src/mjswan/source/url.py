"""A plain http(s) URL as a source, fetched with the standard library."""

from __future__ import annotations

import atexit
import shutil
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


def fetch_file(url: str) -> Path:
    """Download *url* to a fresh temporary directory and return the file's path.

    Not cached: the build copies the file into the app, so it is needed once, and the
    directory goes when the process exits. A server that stalls for a minute fails it.
    """
    name = PurePosixPath(urlparse(url).path).name or "download"
    directory = tempfile.mkdtemp(prefix="mjswan-")
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    target = Path(directory) / name
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:  # noqa: S310 (author's own URL)
        shutil.copyfileobj(response, out)
    return target


__all__ = ["fetch_file"]
