"""The local server answers only for files inside the app directory."""

from __future__ import annotations

import http.client
import http.server
import threading
from pathlib import Path

import pytest

from mjswan.app import _serving_handler


@pytest.fixture
def port(tmp_path: Path):
    app = tmp_path / "dist"
    (app / "empty").mkdir(parents=True)
    (app / "index.html").write_text("<!doctype html>")
    (app / "again.html").symlink_to(app / "index.html")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not part of the build")
    (app / "leak.txt").symlink_to(outside / "secret.txt")
    (app / "fs").symlink_to(outside, target_is_directory=True)

    server = http.server.HTTPServer(("127.0.0.1", 0), _serving_handler(app))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _status(port: int, path: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", path)
        return conn.getresponse().status
    finally:
        conn.close()


@pytest.mark.parametrize("path", ["/", "/index.html", "/again.html"])
def test_serves_the_app_and_links_within_it(port, path):
    assert _status(port, path) == 200


@pytest.mark.parametrize("path", ["/leak.txt", "/fs/", "/fs/secret.txt", "/empty/"])
def test_refuses_links_out_and_directory_listings(port, path):
    assert _status(port, path) == 404
