"""The local server answers only for files inside the app directory, and, bound to
loopback, only requests addressed to it."""

from __future__ import annotations

import http.client
import http.server
import threading
from pathlib import Path

import pytest

from mjswan.app import _hosts_answered, _serving_handler


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
    (app / "inner").mkdir()
    (app / "inner" / "index.html").write_text("<!doctype html>")
    for name, index in (("sub", "index.html"), ("htm", "index.htm")):
        (app / name).mkdir()
        (app / name / index).symlink_to(outside / "secret.txt")

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


@pytest.mark.parametrize("path", ["/", "/index.html", "/again.html", "/inner/"])
def test_serves_the_app_and_links_within_it(port, path):
    assert _status(port, path) == 200


@pytest.mark.parametrize(
    "path", ["/leak.txt", "/fs/", "/fs/secret.txt", "/empty/", "/sub/", "/htm/"]
)
def test_refuses_links_out_and_directory_listings(port, path):
    assert _status(port, path) == 404


@pytest.fixture
def loopback_port(tmp_path: Path):
    app = tmp_path / "dist"
    app.mkdir()
    (app / "manifest.json").write_text("{}")
    handler = _serving_handler(app, _hosts_answered("localhost"))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def _status_for_host(port: int, host: str | None) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.putrequest("GET", "/manifest.json", skip_host=True)
        if host is not None:
            conn.putheader("Host", host)
        conn.endheaders()
        return conn.getresponse().status
    finally:
        conn.close()


# Any port: one forwarded over SSH or by an editor arrives under another.
@pytest.mark.parametrize(
    "host",
    ["localhost:8080", "127.0.0.1:1234", "[::1]:9000", "LOCALHOST", "localhost"],
)
def test_a_loopback_server_answers_requests_addressed_to_it(loopback_port, host):
    assert _status_for_host(loopback_port, host) == 200


# What a page reaches the server with after pointing its own domain at 127.0.0.1.
@pytest.mark.parametrize(
    "host",
    ["evil.test:8080", "localhost.evil.test", "127.0.0.1.nip.io", "[::1", "", None],
)
def test_a_loopback_server_refuses_requests_addressed_elsewhere(loopback_port, host):
    assert _status_for_host(loopback_port, host) == 403


@pytest.mark.parametrize(
    ("bind", "answered"),
    [
        ("localhost", {"localhost", "127.0.0.1", "::1"}),
        ("LocalHost", {"localhost", "127.0.0.1", "::1"}),
        ("127.0.0.1", {"localhost", "127.0.0.1", "::1"}),
        ("127.0.0.2", {"localhost", "127.0.0.1", "::1", "127.0.0.2"}),
        ("::1", {"localhost", "127.0.0.1", "::1"}),
        ("0.0.0.0", None),
        ("", None),
        ("192.168.1.5", None),
        ("devbox.local", None),
    ],
)
def test_only_a_loopback_bind_checks_the_host(bind, answered):
    """Bound anywhere else, Colab's server included, it was opened to the network."""
    assert _hosts_answered(bind) == (None if answered is None else frozenset(answered))
