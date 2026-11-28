"""What the Cloud clients send their requests through: no redirect is followed, and a
token goes only to an https URL or a local server.

Layer: L1 (local HTTP servers on 127.0.0.1; no network).
"""

from __future__ import annotations

import contextlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mjswan.cloud import auth
from mjswan.cloud.auth import AuthError, AuthTransport
from mjswan.cloud.publish import HttpTransport, PublishError, resolve_api_base

REDIRECTS = [301, 302, 303, 307, 308]


@contextlib.contextmanager
def _serving(handler: type[BaseHTTPRequestHandler]):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, format, *args) -> None:
        pass


@contextlib.contextmanager
def _redirecting(code: int):
    """A server that answers every request with a redirect to a second one, which notes
    the ``Authorization`` of each request reaching it."""
    reached: list[str | None] = []

    class Elsewhere(_Quiet):
        def _answer(self) -> None:
            reached.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_GET = do_POST = _answer

    with _serving(Elsewhere) as elsewhere:
        target = f"{elsewhere}/elsewhere"

        class Redirect(_Quiet):
            def _answer(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(code)
                self.send_header("Location", target)
                self.end_headers()

            do_GET = do_POST = _answer

        with _serving(Redirect) as base:
            yield base, target, reached


@pytest.mark.parametrize("code", REDIRECTS)
def test_publish_follows_no_redirect(code):
    with _redirecting(code) as (base, target, reached):
        with pytest.raises(
            PublishError, match=rf"redirect \({code}\) to {re.escape(target)}"
        ):
            HttpTransport().post_json(
                f"{base}/api/simulations/upload-session", {}, "tok"
            )
    assert reached == []


@pytest.mark.parametrize("code", REDIRECTS)
def test_login_follows_no_redirect(code):
    with _redirecting(code) as (base, target, reached):
        with pytest.raises(
            AuthError, match=rf"redirect \({code}\) to {re.escape(target)}"
        ):
            AuthTransport().get(f"{base}/auth/v1/user", {"Authorization": "Bearer tok"})
    assert reached == []


def test_an_error_status_still_comes_back_with_its_body():
    class Unprocessable(_Quiet):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(422)
            self.end_headers()
            self.wfile.write(b'{"error": "bad file", "file": "a.onnx"}')

    with _serving(Unprocessable) as base:
        resp = HttpTransport().post_json(f"{base}/api/simulations/commit", {}, "tok")
    assert resp.status == 422
    assert resp.json() == {"error": "bad file", "file": "a.onnx"}


@pytest.mark.parametrize(
    "url",
    [
        "http://api.mjswan.com",
        "http://localhost.evil.test",
        "http://localhost@evil.test",
        "ftp://api.mjswan.com",
        "file:///tmp/api",
        "https://",
        "api.mjswan.com",
    ],
)
def test_a_token_goes_to_no_url_but_https_or_a_local_server(url, monkeypatch):
    with pytest.raises(PublishError, match="carry your token"):
        resolve_api_base(url)
    monkeypatch.setenv(auth.SUPABASE_URL_ENV_VAR, url)
    with pytest.raises(AuthError, match="carry your token"):
        auth.supabase_url()


@pytest.mark.parametrize(
    "url",
    [
        "https://api-rc.mjswan.com",
        "http://localhost:8787",
        "http://127.0.0.1:8787",
        "http://[::1]:8787",
    ],
)
def test_https_and_a_local_server_pass(url, monkeypatch):
    assert resolve_api_base(url) == url
    monkeypatch.setenv(auth.SUPABASE_URL_ENV_VAR, url)
    assert auth.supabase_url() == url
