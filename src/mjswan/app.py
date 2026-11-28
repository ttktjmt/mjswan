"""MjswanApp class for exporting and running applications.

This module defines the MjswanApp class which represents a built application
that can be saved to disk or launched in a web browser.

See ``context/document.md``.
"""

from __future__ import annotations

import ipaddress
import re
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cloud.publish import PublishResult

#: A Vite bundle file, named by its content hash (`plugins.js` beside it is not).
_HASHED_ASSET = re.compile(r"/assets/[^/]+-[\w-]{8}\.\w+")

#: What a server bound to loopback answers to, whichever of them it was bound with.
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _hosts_answered(bind_host: str) -> frozenset[str] | None:
    """The ``Host`` names a server bound to *bind_host* answers, or ``None`` for any.

    Bound to loopback, a request naming another host comes from a page that pointed its
    own domain here (DNS rebinding). Bound elsewhere, Colab's server included, the server
    was opened to other machines on purpose.
    """
    try:
        loopback = (
            bind_host == "localhost" or ipaddress.ip_address(bind_host).is_loopback
        )
    except ValueError:
        loopback = False
    return _LOOPBACK_NAMES | {bind_host.lower()} if loopback else None


def _host_name(header: str | None) -> str | None:
    """The host a ``Host`` header names, lowercased and without its port."""
    if not header:
        return None
    try:
        return urllib.parse.urlsplit(f"//{header}").hostname
    except ValueError:
        return None


def _serving_handler(app_dir: Path, hosts: frozenset[str] | None = None) -> type:
    """The request handler ``launch`` serves ``app_dir`` with, answering only requests
    addressed to one of ``hosts`` when given."""
    import http.server
    import os
    from http import HTTPStatus

    directory = str(app_dir)
    root = os.path.realpath(directory)

    class CrossOriginIsolatedHandler(http.server.SimpleHTTPRequestHandler):
        """HTTP handler with Cross-Origin Isolation headers for SharedArrayBuffer."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def send_head(self):
            # The port is not checked: a port forwarded over SSH arrives under another.
            if hosts is not None and _host_name(self.headers.get("Host")) not in hosts:
                self.send_error(HTTPStatus.FORBIDDEN, "Unknown host")
                return None
            # translate_path drops `..` but open() follows links, also for the index page
            # the stdlib picks for a directory (both names: index_pages is 3.12+).
            path = self.translate_path(self.path)
            paths = [path]
            if os.path.isdir(path):
                paths += [os.path.join(path, n) for n in ("index.html", "index.htm")]
            for p in map(os.path.realpath, paths):
                if p != root and not p.startswith(root + os.sep):
                    self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                    return None
            return super().send_head()

        def list_directory(self, path):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")

        def send_response(self, code, message=None):
            super().send_response(code, message)
            # Builds reuse unhashed URLs, and Last-Modified can go backwards.
            # A 404 is not kept either: another build may have the file.
            if code == 200 and _HASHED_ASSET.fullmatch(self.path):
                self.send_header("Cache-Control", "max-age=31536000, immutable")
            else:
                self.send_header("Cache-Control", "no-store")

        def end_headers(self):
            # Required for SharedArrayBuffer (used by MuJoCo WASM threading)
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    return CrossOriginIsolatedHandler


def _detect_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


class MjswanApp:
    """A built mjswan application ready to be launched.

    This class encapsulates the built application and provides methods
    for launching it in a web browser.
    """

    def __init__(self, app_dir: Path) -> None:
        self._app_dir = app_dir

    @classmethod
    def from_document(cls, source: str | Path) -> "MjswanApp":
        """The app for a built directory, or for a ``.swn`` document.

        A directory is already an app and is served where it sits, if it holds a
        ``manifest.json``. A document carries no engine, so it is unpacked into a
        temporary directory (removed at exit) with the packaged engine laid over it, built
        first if no matching SPA is cached. One that asks for a plugins module, or holds a
        file no build writes, is refused first.
        """
        import atexit
        import shutil
        import tempfile

        from .build.frontend import TEMPLATE_DIR, ClientBuilder, install_spa
        from .document import (
            MANIFEST_NAME,
            DocumentError,
            author_code,
            is_document,
            read_manifest,
            unpack_document,
        )

        source = Path(source).expanduser()
        if not source.exists():
            raise FileNotFoundError(f"No such app or document: {source}")
        if not is_document(source):
            if not (source / MANIFEST_NAME).is_file():
                raise DocumentError(
                    f"{source} holds no {MANIFEST_NAME}, so it is not a built mjswan app "
                    "and is not served. Give the directory builder.build() wrote, or a "
                    ".swn."
                )
            return cls(source.resolve())

        # A custom-TS module ships with the engine; one a document named would run here.
        if author_code(read_manifest(source)):
            raise DocumentError(
                f"{source.name} asks for a plugins module, as a build with custom-TS MDP "
                "terms does. Their module ships with the engine rather than in the "
                "document, and a document runs no code of its own. Serve the built "
                "directory instead."
            )

        app_dir = Path(tempfile.mkdtemp(prefix="mjswan-swn-"))
        atexit.register(shutil.rmtree, app_dir, ignore_errors=True)
        unpack_document(source, app_dir)
        if (TEMPLATE_DIR / "package.json").exists():
            ClientBuilder(TEMPLATE_DIR).build(base_path="/")
        if not install_spa(app_dir):
            raise RuntimeError(
                f"No built engine at {TEMPLATE_DIR / 'dist'}, so {source.name} cannot be "
                "served. Build an app once with `builder.build()` to produce one."
            )
        return cls(app_dir)

    @property
    def app_dir(self) -> Path:
        """The built directory: the engine plus the expanded simulation document."""
        return self._app_dir

    def save_document(self, path: str | Path | None = None) -> Path:
        """Write the simulation as one ``.swn`` file and return its path.

        The document is the build's data (``manifest.json`` and every project directory)
        packaged as a ZIP of the same tree, carrying no engine. ``path`` defaults to the
        build directory's name with ``.swn``, beside it.
        """
        from .document import write_document

        return write_document(self._app_dir, path)

    def publish(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        token: str | None = None,
        api_base: str | None = None,
    ) -> "PublishResult":
        """Publish this built app's data files to mjswan Cloud.

        Extracts only data files (manifest.json, scene/policy/motion/splat assets,
        traced MDP graphs) from the built ``dist/`` and uploads them via the
        presigned-upload protocol. Refuses builds that use custom-TS MDP terms
        (``uses_custom_js: true``), which mjswan Cloud cannot render.

        Args:
            title: Simulation title. Defaults to the first project's name.
            description: Optional description.
            tags: Optional list of tags.
            token: Supabase access token (GitHub OAuth). Falls back to
                ``$MJSWAN_TOKEN``.
            api_base: Cloud API base URL. Defaults to ``https://api.mjswan.com``.

        Returns:
            The publish result, including the new simulation id.

        Raises:
            mjswan.cloud.publish.PublishError: on validation failure or server rejection.
        """
        from .cloud.publish import DEFAULT_API_BASE, publish_dist

        return publish_dist(
            self._app_dir,
            title=title,
            description=description,
            tags=tags,
            token=token,
            api_base=api_base or DEFAULT_API_BASE,
            on_progress=print,
        )

    def launch(
        self,
        *,
        host: str = "localhost",
        port: int = 8080,
        open_browser: bool = True,
        height: int = 600,
    ) -> None:
        """Launch the application in a local web server.

        Automatically detects Google Colab and displays the viewer as an
        inline iframe. Outside Colab, starts a blocking server and optionally
        opens a browser tab. Bound to a loopback host, the server answers only
        requests addressed to ``localhost``, ``127.0.0.1`` or ``[::1]``.

        Args:
            host: Host to bind the server to (ignored in Colab).
            port: Port to run the server on.
            open_browser: Whether to automatically open a browser (ignored in Colab).
            height: Height of the Colab iframe in pixels (ignored outside Colab).
        """
        if not self._app_dir.exists():
            raise RuntimeError(f"Application directory {self._app_dir} does not exist.")

        import socket
        import socketserver

        def _find_available_port(
            bind_host: str, start_port: int, max_tries: int = 1000
        ) -> int:
            port_try = start_port
            tries = 0
            while tries < max_tries and port_try <= 65535:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    try:
                        s.bind((bind_host, port_try))
                        return port_try
                    except OSError:
                        port_try += 1
                        tries += 1
            raise RuntimeError(f"No available port found starting at {start_port}")

        class _ReusableTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        if _detect_colab():
            bind_host = ""
            chosen_port = _find_available_port(bind_host, port)
            if chosen_port != port:
                print(f"Port {port} unavailable — using port {chosen_port} instead.")
            port = chosen_port

            import threading

            handler = _serving_handler(self._app_dir, _hosts_answered(bind_host))

            def _serve():
                with _ReusableTCPServer((bind_host, port), handler) as httpd:
                    httpd.serve_forever()

            thread = threading.Thread(target=_serve, daemon=True)
            thread.start()
            print(f"Server running on port {port}")

            from google.colab import output  # type: ignore[import]

            output.serve_kernel_port_as_iframe(port, height=str(height))
            return

        import webbrowser

        chosen_port = _find_available_port(host, port)
        if chosen_port != port:
            print(f"Port {port} unavailable — using port {chosen_port} instead.")
        port = chosen_port

        print(f"Starting server at http://{host}:{port}")
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")

        handler = _serving_handler(self._app_dir, _hosts_answered(host))
        try:
            with _ReusableTCPServer((host, port), handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


__all__ = ["MjswanApp"]
