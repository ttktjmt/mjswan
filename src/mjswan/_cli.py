"""CLI entry points for mjswan scripts."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run_module(module_path: str) -> None:
    """Run a module with ``python -m``."""
    project_root = Path(__file__).parent.parent.parent

    result = subprocess.run(
        [sys.executable, "-m", module_path],
        check=False,
        cwd=project_root,
    )
    sys.exit(result.returncode)


def main() -> None:
    """Run examples/demo/main.py"""
    _run_module("examples.demo.main")


def simple() -> None:
    """Run examples/demo/simple.py"""
    _run_module("examples.demo.simple")


def mjlab() -> None:
    """Run examples/mjlab/defaults/main.py"""
    _run_module("examples.mjlab.defaults.main")


def serve() -> None:
    """Launch a pre-built mjswan app from a dist directory.

    Usage: serve <dist-dir>
    """
    if len(sys.argv) < 2:
        print("Usage: serve <dist-dir>", file=sys.stderr)
        sys.exit(1)

    from mjswan.app import mjswanApp

    app = mjswanApp(Path(sys.argv[1]).resolve())
    app.launch()


# ---------------------------------------------------------------------------
# mjswan unified CLI
# ---------------------------------------------------------------------------


def _resolve_app_dir(path: str | None) -> Path:
    if path is not None:
        return Path(path).resolve()
    dist = Path.cwd() / "dist"
    if not dist.exists():
        print(f"Error: dist directory not found at {dist}", file=sys.stderr)
        sys.exit(1)
    return dist


def _cmd_serve(args: argparse.Namespace) -> None:
    from mjswan.app import mjswanApp

    mjswanApp(_resolve_app_dir(args.path)).launch()


def _cmd_cf_pages(args: argparse.Namespace) -> None:
    app_dir = _resolve_app_dir(args.path)
    name: str = args.name or app_dir.parent.name
    wrangler = shutil.which("wrangler")
    if wrangler:
        cmd = [wrangler, "pages", "deploy", str(app_dir), "--project-name", name]
    else:
        npx = shutil.which("npx")
        if not npx:
            print(
                "Error: wrangler or npx not found. Install with: npm install -g wrangler",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd = [npx, "wrangler", "pages", "deploy", str(app_dir), "--project-name", name]
    result = subprocess.run(cmd, check=False)
    sys.exit(result.returncode)


def _cmd_gh_pages(args: argparse.Namespace) -> None:
    app_dir = _resolve_app_dir(args.path)
    ghp = shutil.which("ghp-import")
    if not ghp:
        print(
            "Error: ghp-import not found. Install with: pip install ghp-import",
            file=sys.stderr,
        )
        sys.exit(1)
    result = subprocess.run([ghp, "-p", "-f", str(app_dir)], check=False)
    sys.exit(result.returncode)


def mjswan_cli() -> None:
    """Unified mjswan command-line interface."""
    parser = argparse.ArgumentParser(prog="mjswan", description="mjswan CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Serve a pre-built app directory locally")
    p_serve.add_argument(
        "path", nargs="?", help="Path to app directory (default: ./dist)"
    )

    p_cf = sub.add_parser("cf-pages", help="Deploy app directory to Cloudflare Pages")
    p_cf.add_argument("path", nargs="?", help="Path to app directory (default: ./dist)")
    p_cf.add_argument(
        "--name",
        "-n",
        help="Cloudflare Pages project name (default: parent directory name)",
    )

    p_gh = sub.add_parser("gh-pages", help="Deploy app directory to GitHub Pages")
    p_gh.add_argument("path", nargs="?", help="Path to app directory (default: ./dist)")

    args = parser.parse_args()
    {"serve": _cmd_serve, "cf-pages": _cmd_cf_pages, "gh-pages": _cmd_gh_pages}[
        args.command
    ](args)
