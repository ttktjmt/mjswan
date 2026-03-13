"""CLI entry points for mjswan scripts."""

import subprocess
import sys
from pathlib import Path


def _run_script(script_path: str) -> None:
    """Run a script."""
    project_root = Path(__file__).parent.parent.parent
    script = project_root / script_path

    if not script.exists():
        print(f"Error: {script} not found", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        cwd=project_root,
    )
    sys.exit(result.returncode)


def main() -> None:
    """Run examples/demo/main.py"""
    _run_script("examples/demo/main.py")


def simple() -> None:
    """Run examples/demo/simple.py"""
    _run_script("examples/demo/simple.py")


def splat() -> None:
    """Run examples/demo/splat.py"""
    _run_script("examples/demo/splat.py")


def mjlab() -> None:
    """Run examples/mjlab/mjlab_integration.py"""
    _run_script("examples/mjlab/mjlab_integration.py")


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
