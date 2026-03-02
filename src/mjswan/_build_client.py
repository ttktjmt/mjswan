"""Automatic Node.js environment setup and client build management.

This module handles:
- Creating isolated Node.js environments using nodeenv
- Installing dependencies
- Building TypeScript/JavaScript clients
- Cross-platform compatibility (Windows/macOS/Linux)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["ClientBuilder", "ensure_node_env", "build_client"]


class ClientBuilder:
    """Manages isolated Node.js environment and client builds."""

    NODE_VERSION = "25.5.0"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.nodeenv_dir = self.project_dir / ".nodeenv"

    def _get_node_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "node.exe"
        else:
            return self.nodeenv_dir / "bin" / "node"

    def _get_npm_bin(self) -> Path:
        if sys.platform == "win32":
            return self.nodeenv_dir / "Scripts" / "npm.cmd"
        else:
            return self.nodeenv_dir / "bin" / "npm"

    def _ensure_nodeenv_installed(self) -> None:
        try:
            import nodeenv  # noqa: F401
        except ImportError:
            print("Installing nodeenv...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "nodeenv>=1.9.0"],
                stdout=subprocess.PIPE if not os.getenv("VERBOSE_BUILD") else None,
            )

    def create_env(self, clean: bool = False) -> None:
        if clean and self.nodeenv_dir.exists():
            print(f"Removing existing nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)

        if self.nodeenv_dir.exists():
            node_bin = self._get_node_bin()
            if node_bin.exists():
                try:
                    result = subprocess.run(
                        [str(node_bin), "--version"],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        installed_version = result.stdout.strip().lstrip("v")
                        if installed_version == self.NODE_VERSION:
                            print(f"✓ Node.js {self.NODE_VERSION} already available")
                            return
                except Exception as e:
                    print(f"Warning: Could not verify Node.js version: {e}")

        print(f"Creating Node.js {self.NODE_VERSION} environment in {self.nodeenv_dir}")
        self._ensure_nodeenv_installed()

        # Use nodeenv CLI for robustness across versions
        try:
            cmd = [
                sys.executable,
                "-m",
                "nodeenv",
                str(self.nodeenv_dir),
                "--node",
                self.NODE_VERSION,
            ]
            if os.getenv("VERBOSE_BUILD"):
                cmd.append("--verbose")
            subprocess.check_call(cmd)
        except Exception as e:
            raise RuntimeError(f"Failed to create Node.js environment: {e}")

    def install_dependencies(self) -> None:
        npm_bin = self._get_npm_bin()
        package_lock = self.project_dir / "package-lock.json"
        node_modules = self.project_dir / "node_modules"

        # In CI environments or cross-platform builds, npm ci can fail with optional dependencies
        # Update lock file and node_modules to ensure clean install
        if package_lock.exists():
            package_lock.unlink()
        if node_modules.exists():
            shutil.rmtree(node_modules)

        print("Installing npm dependencies (npm install)...")
        subprocess.check_call([str(npm_bin), "install"], cwd=self.project_dir)

    def sync_version_from_python(self) -> None:
        """Sync package.json version with Python package __version__."""
        from mjswan import __version__

        package_json = self.project_dir / "package.json"
        with open(package_json, "r") as f:
            package_data = json.load(f)

        current_version = package_data.get("version", "0.0.0")
        if current_version != __version__:
            print(f"Updating package.json version: {current_version} → {__version__}")
            package_data["version"] = __version__
            # Remove private field if it exists
            package_data.pop("private", None)
            with open(package_json, "w") as f:
                json.dump(package_data, f, indent=2)
                f.write("\n")

    def run_build_script(
        self, script_name: str = "build", env: dict[str, str] | None = None
    ) -> None:
        npm_bin = self._get_npm_bin()
        package_json = self.project_dir / "package.json"
        with open(package_json) as f:
            package_data = json.load(f)
        if script_name not in package_data.get("scripts", {}):
            raise ValueError(
                f"Script '{script_name}' not found in {package_json}. "
                f"Available scripts: {list(package_data.get('scripts', {}).keys())}"
            )
        print(f"Running npm script: {script_name}")
        build_env = os.environ.copy()
        if env:
            build_env.update(env)
        subprocess.check_call(
            [str(npm_bin), "run", script_name],
            cwd=self.project_dir,
            env=build_env,
        )

    def build(
        self, clean: bool = False, base_path: str = "/", gtm_id: str | None = None
    ) -> None:
        try:
            self.create_env(clean=clean)
            self.sync_version_from_python()
            self.install_dependencies()
            env: dict[str, str] = {"MJSWAN_BASE_PATH": base_path}
            if gtm_id:
                env["MJSWAN_GTM_ID"] = gtm_id
            self.run_build_script("build", env=env)
            print("✓ Build completed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed with exit code {e.returncode}") from e
        except Exception as e:
            raise RuntimeError(f"Build failed: {e}") from e

    def cleanup(self) -> None:
        if self.nodeenv_dir.exists():
            print(f"Cleaning up nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)


def ensure_node_env(
    project_dir: Path, node_version: str = "20.4.0", clean: bool = False
) -> Path:
    builder = ClientBuilder(project_dir)
    builder.create_env(clean=clean)
    return builder.nodeenv_dir


def build_client(
    project_dir: Path, clean: bool = False, script: str = "build", base_path: str = "/"
) -> None:
    builder = ClientBuilder(project_dir)
    builder.build(clean=clean, base_path=base_path)
