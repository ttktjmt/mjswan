"""Automatic Node.js environment setup and client build management.

This module handles:
- Creating isolated Node.js environments using nodeenv
- Installing dependencies
- Building TypeScript/JavaScript clients
- Cross-platform compatibility (Windows/macOS/Linux)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = [
    "ClientBuilder",
    "TEMPLATE_DIR",
    "ensure_node_env",
    "build_client",
    "install_spa",
]

#: The packaged frontend: its sources, and `dist/` once it has been built once.
TEMPLATE_DIR = Path(__file__).parent / "template"

#: What vite leaves in `dist/` for the dev loop alone: the E2E fixture, the cache key.
_SPA_EXCLUDES = frozenset({"fixtures", ".mjswan-build-meta.json"})


def install_spa(dest: Path, template_dir: Path | None = None) -> bool:
    """Copy the built SPA, the engine, into ``dest``; False when none is built yet.

    An app is the engine plus an expanded document (ADR 0006 §8); this lays down the
    engine half, for both the builder and ``MjswanApp.from_document``.
    """
    root = template_dir or TEMPLATE_DIR
    built = root / "dist"
    if not built.is_dir():
        return False
    for item in built.iterdir():
        if item.name in _SPA_EXCLUDES:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    # Ship the license alongside the app.
    license_file = root / "LICENSE"
    if license_file.exists():
        shutil.copy2(license_file, dest / license_file.name)
    return True


class ClientBuilder:
    """Manages isolated Node.js environment and client builds."""

    NODE_VERSION = "24.19.0"

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

    def install_dependencies(self, clean: bool = False) -> None:
        npm_bin = self._get_npm_bin()
        package_lock = self.project_dir / "package-lock.json"
        node_modules = self.project_dir / "node_modules"

        if clean:
            # Force a fresh install by removing the lock file and node_modules.
            # Useful when switching platforms or resolving corrupted installs.
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

    def _has_script(self, script_name: str) -> bool:
        package_json = self.project_dir / "package.json"
        try:
            with open(package_json) as f:
                package_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        return script_name in package_data.get("scripts", {})

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

    def _plugin_alias_args(self) -> list[str]:
        """esbuild `--alias:` args for bundling author custom-MDP terms.

        - Each `mjswan/<sub>` export → its engine source, so author term files
          (which import base classes from `mjswan/event`, `mjswan/observation`, …)
          resolve deterministically when bundled standalone.
        - `three` → a shim that reuses the engine bundle's single three instance
          at runtime, not a bundled duplicate (so instanceof / shared scene /
          raycasting work across the separately-loaded plugin ESM). See
          src/engine/plugin-three-shim.cjs and src/engine/index.ts.
        """
        package_json = self.project_dir / "package.json"
        with open(package_json) as f:
            exports = json.load(f).get("exports", {})
        args: list[str] = []
        for subpath, target in exports.items():
            if subpath in (".", "./manifest") or not isinstance(target, str):
                continue
            name = "mjswan/" + subpath[len("./") :]
            abs_target = (self.project_dir / target).resolve()
            args.append(f"--alias:{name}={abs_target}")
        three_shim = self.project_dir / "src" / "engine" / "plugin-three-shim.cjs"
        args.append(f"--alias:three={three_shim.resolve()}")
        return args

    @staticmethod
    def _collect_custom_terms() -> dict[str, dict[str, Path]]:
        """Map each MDP kind to {ts_name: source_path} for registered ts_src terms."""
        from mjswan.command import _custom_registry as cmd_reg
        from mjswan.envs.mdp.events import _custom_registry as evt_reg
        from mjswan.envs.mdp.observations import _custom_registry as obs_reg
        from mjswan.envs.mdp.terminations import _custom_registry as term_reg

        kinds = {
            "observations": obs_reg,
            "terminations": term_reg,
            "events": evt_reg,
            "commands": cmd_reg,
        }
        result: dict[str, dict[str, Path]] = {}
        for kind, registry in kinds.items():
            entries: dict[str, Path] = {}
            for sentinel in registry.values():
                ts_src = getattr(sentinel, "ts_src", None)
                ts_name = getattr(sentinel, "ts_name", None)
                if ts_src and ts_name:
                    src = Path(ts_src).expanduser().resolve()
                    if not src.exists():
                        raise FileNotFoundError(
                            f"Custom {kind} ts_src not found: {src}"
                        )
                    entries[ts_name] = src
            if entries:
                result[kind] = entries
        return result

    def build_plugins_module(self, dest: Path) -> bool:
        """Bundle author-supplied custom-MDP terms into a standalone ESM at ``dest``.

        Uses esbuild to inline the terms plus their engine base classes into one
        self-contained module (no bare imports), exporting term constructors
        grouped by kind (``events``/``observations``/``terminations``/``commands``)
        — the ``EnginePlugins`` shape the app hands to ``createEngine`` at load.
        The engine bundle is never rebuilt. Returns False when there are no
        custom terms. Needs Node (esbuild), unlike declarative builds.
        """
        terms = self._collect_custom_terms()
        if not terms:
            return False

        # Generate an entry that imports each term and re-exports it grouped by kind.
        by_src: dict[Path, list[str]] = {}
        for names in terms.values():
            for ts_name, src in names.items():
                by_src.setdefault(src, []).append(ts_name)
        lines = ["// Auto-generated plugin entry — do not edit."]
        for src, names in by_src.items():
            lines.append(
                f"import {{ {', '.join(sorted(set(names)))} }} from {json.dumps(str(src))};"
            )
        for kind, names in terms.items():
            pairs = ", ".join(sorted(names.keys()))
            lines.append(f"export const {kind} = {{ {pairs} }};")
        entry = self.project_dir / "src" / ".plugins-entry.ts"
        entry.write_text("\n".join(lines) + "\n")

        esbuild = self.project_dir / "node_modules" / ".bin" / "esbuild"
        if not esbuild.exists():
            # Custom-JS builds need Node; install if a cached SPA skipped it.
            self.create_env()
            self.install_dependencies()

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                [
                    str(esbuild),
                    str(entry),
                    "--bundle",
                    "--format=esm",
                    "--platform=browser",
                    f"--outfile={dest}",
                    *self._plugin_alias_args(),
                ],
                cwd=self.project_dir,
            )
        finally:
            entry.unlink(missing_ok=True)
        return True

    def _build_meta(
        self, base_path: str, gtm_id: str | None, mt: bool, debug: bool
    ) -> dict[str, object]:
        """Cache key for a built SPA: version + build opts + a source fingerprint.

        The SPA is project-independent (custom terms load at runtime via
        plugins.js, ADR 0004 §10), so any build with a matching key is reusable.
        The fingerprint rebuilds the SPA whenever the app codebase changes.
        """
        from mjswan import __version__

        return {
            "version": __version__,
            "base_path": base_path,
            "gtm_id": gtm_id,
            "mt": mt,
            "debug": debug,
            "source": self._source_fingerprint(),
        }

    def _source_fingerprint(self) -> str:
        """SHA-256 over the frontend build inputs (app source + config + declared deps).

        Invalidates the cache whenever the app codebase changes. Excludes derived
        outputs (node_modules / dist / .nodeenv) and package-lock.json (npm may
        rewrite it); package.json's ``version`` is normalized out since it is
        already a separate part of the key.
        """
        inputs: list[Path] = []
        for top in (
            "package.json",
            "vite.config.ts",
            "vite.lib.config.ts",
            "vite.manifest.config.ts",
            "vite.shared.ts",
            "vite.wasm.ts",
            "tsconfig.json",
            "index.html",
        ):
            p = self.project_dir / top
            if p.exists():
                inputs.append(p)
        for root in ("src", "public"):
            base = self.project_dir / root
            if base.exists():
                inputs.extend(p for p in base.rglob("*") if p.is_file())

        digest = hashlib.sha256()
        for path in sorted(
            inputs, key=lambda p: p.relative_to(self.project_dir).as_posix()
        ):
            rel = path.relative_to(self.project_dir).as_posix()
            digest.update(rel.encode())
            if rel == "package.json":
                data = json.loads(path.read_text())
                data.pop("version", None)
                digest.update(json.dumps(data, sort_keys=True).encode())
            else:
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _cached_spa_matches(self, meta: dict[str, object]) -> bool:
        dist = self.project_dir / "dist"
        marker = dist / ".mjswan-build-meta.json"
        if not (dist / "index.html").exists() or not marker.exists():
            return False
        try:
            return json.loads(marker.read_text()) == meta
        except (OSError, json.JSONDecodeError):
            return False

    def build(
        self,
        *,
        clean: bool = False,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
        build_frontend: bool | None = None,
    ) -> None:
        """Build the standalone SPA into ``dist/``.

        ``build_frontend``: True forces a build; False requires a matching cached
        artifact (raises otherwise); None (default) reuses the cache when it
        matches and builds only when it doesn't. The SPA is project-independent,
        so the cache is keyed on the mjswan version + base_path/gtm_id/mt/debug.
        """
        meta = self._build_meta(base_path, gtm_id, mt, debug)
        if not clean and build_frontend is not True and self._cached_spa_matches(meta):
            print("✓ Reusing cached frontend build (dist/)")
            return
        if build_frontend is False:
            raise RuntimeError(
                "build_frontend=False but no matching prebuilt dist/ was found."
            )
        try:
            self.create_env(clean=clean)
            self.sync_version_from_python()
            self.install_dependencies(clean=clean)
            env: dict[str, str] = {"MJSWAN_BASE_PATH": base_path}
            if gtm_id:
                env["MJSWAN_GTM_ID"] = gtm_id
            if mt:
                env["MJSWAN_MT"] = "1"
            if debug:
                env["MJSWAN_DEBUG"] = "1"
            # The standalone app needs only the SPA build; the library build (`mjswan.js`, for
            # mjswan Cloud) comes from the full `build` script during npm publish.
            script = "build:spa" if self._has_script("build:spa") else "build"
            self.run_build_script(script, env=env)
            (self.project_dir / "dist" / ".mjswan-build-meta.json").write_text(
                json.dumps(meta)
            )
            print("✓ Build completed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed with exit code {e.returncode}") from e
        except Exception as e:
            raise RuntimeError(f"Build failed: {e}") from e

    def cleanup(self) -> None:
        if self.nodeenv_dir.exists():
            print(f"Cleaning up nodeenv: {self.nodeenv_dir}")
            shutil.rmtree(self.nodeenv_dir)


def ensure_node_env(project_dir: Path, clean: bool = False) -> Path:
    builder = ClientBuilder(project_dir)
    builder.create_env(clean=clean)
    return builder.nodeenv_dir


def build_client(
    project_dir: Path,
    clean: bool = False,
    script: str = "build",
    base_path: str = "/",
    mt: bool = False,
) -> None:
    builder = ClientBuilder(project_dir)
    builder.build(clean=clean, base_path=base_path, mt=mt)
