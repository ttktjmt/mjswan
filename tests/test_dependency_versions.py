"""Tests for dependency version consistency across Python and npm manifests.

Layer: L1 (manifest parsing only).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_TOML = PROJECT_ROOT / "pyproject.toml"
TEMPLATE_PACKAGE_JSON = PROJECT_ROOT / "src" / "mjswan" / "template" / "package.json"
MODEL_GALLERY = PROJECT_ROOT / "examples" / "demo" / "mujoco_models.py"


def _read_python_mujoco_version() -> str:
    pyproject = PYPROJECT_TOML.read_text()
    match = re.search(
        r'^\s*"mujoco==(?P<version>[^"]+)",\s*$',
        pyproject,
        re.MULTILINE,
    )
    if match is None:
        raise AssertionError(
            f"Could not find Python mujoco dependency in {PYPROJECT_TOML}"
        )
    return match.group("version")


def _read_npm_mujoco_version() -> str:
    """Extract the MuJoCo version from package.json dependencies.

    Version value may be plain semver ("3.7.0") or an npm alias spec
    ("npm:@scope/pkg@3.7.0").
    """
    package_json = json.loads(TEMPLATE_PACKAGE_JSON.read_text())
    deps = package_json.get("dependencies", {})

    version_spec: str | None = None
    for key, value in deps.items():
        if key == "mujoco" or key.endswith("/mujoco"):
            version_spec = value
            break

    if version_spec is None:
        raise AssertionError(
            f"Could not find a mujoco npm dependency in {TEMPLATE_PACKAGE_JSON}. "
            "Expected a key of 'mujoco', '@mujoco/mujoco', '@ttktjmt/mujoco', "
            "or any '<scope>/mujoco'."
        )
    # npm alias form: "npm:@scope/pkg@X.Y.Z" — extract trailing @version
    alias_match = re.search(r"@(\d+\.\d+\.\d+[^\"']*)$", version_spec)
    if alias_match:
        return alias_match.group(1)
    # plain semver, strip optional range prefix (^, ~, =)
    return version_spec.lstrip("^~=")


class TestMuJoCoDependencyVersions:
    def test_python_and_npm_mujoco_versions_match(self):
        python_version = _read_python_mujoco_version()
        npm_version = _read_npm_mujoco_version()

        assert python_version == npm_version, (
            "Python package 'mujoco' and npm package 'mujoco' must use the "
            "same version. Passing MjModel between Python and the frontend is only "
            "guaranteed to work when both sides use the same MuJoCo version, so "
            f"keeping these versions aligned is strongly recommended "
            f"(Python: {python_version}, npm: {npm_version})."
        )

    def test_the_model_gallery_is_fetched_at_the_pinned_release(self):
        """`mjswan demo mujoco` compiles upstream's XML with the installed MuJoCo."""
        match = re.search(
            r'^MUJOCO_TAG = "(?P<tag>[^"]+)"$', MODEL_GALLERY.read_text(), re.MULTILINE
        )

        assert match is not None, f"No MUJOCO_TAG in {MODEL_GALLERY}"
        assert match.group("tag") == _read_python_mujoco_version()
