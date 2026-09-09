"""Tests for the Vite library build (`mjswan.js` createEngine entry).

L3 slow (triggers a frontend build): TestLibBuild
Run with: pytest -m slow -k LibBuild

The library build produces a single self-contained ESM (the headless engine,
no React/Mantine) consumed by mjswan Cloud from a CDN. These tests enforce the
load-bearing invariants:
  - `dist/mjswan.js` exists and exports `createEngine` (and default),
  - every dependency is bundled (no bare imports left to resolve from a CDN),
  - the MuJoCo/ONNX WASM is emitted as co-located files (NOT inlined as base64
    data URLs) and referenced relative to the bundle,
  - ONNX Runtime Web is pointed at the co-located copy, so a host serving `dist/`
    from its own origin fetches nothing third-party (issue #123).
See vite.lib.config.ts and docs/adr/0004-headless-engine-core.md.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

import mjswan
from mjswan._build_client import ClientBuilder

TEMPLATE_DIR = Path(mjswan.__file__).parent / "template"

# Real ESM `import ... from "spec"` / `export ... from "spec"` and `import("spec")`,
# anchored at a statement boundary so matches inside string literals (e.g. React's
# "import it from \"react-dom/client\"" warning) are not mistaken for imports.
_IMPORT_FROM = re.compile(
    r"""(?:^|[;}\n])\s*(?:import|export)\b[^;{}\n]*?\bfrom\s*["']([^"']+)["']"""
)
_DYNAMIC_IMPORT = re.compile(
    r"""(?:^|[;}\n=(,:?&|])\s*import\(\s*["']([^"']+)["']\s*\)"""
)


#: The name the library build gives ORT's runtime wasm. Part of the published file set:
#: a consumer mirroring `dist/` serves this path, and the bundle asks for it by name.
_ORT_WASM_FILE = "ort-wasm-simd-threaded.jsep.wasm"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_bare(spec: str) -> bool:
    return not (
        spec.startswith("./")
        or spec.startswith("../")
        or spec.startswith("data:")
        or spec.startswith("http://")
        or spec.startswith("https://")
    )


@pytest.fixture(scope="class")
def lib_dist() -> Path:
    """Build only the library bundle once for the whole test class."""
    builder = ClientBuilder(TEMPLATE_DIR)
    builder.create_env()
    builder.sync_version_from_python()
    builder.install_dependencies()
    builder.run_build_script("build:lib")
    return TEMPLATE_DIR / "dist"


@pytest.mark.slow
class TestLibBuild:
    def test_mjswan_js_emitted(self, lib_dist: Path):
        assert (lib_dist / "mjswan.js").is_file()

    def test_exports_create_engine(self, lib_dist: Path):
        code = (lib_dist / "mjswan.js").read_text()
        assert re.search(r"\bas createEngine\b", code) or re.search(
            r"export\s*\{[^}]*\bcreateEngine\b", code
        )

    def test_no_bare_imports(self, lib_dist: Path):
        """No `import 'three'`-style specifiers — all resolvable from the CDN."""
        offenders: list[str] = []
        for js in lib_dist.glob("*.js"):
            code = js.read_text()
            specs = _IMPORT_FROM.findall(code) + _DYNAMIC_IMPORT.findall(code)
            offenders.extend(f"{js.name} -> {spec}" for spec in specs if _is_bare(spec))
        assert not offenders, f"bare imports found: {offenders}"

    def test_wasm_co_located_not_inlined(self, lib_dist: Path):
        """Main-thread WASM is co-located, never inlined as a base64 data URL.

        `extractInlinedWasmPlugin` in vite.lib.config.ts pulls every
        `new URL('data:application/wasm…', import.meta.url)` back out into a
        flat dist/ file. It deliberately leaves ONE class inlined: Spark's
        Gaussian Splat sorter runs in a classic Blob worker whose base is
        `self.location.href`, where `import.meta` is a syntax error — that
        dormant single-threaded worker keeps its base64 WASM. So forbid only
        `import.meta.url`-based (MuJoCo/ONNX main-thread) inlining.
        """
        wasm_files = list(lib_dist.glob("*.wasm"))
        assert wasm_files, "no co-located .wasm files emitted in dist/"
        inlined = re.compile(
            r"""new URL\(\s*(["'`])data:application/wasm;base64,"""
            r"""[A-Za-z0-9+/=]+\1\s*,\s*([^)]*)\)"""
        )
        for js in lib_dist.glob("*.js"):
            for m in inlined.finditer(js.read_text()):
                base = m.group(2)
                assert "import.meta.url" not in base, (
                    f"{js.name} still inlines main-thread WASM as a data URL "
                    f"(base {base!r}); it must be extracted to a co-located file"
                )

    def test_wasm_referenced_relative_to_bundle(self, lib_dist: Path):
        """WASM is fetched via `new URL('./x.wasm', import.meta.url)`."""
        found = False
        for js in lib_dist.glob("*.js"):
            if re.search(
                r"""new URL\(\s*["'`]\./[^"'`]*\.wasm["'`]\s*,\s*import\.meta\.url""",
                js.read_text(),
            ):
                found = True
                break
        assert found, "no co-located `new URL('./*.wasm', import.meta.url)` reference"

    def test_ort_wasm_co_located(self, lib_dist: Path):
        """ORT's runtime wasm ships beside the bundle, byte-for-byte as installed.

        `src/core/onnx/ortEnv.ts` points `ort.env.wasm.wasmPaths` at this file, which is
        what lets a host serving `dist/` run a policy with no third-party request.
        Comparing bytes also pins the ORT version to the one the build resolved from the
        lockfile, rather than a range something else resolves at request time.
        """
        emitted = lib_dist / _ORT_WASM_FILE
        assert emitted.is_file(), f"{_ORT_WASM_FILE} not emitted beside mjswan.js"
        installed = (
            TEMPLATE_DIR / "node_modules" / "onnxruntime-web" / "dist" / _ORT_WASM_FILE
        )
        assert _sha256(emitted) == _sha256(installed), (
            "the co-located ORT wasm is not the installed one — the extract plugin "
            "matched the wrong asset"
        )

    def test_ort_wasm_paths_names_the_file_never_a_prefix(self, lib_dist: Path):
        """`wasmPaths` names the wasm; a URL prefix would be a third-party script fetch.

        The distinction is the whole point: given a prefix, ORT dynamic-imports
        `ort-wasm-simd-threaded.jsep.mjs` from it — executable code, from whatever origin
        the prefix names, on every policy-driven scene — while naming only the wasm keeps
        ORT on the loader already inlined in the bundle. This build set a jsDelivr prefix
        until issue #123.
        """
        code = (lib_dist / "mjswan.js").read_text()
        named = re.compile(
            r"""wasmPaths\s*=\s*\{\s*wasm\s*:\s*new URL\(\s*["'`]\./"""
            + re.escape(_ORT_WASM_FILE)
        )
        assert named.search(code), "wasmPaths does not name the co-located ORT wasm"
        prefix = re.search(r"""wasmPaths\s*=\s*["'`](https?://[^"'`]*)""", code)
        assert prefix is None, (
            f"wasmPaths is set to a URL prefix ({prefix.group(1)}), which makes ORT "
            "fetch its .mjs loader from there"
        )

    def test_no_bundled_react_or_mantine(self, lib_dist: Path):
        # The engine entry drops the React/Mantine chrome; the CDN bundle must
        # not carry it (ADR 0004 §11). A cheap proxy: React's dev-warning prefix.
        code = (lib_dist / "mjswan.js").read_text()
        assert "Warning: React" not in code and "@mantine" not in code

    def test_our_console_messages_are_stripped(self, lib_dist: Path):
        """`Builder(debug=False)` promises a console-quiet bundle; assert the artifact.

        Greps the bundle rather than the config, because the config lied once already:
        `esbuild: {drop: ['console']}` stopped working at Vite 8 and every `console.*`
        shipped anyway. The tags come from the sources, so a new one is covered the day
        it is written.

        mjswan's own messages only — a dependency taking a reference
        (`console.log.bind(console)`) is not a call and cannot be dropped.
        """
        tag = re.compile(r"console\.\w+\(\s*[`'\"]?\s*(\[[A-Za-z][A-Za-z0-9_]*\])")
        tags = {
            match
            for source in (TEMPLATE_DIR / "src").rglob("*.ts")
            if "__tests__" not in source.parts
            for match in tag.findall(source.read_text())
        }
        assert len(tags) > 5, f"the scan found only {tags} — it stopped matching"

        code = (lib_dist / "mjswan.js").read_text()
        assert [t for t in sorted(tags) if t in code] == []

    def test_no_unfolded_process_env_node_env(self, lib_dist: Path):
        """The bundle must be browser-self-contained: no unfolded `process`.

        mjswan Cloud loads this bundle straight from a CDN with `@vite-ignore`,
        so the consuming bundler never substitutes globals away. An eager,
        unguarded `process.env.NODE_ENV` (shipped by React/Mantine dev checks)
        therefore throws `ReferenceError: process is not defined` at mount time
        in the browser. `vite.lib.config.ts` statically folds it to
        "production"; this asserts none survives. The engine — not the host —
        owns this invariant (the host must not need a `process` shim). See
        vite.lib.config.ts `define` and mjswan-cloud ADR 0001.

        Residual `process.*` references (e.g. setimmediate's `process.nextTick`)
        are allowed only because they sit behind runtime guards and never
        evaluate single-threaded in the browser.
        """
        offenders = [
            js.name
            for js in lib_dist.glob("*.js")
            if "process.env.NODE_ENV" in js.read_text()
        ]
        assert not offenders, (
            f"unfolded process.env.NODE_ENV in {offenders}; the Vite `define` "
            "in vite.lib.config.ts must fold it to a literal so the CDN-loaded "
            "engine needs no host-side `process` shim."
        )


@pytest.fixture(scope="class")
def manifest_dist() -> Path:
    """Build the `mjswan/manifest` CDN bundle once for the whole test class."""
    builder = ClientBuilder(TEMPLATE_DIR)
    builder.create_env()
    builder.install_dependencies()
    builder.run_build_script("build:manifest")
    return TEMPLATE_DIR / "dist"


@pytest.mark.slow
class TestManifestBuild:
    """`dist/manifest.js` is a standalone, CDN-loadable parser (ADR 0004 §9/§11).

    mjswan Cloud imports it from jsDelivr alongside the engine bundle, so it must
    be self-contained (no bare imports) and export `parseManifest`.
    """

    def test_manifest_js_emitted(self, manifest_dist: Path):
        assert (manifest_dist / "manifest.js").is_file()

    def test_exports_parse_manifest(self, manifest_dist: Path):
        assert "parseManifest" in (manifest_dist / "manifest.js").read_text()

    def test_no_bare_imports(self, manifest_dist: Path):
        code = (manifest_dist / "manifest.js").read_text()
        specs = _IMPORT_FROM.findall(code) + _DYNAMIC_IMPORT.findall(code)
        offenders = [s for s in specs if _is_bare(s)]
        assert not offenders, f"bare imports in manifest.js: {offenders}"
