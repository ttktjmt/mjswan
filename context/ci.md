# Tests, CI, release and deployment

## Tests

`tests/` is the pytest suite, plus the `dump_*_fixture.py` generators for the TypeScript parity fixtures. `@pytest.mark.slow` marks a test that triggers a full frontend (npm + Vite) build; it is excluded from pre-commit (`pytest -m "not slow"`), and unmarked tests are fast and always run. `make test` runs everything. The template's own suite is vitest (`frontend.yml`); `e2e/` holds Playwright specs against a real canvas. The parity layers, and what each proves, are in [tracing.md](tracing.md).

## Workflows

- **`pytest.yml`** runs `pytest -m "not slow"` across Python 3.11 / 3.12 / 3.13 with the `dev` extra only. Its `core` job installs mjswan with no extra and runs `tests/test_core_install.py` (import, a model-only build, and the extra each source names), which skips anywhere an extra is installed. Because the default job has no mjlab or torch, every parity test is skipped there.
- **`parity.yml`** is separate and heavier: it installs the `examples` extras (mjlab, torch, onnxruntime), caches warp's CPU kernels, and runs the ONNX numeric-parity sweep over the reference mjlab tasks. It runs on `workflow_dispatch`, on a pull request that touches `compile/`, `build/mdp/`, `mjlab/env.py`, the parity tests or `uv.lock` (what can silently change a traced term's numbers), and weekly. Weekly because that cadence targets *upstream* drift: mjlab moves fast, and a changed term definition would otherwise surface as a mysterious runtime difference rather than a failing check.
- **`frontend.yml`** lints the template and runs its vitest suite, sharing one dependency install.
- **`release.yml`** writes one version into `_version.py`, `package.json` and `package-lock.json`, commits it to main for a stable release, tags it, and calls `publish-pypi.yml` and `publish-npm.yml`, which set the version from the input again so a prerelease needs no commit. The Python and npm versions are therefore one string by construction ([document.md](document.md) relies on it).
- Also: `ruff.yml`, `deploy.yml` (below), `sync-contributors.yml`.

Every workflow installs Python dependencies from `uv.lock` alone (`uv sync --locked`, `uv run --locked`), so a release published since the last lock never reaches CI, torch included (the locked Linux wheel is the CUDA build). The `core` job's pytest is the `test` dependency group, locked like an extra but not published as one. Every action is pinned to a commit SHA with its release tag beside it. How the lock admits new releases is in [dependencies.md](dependencies.md).

## Deployment

The demo app is built by `examples/demo/main.py` and deployed to GitHub Pages by `deploy.yml` on every push to `main` that touches relevant paths. It installs only the `demo` extra and needs no credentials: models, checkpoints and skins come from the public Hub repository `ttktjmt/mjswan` (the comment beside `HF_REPO` in `main.py` lists how to mirror a task's W&B checkpoints there), and WBC-Mjlab's policy and clips from its deploy repository on GitHub at a pinned commit. The `MJSWAN_BASE_PATH` and `MJSWAN_NO_LAUNCH` env vars, which `main.py` reads, control the build. Cloudflare Pages builds a preview of the same demo for every pull request. The GentleHumanoid and MuscleMimic demos are deployed to Cloudflare Pages too; their sources left this repository with `examples/mjlab/` and `examples/demo/gentle_humanoid/`. `mjswan publish` is the alternative to hosting a `dist/` at all ([cloud.md](cloud.md)).
