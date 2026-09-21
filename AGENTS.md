# AGENTS.md

## Project

mjswan packages browser-based MuJoCo simulations with real-time policy control into an interactive static web app. It has two sides:

- **Python** ([src/mjswan/](src/mjswan/)) — `Builder` / `Project` / `Scene` API that bundles models, policies, and UI config into a static site.
- **Frontend template** ([src/mjswan/template/](src/mjswan/template/)) — TypeScript + three.js + mujoco-wasm client that the Python build step bundles.

See [CONTEXT.md](CONTEXT.md) for a full codebase map, object model, module descriptions, and tooling reference.

## Philosophy

- Write clean, readable, maintainable code.
- Don't reinvent what already exists upstream. Prefer mjlab ([GitHub](https://github.com/mujocolab/mjlab), [local](.venv/lib/python3.12/site-packages/mjlab)) or other dependencies over new boilerplate in mjswan.
- Keep comments minimal and concise. Only comment the non-obvious *why*, not the *what*. When touching existing comments, strip anything redundant or self-evident.

## Layout

- [src/mjswan/](src/mjswan/) — package source: the object model at the root (`Builder`, the `*Handle` / `*Config` pairs, `cli.py`) over one package per layer (`build/`, `compile/`, `mjlab/`, `source/`, `document/`, `cloud/`, `license/`, and the mjlab-mirror `managers/` / `envs/mdp/`). [docs/adr/0008](docs/adr/0008-package-layout.md) gives the rules: imports point downward, mjswan's own names are singular, a file never repeats its package's name.
- [src/mjswan/template/](src/mjswan/template/) — frontend source (Vite + React + three.js + mujoco-wasm).
- [examples/](examples/) — `demo`, `tutorial`, `mjlab`, `colab` runnable examples.
  Python only: assets are fetched at run time, and a pre-commit hook rejects a
  binary committed under `examples/`.
- [tests/](tests/) — pytest suite. `slow`-marked tests are opt-out (see below).
- [docs/](docs/) — zensical (MkDocs-based) site published to Read the Docs. Build with `make docs-build`; serve locally with `make docs-serve`.

## Python workflow

Use `uv` instead of bare `python`/`pip`. Prefer the [Makefile](Makefile) targets.

Tests are configured with `slow` opt-out: `make test` runs everything, but pre-commit runs `pytest -m "not slow"` for speed.
