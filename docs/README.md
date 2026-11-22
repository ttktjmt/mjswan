# mjswan Documentation

Source for the mjswan documentation site, hosted on
[Read the Docs](https://mjswan.readthedocs.io/).

Built with [zensical](https://zensical.org) (the MkDocs Material successor). Configuration
lives in [`zensical.toml`](zensical.toml); pages live under [`docs/`](docs/).

## Commands

From the repository root:

```bash
make docs-serve    # live-reloading server on http://localhost:8000
make docs-build    # one-off build into docs/site/
```

Or directly, e.g. to pick a different port:

```bash
uv run --with-requirements requirements.txt --directory docs zensical serve -a localhost:8123
```

The build validates internal links and reports broken ones as warnings; treat them as
errors.

## Layout

```
docs/
├── zensical.toml          site config: nav, theme, markdown extensions
├── requirements.txt       docs build dependencies (zensical)
├── .readthedocs.yaml      Read the Docs build definition
└── docs/
    ├── index.md
    ├── getting-started/   installation, quickstart, core concepts, examples, CLI
    ├── guides/            mjlab, MDP terms, build internals, deployment, cloud, embedding
    ├── skills/            agent skills: an overview (index.md) and one page per skill
    ├── api/               Python and TypeScript API reference
    └── resources.md
```

A new skill takes a page in `docs/skills/`, a row in the table on `skills/index.md`, and an
entry under `Skills` in the nav.
