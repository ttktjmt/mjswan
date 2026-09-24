---
icon: octicons/download-16
---

# Installation

mjswan can be installed as a Python package (the primary workflow) or as an npm package for JavaScript/TypeScript projects.

<div class="grid cards" markdown>

-   [:simple-python: &nbsp; __Python package__](#python-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via pip to quickly build and share interactive MuJoCo simulations

-   [:simple-javascript: &nbsp; __JavaScript package__](#javascript-installation){ style="text-decoration: none; color: inherit;" }

    ---

    Install via npm for custom web applications with TypeScript support

-   [:simple-github: &nbsp; __GitHub Source__](#github-source){ style="text-decoration: none; color: inherit;" }

    ---

    Clone the repository for development and contributing

>   :simple-docker: &nbsp; __Docker / Cluster__
>   ---
>   Not supported.

</div>

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.13 |
| Platform | macOS (Apple Silicon) or Linux (x86-64) |
| Browser | Any modern browser with WebAssembly and WebGL |
| Node.js | 24+ (npm installation only) |

## Python Installation

```bash
pip install mjswan
```

That is the whole pipeline: `mujoco`, `onnx`, `typer`, `rich`, and `nodeenv` for the
frontend build. It bundles any MuJoCo model and any ONNX policy you already have on disk.

**Where assets come from is an extra**, one per backend, and `import mjswan` needs none
of them:

```bash
pip install 'mjswan[wandb]'  # add_policy_wandb(only_latest=True) / add_motion_wandb
pip install 'mjswan[hf]'     # add_scene_hf / add_policy_hf / add_motion_hf / add_splat_hf
pip install 'mjswan[mjlab]'  # add_scene_mjlab, and tracing MDP terms (mjlab + torch)
```

They combine: `pip install 'mjswan[wandb,mjlab]'` is the W&B checkpoint workflow, which
`add_policy_wandb` runs by default and checks for when it is called.

!!! warning "Policies with MDP terms need the `mjlab` extra"
    mjswan compiles observation, termination, event and command terms to ONNX at build
    time, which runs `torch.onnx.export` against a live mjlab environment — so a policy
    carrying any of those needs `mjlab` and `torch` installed. Both are **build-time
    only**; neither ships to the browser, and a model-only scene needs neither. See
    [How the Build Works](../guides/how-it-works.md).

    ```bash
    pip install 'mjswan[mjlab]'
    ```

Three more extras: two for working *on* mjswan rather than with it, and one for running its
examples:

```bash
pip install 'mjswan[check]'     # ruff, ty, pyright, what `make check` runs
pip install 'mjswan[dev]'       # the above, plus pytest, pre-commit, and every source
pip install 'mjswan[examples]'  # what examples/ needs: every source, plus onnxruntime
```

`examples` is `wandb`, `hf` and `mjlab` together, plus `onnxruntime` for the numeric
parity checks. torch makes it a large download.

## JavaScript Installation

```bash
npm install mjswan
```

Or with yarn:

```bash
yarn add mjswan
```

The npm package provides the browser-side runtime — MuJoCo WASM, three.js rendering, and
ONNX Runtime Web behind a `createEngine` API. It is independent of the Python package and
has no Python dependency. Use it to embed a simulation in an app you own, or to author
custom MDP terms in TypeScript; see the [Engine API](../api/engine.md).

## GitHub Source

Clone the repository and install all dependencies with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ttktjmt/mjswan.git
cd mjswan
uv sync --all-extras
```

To run the bundled demo after cloning:

```bash
uv run mjswan demo          # lists the bundled demos
uv run mjswan demo simple   # runs one of them
```

Common Makefile targets while developing:

| Target | What it does |
|---|---|
| `make sync` | Install/refresh all dependencies with `uv` |
| `make check` | Format and lint in place (`make format`), then type check |
| `make test` | Full pytest suite (`test-all` runs `check` first) |
| `make docs-serve` | Live-reloading documentation server |

See [docs/README.md](https://github.com/ttktjmt/mjswan/blob/main/docs/README.md){:target="_blank"} for the documentation workflow.
