---
icon: octicons/terminal-16
---

# Command-line interface

`pip install mjswan` exposes a single Typer-based command:

```bash
mjswan --help
```

| Subcommand | Purpose |
|---|---|
| [`mjswan view`](#mjswan-view) | Build and launch a viewer for a MuJoCo XML / MJCF file |
| [`mjswan serve`](#mjswan-serve) | Serve a pre-built `dist/` directory, or a `.swn` document |
| [`mjswan new`](#mjswan-new) | Scaffold a new project from a template |
| [`mjswan demo`](#mjswan-demo) | List the built-in demos, or run one |
| [`mjswan info`](#mjswan-info) | Show a tree of projects / scenes / MDPs / policies and asset sizes, for a `dist/` or a `.swn` |
| [`mjswan publish`](#mjswan-publish) | Upload a built `dist/` or a `.swn` to mjswan Cloud |
| [`mjswan login`](#mjswan-login-whoami-logout) | Sign in to mjswan Cloud via GitHub |
| [`mjswan whoami`](#mjswan-login-whoami-logout) | Show the signed-in account |
| [`mjswan logout`](#mjswan-login-whoami-logout) | Remove the stored session |

## `mjswan view`

```bash
mjswan view <model.xml> [--name SCENE] [--port 8080] [--host localhost] [--no-open]
```

Build a temporary app around the supplied MuJoCo XML/MJCF file and launch a local viewer.

| Option | Default | Description |
|---|---|---|
| `<model>` | — | Path to a MuJoCo XML / MJCF file. |
| `--name` | `Scene` | Display name shown in the viewer. |
| `--port` | `8080` | HTTP server port. |
| `--host` | `localhost` | HTTP bind address. |
| `--no-open` | `false` | Do not open the browser automatically. |

The build is written to a temporary directory and discarded when the server exits.

## `mjswan serve`

```bash
mjswan serve <dist-dir | document.swn> [--port 8080] [--host localhost] [--no-open] [--height 600]
```

Serve a pre-built `dist/` directory with the COOP/COEP headers set. Use this to re-launch an app you previously built with `builder.build()`.

A `.swn` works too. A document holds the simulation but no engine, so mjswan expands it into a
temporary directory, lays the packaged engine over it and serves that; the directory goes away when
the command exits. A document built with custom-JS MDP terms is refused, since their runtime module
ships with the engine rather than in the document — serve its built directory instead.

| Option | Default | Description |
|---|---|---|
| `<dist-dir \| document.swn>` | — | A built mjswan `dist/` directory, or a `.swn` document. |
| `--port` | `8080` | HTTP server port. |
| `--host` | `localhost` | HTTP bind address. |
| `--no-open` | `false` | Do not open the browser automatically. |
| `--height` | `600` | Colab iframe height in pixels (ignored outside Colab). |

## `mjswan new`

```bash
mjswan new <name> [--template hello-world|policy|mjlab]
```

Scaffold a new mjswan project in a directory named `<name>`.

| Template | What you get |
|---|---|
| `hello-world` (default) | A `main.py` that builds a single scene from `model.xml` and a minimal `model.xml`. |
| `policy` | The hello-world template plus an `add_policy(name="Policy", policy=...)` call expecting a `policy.onnx` file alongside `main.py`. |
| `mjlab` | A one-line `mjswan.Builder.from_mjlab("go2_flat").build()` script. Requires [`mjlab`](../guides/mjlab.md) installed. |

After scaffolding:

```bash
cd <name>
python main.py
```

## `mjswan demo`

```bash
mjswan demo [name]
```

With no name it lists the bundled demos; with one it runs that demo. They live under
`examples/`, so this needs a source checkout, and they fetch what they show (from mjlab,
from the Hugging Face Hub, or from upstream), so the first run downloads and later ones
read the cache.

| Demo | Source | Needs |
|---|---|---|
| `main` | `examples/demo/main.py`, the demo deployed to GitHub Pages | `mjswan[hf,mjlab]` |
| `simple` | `examples/demo/simple.py`, one mjlab task, one checkpoint | `mjswan[hf,mjlab]` |
| `mujoco` | `examples/demo/mujoco_models.py`, MuJoCo's `<replicate>` gallery, no policy | core only |

The bare command lists rather than running one, because each of these downloads.

## `mjswan info`

```bash
mjswan info <dist-dir | document.swn>
```

Print a tree summary of a built `dist/` directory or a `.swn` document: projects, scenes
(with `.mjz`/`.mjb` size), each scene's MDPs (with their traced graph count) and policies
(with `.onnx` size). Useful for spotting which assets are pushing your build toward the
GitHub Pages 1 GB limit.

Example output:

```
mjswan app — dist  v0.9.3, format 1
└── My Robots  [my_robots]
    └── G1  g1/scene.mjz  (3.2 MB)
        ├── MDP: mdp_0  3 graph(s)
        └── Policy: Locomotion  mdp=mdp_0  (1.1 MB)
Total scene+policy assets: 4.3 MB
```

## `mjswan publish`

```bash
mjswan publish <dist-dir | document.swn> [--title TITLE] [--description TEXT] [--tag TAG]... \
                                         [--token TOKEN] [--api-base URL]
```

Upload a built `dist/` directory's data files, or a `.swn` document, to
[mjswan Cloud](../guides/publishing.md) and print the resulting simulation URL. Only the
simulation document travels — `manifest.json`, the scene/policy/motion/splat assets and
traced graphs — never the compiled JavaScript, since Cloud renders them with its own copy
of the engine. A `.swn` uploads exactly the file set its directory would.

| Option | Default | Description |
|---|---|---|
| `<dist-dir>` | — | A built mjswan `dist/` directory, or a `.swn` written by `app.save_document()`. |
| `--title` | first project's name | Simulation title. |
| `--description` | — | Optional description. |
| `--tag` | — | Tag to attach. Repeatable. |
| `--token` | `$MJSWAN_TOKEN` | Access token, for CI. Skips the interactive login. |
| `--api-base` | `$MJSWAN_API_BASE`, then `https://api.mjswan.com` | Cloud API base URL. |

If you are not signed in and no token is supplied, the browser login runs first
automatically. Builds that use custom-JavaScript MDP terms are rejected: Cloud cannot
execute author-supplied code.

## `mjswan login` / `whoami` / `logout`

```bash
mjswan login [--no-open]   # GitHub OAuth over a loopback redirect
mjswan whoami              # print the signed-in account
mjswan logout              # remove the stored session
```

`--no-open` prints the authorization URL instead of opening a browser — useful over SSH.
The session is stored locally; `mjswan whoami` exits non-zero when there is none, or when
one exists locally but is no longer valid server-side.
