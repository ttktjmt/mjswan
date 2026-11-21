# CONTEXT.md

The map of the codebase. Each area's rules, reasons and pitfalls are one level down, in [context/](context/).

## What mjswan is

mjswan is a Python framework that packages browser-based MuJoCo simulations with real-time ONNX policy control into interactive static web apps. The Python side builds the bundle; the browser client is TypeScript / React / three.js running mujoco-wasm for physics and onnxruntime-web for inference. Published on PyPI and npm; demos hosted on GitHub Pages and Cloudflare Pages, with optional one-command publishing to mjswan Cloud.

The defining architectural fact: mjswan reimplements none of mjlab's MDP term functions. A task's **real Python function objects** (observations, terminations, events, commands, and a task's own action `process`) are traced to ONNX at build time with `torch.onnx.export` and run in the browser by ONNX Runtime Web, beside the policy. Orchestration and every action's per-substep apply are native TypeScript. Anything that cannot be traced fails the build loudly rather than being silently dropped.

## Second level: `context/`

| File | Covers |
|---|---|
| [package-layout.md](context/package-layout.md) | the layers and import direction, naming rules, mjlab's three relations, the old → new module table |
| [object-model.md](context/object-model.md) | `Builder` and the `*Handle` / `*Config` pairs: what each owns, `MdpConfig` identity and ids, the public API |
| [build.md](context/build.md) | `build/`: the pipeline's order, manifest entries, where graphs land, `frontend.py` and the SPA cache, `scene.mjz` |
| [tracing.md](context/tracing.md) | `compile/`: the two passes, slot namespaces, the trace env, stateful terms, randomness, fusion, traced action process and native apply, what cannot be traced, parity, declined and rejected options |
| [mdp.md](context/mdp.md) | `managers/` and `envs/mdp/`: the mirror cfgs, `*Binding` escape hatches, command terms, the muscle action term |
| [mjlab.md](context/mjlab.md) | `mjlab/` adapters, the trace envs, the runner, `onnx_meta` pitfalls, and `source/` (W&B, Hub, URL) |
| [document.md](context/document.md) | the tree and `.swn`, manifest rules, ids, the `in_keys` / `out_keys` slot tables, `format` versus `version`, an MDP owning its events and the MDP switch, `app.py` |
| [licenses.md](context/licenses.md) | `license/`: file locations, detection, identification and tiers, why files |
| [cloud.md](context/cloud.md) | `cloud/`: what `publish` sends and refuses, content-addressed pieces, login |
| [skins.md](context/skins.md) | `skin.py`, `rig/`, `core/skin/`: characters drawn outside the model |
| [frontend.md](context/frontend.md) | `template/`: the engine API, the step loop, `src/core/`, graphics, interaction with pick and preview, the three builds, plugins and custom TS |
| [ci.md](context/ci.md) | tests, workflows, release, deployment |
| [dependencies.md](context/dependencies.md) | pins and why, MuJoCo on both sides, how releases enter the lock |

## Repository layout

```
src/mjswan/          Python package: the object model at the root, one package per layer below
  __init__.py          The public API: Builder / MjswanApp, the *Handle / *Config pairs, cfgs
  builder.py           Builder: the fluent entry point; build() hands off to build/pipeline.py
  app.py               MjswanApp: launch / serve / save_document / publish a built app
  project.py           ProjectConfig / ProjectHandle (add_scene, add_scene_mjlab, set_license)
  scene.py             SceneConfig / SceneHandle (add_policy, add_policy_wandb / _hf, splats)
  policy.py            PolicyConfig / PolicyHandle; the ONNX slot and width checks a policy passes
  motion.py            MotionConfig / MotionHandle; attaching a tracking task's clip
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  skin.py              SkinConfig / SkinHandle: a rigged character (.vrm or .glb) drawn over the
                       robot, posed from its bodies, outside the model; and ModelledSkinConfig,
                       a .glb modelled on the robot, fixed to its bodies
  viewer.py            ViewerConfig: the camera, and the graphics settings (colour management, tone mapping)
  mdp.py               MdpConfig: the five term sets a policy runs against, as one shared unit
  cli.py               Typer-based `mjswan` CLI
  _version.py          __version__ (read by hatch; the only underscore module at the root)
  build/               What Builder.build() does; the one place that knows the manifest's shape
    pipeline.py          write_app: frontend build, then per scene pack → assets → trace → entries
    manifest.py          project / scene / MDP / policy entries, and the manifest.json write
    asset.py             a scene's assets/: motion clips (deduplicated by content), splats, skins
    skin.py              a skin's manifest entry: node indices and measured turns, or a
                         modelled skin's joints and bodies; no glTF names
    credit.py            a scene's `credits` and a project's `work`, and the names in `licenses/`
    mjz.py               scene.mjz: MjSpec plus every asset, DEFLATE-compressed, paths rewritten
    frontend.py          the Python side of template/: nodeenv + Vite build, install_spa
    mdp/                 one .onnx per traced term and the JSON entry that names it,
                         observation · termination · event · command · action, with graph.py
                         (where a graph lands, the guarded write), provenance, sensor, binding
  compile/             ONNX tracing + numeric parity; returns bytes, writes no file
    slot · proxy · record · replay · export      the two passes and what they share
    term · event · command · action · group      one trace per term kind
    model.py             an event's env.sim.model: reads become slots, writes outputs
    gate.py              env sets a draw selects, kept whole and gated
    parity.py            live mjlab env vs exported graphs, term by term, step by step
    rng.py               build-time RNG spy/replay (DrawRecorder / ReplayRng)
  mjlab/               Everything that reads mjlab's own objects (mjlab imported lazily)
    observation · termination · command · action · event   per-manager adapters
    task.py · runner.py · env.py · sim.py · gui.py · onnx_meta.py · detect.py
  source/              A reference in, local files out: wandb.py, hf.py (file, dir, ONNX),
                       url.py (a plain http(s) URL)
  document/            The .swn document: container.py, manifest.py, ids.py, and skin.py, a
                       skin's file as a document carries it (one binary glTF)
  rig/                 Fitting a skin to its robot: humanoid.py (VRM bones, presets), pose.py
                       (rest pose, root height, hips, limb lengths, the body each joint of a
                       modelled skin rests on), guess.py (`mjswan skin map`'s drafts)
  cloud/               mjswan Cloud: publish.py, blob.py (the cutter), auth.py (loopback PKCE
                       OAuth), transport.py
  license/             LICENSE / NOTICE files in the build: path, spdx, attribution; vrm.py,
                       a VRM's license as its meta states it
  managers/            mjlab-mirror cfgs: observation / event / action / termination / command
  envs/mdp/            mjlab-mirror MDP side: actions/ (real cfgs), the *Binding escape hatches,
                       commands.py (ui_command presets, mjlab bindings, motion reset jitter)
  template/            TypeScript frontend (Vite + React + three.js + mujoco-wasm)

context/             Second-level contributor context, one file per area (index above)

examples/            Runnable examples. Python only: assets are fetched at run time
  demo/                main.py (the deployed app: mjlab Tasks + Showcase, everything
                       from mjlab or the Hub, skins included, but WBC-Mjlab's policy
                       and clips, from its deploy repo at a pinned commit; the `demo`
                       extra is what it imports), simple.py (one task, one checkpoint),
                       minimum_policy.py (the smallest complete policy), mujoco_models.py
                       (a gallery, no policy) and playground.py (MuJoCo Playground:
                       SO101PickCube and SO101PickCubeVision as mjlab tasks driven by the
                       policies examples/train trained, the latter through the wrist
                       camera, and the tabletop scenes, no policy, its files from GitHub at
                       pinned commits). `mjswan demo` runs the four named ones;
                       minimum_policy.py is read, not run
  colab/               Google Colab notebooks (demo, anymal_c_velocity)
  train/               so101_pick_cube.py: trains MuJoCo Playground 1.0.1's SO101PickCube, or
                       with --task vision SO101PickCubeVision, on one free GPU and exports the
                       ONNX policy, its contract and a reference rollout for an mjswan port;
                       the Colab and molab notebooks beside it run it

                     The W&B-driven examples (defaults, g1_spinkick, myosuite,
                     musclemimic, unitree_rl, gentle_humanoid) live in
                     mjswan_playground, which carries their pinned dependencies

tests/               pytest suite + dump_*_fixture.py generators for the TS parity fixtures
skills/              Agent skills this repo publishes (mjlab-to-mjswan: port an mjlab task)
docs/                zensical (MkDocs-based) site → Read the Docs
typings/             MuJoCo stub generator script
scripts/             Maintenance scripts (sync_contributors.py)
assets/              Demo GIF and banner SVG
```

## Python object model (fluent API)

```
Builder(base_path, gtm_id, mt, debug)
  ├── Builder.from_mjlab(task_id, run_path=..., play=...) → Builder  # classmethod factory
  ├── .add_project_mjlab(task_id, run_path=..., play=...) → ProjectHandle
  └── .add_project(name, default=False) → ProjectHandle      # id = name2id(name)
        ├── .add_scene_mjlab(task_id, play=..., env_cfg=..., events=...) → SceneHandle
        ├── .add_scene_hf(repo_id, path, name=..., ...) → SceneHandle   # MJCF + its meshes
        └── .add_scene(name, model|spec, metadata, control_dt, events) → SceneHandle
              ├── .add_policy(name, policy, mdp=MdpConfig | the five term sets,
              │               in_keys=, out_keys=, ...) → PolicyHandle
              │     └── .add_motion(...) / .add_motion_wandb(...) / .add_motion_hf(...) → MotionHandle
              ├── .add_policy_wandb(run_path, ...) → list[PolicyHandle]   # one MdpConfig per call
              ├── .add_policy_hf(repo_id, ...) → list[PolicyHandle]       # one MdpConfig per call
              ├── .add_splat(name, source|url, ...) → SplatHandle
              ├── .add_splat_hf(repo_id, filename, ...) → SplatHandle  # downloaded, then bundled
              ├── .set_viewer(ViewerConfig)
              ├── .add_skin(name, SkinConfig | ModelledSkinConfig) → SkinHandle  # a .vrm or .glb over the robot; outside the model
              ├── .set_events(events)             # the scene's default events for its policies
              └── .set_trace_env(env)             # required for a non-mjlab policy scene

builder.build(output_dir) → MjswanApp
MjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
MjswanApp.save_document(path=None) → Path    # the data as one .swn (ZIP of the tree)
MjswanApp.publish(title=..., tags=...)       # → mjswan Cloud; a .swn publishes too
```

- Each `*Handle` wraps a `*Config` dataclass: the handle is the fluent API, the config is the serializable state. `__init__.py` is the canonical public API.
- Every project / scene / MDP / policy / splat / skin has `id = name2id(name)`, unique within its parent; ids are the build's directories and the `?project=` / `?scene=` / `?policy=` / `?skin=` values. A collision renames the second `<name>_1` with a RuntimeWarning rather than failing ([document.md](context/document.md)).
- A policy scene needs two things a model-only scene does not: `control_dt` (a wrong control rate raises nothing at playback) and a trace env (`add_scene_mjlab` supplies both; a plain `add_scene` needs `set_trace_env(build_single_entity_trace_env(spec_fn))`). See [object-model.md](context/object-model.md) and [tracing.md](context/tracing.md).

## Layering

Imports only point downward ([package-layout.md](context/package-layout.md)):
`cli.py` → `app.py` / `cloud/` → the object model at the root → `build/` → `compile/` →
`mjlab/` · `source/` · `rig/` → `managers/` · `envs/mdp/` · `document/` · `license/`.
`build/` alone knows the manifest's shape; `compile/` writes no file and never imports `mjlab/`; `cloud/` never imports `build/`. The root imports the build pipeline and `compile/` lazily and `mjlab/` imports mjlab only inside its functions, so `import mjswan` never costs a Node build and, in a core install, loads no source backend. mjswan's own names are singular; the mjlab-mirror paths keep mjlab's spelling because the import path is the contract.

## CLI entry points

The primary CLI is `mjswan` (Typer-based, defined in `cli.py:app`). Subcommands:

| Subcommand | Description |
|------------|-------------|
| `mjswan view <model.xml>` | Build and launch a viewer for a MuJoCo XML/MJCF file |
| `mjswan serve <dist-dir \| document.swn>` | Serve a pre-built `dist/` directory, or a `.swn` document |
| `mjswan new <name> [--template hello-world\|policy\|mjlab]` | Scaffold a new project from a template |
| `mjswan demo [name]` | List the built-in demos, or run one (`main`, `simple`, `mujoco`, `playground`) |
| `mjswan info <dist-dir \| document.swn>` | Show a tree of projects/scenes/policies and asset sizes, each license file and skin with its license |
| `mjswan publish <dist-dir>` | Upload a built dist's data files to mjswan Cloud (rejects custom-TS builds) |
| `mjswan skin map <file> [--robot <xml \| mjb \| task>]` | Draft a skin's bone maps against a robot, as a `SkinConfig` (or a `ModelledSkinConfig`) to check |
| `mjswan login` / `whoami` / `logout` | mjswan Cloud session (loopback GitHub OAuth) |
| `mjswan version` / `--version` | Print the installed mjswan version |

## Tooling and workflow

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management and script runner: use instead of bare `python` / `pip` |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting (pinned exactly, not floored: see the pyproject comment) |
| `pyright` / `ty` | Type checking (`make type` runs both over `src/mjswan`; `examples/` and `typings/` are out of scope, the latter a search path for ty) |
| `pytest` | Tests (`make test`) |
| `pre-commit` | Hooks: trailing-whitespace, end-of-file-fixer, ruff, ruff-format, npmrc secret scan, pytest (not slow), eslint (the template's locked one, so `npm ci` in `src/mjswan/template` first) |
| `zensical` | Docs site builder (MkDocs-based): `make docs-build` / `make docs-serve` |

Key Makefile targets: `sync`, `format`, `type`, `check`, `test`, `test-all`, `docs-build`, `docs-serve`.

## Tests and CI

`@pytest.mark.slow` marks a test that runs a full frontend (npm + Vite) build; pre-commit skips those (`pytest -m "not slow"`), `make test` runs everything. `pytest.yml` runs the fast suite on Python 3.11 to 3.13 plus a `core` job with no extra; `parity.yml` runs the ONNX numeric-parity sweep against live mjlab on relevant pull requests and weekly; `frontend.yml` lints the template and runs vitest. Every workflow installs from `uv.lock` alone. Details in [ci.md](context/ci.md); the parity layers in [tracing.md](context/tracing.md).

## Dependencies

Core: `mujoco==3.11.0` (exact, following mjlab's pin), `onnx`, `nodeenv`, `rich`, `typer`. One extra per source (`wandb`, `hf`), `mjlab` (mjlab + torch, needed at build time for any traced policy, never shipped to the browser), plus `check`, `dev`, `demo`, `examples`. The browser's MuJoCo is the template's `@ttktjmt/mujoco` alias, held to the same MuJoCo as the Python pin. Python 3.11 to 3.13. Details and reasons in [dependencies.md](context/dependencies.md).

## Deployment

`deploy.yml` builds `examples/demo/main.py` and deploys it to GitHub Pages on every relevant push to `main`, with no credentials (models, checkpoints and skins come from the public Hub repository `ttktjmt/mjswan`). Cloudflare Pages builds a preview of the same demo for every pull request and hosts the GentleHumanoid and MuscleMimic demos. `mjswan publish` is the alternative to hosting a `dist/` at all. Details in [ci.md](context/ci.md).
