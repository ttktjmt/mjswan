# CONTEXT.md

## What mjswan is

mjswan is a Python framework that packages browser-based MuJoCo simulations with real-time ONNX policy control into interactive static web apps. The Python side builds the bundle; the browser client is TypeScript/React/three.js running mujoco-wasm for physics and onnxruntime-web for inference. Published on PyPI and npm; demos hosted on GitHub Pages and Cloudflare Pages, with optional one-command publishing to mjswan Cloud.

The defining architectural fact (ADR 0005): mjswan reimplements none of mjlab's MDP term functions. A task's **real Python function objects** — observations, terminations, events, commands — are traced to ONNX at build time with `torch.onnx.export` and run in the browser by ONNX Runtime Web, beside the policy. Only Action is native TypeScript. Anything that cannot be traced fails the build loudly rather than being silently dropped.


## Repository layout

```
src/mjswan/          Python package source
  builder.py           Builder — top-level entry point; orchestrates the whole build
  app.py               MjswanApp — launch / serve / publish built apps
  publish.py           Publish a built dist/ to mjswan Cloud (presigned-upload protocol)
  auth.py              mjswan Cloud login — gh-style loopback OAuth (PKCE) via Supabase
  project.py           ProjectConfig / ProjectHandle
  scene.py             SceneConfig / SceneHandle
  policy.py            PolicyConfig / PolicyHandle
  motion.py            MotionConfig / MotionHandle
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  command.py           Command terms (Slider, Button, Checkbox, velocity_command, ui_command)
  viewer.py            ViewerConfig
  trace_env.py         Minimal live envs for tracing (build_single_entity_trace_env)
  utils.py             ZIP-DEFLATE bundling, XML path rewriting, name2id slug helper
  wandb_io.py          W&B checkpoint / motion artifact downloads
  _onnx_build.py       Bridges term cfg dataclasses → compile/ tracer; writes .onnx + JSON
  _cli.py              Typer-based `mjswan` CLI + legacy entry points
  _build_client.py     Frontend build orchestration (npm/vite, managed nodeenv)
  _compat.py           Deprecated pre-0.8 aliases (methods, classes, modules). Remove in 0.9
  compile/             ONNX tracing + numeric parity (ADR 0005)
    tracer.py            trace_term / trace_event_term / trace_command_term
    serialize.py         traced command → policy.json entry + JSON schema
    parity.py            live mjlab env vs exported graphs, term by term, step by step
    rng.py               build-time RNG spy/replay (DrawRecorder / ReplayRng)
  adapters/            mjlab soft-dependency adapter + compat helpers
  envs/mdp/            MDP config side: actions/ (real cfgs) + Binding escape hatches
  managers/            observation / event / action / termination manager cfgs
  template/            TypeScript frontend (Vite + React + three.js + mujoco-wasm)

examples/            Runnable examples
  demo/                main demo (deployed to GitHub Pages) + simple, splat, muscle,
                       and gentle_humanoid/ (11 traced terms, deployed separately)
  mjlab/               defaults (7 reference tasks + commands/events/terminations
                       registrations), g1_spinkick, myosuite, musclemimic, unitree_rl
  colab/               Google Colab notebooks (demo, anymal_c_velocity)
  tutorial/            hello_world, minimum_policy, mujoco_models

tests/               pytest suite + dump_*_fixture.py generators for the TS parity fixtures
skills/              Agent skills this repo publishes (mjlab-to-mjswan: port an mjlab task)
docs/                zensical (MkDocs-based) site → Read the Docs; adr/ design records
typings/             MuJoCo stub generator script
scripts/             Maintenance scripts (sync_contributors.py)
assets/              Demo GIF and banner SVG
```


## Python object model (fluent API)

```
Builder(base_path, gtm_id, mt, debug)
  ├── Builder.from_mjlab(task_id, run_path=..., play=...) → Builder  # classmethod factory
  ├── .add_project_mjlab(task_id, run_path=..., play=...) → ProjectHandle
  └── .add_project(name, id) → ProjectHandle
        ├── .add_scene_mjlab(task_id, play=..., env_cfg=..., events=...) → SceneHandle
        └── .add_scene(name, model|spec, metadata, control_dt, events) → SceneHandle
              ├── .add_policy(name, policy, ...) → PolicyHandle
              │     └── .add_motion(...) / .add_motion_wandb(...) → MotionHandle
              ├── .add_policy_wandb(run_path, ...) → list[PolicyHandle]
              ├── .add_splat(name, source|url, ...) → SplatHandle
              ├── .set_viewer(ViewerConfig)
              ├── .set_events(events)
              └── .set_trace_env(env)          # required for a non-mjlab policy scene

builder.build(output_dir) → MjswanApp
MjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
MjswanApp.publish(title=..., tags=...)       # → mjswan Cloud
```

`Builder.from_mjlab(task_id, run_path=...)` is the one-liner shortcut for the common "visualize a single mjlab task" pattern; it delegates to the instance method `Builder.add_project_mjlab`, which creates a project, adds an mjlab scene, and optionally attaches all `model_*.pt` checkpoints from one or more W&B runs (converted to ONNX via mjlab+torch). For finer control, build manually: `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb(...)`.

Each `*Handle` wraps a `*Config` dataclass — the handle is the fluent API, the config is the serializable state.

**Two build-time requirements a policy scene has and a model-only scene does not:**

- `control_dt` — seconds per control step (mjlab's `timestep * decimation`). The model carries only the physics timestep, and a wrong control rate raises nothing at playback, so the builder requires it. `add_scene_mjlab` fills it in from the task.
- a **trace env** — tracing needs a live env to read shapes from and resolve `SceneEntityCfg` regexes against. `add_scene_mjlab` builds one at build time; a plain `add_scene` scene with traced terms needs `set_trace_env(build_single_entity_trace_env(spec_fn))` or the build raises.

The package's `__init__.py` is the canonical public API. Re-exports cover: `Builder` / `MjswanApp`; the five `*Handle` and `*Config` pairs; mjlab-compatible MDP cfgs (`ObservationGroupCfg`, `ObservationTermCfg`, `ActionTermCfg`, `JointPositionActionCfg`, `JointEffortActionCfg`, `TerminationTermCfg`); command UI (`SliderConfig`/`ButtonConfig`/`CheckboxConfig` and their `Slider`/`Button`/`Checkbox` aliases, `SliderRangeConfig`, `CommandTermConfig`, `CommandBinding`, `CommandUiConfig`, `ui_command`, `velocity_command`); the `MdpBinding` umbrella type and the `register_observation` / `register_event` / `register_termination` / `register_command` hooks; and `build_single_entity_trace_env`.


## Key modules

### `builder.py` — `Builder`
Main entry point. Accumulates `ProjectConfig` objects, calls `ClientBuilder` to invoke the Vite frontend build, then per scene: builds or reuses the trace env, traces every term via `_onnx_build`, and writes `config.json` + per-scene DEFLATE-compressed ZIPs (via `utils.to_zip_deflated`, since `mujoco.to_zip` stores entries uncompressed) plus policy/motion/splat assets and traced `.onnx` graphs into the output directory.

### `app.py` — `MjswanApp`
Wraps a built `dist/` directory. `launch()` starts a stdlib HTTP server (COOP/COEP headers required for SharedArrayBuffer / MuJoCo WASM threading); detects Google Colab and displays an inline iframe instead. `publish()` delegates to `publish.py`.

### `publish.py` / `auth.py` — mjswan Cloud
`publish_dist()` uploads only *data* files (`.json`, `.mjz`, `.onnx`, `.npz`, `.ply`, `.spz`; 50 MB/file, 200 MB total, 64 files) via a presigned-upload protocol — never the compiled JS, since Cloud loads a pinned engine from its own CDN. It refuses a build with `uses_custom_js: true`, which Cloud cannot render. `auth.py` is a `gh`-style loopback PKCE OAuth flow against Supabase's GitHub OAuth, persisting a rotating refresh token locally; `MJSWAN_TOKEN` bypasses it for CI.

### `policy.py` — `PolicyConfig` / `PolicyHandle`
Holds an `onnx.ModelProto` plus observation groups, action terms, termination terms, commands, and motion references. Compatible with mjlab config classes via the adapter layer. Serialized to a per-policy `<name>.json` at build time, alongside the traced `obs/`, `term/` and `command/` graphs it references.

### `command.py`
Defines command terms consumed by policies: `SliderConfig` (with `enabled_when` and an `adjustable_range` → `SliderRangeConfig` companion slider), `CheckboxConfig`, `ButtonConfig`, `CommandUiConfig`, `CommandTermConfig`, `CommandBinding`, and the `CommandInput` union. `velocity_command()` builds the standard locomotion 3-DoF velocity command; `ui_command()` builds a generic UI-driven term. A real mjlab command class (velocity, lifting, tracking RSI jitter) is traced like any other term — its hidden state is promoted to explicit graph I/O — and registered with `register_command`. `default_viz()` restates what mjlab's `_debug_vis_impl` draws for mjlab's own command classes as data (`core/command/debugViz.ts` evaluates it), so a `debug_vis=True` task gets its arrows and markers without the author declaring any; one mjswan has no drawing for warns at build time. The panel lists each drawing term in its **Debug Viz** section (`engine.debugVis.set`), on by default as mjlab's viewers are.

### `scene.py` — `SceneConfig` / `SceneHandle`
A scene owns one MuJoCo model (as `MjModel` → binary `.mjb` or `MjSpec` → XML), its `control_dt`, its scene-scoped events, its trace env, zero or more policies, and zero or more Gaussian splat backgrounds.

### `splat.py` — `SplatConfig` / `SplatHandle`
Configures a 3D Gaussian Splat (`.spz` format) background: scale, position offsets, Euler rotations, optional collider mesh URL, and a `control` flag exposing live calibration sliders.

### `viewer.py` — `ViewerConfig`
Camera parameters (lookat, distance, fovy, elevation, azimuth) + tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

### `trace_env.py`
`build_single_entity_trace_env(spec_fn)` builds a minimal single-entity `ManagerBasedRlEnv` out of mjlab's own `Entity`/`Scene` — no reimplemented kinematics. It configures no managers and is never stepped; it is only the tracer's `env.scene[name].data.<field>` read/write target. Joint defaults come from the model's first keyframe, matching what the browser resets to. `build_mjlab_env()` grows `nconmax`/`njmax` until a task's env fits a single-env re-use. `TraceCommandManager` supplies trace-time stand-ins for commands the browser owns (a `ui_command` has no Python side).

### `compile/` — the tracer (ADR 0005)
- `tracer.py`: runs each `func(env, **params)` once against a recording proxy to discover its reads, classifies each as time-varying state (a graph input, or "slot") or a model-derived constant (baked in), then exports an `nn.Module` via `torch.onnx.export`. Three slot namespaces: an entity `data` field, a named sensor's `sensordata` window, and a live command's state field. Structured sensors (`RayCastSensor`) contribute one slot per field read. Only `_STATIC_DATA_FIELDS` are treated as constant — anything unrecognized errs toward a graph input, so a missing runtime input fails loudly instead of returning stale values.
- `serialize.py`: a traced command's `policy.json` entry. One generic browser-side `OnnxCommand` handler interprets every command from this data, so the entry declares state fields (names, shapes, **and initial values**), `rand_dim`, dynamic reads, and any `entity_write`.
- `rng.py`: build-time RNG spy/replay. Patches the term function's *own* module globals (mjlab binds the name at import time), records mjlab's real draws, replays those exact values into the graph's `rand` input so parity does not diverge on randomness alone. Unrelated to the runtime's seeded PRNG.
- `parity.py`: steps a live env with a seeded action sequence and feeds the same raw state through each exported graph via `onnxruntime` (not torch), asserting `allclose` for every term at every step.

### `_onnx_build.py`
Bridges the term config dataclasses to `compile/`. Traces each plain-callable body against the scene's live env, writes `.onnx` bytes under `<scene_dir>/{obs,term,command,event}/`, and returns the manifest-shaped JSON entry. Emits **fused** graphs: one per observation group (clip-then-scale folded in, mjlab's order) and one per termination group (a bool lane per term, so per-term reset reasons survive). A group with a legacy `*Binding` term or per-term history does not fuse; a lone traced termination is deliberately left unfused. Startup DR that perturbs `mjModel` rather than `mjData` (`geom_friction`, `body_com_offset`, `geom_rgba`) carries no graph — it emits a descriptor the browser applies once from the seeded PRNG, keyed by entity *names* since the browser compiles its own model.

### `adapters/`
- `mjlab_adapter.py`: Converts mjlab types to mjswan equivalents. Obs/term/event bodies are *not* name-resolved to mirrors any more — mjlab's own functions are traced directly. What remains is config-shape adaptation, observation-group key resolution via the task's runner config (`rl_cfg.obs_groups["actor"]`), `clip_actions`, and action-scale resolution.
- `gui_spy.py`: runs mjlab's own `CommandTerm.create_gui` against a recording stand-in, so the browser control panel's slider ranges come from mjlab's declaration instead of a hand-copied duplicate that drifts.
- `mjlab_compat.py`: Monkey-patches `MujocoCfg.apply_to_spec()` onto mjlab when needed.

### `envs/mdp/` and `managers/`
mjlab-compatible MDP layer, with a deliberate asymmetry:

- **`envs/mdp/actions/`** carries *real, directly usable* config classes, because Action is permanently native (ADR 0005 §7): `JointPositionActionCfg`, `JointEffortActionCfg`, `MuscleActivationActionCfg` are supported; `JointVelocityActionCfg`, `TendonLength/Velocity/EffortActionCfg` and `SiteEffortActionCfg` are exported so mjlab configs import cleanly but raise `NotImplementedError` at build time. `stiffness`/`damping` are mjswan-specific — the browser computes PD externally for motor actuators with `biastype=none`.
- **`envs/mdp/{observations,terminations,events}.py`** carry only the `*Binding` escape hatch and its `register_*` registry. Pass mjlab's own functions instead. The one event mjswan owns is `reset_root_state_on_flat_patch`, which replaces mjlab's untraceable `reset_root_state_from_flat_patches` (it draws with `torch.randint` and indexes per-env terrain tensors); `apply_terrain_spawn` swaps it in, since the browser runs one env where mjlab spreads many across the terrain.
- **`managers/`** holds the config-side counterparts (`observation_manager`, `event_manager`, `action_manager`, `termination_manager`). `ObservationTermCfg` adds `history_steps` (sparse look-back offsets) and `history_interleaved` (Isaac joint-major layout) beyond mjlab's fields; training-only fields (`noise`, `delay_*`, `enable_corruption`) are accepted and ignored.

A term with no browser implementation **fails the build**, naming both ways out (a trace-friendly `register_*` replacement callable, or a `ts_src` TS class). Three event terms are exempt because there is provably nothing to write: `randomize_terrain`, `encoder_bias`, and a root-state write onto a fixed-base entity.

**Muscle action term.** `MuscleActivationActionCfg` drives MuJoCo muscle actuators. `normalize=True` (default) applies the canonical MyoSuite sigmoid `σ(5(scale·a + offset − 0.5))` to map policy outputs into excitation in (0, 1); `normalize=False` clips `scale·a + offset` to [0, 1]. The semantics mirror myosuite4's `MuscleActionTermCfg.normalize`. The mjlab adapter translates `MyoMuscleActivationActionCfg` (the class actually used by every myo* mjlab task) to `MuscleActivationActionCfg`; see [docs/adr/0002](./docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md).

### `_build_client.py`
Orchestrates the Node.js / Vite frontend build. Manages a local `nodeenv` if Node isn't available system-wide.

### `wandb_io.py`
Downloads `model_*.pt` checkpoints and motion `.npz` artifacts from Weights & Biases runs. Used by `SceneHandle.add_policy_wandb()` and `PolicyHandle.add_motion_wandb()`.

### `utils.py`
Asset bundling and path helpers. `to_zip_deflated()` is the per-scene packager: it collects mesh/texture/hfield/skin files from disk (with `spec.assets` fallback for mjlab's prefixed-key layout), encodes buffer-only textures as PNGs, rewrites the MuJoCo XML so meshdir/texturedir hints are eliminated and all paths are ZIP-safe, and writes a DEFLATE-compressed ZIP that JSZip decodes on the client. `name2id()` is the lowercase-underscore slug helper used everywhere project / scene / policy IDs are derived from human-readable names.

### `_compat.py`
Deprecated pre-0.8 aliases, imported for its side effects by `__init__.py`. Method aliases (`add_mjlab_scene`, `add_policy_from_wandb`, `set_viewer_config`, `add_splat_section`), the `mjswanApp` class name, `register_obs_func` / `register_termination_func` / `register_event_func` / `register_command_term`, binding class aliases (`ObsBinding`, `TermFunc`, `CommandTermSpec`, …), and module aliases (`mjswan.viewer_config`, `mjswan.wandb_utils`). Functions warn; class aliases cannot. Remove in 0.9.


## Frontend (`src/mjswan/template/`)

TypeScript + React + Vite + three.js. Built by `Builder.build()` via `_build_client.py`. The browser client:
- Loads the MuJoCo WASM module and runs physics in a Web Worker.
- Runs the policy **and every traced MDP term body** via onnxruntime-web.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR), including tracked hands as bodies inside the sim (opt-in).
- Reads `config.json` to discover projects/scenes/policies at runtime.

`src/core/` mirrors mjlab's layout — `observation/`, `termination/`, `action/`, `event/`, `command/` — plus `onnx/` (`session.ts` caches split by lifetime, `runQueue.ts`, `slotReader.ts` serving graph inputs from `mjModel`/`mjData`, `raycast.ts` casting height-scan rays with `mj_ray`, `contact.ts`, `graphRefs.ts`), `rng.ts` (xoshiro128\*\*, snapshot-able), `policy/`, `scene/`, `xr/` (`handMocap.ts` injects a capsule per hand bone and writes `mocap_pos`/`mocap_quat`), and `engine/` (`runtime.ts` step loop, `resetChain.ts`, `viewer_config.ts`). Each manager is native orchestration around ONNX term bodies: `FusedObservation`/`OnnxObservation`/`NativeObservation`/`HistoryObservation`, `FusedTermination`/`OnnxTermination`/`TimeOutTermination`, `OnnxEvent` + `triggers.ts` + `entityWrite.ts` + `modelFieldDr.ts`, `OnnxCommand`. Action is fully native (`action/applyAction.ts`).

The step loop follows mjlab's `ManagerBasedRlEnv.step` ordering exactly — forward → command → event → obs → action → physics → term → reset → forward — with **one** `mj_forward` per iteration. The reset chain mirrors `_reset_idx`: `event(mode="reset")` → observation/action reset → command resample, all awaited in config order (writes are last-writer-wins by config order, so concurrency would make that machine-dependent).

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).

The template has three Vite build outputs (all written to `template/dist/`):
- **SPA** (`vite.config.ts`, `npm run build:spa`) — the standalone app the Python `Builder` assembles. Entry `src/index.tsx`.
- **Library** (`vite.lib.config.ts`, `npm run build:lib`) — a single self-contained ESM `dist/mjswan.js` exposing `createEngine(element, options?)` (entry `src/engine/index.ts`), consumed by mjswan Cloud from a CDN. Every dependency is bundled (no bare imports) and the MuJoCo/ONNX WASM is emitted as co-located files referenced via `new URL('./x.wasm', import.meta.url)`. Vite lib mode force-inlines those WASM as base64; a `generateBundle` plugin extracts them back into co-located files. See mjswan-cloud ADR 0001.
- **Manifest** (`vite.manifest.config.ts`) — `dist/manifest.js`, the `mjswan/manifest` catalog parser as a standalone CDN-loadable ESM.

`npm run build` runs all of them. The public engine API (ADR 0004) is bytes-in / snapshot-out: `loadScene` / `setPolicy` / `setSplat` / `setMotion`, `camera` / `commands` verbs, `subscribe`, `captureThumbnail`, `dispose`, with `termSeed` in and out for session replay. It never fetches — `mjswan/manifest` (`parseManifest(config, byteSource)`) turns a `config.json` into a lazy catalog and the app owns the fetching.


## CLI entry points

The primary CLI is `mjswan` (Typer-based, defined in `_cli.py:app`). Subcommands:

| Subcommand | Description |
|------------|-------------|
| `mjswan view <model.xml>` | Build and launch a viewer for a MuJoCo XML/MJCF file |
| `mjswan serve <dist-dir>` | Serve a pre-built `dist/` directory |
| `mjswan new <name> [--template hello-world\|policy\|mjlab]` | Scaffold a new project from a template |
| `mjswan demo [name]` / `--list` | Run a built-in demo (`simple`, `main`, `mjlab`) |
| `mjswan info <dist-dir>` | Show a tree of projects/scenes/policies and asset sizes |
| `mjswan publish <dist-dir>` | Upload a built dist's data files to mjswan Cloud (rejects custom-JS builds) |
| `mjswan login` / `whoami` / `logout` | mjswan Cloud session (loopback GitHub OAuth) |

Legacy entry points (kept for backward compatibility): `main`, `simple`, `mjlab`, `serve <dist-dir>` — each runs the corresponding `examples/` module.


## Tooling and workflow

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management and script runner — use instead of bare `python`/`pip` |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting (pinned exactly, not floored — see pyproject comment) |
| `pyright` / `ty` | Type checking |
| `pytest` | Tests (`make test`) |
| `pre-commit` | Hooks: trailing-whitespace, end-of-file-fixer, ruff, ruff-format, npmrc secret scan, pytest (not slow), eslint |
| `zensical` | Docs site builder (MkDocs-based) — `make docs-build` / `make docs-serve` |

Key Makefile targets: `sync`, `format`, `type`, `check`, `test`, `test-all`, `docs-build`, `docs-serve`.


## Tests and CI

`@pytest.mark.slow` triggers a full frontend (npm + Vite) build and is excluded from pre-commit (`pytest -m "not slow"`); unmarked tests are fast and always run.

Workflows: `pytest.yml` runs `pytest -m "not slow"` across Python 3.10 / 3.11 / 3.12 with the `dev` extra only. `parity.yml` is separate and heavier — it installs the `examples` extras (~2 GB), caches warp's CPU kernels, and runs the ONNX numeric-parity sweep over the reference mjlab tasks on `workflow_dispatch` plus weekly. Weekly because the cadence targets *upstream* drift: mjlab moves fast, and a changed term definition would otherwise surface as a mysterious runtime difference rather than a failing check. Trigger it by hand for a change to `compile/` or the serializers. Also: `eslint.yml`, `ruff.yml`, `deploy.yml`, `publish-pypi.yml`, `publish-npm.yml`, `release.yml`, `sync-contributors.yml`.

Parity is layered, and each layer covers something the others cannot:

| Check | What it proves |
|---|---|
| `tests/test_onnx_parity.py` | each traced graph reproduces its mjlab term (7 tasks, 73 terms, worst max\|Δ\|≈9e-08) |
| `tests/test_onnx_command_parity.py` | every reference task's traced command over 16 replayed draws |
| `core/onnx/__tests__/slotReaderParity.test.ts` | the reader hands those graphs mjlab's numbers (fixture from `dump_slot_fixture.py`) |
| `core/engine/__tests__/rolloutParity.test.ts` | the layers **composed** — state → reader → real ORT session → fused obs → group vector (fixture from `dump_rollout_fixture.py`) |
| `tests/test_artifact_hygiene.py` | no training-only manager keys and no Python term source in the emitted bundle |

Rollout parity **replays** mjlab's states rather than co-simulating: mjlab integrates with `mujoco_warp` while the browser runs MuJoCo's own WASM build, so a free-running comparison would measure MuJoCo against itself.


## Dependencies

Core: `mujoco==3.8.1`, `onnx>=1.20.0`, `nodeenv>=1.9.1`, `rich>=13.0.0`, `wandb>=0.23.1`, `typer>=0.12.0`.
Dev extras: `pyright`, `ruff==0.16.0` (pinned), `pre-commit`, `pytest`.
Examples extras: `mjlab==1.5.3`, `torch>=2.9.1`, `onnxruntime>=1.21.0`, `robot-descriptions`, `playground`, `myosuite`, `gymnasium`.

**`mjlab` + `torch` are build-time requirements for any policy carrying traced MDP terms** — tracing runs `torch.onnx.export` against a live mjlab env. Neither ships to the browser, and a model-only scene needs neither. `onnxruntime` (not `onnxruntime-web`) is the parity harness's runtime.

mjlab itself pulls in `mujoco-mjx==3.8.1` and `mujoco-warp>=3.8.0.3` (3.8.0.3 switched from `mjENBL_MULTICCD` to a `DisableBit`, restoring compat with stable mujoco 3.8.1).

Python 3.10–3.12 only (`labmaze`, transitive via myosuite → dm-control, has no cp313 wheel).


## Deployment

The demo app is built by `examples/demo/main.py` and deployed to GitHub Pages via the `deploy.yml` workflow on every push to `main` that touches relevant paths. The `MJSWAN_BASE_PATH` and `MJSWAN_NO_LAUNCH` env vars control the build. The GentleHumanoid and MuscleMimic demos are deployed to Cloudflare Pages. `mjswan publish` is the alternative to hosting a `dist/` at all.


## Design records

`docs/adr/` holds the ADRs, deliberately outside the published site (contributor documentation citing files and line numbers). ADR 0005 and its companion implementation brief are the ones to read before touching `compile/`, `_onnx_build.py`, or the frontend's manager layer — the brief carries a per-item status table including the things measured and **declined** (event fusion, input-tensor pre-allocation, `source_url` provenance, bit-for-bit replay), each with the reasoning. Do not re-add those as unfinished requirements.
