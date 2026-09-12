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
  licenses/            LICENSE / NOTICE files in the build (ADR 0007): naming rule, SPDX
                       identifier + tier table (mirrors Cloud's @mjswan/licenses), detection
                       beside a model, known-assets table, SPDX texts as package data
  project.py           ProjectConfig / ProjectHandle
  scene.py             SceneConfig / SceneHandle
  policy.py            PolicyConfig / PolicyHandle
  motion.py            MotionConfig / MotionHandle
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  command.py           Command terms (Slider, Button, Checkbox, velocity_command, ui_command)
  viewer.py            ViewerConfig
  trace_env.py         Minimal live envs for tracing (build_single_entity_trace_env)
  utils.py             ZIP-DEFLATE bundling, XML path rewriting, name2id slug + scoped unique ids
  mdp.py               MdpConfig: the five term sets a policy runs against, as one shared unit
  document.py          The .swn simulation document: list / pack / unpack the built tree,
                       and DOCUMENT_FORMAT, the layout version the manifest stamps
  wandb_io.py          W&B checkpoint / motion artifact downloads
  _onnx_build.py       Bridges term cfg dataclasses → compile/ tracer; writes .onnx + JSON
  _graph_io.py         Bundle path a traced graph gets, and the guarded write to it
  _cli.py              Typer-based `mjswan` CLI + legacy entry points
  _build_client.py     Frontend build orchestration (npm/vite, managed nodeenv)
  _compat.py           Deprecated pre-0.8 aliases (methods, classes, modules). Remove in 0.9
  compile/             ONNX tracing + numeric parity (ADR 0005)
    tracer.py            trace_term / trace_event_term / trace_command_term
    serialize.py         traced command → manifest entry + JSON schema
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
  └── .add_project(name, default=False) → ProjectHandle      # id = name2id(name)
        ├── .add_scene_mjlab(task_id, play=..., env_cfg=..., events=...) → SceneHandle
        └── .add_scene(name, model|spec, metadata, control_dt, events) → SceneHandle
              ├── .add_policy(name, policy, mdp=MdpConfig | the five term sets,
              │               in_keys=, out_keys=, ...) → PolicyHandle
              │     └── .add_motion(...) / .add_motion_wandb(...) → MotionHandle
              ├── .add_policy_wandb(run_path, ...) → list[PolicyHandle]   # one MdpConfig per call
              ├── .add_splat(name, source|url, ...) → SplatHandle
              ├── .set_viewer(ViewerConfig)
              ├── .set_events(events)             # the scene's default events for its policies
              └── .set_trace_env(env)             # required for a non-mjlab policy scene

builder.build(output_dir) → MjswanApp
MjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
MjswanApp.save_document(path=None) → Path    # the data as one .swn (ZIP of the tree)
MjswanApp.publish(title=..., tags=...)       # → mjswan Cloud; a .swn publishes too
```

Every project / scene / MDP / policy / splat has an `id = name2id(name)`, unique within its parent (collision → `<id>_1` + RuntimeWarning); ids are the directories in the build and the `?project=` / `?scene=` / `?policy=` values (ADR 0006 §4).

`Builder.from_mjlab(task_id, run_path=...)` is the one-liner shortcut for the common "visualize a single mjlab task" pattern; it delegates to the instance method `Builder.add_project_mjlab`, which creates a project, adds an mjlab scene, and optionally attaches all `model_*.pt` checkpoints from one or more W&B runs (converted to ONNX via mjlab+torch). For finer control, build manually: `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb(...)`.

Each `*Handle` wraps a `*Config` dataclass — the handle is the fluent API, the config is the serializable state.

**Two build-time requirements a policy scene has and a model-only scene does not:**

- `control_dt` — seconds per control step (mjlab's `timestep * decimation`). The model carries only the physics timestep, and a wrong control rate raises nothing at playback, so the builder requires it. `add_scene_mjlab` fills it in from the task.
- a **trace env** — tracing needs a live env to read shapes from and resolve `SceneEntityCfg` regexes against. `add_scene_mjlab` builds one at build time; a plain `add_scene` scene with traced terms needs `set_trace_env(build_single_entity_trace_env(spec_fn))` or the build raises.

The package's `__init__.py` is the canonical public API. Re-exports cover: `Builder` / `MjswanApp`; the five `*Handle` and `*Config` pairs; mjlab-compatible MDP cfgs (`ObservationGroupCfg`, `ObservationTermCfg`, `ActionTermCfg`, `JointPositionActionCfg`, `JointEffortActionCfg`, `TerminationTermCfg`); command UI (`SliderConfig`/`ButtonConfig`/`CheckboxConfig` and their `Slider`/`Button`/`Checkbox` aliases, `SliderRangeConfig`, `CommandTermConfig`, `CommandBinding`, `CommandUiConfig`, `ui_command`, `velocity_command`); the `MdpBinding` umbrella type and the `register_observation` / `register_event` / `register_termination` / `register_command` hooks; and `build_single_entity_trace_env`.


## Key modules

### `builder.py` — `Builder`
Main entry point. Accumulates `ProjectConfig` objects, calls `ClientBuilder` to invoke the Vite frontend build, then per scene: builds or reuses the trace env, traces each `MdpConfig` once via `_onnx_build` into `<project-id>/<scene-id>/mdp/<mdp-id>/`, writes the scene's DEFLATE-compressed ZIP (via `utils.to_zip_deflated`, since `mujoco.to_zip` stores entries uncompressed), `policy/<policy-id>.onnx` and `assets/` (motions, splats), and finally the one `manifest.json` at the output root — `{format, version, uses_custom_js, plugins?, projects}` (ADR 0006). A `config_path` sidecar contributes a checkpoint's own defaults to its policy entry; its slot tables are ignored with a warning, since `in_keys`/`out_keys` are declared on `add_policy` and checked against the network there and against the MDP's observation groups at build time.

### `app.py` — `MjswanApp`
Wraps a built `dist/` directory: the engine plus the expanded simulation document. `from_document()` takes either form — a directory is served where it sits, a `.swn` is unpacked to a temporary directory with the packaged engine (`_build_client.install_spa`) laid over it, refusing a custom-JS document whose plugin module ships with the engine. `launch()` starts a stdlib HTTP server (COOP/COEP headers required for SharedArrayBuffer / MuJoCo WASM threading); detects Google Colab and displays an inline iframe instead. `save_document()` writes the document as one `.swn` via `document.py`; `publish()` delegates to `publish.py`.

### `document.py` — the `.swn` simulation document
The built tree — `manifest.json` over `<project-id>/<scene-id>/` — *is* the document; `.swn` is that tree as a ZIP (manifest first, `.mjz`/`.npz`/`.spz` stored, the rest deflated). `document_files()` lists it off the manifest, never off what else is on disk, so the SPA's `assets/` is never mistaken for data; `unpack_document()` confines entries to the target; `as_directory()` gives `publish` and `mjswan info` one code path for either form (ADR 0006 §8).

### `publish.py` / `auth.py` — mjswan Cloud
`publish_dist()` takes a built directory or a `.swn` and uploads only *data* files (`.json`, `.mjz`, `.onnx`, `.npz`, `.ply`, `.spz`; 50 MB/file, 200 MB total, 64 files) via a presigned-upload protocol — never the compiled JS, since Cloud loads a pinned engine from its own CDN. It refuses a build with `uses_custom_js: true`, which Cloud cannot render. It also admits license files by name (`<project-id>/LICENSE` / `NOTICE`, `<project-id>/<scene-id>/LICENSE.<component>` / `NOTICE.<component>`; never the root `LICENSE`, the engine's) as `text/plain`, prints each with its identified license, warns for a restricted one and refuses one that forbids redistribution before any network call (ADR 0007 §3). `auth.py` is a `gh`-style loopback PKCE OAuth flow against Supabase's GitHub OAuth, persisting a rotating refresh token locally; `MJSWAN_TOKEN` bypasses it for CI.

### `licenses/` — license files in the build
The naming rule, the SPDX identifier (first-line tag, else the standard texts' headers, else the recognised non-redistributable licenses, else `LicenseRef-custom`) and the tier table (`notice` / `restricted` / `blocked`), kept in step by hand with mjswan Cloud's `@mjswan/licenses`. `detect_attributions(spec_asset_directories(spec))` copies `LICENSE*` / `COPYING*` / `NOTICE*` found beside a model or its meshes (nearest level wins, two parents at most, de-duplicated by content); `KNOWN_ASSETS` generates a tagged text for models that ship no file (Unitree, ANYbotics). `Attribution` is what a scene carries and `_save_web` writes; `declare()` is what `publish` and `info` read back (ADR 0007).

### `mdp.py` — `MdpConfig`
The five term sets (observations, actions, terminations, commands, events) as one unit, shared by identity: policies handed the same object share one MDP, traced once into `mdp/<mdp-id>/`. A named MDP is `name2id(name)`; one the `add_policy` term-set kwargs built for a single policy takes that policy's id (`mdp/locomotion/` beside `policy/locomotion.onnx`); one shared by hand, as `add_policy_wandb` does for a run's checkpoints, is `mdp_<n>` per scene in first-use order. The first policy to use one fills its unset fields from the scene's env config and adapts mjlab types in place (ADR 0006 §3).

### `policy.py` — `PolicyConfig` / `PolicyHandle`
Holds an `onnx.ModelProto`, its `MdpConfig`, the checkpoint's own metadata (`policy_joint_names`, `default_joint_pos`, `encoder_bias`, `clip_actions`), motion references, and the slot tables `in_keys` / `out_keys` — positional maps onto the network's inputs and outputs (`RUNTIME_INPUT_SLOTS` names the tensors the runtime synthesizes: `is_init`, `adapt_hx`, `time_step`). Serialized as one entry of the scene's `policies` list in `manifest.json`, pointing at `policy/<id>.onnx` and at its MDP by id; a table equal to the default (`["actor"]`, `["action"]`) is omitted.

### `command.py`
Defines command terms consumed by policies: `SliderConfig` (with `enabled_when` and an `adjustable_range` → `SliderRangeConfig` companion slider), `CheckboxConfig`, `ButtonConfig`, `CommandUiConfig`, `CommandTermConfig`, `CommandBinding`, and the `CommandInput` union. `velocity_command()` builds the standard locomotion 3-DoF velocity command; `ui_command()` builds a generic UI-driven term. A real mjlab command class (velocity, lifting, tracking RSI jitter) is traced like any other term — its hidden state is promoted to explicit graph I/O — and registered with `register_command`. `default_viz()` restates what mjlab's `_debug_vis_impl` draws for mjlab's own command classes as data (`core/command/debugViz.ts` evaluates it), so a `debug_vis=True` task gets its arrows and markers without the author declaring any; one mjswan has no drawing for warns at build time. The panel lists each drawing term in its **Debug Viz** section (`engine.debugVis.set`), on by default as mjlab's viewers are.

### `scene.py` — `SceneConfig` / `SceneHandle`
A scene owns one MuJoCo model (as `MjModel` → binary `.mjb` or `MjSpec` → XML), its `control_dt`, its default events (what a policy's MDP gets when it declares none), its trace env, the `MdpConfig`s its policies use (registered on first use, which fixes their ids), zero or more policies, zero or more Gaussian splat backgrounds, and its `attributions`: the third-party components written as `LICENSE.<component>` / `NOTICE.<component>` (detected at `add_scene` or declared with `add_attribution`). `add_policy` checks a policy's slot tables against its network: a multi-input network without `in_keys` is refused there.

### `splat.py` — `SplatConfig` / `SplatHandle`
Configures a 3D Gaussian Splat (`.spz` format) background: scale, position offsets, Euler rotations, optional collider mesh URL, and a `control` flag exposing live calibration sliders.

### `viewer.py` — `ViewerConfig`
Camera parameters (lookat, distance, fovy, elevation, azimuth) + tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

### `trace_env.py`
`build_single_entity_trace_env(spec_fn)` builds a minimal single-entity `ManagerBasedRlEnv` out of mjlab's own `Entity`/`Scene` — no reimplemented kinematics. It configures no managers and is never stepped; it is only the tracer's `env.scene[name].data.<field>` read/write target. Joint defaults come from the model's first keyframe, matching what the browser resets to. `build_mjlab_env()` grows `nconmax`/`njmax` until a task's env fits a single-env re-use. `TraceCommandManager` supplies trace-time stand-ins for commands the browser owns (a `ui_command` has no Python side).

### `compile/` — the tracer (ADR 0005)
- `tracer.py`: runs each `func(env, **params)` once against a recording proxy to discover its reads, classifies each as time-varying state (a graph input, or "slot") or a model-derived constant (baked in), then exports an `nn.Module` via `torch.onnx.export`. Four slot namespaces: a raw `mjData` field (`sim`), an entity `data` field, a named sensor's `sensordata` window, and a live command's state field. Raw `mjData` is the foundation: an `EntityData` property the browser reads natively (`READER_FIELDS`) is one value slot — the shortcut — and any other property is traced through a copy of the real `EntityData` whose `data` is the sim proxy, so mjlab's property math lands in the graph and its raw reads become `sim` slots. A `sim` slot is narrowed to the rows the term indexes when discovery can tell which (the raw field is handed out as a tensor subclass that notes each `[:, ids]` read; replay remaps those ids to positions, so a read of exactly the rows is the input itself and no Gather is emitted), unioned across a group's terms, and shipped whole otherwise. Structured sensors (`RayCastSensor`) contribute one slot per field read. Only `_STATIC_DATA_FIELDS` are treated as constant — anything unrecognized errs toward a graph input, so a missing runtime input fails loudly instead of returning stale values.
- `serialize.py`: a traced command's manifest entry. One generic browser-side `OnnxCommand` handler interprets every command from this data, so the entry declares state fields (names, shapes, **and initial values**), `rand_dim`, dynamic reads, and any `entity_write`.
- `rng.py`: build-time RNG spy/replay. Patches the term function's *own* module globals (mjlab binds the name at import time), records mjlab's real draws, replays those exact values into the graph's `rand` input so parity does not diverge on randomness alone. Unrelated to the runtime's seeded PRNG.
- `parity.py`: steps a live env with a seeded action sequence and feeds the same raw state through each exported graph via `onnxruntime` (not torch), asserting `allclose` for every term at every step.

### `_onnx_build.py`
Bridges the term config dataclasses to `compile/`. Traces each plain-callable body against the scene's live env, writes `.onnx` bytes under `<scene_dir>/mdp/<mdp-id>/{obs,term,command,event}/`, and returns the manifest-shaped JSON entry. Emits **fused** graphs: one per observation group (clip-then-scale folded in, mjlab's order) and one per termination group (a bool lane per term, so per-term reset reasons survive). A group with a legacy `*Binding` term or per-term history does not fuse; a lone traced termination is deliberately left unfused. Startup DR that perturbs `mjModel` rather than `mjData` (`geom_friction`, `body_com_offset`, `geom_rgba`) carries no graph — it emits a descriptor the browser applies once from the seeded PRNG, keyed by entity *names* since the browser compiles its own model.

### `_graph_io.py`
The one place that names a traced graph's file and writes it. A ref is the path the manifest carries, resolved by the browser against the scene directory, so the string and the location on disk are the same thing. Graphs are scoped `mdp/<mdp-id>/`: a group or term name is unique within an MDP but not within a scene, and nearly every MDP names its one observation group `actor` — the default slot (`mjlab_adapter.DEFAULT_OBS_GROUP_KEY`). The write refuses to replace a *different* graph already at one path, so a serializer that ever loses its scope fails the build instead of shipping a document where one MDP loads another's graph.

### `adapters/`
- `mjlab_adapter.py`: Converts mjlab types to mjswan equivalents. Obs/term/event bodies are *not* name-resolved to mirrors any more — mjlab's own functions are traced directly. What remains is config-shape adaptation, reducing mjlab's network-keyed observation dict to the actor's group via the task's runner config (`rl_cfg.obs_groups`: only groups it attributes to a network are renamed or dropped, so an author's extra slots survive), `clip_actions`, and action-scale resolution.
- `gui_spy.py`: runs mjlab's own `CommandTerm.create_gui` against a recording stand-in, so the browser control panel's slider ranges come from mjlab's declaration instead of a hand-copied duplicate that drifts.
- `mjlab_compat.py`: Monkey-patches `MujocoCfg.apply_to_spec()` onto mjlab when needed.

### `envs/mdp/` and `managers/`
mjlab-compatible MDP layer, with a deliberate asymmetry:

- **`envs/mdp/actions/`** carries *real, directly usable* config classes, because Action is permanently native (ADR 0005 §7): `JointPositionActionCfg`, `JointEffortActionCfg`, `MuscleActivationActionCfg` are supported; `JointVelocityActionCfg`, `TendonLength/Velocity/EffortActionCfg` and `SiteEffortActionCfg` are exported so mjlab configs import cleanly but raise `NotImplementedError` at build time. `stiffness`/`damping` are mjswan-specific — the browser computes PD externally for motor actuators with `biastype=none`.
- **`envs/mdp/{observations,terminations,events}.py`** carry only the `*Binding` escape hatch and its `register_*` registry. Pass mjlab's own functions instead. The one event mjswan owns is `reset_root_state_on_flat_patch`, which replaces mjlab's untraceable `reset_root_state_from_flat_patches` (it draws with `torch.randint` and indexes per-env terrain tensors); `apply_terrain_spawn` swaps it in, since the browser runs one env where mjlab spreads many across the terrain.
- **`managers/`** holds the config-side counterparts (`observation_manager`, `event_manager`, `action_manager`, `termination_manager`). `ObservationTermCfg` adds `history_steps` (sparse look-back offsets) and `history_interleaved` (Isaac joint-major layout) beyond mjlab's fields; training-only fields (`noise`, `delay_*`, `enable_corruption`) are accepted and ignored.

A term with no browser implementation **fails the build**, naming both ways out (a trace-friendly `register_*` replacement callable, or a `ts_src` TS class). Three event terms are exempt because there is provably nothing to write: `randomize_terrain`, `encoder_bias`, and a root-state write onto a fixed-base entity.

**Muscle action term.** `MuscleActivationActionCfg` drives MuJoCo muscle actuators. `action_mode` (spelled as mjlab/myosuite spell it, so the adapter copies it across) picks the mapping from `raw = scale·a + offset`: `sigmoid` (default) applies the canonical MyoSuite sigmoid `σ(5(raw − 0.5))` for excitation in (0, 1); `excitation` clips `raw` to [0, 1]; `direct` clips `raw` to each actuator's own `ctrlrange` and writes it unchanged, for a checkpoint trained against the raw control (a myosuite muscle model declares `ctrlrange="-1 1"` and uses the negative half). `normalize` is the legacy spelling of the first two. The mjlab adapter translates `MyoMuscleActivationActionCfg` (the class actually used by every myo* mjlab task) to `MuscleActivationActionCfg`; see [docs/adr/0002](./docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md).

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
- Loads the MuJoCo WASM module and steps physics on the main thread, yielding to the browser each control step (`engine/yieldToBrowser.ts`).
- Runs the policy **and every traced MDP term body** via onnxruntime-web — the policy on WebGPU where the browser has it (ORT falls back to wasm itself), term graphs always on wasm since each is small and a step runs many.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR), including tracked hands as bodies inside the sim (opt-in).
- Reads `manifest.json` to discover projects/scenes/MDPs/policies at runtime; `?project=` / `?scene=` / `?policy=` take ids (`sanitizeName` = `name2id`, pinned by a shared case table).

`src/core/` mirrors mjlab's layout — `observation/`, `termination/`, `action/`, `event/`, `command/` — plus `onnx/` (`session.ts` caches split by lifetime, `runQueue.ts`, `slotReader/` serving graph inputs from `mjModel`/`mjData` — `indexing.ts` resolves an entity's elements as mjlab's `EntityIndexing` does, `fields/` holds every `EntityData` property as a reader, one file per section of mjlab's `entity/data.py`, and `index.ts` dispatches a slot by namespace — `raycast.ts` casting height-scan rays with `mj_ray`, `contact.ts`, `graphRefs.ts`), `rng.ts` (xoshiro128\*\*, snapshot-able), `policy/`, `scene/`, `xr/` (`handMocap.ts` injects a capsule per hand bone and writes `mocap_pos`/`mocap_quat`), and `engine/` (`runtime.ts` step loop, `yieldToBrowser.ts`, `resetChain.ts`, `viewer_config.ts`). Each manager is native orchestration around ONNX term bodies: `FusedObservation`/`OnnxObservation`/`NativeObservation`/`HistoryObservation`, `FusedTermination`/`OnnxTermination`/`TimeOutTermination`, `OnnxEvent` + `triggers.ts` + `entityWrite.ts` + `modelFieldDr.ts`, `OnnxCommand`. Action is fully native (`action/applyAction.ts`).

The step loop follows mjlab's `ManagerBasedRlEnv.step` ordering exactly — forward → command → event → obs → action → physics → term → reset → forward — with **one** `mj_forward` per iteration. The reset chain mirrors `_reset_idx`: `event(mode="reset")` → observation/action reset → command resample, all awaited in config order (writes are last-writer-wins by config order, so concurrency would make that machine-dependent).

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).

The template has three Vite build outputs (all written to `template/dist/`):
- **SPA** (`vite.config.ts`, `npm run build:spa`) — the standalone app the Python `Builder` assembles. Entry `src/index.tsx`.
- **Library** (`vite.lib.config.ts`, `npm run build:lib`) — a single self-contained ESM `dist/mjswan.js` exposing `createEngine(element, options?)` (entry `src/engine/index.ts`), consumed by mjswan Cloud from a CDN. Every dependency is bundled (no bare imports) and the MuJoCo/ONNX WASM is emitted into `dist/assets/`, referenced via `new URL('./assets/x.wasm', import.meta.url)`. Vite lib mode force-inlines those WASM as base64; a `generateBundle` plugin extracts them back out, recognizing each by content digest so it keeps its upstream basename (mjswan-cloud ADR 0001). The SPA build names files the same way into the same directory, so identical bytes land on one path instead of shipping 40 MiB twice; only `mjswan.js` sits at the root, addressed from outside as the npm entry. `src/core/onnx/ortEnv.ts` points `ort.env.wasm.wasmPaths` at ORT's copy, so a host serving `dist/` fetches nothing third-party (issue #123). `vite.wasm.ts` carries the sources and the ORT assumptions this rests on.
- **Manifest** (`vite.manifest.config.ts`) — `dist/manifest.js`, the `mjswan/manifest` catalog parser as a standalone CDN-loadable ESM.

`npm run build` runs all of them. The public engine API (ADR 0004) is bytes-in / snapshot-out: `loadScene` / `setPolicy` / `setSplat` / `setMotion`, `camera` / `commands` verbs, `subscribe`, `captureThumbnail`, `dispose`, with `termSeed` in and out for session replay. It never fetches — `mjswan/manifest` (`parseManifest(manifest, byteSource)`) turns a `manifest.json` into a lazy catalog (refusing a `format` newer than it knows) and the app owns the fetching. A policy switch is an MDP switch: the runtime restores the model fields the previous MDP's startup randomization changed, reseeds the term PRNG, builds the new MDP's `EventManager` and runs its startup events (ADR 0006 §9).


## CLI entry points

The primary CLI is `mjswan` (Typer-based, defined in `_cli.py:app`). Subcommands:

| Subcommand | Description |
|------------|-------------|
| `mjswan view <model.xml>` | Build and launch a viewer for a MuJoCo XML/MJCF file |
| `mjswan serve <dist-dir \| document.swn>` | Serve a pre-built `dist/` directory, or a `.swn` document |
| `mjswan new <name> [--template hello-world\|policy\|mjlab]` | Scaffold a new project from a template |
| `mjswan demo [name]` / `--list` | Run a built-in demo (`simple`, `main`, `mjlab`) |
| `mjswan info <dist-dir \| document.swn>` | Show a tree of projects/scenes/policies and asset sizes |
| `mjswan publish <dist-dir>` | Upload a built dist's data files to mjswan Cloud (rejects custom-JS builds) |
| `mjswan login` / `whoami` / `logout` | mjswan Cloud session (loopback GitHub OAuth) |

Legacy entry points (kept for backward compatibility): `main`, `simple`, `mjlab`, `serve <dist-dir>` — each runs the corresponding `examples/` module.


## Tooling and workflow

| Tool | Purpose |
|------|---------|
| `uv` | Dependency management and script runner — use instead of bare `python`/`pip` |
| `hatchling` | Build backend |
| `ruff` | Linting and formatting (pinned exactly, not floored — see pyproject comment) |
| `pyright` / `ty` | Type checking (`make type` runs both over `src/mjswan`; `examples/` and `typings/` are out of scope, the latter a search path for ty) |
| `pytest` | Tests (`make test`) |
| `pre-commit` | Hooks: trailing-whitespace, end-of-file-fixer, ruff, ruff-format, npmrc secret scan, pytest (not slow), eslint |
| `zensical` | Docs site builder (MkDocs-based) — `make docs-build` / `make docs-serve` |

Key Makefile targets: `sync`, `format`, `type`, `check`, `test`, `test-all`, `docs-build`, `docs-serve`.


## Tests and CI

`@pytest.mark.slow` triggers a full frontend (npm + Vite) build and is excluded from pre-commit (`pytest -m "not slow"`); unmarked tests are fast and always run.

Workflows: `pytest.yml` runs `pytest -m "not slow"` across Python 3.10 / 3.11 / 3.12 with the `dev` extra only. `parity.yml` is separate and heavier — it installs the `examples` extras (~2 GB), caches warp's CPU kernels, and runs the ONNX numeric-parity sweep over the reference mjlab tasks on `workflow_dispatch` plus weekly. Weekly because the cadence targets *upstream* drift: mjlab moves fast, and a changed term definition would otherwise surface as a mysterious runtime difference rather than a failing check. Trigger it by hand for a change to `compile/` or the serializers. `frontend.yml` lints the template and runs its vitest suite, sharing one dependency install. Also: `ruff.yml`, `deploy.yml`, `publish-pypi.yml`, `publish-npm.yml`, `release.yml`, `sync-contributors.yml`.

Parity is layered, and each layer covers something the others cannot:

| Check | What it proves |
|---|---|
| `tests/test_onnx_parity.py` | each traced graph reproduces its mjlab term (7 tasks, 73 terms, worst max\|Δ\|≈9e-08) |
| `tests/test_onnx_command_parity.py` | every reference task's traced command over 16 replayed draws |
| `core/onnx/__tests__/slotReaderParity.test.ts` | the reader hands those graphs mjlab's numbers, for every `EntityData` property and each entity's `EntityIndexing` (fixture from `dump_slot_fixture.py`: two tasks and a synthetic tendon-driven entity) |
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

`docs/adr/` holds the ADRs, deliberately outside the published site (contributor documentation citing files and line numbers). ADR 0005 and its companion implementation brief are the ones to read before touching `compile/`, `_onnx_build.py`, or the frontend's manager layer — the brief carries a per-item status table including the things measured and **declined** (event fusion, input-tensor pre-allocation, `source_url` provenance, bit-for-bit replay), each with the reasoning. Do not re-add those as unfinished requirements. [ADR 0006](./docs/adr/0006-swn-simulation-document.md) supersedes ADR 0005 §1: the build output becomes a `.swn` document with a single root `manifest.json`, an `MdpConfig` unit that owns all five managers (events included), and a `<project-id>/<scene-id>/` layout. It is design-stage — the code still writes `assets/config.json`.
