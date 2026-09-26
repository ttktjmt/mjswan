# CONTEXT.md

## What mjswan is

mjswan is a Python framework that packages browser-based MuJoCo simulations with real-time ONNX policy control into interactive static web apps. The Python side builds the bundle; the browser client is TypeScript/React/three.js running mujoco-wasm for physics and onnxruntime-web for inference. Published on PyPI and npm; demos hosted on GitHub Pages and Cloudflare Pages, with optional one-command publishing to mjswan Cloud.

The defining architectural fact (ADR 0005): mjswan reimplements none of mjlab's MDP term functions. A task's **real Python function objects** — observations, terminations, events, commands — are traced to ONNX at build time with `torch.onnx.export` and run in the browser by ONNX Runtime Web, beside the policy. Only Action is native TypeScript. Anything that cannot be traced fails the build loudly rather than being silently dropped.


## Repository layout

```
src/mjswan/          Python package source: the object model at the root, one package per
                     layer below it (rules and the old → new table: docs/adr/0008)
  __init__.py          The public API: Builder / MjswanApp, the *Handle / *Config pairs, cfgs
  builder.py           Builder: the fluent entry point; build() hands off to build/pipeline.py
  app.py               MjswanApp: launch / serve / save_document / publish a built app
  project.py           ProjectConfig / ProjectHandle (add_scene, add_scene_mjlab, set_license)
  scene.py             SceneConfig / SceneHandle (add_policy, add_policy_wandb / _hf, splats)
  policy.py            PolicyConfig / PolicyHandle; the ONNX slot and width checks a policy passes
  motion.py            MotionConfig / MotionHandle; attaching a tracking task's clip
  splat.py             SplatConfig / SplatHandle (Gaussian Splat)
  viewer.py            ViewerConfig
  mdp.py               MdpConfig: the five term sets a policy runs against, as one shared unit
  cli.py               Typer-based `mjswan` CLI
  _version.py          __version__ (read by hatch; the only underscore module at the root)
  build/               What Builder.build() does; the one place that knows the manifest's shape
    pipeline.py          write_app: frontend build, then per scene pack → assets → trace → entries
    manifest.py          project / scene / MDP / policy entries, and the manifest.json write
    asset.py             a scene's assets/: motion clips (deduplicated by content) and splats
    mjz.py               scene.mjz: MjSpec plus every asset, DEFLATE-compressed, paths rewritten
    frontend.py          the Python side of template/: nodeenv + Vite build, install_spa
    mdp/                 one .onnx per traced term and the JSON entry that names it,
                         observation · termination · event · command · action, with graph.py
                         (where a graph lands, the guarded write), provenance, sensor, binding
  compile/             ONNX tracing + numeric parity (ADR 0005); returns bytes, writes no file
    slot · proxy · record · replay · export      the two passes and what they share
    term · event · command · group · native      one trace per term kind; terms needing none
    parity.py            live mjlab env vs exported graphs, term by term, step by step
    rng.py               build-time RNG spy/replay (DrawRecorder / ReplayRng)
  mjlab/               Everything that reads mjlab's own objects (mjlab imported lazily)
    observation · termination · command · action · event   per-manager adapters
    task.py · runner.py · env.py · sim.py · gui.py · onnx_meta.py · detect.py
  source/              A reference in, local files out: wandb.py, hf.py (file, dir, ONNX)
  document/            The .swn document (ADR 0006): container.py, manifest.py, ids.py
  cloud/               mjswan Cloud: publish.py, auth.py (loopback PKCE OAuth), transport.py
  license/             LICENSE / NOTICE files in the build (ADR 0007): path, spdx, attribution
  managers/            mjlab-mirror cfgs: observation / event / action / termination / command
  envs/mdp/            mjlab-mirror MDP side: actions/ (real cfgs), the *Binding escape hatches,
                       commands.py (ui_command presets, trace-friendly rewrites, mjlab bindings)
  template/            TypeScript frontend (Vite + React + three.js + mujoco-wasm)

examples/            Runnable examples. Python only: assets are fetched at run time
  demo/                main.py (the deployed app: mjlab Tasks + Showcase, everything
                       from mjlab or the Hub), simple.py (one task, one checkpoint),
                       minimum_policy.py (the smallest complete policy) and
                       mujoco_models.py (a gallery, no policy). `mjswan demo` runs
                       the three named ones; minimum_policy.py is read, not run
  colab/               Google Colab notebooks (demo, anymal_c_velocity)

                     The W&B-driven examples (defaults, g1_spinkick, myosuite,
                     musclemimic, unitree_rl, gentle_humanoid) live in
                     mjswan_playground, which carries their pinned dependencies

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
              ├── .set_events(events)             # the scene's default events for its policies
              └── .set_trace_env(env)             # required for a non-mjlab policy scene

builder.build(output_dir) → MjswanApp
MjswanApp.launch(host, port, open_browser)   # blocking; Colab-aware
MjswanApp.save_document(path=None) → Path    # the data as one .swn (ZIP of the tree)
MjswanApp.publish(title=..., tags=...)       # → mjswan Cloud; a .swn publishes too
```

Every project / scene / MDP / policy / splat has an `id = name2id(name)`, unique within its parent; ids are the directories in the build and the `?project=` / `?scene=` / `?policy=` values (ADR 0006 §4). On a collision the second is renamed `<name>_1` with id `<id>_1` and a RuntimeWarning (`document.ids.assign_name`), so the viewer never lists two alike and the id stays `name2id(name)`; a policy's motions, which have no id, are renamed the same way.

`Builder.from_mjlab(task_id, run_path=...)` is the one-liner shortcut for the common "visualize a single mjlab task" pattern; it delegates to the instance method `Builder.add_project_mjlab`, which creates a project, adds an mjlab scene, and optionally attaches all `model_*.pt` checkpoints from one or more W&B runs (converted to ONNX via mjlab+torch). For finer control, build manually: `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb(...)`. `hf_repo_id=` is the same shortcut against a Hugging Face Hub repository, which holds an exported ONNX rather than training state and so needs no conversion.

Each `*Handle` wraps a `*Config` dataclass — the handle is the fluent API, the config is the serializable state.

**Two build-time requirements a policy scene has and a model-only scene does not:**

- `control_dt` — seconds per control step (mjlab's `timestep * decimation`). The model carries only the physics timestep, and a wrong control rate raises nothing at playback, so the builder requires it. `add_scene_mjlab` fills it in from the task.
- a **trace env** — tracing needs a live env to read shapes from and resolve `SceneEntityCfg` regexes against. `add_scene_mjlab` builds one at build time; a plain `add_scene` scene with traced terms needs `set_trace_env(build_single_entity_trace_env(spec_fn))` or the build raises.

The package's `__init__.py` is the canonical public API. Re-exports cover: `Builder` / `MjswanApp`; the five `*Handle` and `*Config` pairs; mjlab-compatible MDP cfgs (`ObservationGroupCfg`, `ObservationTermCfg`, `ActionTermCfg`, `JointPositionActionCfg`, `JointEffortActionCfg`, `TerminationTermCfg`); command UI (`SliderConfig`/`ButtonConfig`/`CheckboxConfig` and their `Slider`/`Button`/`Checkbox` aliases, `SliderRangeConfig`, `CommandTermConfig`, `CommandBinding`, `CommandUiConfig`, `ui_command`, `velocity_command`); the `MdpBinding` umbrella type and the `register_observation` / `register_event` / `register_termination` / `register_command` hooks; and `build_single_entity_trace_env`.


## Key modules

Five layers, and imports only point downward ([ADR 0008](./docs/adr/0008-package-layout.md)):
`cli.py` → `app.py` / `cloud/` → the object model at the root → `build/` → `compile/` →
`mjlab/` · `source/` → `managers/` · `envs/mdp/` · `document/` · `license/`. The root imports
the build pipeline and `compile/` lazily, and `mjlab/` imports mjlab only inside its
functions, so `import mjswan` never costs a Node build and, in a core install, loads no
source backend (`tests/test_core_install.py`). With the `mjlab` extra installed it does
load torch and mjlab: `envs/mdp/` binds mjlab's samplers at module level for the RNG spy.
mjswan's own package and module names are singular; the mjlab-mirror paths
(`managers/observation_manager.py`, `envs/mdp/observations.py`) keep mjlab's spelling because
the import path is the contract. The same short name recurring under `mjlab/`, `build/mdp/`
and `compile/` (`observation`, `termination`, `event`, `command`, `action`) is deliberate: one
manager kind, one module per stage: always read qualified by its package.

### `builder.py` — `Builder`
Main entry point. Accumulates `ProjectConfig` objects; `build()` resolves the output path
against the caller's file and hands the projects to `build.pipeline.write_app`.
`from_mjlab` / `add_project_mjlab` are the one-liner for a single mjlab task.

### `build/`: what `Builder.build()` does
- `pipeline.py`: `write_app` builds the frontend first (`ClientBuilder`, reusing a cached SPA when it matches), lays the engine down (`install_spa`), then per scene: runs pending checkpoint conversions, validates muscle action terms, writes the scene as DEFLATE-compressed `scene.mjz` (`mjz.to_zip_deflated`, since `mujoco.to_zip` stores entries uncompressed) or `.mjb`, copies clips and splats into `assets/` (`asset.py`; a tracking task's env is pointed at the bundled clip before it is built), builds or reuses the trace env, traces each `MdpConfig` once into `mdp/<mdp-id>/`, writes `policy/<policy-id>.onnx`, and finally the one `manifest.json` at the output root: `{format, version, uses_custom_js, plugins?, projects}` (ADR 0006). `write_mt_headers` writes the COOP/COEP `_headers` file for `mt=True`.
- `manifest.py`: the entries `mdp_entry` (calls `build.mdp`), `policy_entry`, `scene_entry` and `splat_entry`, plus `write_manifest`. A `config_path` sidecar (`load_sidecar`) contributes a checkpoint's own defaults to its policy entry; its slot tables are ignored with a warning, since `in_keys` / `out_keys` are declared on `add_policy` and checked against the network there and against the MDP's observation groups here.
- `mdp/`: one module per term kind, each tracing plain-callable bodies via `compile/` against the scene's live env, writing `.onnx` bytes under `<scene_dir>/mdp/<mdp-id>/{obs,term,command,event}/` (`graph.py`: the ref the manifest carries *is* the path on disk, scoped by MDP so two MDPs' `actor` groups cannot collide, and the write refuses to replace a *different* graph already at one path) and returning the manifest-shaped entry. Emits **fused** graphs: one per observation group (clip-then-scale folded in, mjlab's order) and one per termination group (a bool lane per term, so per-term reset reasons survive). A group with a `*Binding` term or per-term history does not fuse; a lone traced termination is deliberately left unfused. Startup DR that perturbs `mjModel` rather than `mjData` (`geom_friction`, `body_com_offset`, `geom_rgba`, `geom_size`) carries no graph: `event.py` emits a descriptor the browser applies once from the seeded PRNG, keyed by entity *names* since the browser compiles its own model. `provenance.py` records which function each entry came from (`func` / `doc` / `params`, and the `metadata_props` stamped into the graph); `sensor.py` describes the raycast and contact sensors a graph reads; `action.py` merges each action term's `to_dict()` over the sidecar's authored block (a motor robot's PD gains).
- `frontend.py`: the Python side of `template/`, which stays where npm, Vite, CI and hatch expect it. Manages a local `nodeenv` if Node isn't available system-wide, runs the Vite build, bundles author `ts_src` terms into `plugins.js` with esbuild; `uses_custom_js` is what the manifest's flag reads.

### `app.py` — `MjswanApp`
Wraps a built `dist/` directory: the engine plus the expanded simulation document. `from_document()` takes either form: a directory is served where it sits, a `.swn` is unpacked to a temporary directory with the packaged engine (`build.frontend.install_spa`) laid over it, refusing a custom-JS document whose plugin module ships with the engine. `launch()` starts a stdlib HTTP server (COOP/COEP headers required for SharedArrayBuffer / MuJoCo WASM threading); detects Google Colab and displays an inline iframe instead. `save_document()` writes the document as one `.swn` via `document/`; `publish()` delegates to `cloud.publish`.

### `document/`: the `.swn` simulation document
The built tree (`manifest.json` over `<project-id>/<scene-id>/`) *is* the document; `.swn` is that tree as a ZIP (manifest first, `.mjz`/`.npz`/`.spz` stored, the rest deflated). `container.py`: `document_files()` lists it off the manifest, never off what else is on disk, so the SPA's `assets/` is never mistaken for data; `unpack_document()` confines entries to the target; `as_directory()` gives `publish` and `mjswan info` one code path for either form (ADR 0006 §8). `manifest.py`: `DOCUMENT_FORMAT`, `MANIFEST_NAME` and the slot defaults (`DEFAULT_IN_KEYS`, `DEFAULT_OUT_KEYS`, `RUNTIME_INPUT_SLOTS`): the format constants `build/` writes and `app`/`cloud` read. `ids.py`: `name2id()` is the lowercase-underscore slug every project / scene / MDP / policy id derives from, `unique_id` the scoped `_N` suffix, `assign_name` the rename every listed object takes (name and id together, with a RuntimeWarning), and `assign_id` the id-only form MDP ids take (ADR 0006 §4).

### `cloud/`: mjswan Cloud
`publish.py`: `publish_dist()` takes a built directory or a `.swn` and uploads only *data* files (`.json`, `.mjz`, `.onnx`, `.npz`, `.ply`, `.spz`; 50 MB/file, 200 MB total, 64 files) via a presigned-upload protocol, never the compiled JS, since Cloud loads a pinned engine from its own CDN. It refuses a build with `uses_custom_js: true`, which Cloud cannot render. It also admits license files by name (`<project-id>/LICENSE` / `NOTICE`, `<project-id>/<scene-id>/LICENSE.<component>` / `NOTICE.<component>`; never the root `LICENSE`, the engine's) as `text/plain`, prints each with its identified license, warns for a restricted one and refuses one that forbids redistribution before any network call (ADR 0007 §3). `auth.py` is a `gh`-style loopback PKCE OAuth flow against Supabase's GitHub OAuth, persisting a rotating refresh token locally; `MJSWAN_TOKEN` bypasses it for CI. `transport.py` holds the user agent both send.

### `license/`: license files in the build
`path.py` is the naming rule, `spdx.py` the identifier (first-line tag, else the standard texts' headers, else the recognised non-redistributable licenses, else `LicenseRef-custom`) and the tier table (`notice` / `restricted` / `blocked`), kept in step by hand with mjswan Cloud's `@mjswan/licenses`, plus the six generatable SPDX texts under `templates/`. `attribution.py`: `detect_attributions(spec_asset_directories(spec))` copies `LICENSE*` / `COPYING*` / `NOTICE*` found beside a model or its meshes (nearest level wins, two parents at most, de-duplicated by content); `KNOWN_ASSETS` generates a tagged text for models that ship no file (Unitree, ANYbotics). `Attribution` is what a scene carries and the build writes; `declare()` is what `publish` and `info` read back (ADR 0007).

### `mdp.py` — `MdpConfig`
The five term sets (observations, actions, terminations, commands, events) as one unit, shared by identity: policies handed the same object share one MDP, traced once into `mdp/<mdp-id>/`. A named MDP is `name2id(name)`; one the `add_policy` term-set kwargs built for a single policy takes that policy's id (`mdp/locomotion/` beside `policy/locomotion.onnx`); one shared by hand, as `add_policy_wandb` does for a run's checkpoints, is `mdp_<n>` per scene in first-use order. The first policy to use one fills its unset fields from the scene's env config and adapts mjlab types in place (ADR 0006 §3).

### `policy.py` — `PolicyConfig` / `PolicyHandle`
Holds an `onnx.ModelProto`, its `MdpConfig`, the checkpoint's own metadata (`policy_joint_names`, `default_joint_pos`, `encoder_bias`, `clip_actions`), motion references, and the slot tables `in_keys` / `out_keys`, positional maps onto the network's inputs and outputs (`RUNTIME_INPUT_SLOTS` names the tensors the runtime synthesizes: `is_init`, `adapt_hx`, `time_step`). Also the checks a network passes on `add_policy`: `check_slot_tables` against its inputs and outputs, `onnx_output_width` against the actuated joints. Serialized as one entry of the scene's `policies` list in `manifest.json`, pointing at `policy/<id>.onnx` and at its MDP by id; a table equal to the default (`["actor"]`, `["action"]`) is omitted. `add_motion_wandb` / `add_motion_hf` call `source.*`.

### `managers/command_manager.py` and `envs/mdp/commands.py`: command terms
`command_manager.py` defines the configs consumed by policies: `SliderConfig` (with `enabled_when` and an `adjustable_range` → `SliderRangeConfig` companion slider), `CheckboxConfig`, `ButtonConfig`, `CommandUiConfig`, `CommandTermConfig`, `CommandBinding`, the `CommandInput` union, and `register_command`, which binds an mjlab command-cfg *class name* to a `CommandBinding`. `CommandTermConfig` keeps mjswan's own name: it is not mjlab's `CommandTermCfg`, since it also carries the browser UI and the pending trace. `envs/mdp/commands.py` has the terms mjswan supplies (`velocity_command()` builds the standard locomotion 3-DoF slider command, `ui_command()` a generic UI-driven term) and the bindings for mjlab's: `UniformVelocityCommandCfg` through a trace-friendly rewrite of its body (a real mjlab command class is traced like any other term, its hidden state promoted to explicit graph I/O), `MotionCommandCfg` as the native `TrackingCommand` with only its reset jitter traced. A binding's `viz` restates what mjlab's `_debug_vis_impl` draws as data (`core/command/debugViz.ts` evaluates it), so a `debug_vis=True` velocity task gets its arrows without the author declaring any; a term with no drawing warns at build time. A command class only one task uses (`LiftingCommandCfg`) is registered by that task's port, as `examples/demo/main.py` does. The panel lists each drawing term in its **Debug Viz** section (`engine.debugVis.set`), on by default as mjlab's viewers are.

### `scene.py` — `SceneConfig` / `SceneHandle`
A scene owns one MuJoCo model (as `MjModel` → binary `.mjb` or `MjSpec` → XML), its `control_dt`, its default events (what a policy's MDP gets when it declares none), its trace env, the `MdpConfig`s its policies use (registered on first use, which fixes their ids), zero or more policies, zero or more Gaussian splat backgrounds, and its `attributions`: the third-party components written as `LICENSE.<component>` / `NOTICE.<component>` (detected at `add_scene` or declared with `add_attribution`). `add_policy_wandb` fetches a run's checkpoints (`source.wandb.fetch_checkpoints`) and converts each with `mjlab.runner.export_checkpoint`; `add_policy_hf` fetches a Hub repository's `.onnx` (`source.hf`) and fills the joint mapping from the task's action terms, else from `mjlab.onnx_meta`. Each call opens the scene on the highest-step checkpoint it added unless the scene already has a default, and a later `add_policy(default=True)` takes over. `add_policy_wandb(only_latest=False)` converts at build time but checks for wandb, mjlab and torch when called, and skips, with a RuntimeWarning, a checkpoint an earlier run in the list already added.

### `splat.py` — `SplatConfig` / `SplatHandle`
Configures a 3D Gaussian Splat (`.spz` format) background: scale, position offsets, Euler rotations, optional collider mesh URL, and a `control` flag exposing live calibration sliders.

### `viewer.py` — `ViewerConfig`
Camera parameters (lookat, distance, fovy, elevation, azimuth) + tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

### `mjlab/`: everything that reads mjlab's own objects
mjlab stays a soft dependency: it is imported lazily inside the functions that need it, and `onnx_meta.py` needs it not at all.
- `observation.py`, `termination.py`, `command.py`, `action.py`, `event.py`: the per-manager adapters. Obs/term/event bodies are *not* name-resolved to mirrors: mjlab's own functions are traced directly. What remains is config-shape adaptation: reducing mjlab's network-keyed observation dict to the actor's group via the task's runner config (`rl_cfg.obs_groups`: only groups it attributes to a network are renamed or dropped, so an author's extra slots survive), `clip_actions`, action-scale and PD-gain resolution, translating `MyoMuscleActivationActionCfg` to `MuscleActivationActionCfg`, and `apply_terrain_spawn`, which swaps mjlab's untraceable `reset_root_state_from_flat_patches` for mjswan's patch-based reset (the browser runs one env where mjlab spreads many across the terrain).
- `task.py`: what a scene takes from `env_cfg` (the control rate, terrain data, entity specs, the viewer config).
- `runner.py`: what playback takes from `rl_cfg` (`resolve_runner_defaults`), and how a `model_*.pt` becomes ONNX (`create_pt_onnx_export_context`, `export_checkpoint`), aligning the observation normalizer with the checkpoint.
- `env.py`: `build_single_entity_trace_env(spec_fn)` builds a minimal single-entity `ManagerBasedRlEnv` out of mjlab's own `Entity`/`Scene`, no reimplemented kinematics. It configures no managers and is never stepped; it is only the tracer's `env.scene[name].data.<field>` read/write target. Joint defaults come from the model's first keyframe, matching what the browser resets to. `build_mjlab_env()` grows `nconmax`/`njmax` until a task's env fits a single-env re-use. `TraceCommandManager` supplies trace-time stand-ins for commands the browser owns (a `ui_command` has no Python side).
- `gui.py`: runs mjlab's own `CommandTerm.create_gui` against a recording stand-in, so the browser control panel's slider ranges come from mjlab's declaration instead of a hand-copied duplicate that drifts.
- `sim.py`: applies a task's `MujocoCfg` sim options to a spec (monkey-patching `MujocoCfg.apply_to_spec()` onto mjlab when needed).
- `onnx_meta.py`: parses the `metadata_props` mjlab's `attach_metadata_to_onnx` writes into an exported policy: joint names, rest pose, action scale, and a description of each observation term. Plain strings, so it needs neither mjlab nor torch, which is what lets a Hub-fetched ONNX be added on the light path. The encoding is lossy (mjlab formats list values with `{:.3f}`; ints and bools arrive as `"1.000"`), and `joint_names` covers *every* joint of the robot while the network emits one action per actuated joint (each action term's joints in joint order, one term after another), so `add_policy_hf` takes the order from the task's action terms where it has them and otherwise pairs the metadata against the scene's own model before using it. Observation terms are named but not carried (the functions mjswan traces are not in the file) so an observation group is never reconstructed from this.
- `detect.py`: `is_from_mjlab`, the duck-typed check the adapters share.

### `source/`: a reference in, local files out
Each module backs one `add_<layer>_<source>()` family, knows nothing about what the files mean, and carries its own optional dependency: one extra per backend (`mjswan[wandb]`, `mjswan[hf]`), imported inside a function so `import mjswan` touches neither. `wandb.py`: `resolve_run_path`, `fetch_onnx`, `fetch_motion_npz` (from a run or an artifact) and `fetch_checkpoints`, a context manager yielding a run's `model_*.pt` files for `mjlab.runner` to convert. `hf.py`: the same verbs against the Hugging Face Hub, which holds the finished artifact, so neither mjlab nor torch is involved. With no filename given, `policy.onnx` then `final.onnx` then the repository's single `.onnx` (the names the Hub's own mjlab download query counts) and several unnamed candidates raise rather than pick. `fetch_dir` is `fetch_file` for an asset that is several files: a MuJoCo model is an MJCF plus the meshes it resolves relative to itself, so `add_scene_hf` brings the XML's whole directory down (`snapshot_download` scoped to it) and compiles the spec where it lands, license detection included. A splat is the opposite case (one opaque file) so `add_splat_hf` is `fetch_file` handed to `add_splat(source=)`, bundling the `.spz` the way a local one is.

### `compile/` — the tracer (ADR 0005)
Runs each `func(env, **params)` once against a recording proxy to discover its reads, classifies each as time-varying state (a graph input, or "slot") or a model-derived constant (baked in), then exports an `nn.Module` via `torch.onnx.export`. Returns bytes and `*Export` records; writing them is `build/mdp/`'s job.
- `slot.py`: what a term may read off `env`, and how each read is named. Four slot namespaces: a raw `mjData` field (`sim`), an entity `data` field, a named sensor's `sensordata` window, and a live command's state field. An `EntityData` property the browser reads natively (`READER_FIELDS`) is one value slot; any other property is traced through a copy of the real `EntityData` whose `data` is the sim proxy, so mjlab's property math lands in the graph and its raw reads become `sim` slots. Only `_STATIC_DATA_FIELDS` are treated as constant: anything unrecognized errs toward a graph input, so a missing runtime input fails loudly instead of returning stale values. Discovery and replay share one contract: a term may read `env.scene[...]`, `env.command_manager`, `env.sim.data` (the same object as `entity.data.data`) and the constants in `_FORWARDED_ENV_ATTRS`; any other `env` read raises `UnsupportedEnvRead`, so nothing bakes a constant by reaching the real env.
- `proxy.py`: the stand-ins both passes share, which subclass the real objects so a term's `isinstance` checks keep passing.
- `record.py`: the discovery pass. A `sim` slot is narrowed to the rows the term indexes when discovery can tell which, unioned across a group's terms, and shipped whole otherwise. Structured sensors (`RayCastSensor`, `ContactSensor`) contribute one slot per field read. Event and command bodies record under tagged keys and have their `write_*_to_sim` calls captured.
- `replay.py`: the replay pass, serving the recorded slots back while torch traces; a read discovery never saw raises.
- `export.py`: classification, narrowing, constant buffers and the `torch.onnx.export` call, plus the single-env guard and the three vetted exporter warnings it silences.
- `native.py`: terms the runtime evaluates itself (`last_action`, `generated_commands`, mjlab's `time_out` by function name), and `action_term_offset`.
- `term.py` / `event.py` / `command.py` / `group.py`: one trace per kind, a value term (a termination that reads nothing fails the build, a baked observation warns by name), an event (writes become outputs, randomness an explicit `rand` input), a stateful command (hidden state promoted to graph I/O), and the fused observation and termination groups.
- `rng.py`: build-time RNG spy/replay. Patches the term function's *own* module globals (mjlab binds the name at import time), records mjlab's real draws, replays those exact values into the graph's `rand` input so parity does not diverge on randomness alone. Unrelated to the runtime's seeded PRNG.
- `parity.py`: steps a live env with a seeded action sequence and feeds the same raw state through each exported graph via `onnxruntime` (not torch), asserting `allclose` for every term at every step.

### `envs/mdp/` and `managers/`
mjlab-compatible MDP layer, with a deliberate asymmetry:

- **`envs/mdp/actions/`** carries *real, directly usable* config classes, because Action is permanently native (ADR 0005 §7): `JointPositionActionCfg`, `JointEffortActionCfg`, `MuscleActivationActionCfg` are supported; `JointVelocityActionCfg`, `TendonLength/Velocity/EffortActionCfg` and `SiteEffortActionCfg` are exported so mjlab configs import cleanly but raise `NotImplementedError` at build time. `stiffness`/`damping` are mjswan-specific: the browser computes PD externally for motor actuators with `biastype=none`. The base `ActionTermCfg` is defined in `managers/action_manager.py`, where mjlab defines its counterpart.
- **`envs/mdp/{observations,terminations,events}.py`** carry only the `*Binding` escape hatch and its `register_*` registry. Pass mjlab's own functions instead. The one event mjswan owns is `reset_root_state_on_flat_patch`, which replaces mjlab's untraceable `reset_root_state_from_flat_patches` (it draws with `torch.randint` and indexes per-env terrain tensors); `mjlab/event.py`'s `apply_terrain_spawn` swaps it in.
- **`envs/mdp/commands.py`** is the exception that carries bodies: the `ui_command` presets and the trace-friendly rewrites described above.
- **`managers/`** holds the config-side counterparts (`observation_manager`, `event_manager`, `action_manager`, `termination_manager`, `command_manager`). `ObservationTermCfg` adds `history_steps` (sparse look-back offsets) and `history_interleaved` (Isaac joint-major layout) beyond mjlab's fields; training-only fields (`noise`, `delay_*`, `enable_corruption`) are accepted and ignored.

A term with no browser implementation **fails the build**, naming both ways out (a trace-friendly `register_*` replacement callable, or a `ts_src` TS class). Three event terms are exempt because there is provably nothing to write: `randomize_terrain`, `encoder_bias`, and a root-state write onto a fixed-base entity.

**Muscle action term.** `MuscleActivationActionCfg` drives MuJoCo muscle actuators. `action_mode` (named as mjlab/myosuite name it, so the adapter copies it across) picks the mapping from `raw = scale·a + offset`: `sigmoid` (default) applies the canonical MyoSuite sigmoid `σ(5(raw − 0.5))` for excitation in (0, 1); `excitation` clips `raw` to [0, 1]; `direct` clips `raw` to each actuator's own `ctrlrange` and writes it unchanged, for a checkpoint trained against the raw control (a myosuite muscle model declares `ctrlrange="-1 1"` and uses the negative half). `normalize` is the legacy spelling of the first two. The mjlab adapter translates `MyoMuscleActivationActionCfg` (the class actually used by every myo* mjlab task) to `MuscleActivationActionCfg`; see [docs/adr/0002](./docs/adr/0002-muscle-action-term-aligned-with-myomuscleactivationactioncfg.md).


## Frontend (`src/mjswan/template/`)

TypeScript + React + Vite + three.js. Built by `Builder.build()` via `build/frontend.py`. The browser client:
- Loads the MuJoCo WASM module and steps physics on the main thread, yielding to the browser each control step (`engine/yieldToBrowser.ts`).
- Runs the policy **and every traced MDP term body** via onnxruntime-web's CPU build (`onnxruntime-web/wasm`), all on wasm: the build with WebGPU kernels is a 26.5 MiB wasm, over the 25 MiB per file Cloudflare Pages serves.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR and passthrough AR), entered from the host's own UI through `engine.xr`: the engine draws no button. Tracked hands can join the sim as bodies, compiled in at the next model build after the switch is turned on.
- Reads `manifest.json` to discover projects/scenes/MDPs/policies at runtime; `?project=` / `?scene=` / `?policy=` take ids (`sanitizeName` = `name2id`, pinned by a shared case table).

`src/core/` mirrors mjlab's layout — `observation/`, `termination/`, `action/`, `event/`, `command/` — plus `onnx/` (`session.ts` caches split by lifetime, `runQueue.ts`, `slotReader/` serving graph inputs from `mjModel`/`mjData` — `indexing.ts` resolves an entity's elements as mjlab's `EntityIndexing` does, `fields/` holds every `EntityData` property as a reader, one file per section of mjlab's `entity/data.py`, and `index.ts` dispatches a slot by namespace — `raycast.ts` casting height-scan rays with `mj_ray`, `contact.ts`, `graphRefs.ts`), `rng.ts` (xoshiro128\*\*, snapshot-able), `policy/`, `scene/`, `xr/` (`handMocap.ts` injects a capsule per hand bone and writes `mocap_pos`/`mocap_quat`; `session.ts` holds what each session asks for and when a scene can take the hands), and `engine/` (`runtime.ts` step loop, `yieldToBrowser.ts`, `resetChain.ts`, `viewer_config.ts`). Each manager is native orchestration around ONNX term bodies: `FusedObservation`/`OnnxObservation`/`NativeObservation`/`HistoryObservation`, `FusedTermination`/`OnnxTermination`/`TimeOutTermination`, `OnnxEvent` + `triggers.ts` + `entityWrite.ts` + `modelFieldDr.ts`, `OnnxCommand`. Action is fully native (`action/applyAction.ts`).

Two more directories under `src/core/` are viewer-only, outside any MDP: `interaction/` (the pointer's view, pull, push and grab modes behind one `InteractionManager`, the only writer of `xfrc_applied`) and `grab/` (`weldHold.ts`, the weld hold the pointer and the tracked hands share, each with its own trigger). Neither reaches Python, the manifest or the `.swn` document.

The step loop follows mjlab's `ManagerBasedRlEnv.step` ordering exactly — forward → command → event → obs → action → physics → term → reset → forward — with **one** `mj_forward` per iteration. The reset chain mirrors `_reset_idx`: `event(mode="reset")` → observation/action reset → command resample, all awaited in config order (writes are last-writer-wins by config order, so concurrency would make that machine-dependent).

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).

The template has three Vite build outputs (all written to `template/dist/`):
- **SPA** (`vite.config.ts`, `npm run build:spa`) — the standalone app the Python `Builder` assembles. Entry `src/index.tsx`.
- **Library** (`vite.lib.config.ts`, `npm run build:lib`) — a single self-contained ESM `dist/mjswan.js` exposing `createEngine(element, options?)` (entry `src/engine/index.ts`), consumed by mjswan Cloud from a CDN. Every dependency is bundled (no bare imports) and the MuJoCo/ONNX WASM is emitted into `dist/assets/`, referenced via `new URL('./assets/x.wasm', import.meta.url)`. Vite lib mode force-inlines those WASM as base64; a `generateBundle` plugin extracts them back out, recognizing each by content digest so it keeps its upstream basename (mjswan-cloud ADR 0001). The SPA build names files the same way into the same directory, so identical bytes land on one path instead of shipping 40 MiB twice; only `mjswan.js` sits at the root, addressed from outside as the npm entry. `src/core/onnx/ortEnv.ts` points `ort.env.wasm.wasmPaths` at ORT's copy, so a host serving `dist/` fetches nothing third-party (issue #123). `vite.wasm.ts` carries the sources and the ORT assumptions this rests on.
- **Manifest** (`vite.manifest.config.ts`) — `dist/manifest.js`, the `mjswan/manifest` catalog parser as a standalone CDN-loadable ESM.

`npm run build` runs all of them. The public engine API (ADR 0004) is bytes-in / snapshot-out: `loadScene` / `setPolicy` / `setSplat` / `setMotion`, `camera` / `commands` verbs, `subscribe`, `captureThumbnail`, `dispose`, with `termSeed` in and out for session replay. It never fetches — `mjswan/manifest` (`parseManifest(manifest, byteSource)`) turns a `manifest.json` into a lazy catalog (refusing a `format` newer than it knows) and the app owns the fetching. A policy switch is an MDP switch: the runtime restores the model fields the previous MDP's startup randomization changed, reseeds the term PRNG, builds the new MDP's `EventManager` and runs its startup events (ADR 0006 §9).


## CLI entry points

The primary CLI is `mjswan` (Typer-based, defined in `cli.py:app`). Subcommands:

| Subcommand | Description |
|------------|-------------|
| `mjswan view <model.xml>` | Build and launch a viewer for a MuJoCo XML/MJCF file |
| `mjswan serve <dist-dir \| document.swn>` | Serve a pre-built `dist/` directory, or a `.swn` document |
| `mjswan new <name> [--template hello-world\|policy\|mjlab]` | Scaffold a new project from a template |
| `mjswan demo [name]` | List the built-in demos, or run one (`main`, `simple`, `mujoco`) |
| `mjswan info <dist-dir \| document.swn>` | Show a tree of projects/scenes/policies and asset sizes |
| `mjswan publish <dist-dir>` | Upload a built dist's data files to mjswan Cloud (rejects custom-JS builds) |
| `mjswan login` / `whoami` / `logout` | mjswan Cloud session (loopback GitHub OAuth) |


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

Workflows: `pytest.yml` runs `pytest -m "not slow"` across Python 3.10 / 3.11 / 3.12 / 3.13 with the `dev` extra only, and a `core` job installs mjswan with no extra to run `tests/test_core_install.py` (import, a model-only build, and the extra each source names), which skips anywhere an extra is installed. `parity.yml` is separate and heavier: it installs the `examples` extras (~2 GB), caches warp's CPU kernels, and runs the ONNX numeric-parity sweep over the reference mjlab tasks on `workflow_dispatch`, on a pull request that touches `compile/`, `build/mdp/`, `mjlab/env.py`, the parity tests or `uv.lock`, and weekly. Weekly because that cadence targets *upstream* drift: mjlab moves fast, and a changed term definition would otherwise surface as a mysterious runtime difference rather than a failing check. `frontend.yml` lints the template and runs its vitest suite, sharing one dependency install. `release.yml` writes one version into `_version.py`, `package.json` and `package-lock.json`, commits it to main for a stable release, tags it, and calls `publish-pypi.yml` and `publish-npm.yml`, which set the version from the input again so a prerelease needs no commit. Also: `ruff.yml`, `deploy.yml`, `sync-contributors.yml`.

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

Core: `mujoco==3.11.0`, `onnx>=1.20.0`, `nodeenv>=1.9.1`, `rich>=13.0.0`, `typer>=0.12.0`.
Tooling pins (the `check` extra): `ruff==0.16.0` and `ty==0.0.77` exactly, `pyright` floored.
Extras, one per asset source plus tooling: `wandb`, `hf` (`huggingface-hub`), `mjlab` (`mjlab==1.6.0` + `torch>=2.9.1`), `check` (ruff/ty/pyright), `dev` (`check` + every source + pytest/pre-commit), `examples` (the sources + `onnxruntime`).

**`mjlab` + `torch` are build-time requirements for any policy carrying traced MDP terms** — tracing runs `torch.onnx.export` against a live mjlab env. Neither ships to the browser, and a model-only scene needs neither. `onnxruntime` (not `onnxruntime-web`) is the parity harness's runtime.

The core `mujoco` pin is exact and follows mjlab's, which is what lets `mjswan[mjlab]` resolve at all: mjlab 1.6.0 requires `mujoco~=3.11.0` and brings `mujoco-warp` with it. Holding the core at 3.8.1 while mjlab moved to 3.10 and then 3.11 is what the `[tool.uv] override-dependencies` block existed for; aligning the pin retired it, and with it the resolution that only ever worked inside this repo. The browser's own engine is the template's `mujoco` dependency, an alias pinned to `@ttktjmt/mujoco` at the same version: a `.mjz` carries MJCF the browser compiles itself, but a scene saved as `.mjb` loads only in the MuJoCo that wrote it. `tests/test_dependency_versions.py` holds the two pins together and `src/engine/__tests__/mujocoPackage.test.ts` checks the alias is what installs. The alias is a fork because MuJoCo's own `@mujoco/mujoco` throws on every `mjtBool` array from 3.9.0 on; `@ttktjmt/mujoco` is 3.11.0 with the generator fix of google-deepmind/mujoco#3616, and the alias goes back to `@mujoco/mujoco` once a release carries it and mjlab's pin reaches that release. The template's `.npmrc` refuses any package younger than three days (`min-release-age`), which a fresh fork release is: its lockfile entry was written with that check off once, and `npm install` and `npm ci` then install from the lockfile without it.

Python 3.10 to 3.13. The old `<3.13` cap was taken for `labmaze` (transitive via myosuite →
dm-control, no cp313 wheel); myosuite left with the #128 examples cleanup, so the cap and
the `h5py` override beside it are gone and the pytest matrix runs 3.13.


## Deployment

The demo app is built by `examples/demo/main.py` and deployed to GitHub Pages via the `deploy.yml` workflow on every push to `main` that touches relevant paths. It installs only the `hf` and `mjlab` extras and needs no credentials: models and checkpoints come from the public Hub repository `ttktjmt/mjswan`, and the comment beside `HF_REPO` in `main.py` lists how to mirror a task's W&B checkpoints there. The `MJSWAN_BASE_PATH` and `MJSWAN_NO_LAUNCH` env vars, which `main.py` reads, control the build. Cloudflare Pages builds a preview of the same demo for every pull request. The GentleHumanoid and MuscleMimic demos are deployed to Cloudflare Pages too; their sources left this repository with `examples/mjlab/` and `examples/demo/gentle_humanoid/`. `mjswan publish` is the alternative to hosting a `dist/` at all.


## Design records

`docs/adr/` holds the ADRs, deliberately outside the published site (contributor documentation citing files and line numbers; ADRs before 0008 cite the module paths of their time; [ADR 0008](./docs/adr/0008-package-layout.md) maps them to the current layout). ADR 0005 and its companion implementation brief are the ones to read before touching `compile/`, `build/mdp/`, or the frontend's manager layer: the brief carries a per-item status table including the things measured and **declined** (event fusion, input-tensor pre-allocation, `source_url` provenance, bit-for-bit replay), each with the reasoning. Do not re-add those as unfinished requirements. [ADR 0006](./docs/adr/0006-swn-simulation-document.md) supersedes ADR 0005 §1: the build output becomes a `.swn` document with a single root `manifest.json`, an `MdpConfig` unit that owns all five managers (events included), and a `<project-id>/<scene-id>/` layout.
