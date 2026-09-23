# Package layout: one package per layer, imports point downward

> Status: **Accepted**, a pre-1.0 change to the *Python package's module layout*
> only. No build output, manifest key, traced graph or `format` value changes; the
> public API in `mjswan/__init__.py` is unchanged. Earlier ADRs cite the paths of
> their time and are not rewritten; §Old → new maps every moved module.
> Implemented as six commits, one per phase (§Phased execution plan), on top of the
> Hugging Face source (`add_policy_hf`).

## Context

`src/mjswan/` had grown by accretion. Twenty-four modules sat at the root, six of them
underscore-private (`_onnx_build.py`, `_graph_io.py`, `_build_client.py`, `_cli.py`,
`_compat.py`, and the version in `__init__.py`), beside an `adapters/` package that
held some of the mjlab-facing code while `trace_env.py`, `mjlab_onnx_meta.py` and
`wandb_io.py` held the rest. Reading the root told a newcomer nothing about which
modules were the author-facing object model and which were the machinery behind
`Builder.build()`.

Four things were wrong with more than the names:

- **The manifest's shape was known in five places.** `_onnx_build.py` wrote the MDP
  entries, `compile/serialize.py` the command entry, `builder.py` the project, scene
  and policy entries, `managers/action_manager.py` the action block, and `command.py`
  the tracking-command params. `compile/`, meant to be the pure tracing layer, wrote
  files.
- **`compile/tracer.py` was 2,457 lines** covering four kinds of term and both
  tracing passes, so the recording proxies, the replay proxies, the export mechanics
  and the four trace functions could only be read as one file.
- **Fetching and converting were one module.** `wandb_io.py` downloaded `model_*.pt`
  files *and* rebuilt a live mjlab env to convert them with torch. When
  `add_policy_hf` added a second source (the Hub, which holds the finished `.onnx`
  and needs no conversion), the asymmetry had nowhere to live.
- **The base action cfg lived in the wrong package.** `ActionTermCfg` was defined in
  `envs/mdp/actions/actions.py` while `managers/action_manager.py` held only a
  serializer, the reverse of the mjlab layout `managers/` exists to mirror.

`_compat.py` carried the pre-0.8 aliases due to go in 0.9. A restructure that keeps
them would have to keep every old module path importable too, which is the layout
this ADR replaces.

## Decision

### 1. Five layers, and imports only point downward

```
cli.py  →  app.py · cloud/  →  the object model at the root  →  build/  →  compile/
                                                                 ↓            ↓
                                              mjlab/ · source/ · managers/ · envs/mdp/ · document/ · license/
```

- **`cli.py`** parses arguments and calls the layers below.
- **`app.py` and `cloud/`** serve, package and publish a *built* document. They read
  `document/` and `license/` and never import `build/` or `compile/`.
- **The root** is the fluent object model: `Builder`, the `*Handle` / `*Config` pairs,
  `MdpConfig`, `ViewerConfig`. `Builder.build()` delegates to `build.pipeline`;
  `add_policy_wandb` / `add_policy_hf` delegate to `source/` and `mjlab/`. Both
  imports are lazy, so `import mjswan` costs neither torch nor a Node build.
- **`build/`** is what `Builder.build()` does, and the one place that knows the shape
  of a manifest entry. It writes every file of a document.
- **`compile/`** traces a term body to ONNX bytes against an env it is handed. It
  writes no file and imports nothing from `mjlab/`.
- **`mjlab/`**, **`source/`**, **`managers/`**, **`envs/mdp/`**, **`document/`** and
  **`license/`** are leaves: each depends on the object model at most through
  `TYPE_CHECKING` imports.

A module may import from its own layer or below; an import that would point up (a
manager cfg importing `build/`, say) is the signal that code sits in the wrong
package.

### 2. The root is the object model, and nothing else

Every module that stays at the root is something an author holds: `builder.py`,
`app.py`, `project.py`, `scene.py`, `policy.py`, `motion.py`, `splat.py`, `viewer.py`,
`mdp.py`, plus the entry point `cli.py`. The only underscore module is `_version.py`,
which `hatch` reads. Helpers that only the build needs (`_onnx_build`, `_graph_io`,
`_build_client`, the ZIP packer in `utils.py`) moved under `build/`; the id helpers
`name2id` / `unique_id` / `assign_id` moved to `document/ids.py`, since an id is part of
the document's naming rules.

### 3. Naming rules

- **Packages and modules mjswan names are singular**: `build/`, `source/`, `license/`,
  `document/`, `build/asset.py`, `compile/slot.py`, `mjlab/observation.py`.
- **mjlab-mirror paths keep mjlab's spelling**: `managers/observation_manager.py`,
  `envs/mdp/observations.py`, `envs/mdp/actions/`. Their import paths are a public
  contract (`from mjswan.managers.observation_manager import ObservationTermCfg`
  reads as mjlab's), so they are the one place plural names stay.
- **A file never repeats its package's name**: `mjlab/observation.py`, not
  `adapters/mjlab_adapter.py`; `build/frontend.py`, not `build/build_client.py`. The
  package qualifies the module, and the same short name recurring under `mjlab/`,
  `build/mdp/` and `compile/` (`observation`, `termination`, `event`, `command`,
  `action`) is the intended symmetry: one manager kind, one module per stage.
- **A name shared across modules is public.** Splitting a module exposes helpers to
  siblings; those lose their underscore (`term_provenance`, `resolved_params`,
  `require_ts_src`, `is_from_mjlab`, `serialize_motion_command`) rather than being
  imported as private names. What one module keeps to itself stays private.

### 4. mjlab in three relations

- **Mirror** (`managers/`, `envs/mdp/`): mjlab's import paths, mjswan's classes.
  `ActionTermCfg` moves to `managers/action_manager.py`, where mjlab defines its
  counterpart; `command.py` becomes `managers/command_manager.py` (the configs,
  `CommandBinding`, `register_command`) and `envs/mdp/commands.py` (the `ui_command`
  presets, the trace-friendly rewrites, and the bindings for mjlab's command classes).
  `CommandTermConfig` keeps its name: it is not mjlab's `CommandTermCfg`, since it also
  carries the browser-side UI and the pending trace.
- **Speak** (`mjlab/`): everything that reads mjlab's own objects, one module per
  concern: the per-manager adapters, `task.py` (what a scene takes from `env_cfg`),
  `runner.py` (playback defaults and `.pt` → ONNX export), `env.py` (the trace envs),
  `sim.py`, `gui.py`, `onnx_meta.py`, `detect.py`. mjlab itself is imported lazily
  inside the functions that need it.
- **Use** (the root, `build/`): a lazy import of `mjlab/` at the point of need, never
  at module import time.

### 5. The manifest's shape is known in one place

`build/manifest.py` assembles the project, scene, MDP and policy entries and writes
`manifest.json`; `build/mdp/` writes one `.onnx` per traced term and returns the entry
that names it, one module per term kind, with `graph.py` (where a graph lands and the
guarded write), `provenance.py`, `sensor.py` and `binding.py` for what they share.
`compile/serialize.py`'s command entry and `managers/action_manager.py`'s action block
both move here. `compile/` returns bytes and records (`TermExport`, `EventExport`,
`CommandExport`, `GroupExport`) and nothing else.

`compile/` itself splits by pass and by kind: `slot.py` (what a term may read, and how
each read is named), `proxy.py` (the stand-ins both passes share), `record.py` (the
discovery pass), `replay.py` (the replay pass), `export.py` (classification, narrowing,
constants, the `torch.onnx.export` call), `native.py` (terms that need no graph), and
`term.py` / `event.py` / `command.py` / `group.py` (one trace each). The package's
public names are unchanged.

### 6. `source/` fetches; `mjlab/runner.py` converts

`source/wandb.py` and `source/hf.py` turn a reference into local files and know
nothing about what the files mean. The `.pt` → ONNX conversion that `wandb_io.py`
did inline is `mjlab.runner.export_checkpoint`, fed by `source.wandb.fetch_checkpoints`;
the Hub path downloads an `.onnx` and stops. Each source backs one
`add_<layer>_<source>()` family and its functions share one vocabulary
(`fetch_onnx`, `fetch_motion_npz`) without a `_from_wandb` / `_from_hf` suffix, since
the module already says which.

### 7. No compatibility layer

`_compat.py` and its aliases go, as do the legacy console scripts (`mjswan-main`,
`mjswan-simple`, `mjswan-mjlab`, `mjswan-serve`; `mjswan demo` and `mjswan serve`
remain). Old module paths are not kept importable: this is a pre-1.0 restructure, and
keeping shims would mean keeping the layout they point at. The table below is the
migration guide.

## Considered options

| Option | Why not |
|---|---|
| Keep the flat root and only rename the private modules | The five places that knew the manifest's shape, and the 2,457-line tracer, stay. Names were the symptom. |
| `emit/` for what became `build/` | A compiler's word, and not what an author reads in `Builder.build()`. `bundle` means both the output and the JS bundle in this repository; `serialize` is too narrow for running npm and writing a ZIP; `export` collides with `torch.onnx.export`. |
| `adapter/mjlab/` instead of `mjlab/` | One level deeper for one adapter target, and the package also holds things that are not adapters (the trace envs, the runner, the ONNX metadata reader). |
| Move `envs/mdp/` and `managers/` under `mjlab/` too | Their paths are the mirror contract; moving them breaks `from mjswan.managers... import` for no gain. |
| Rename `CommandTermConfig` to `UiCommandTermCfg` | It also carries traced mjlab commands (`pending_trace`), so "UI" would be wrong; and it is not mjlab's `CommandTermCfg`, so that name would mislead. |
| Rename `mdp.py` to avoid the `envs/mdp/` clash | `MdpConfig` is the object model's unit and `envs/mdp/` is mjlab's path; both are correct where they are. |
| Move `template/` under `build/` | npm, Vite, CI and `hatch` all point at `src/mjswan/template`; the Python side of it is `build/frontend.py`, which says so. |
| Keep `_compat.py` and old import paths as shims | Every old path would have to stay importable, which is the layout being replaced. |

## Consequences

- `import mjswan` imports neither torch, mjlab, `onnx` nor `rich`: `builder.py` no
  longer imports the build pipeline at module level, and the three modules the
  object model does reach (`policy.py`, `scene.py`, `mjlab/runner.py`) take `onnx`
  under `TYPE_CHECKING`: they annotate a `ModelProto` but read it duck-typed.
  Someone opening a MuJoCo model in the viewer with no policy pays for `mujoco` and
  `numpy` and nothing else.
- A reader finds a manifest key in `build/manifest.py` or `build/mdp/<kind>.py` and
  nowhere else; a tracing question in `compile/<pass or kind>.py`; anything that reads
  an mjlab object in `mjlab/`.
- Import paths change (§Old → new). Examples, docs, the `mjlab-to-mjswan` skill and
  `parity.yml`'s path filter are updated with them.
- Tests follow the modules they cover (`test_build_*.py`, `test_mjlab_*.py`,
  `test_source_*.py`, `test_document_ids.py`), and patch the module that now owns a
  name (`mjswan.build.pipeline.ClientBuilder`, `mjswan.source.wandb.fetch_onnx`).
- Two pieces of dead code are gone: `Builder._policy_filename` (never called) and the
  old `serialize_actions`, which now does the authored-config merge the Builder did
  inline.

## Old → new

| Was | Is |
|---|---|
| `__init__.__version__` (defined there) | `_version.py` (re-exported from `__init__`) |
| `_compat.py` | removed |
| `_cli.py` | `cli.py` |
| `_build_client.py` | `build/frontend.py` (+ `uses_custom_js`, from `builder.py`) |
| `_graph_io.py` | `build/mdp/graph.py` |
| `_onnx_build.py` | `build/mdp/{observation,termination,event,command,action}.py`, with `provenance.py`, `sensor.py`, `binding.py` |
| `compile/serialize.py` | `build/mdp/command.py` |
| `managers/action_manager.serialize_actions` | `build/mdp/action.py` |
| `builder.py` (`_save_web`, `_SceneSteps`, validation, license writing, `_save_mt_headers`) | `build/pipeline.py` (`write_app`, `write_mt_headers`) |
| `builder.py` (`_serialize_mdp`, `_serialize_policy_entry`, `_scene_entry`, `_save_manifest`, `_load_sidecar`, `_build_splat_config_dict`) | `build/manifest.py` (`mdp_entry`, `policy_entry`, `scene_entry`, `write_manifest`, `load_sidecar`, `splat_entry`) |
| `builder.py` (`_write_scene_motions`, `_motion_key`, `_point_env_cfg_at_bundled_motion`, the splat copy) | `build/asset.py` |
| `utils.py` (ZIP packing) | `build/mjz.py` |
| `utils.py` (`name2id`, `unique_id`, `assign_id`) | `document/ids.py` |
| `document.py` | `document/container.py`; `DOCUMENT_FORMAT`, `MANIFEST_NAME`, the slot defaults in `document/manifest.py` |
| `policy.py` (`RUNTIME_INPUT_SLOTS`, `DEFAULT_IN_KEYS`, `DEFAULT_OUT_KEYS`) | `document/manifest.py` (re-exported from `policy.py`) |
| `publish.py`, `auth.py` | `cloud/publish.py`, `cloud/auth.py` (+ `cloud/transport.py` for the user agent) |
| `licenses/` | `license/{path,spdx,attribution}.py` |
| `adapters/mjlab_adapter.py` | `mjlab/{observation,termination,command,action,event}.py`, `mjlab/task.py`, `mjlab/detect.py` |
| `adapters/mjlab_compat.py` | `mjlab/sim.py` |
| `adapters/gui_spy.py` | `mjlab/gui.py` |
| `trace_env.py` | `mjlab/env.py` |
| `mjlab_onnx_meta.py` | `mjlab/onnx_meta.py` |
| `wandb_io.py` (downloads) | `source/wandb.py` (`resolve_run_path`, `fetch_onnx`, `fetch_motion_npz`, `fetch_checkpoints`) |
| `wandb_io.py` (`.pt` → ONNX) | `mjlab/runner.py` (`create_pt_onnx_export_context`, `export_checkpoint`, `resolve_runner_defaults`) |
| `hf_io.py` | `source/hf.py` (`fetch_onnx`, `fetch_motion_npz`, `fetch_file`) |
| `command.py` (configs, `CommandBinding`, `register_command`) | `managers/command_manager.py` |
| `command.py` (`ui_command`, `velocity_command`, the `MotionCommandCfg` binding) | `envs/mdp/commands.py` |
| `command.py` (`default_viz`) | each binding's `viz` (`envs/mdp/commands.py`) |
| `envs/mdp/events.apply_terrain_spawn` | `mjlab/event.py` |
| `envs/mdp/actions/actions.ActionTermCfg` | `managers/action_manager.py` (re-exported from `envs/mdp/actions`) |
| `scene.py` (ONNX slot / width checks) | `policy.py` |
| `scene.py` (tracking-motion helpers) | `motion.py` |
| `compile/tracer.py` | `compile/{slot,proxy,record,replay,export,native,term,event,command,group}.py` |

## Phased execution plan

1. **Tidy the root**: `_version.py`; `_compat.py` and the legacy console scripts
   removed; `ActionTermCfg` into `managers/action_manager.py`.
2. **The torch-free side**: `document/`, `cloud/`, `license/`.
3. **The mjlab side**: `mjlab/` and `source/`; fetching split from converting.
4. **`build/`**: `_onnx_build`, `_graph_io`, `compile/serialize`, the Builder's
   private methods, `utils.py`'s packer and `_build_client` under one package.
5. **`compile/`**: `tracer.py` split by pass and by term kind.
6. **The root finish**: `command.py` into the mirror packages, `cli.py`, this ADR,
   `CONTEXT.md` and the changelog.

Each phase kept ruff clean, left `ty` and `pyright` at or below their previous
diagnostic counts, and passed the fast suite; the parity sweep ran once at the end.

## Acceptance criteria

- No module at the root but `_version.py` starts with an underscore, and every root
  module is part of the object model or the CLI.
- No import points upward through the layers of §1; `import mjswan` imports neither
  torch nor mjlab.
- The shape of a manifest entry appears in `build/` and nowhere else; `compile/`
  writes no file.
- `mjswan/__init__.py`'s `__all__` is unchanged, and the mirror import paths under
  `managers/` and `envs/mdp/` still resolve.
- The fast suite and the parity sweep pass; `ty` and `pyright` report no new
  diagnostics.
