# Package layout

How `src/mjswan/` is split into packages, which way imports may point, and how modules are named. Read this before adding a module or moving one.

## Layers

```
cli.py → app.py · cloud/ → the object model at the root → build/ → compile/
       → mjlab/ · source/ · rig/ → managers/ · envs/mdp/ · document/ · license/
```

A module may import from its own layer or below. An import that would point up (a manager cfg importing `build/`, say) is the signal that code sits in the wrong package.

- **`cli.py`** parses arguments and calls the layers below.
- **`app.py` and `cloud/`** serve, package and publish a *built* document. They read `document/` and `license/` and never import `compile/`. `cloud/` never imports `build/` either; `app.py` reaches `build/frontend.py` (lazily) to lay the engine beside a `.swn`, building it when no matching one is cached.
- **The root** is the object model an author holds: `builder.py`, `app.py`, `project.py`, `scene.py`, `policy.py`, `motion.py`, `splat.py`, `skin.py`, `viewer.py`, `mdp.py`, plus `cli.py`. `Builder.build()` delegates to `build.pipeline`; `add_policy_wandb` / `add_policy_hf` delegate to `source/` and `mjlab/`. Both imports are lazy. The only underscore module is `_version.py`, which hatch reads.
- **`build/`** is what `Builder.build()` does: it writes every file of a document and is the one place that knows the shape of a manifest entry ([build.md](build.md)).
- **`compile/`** traces a term body to ONNX bytes against an env it is handed. It writes no file and imports nothing from `mjlab/`; its only mjswan imports are the `envs/mdp/` registries its parity harness resolves terms through ([tracing.md](tracing.md)).
- **Leaves**: `mjlab/`, `source/`, `rig/`, `managers/`, `envs/mdp/`, `document/`, `license/`. Each depends on the object model at most through `TYPE_CHECKING` imports.

### What `import mjswan` costs

The root imports the build pipeline and `compile/` lazily, and `mjlab/` imports mjlab only inside its functions, so `import mjswan` never costs a Node build and, in a core install, loads no source backend (`tests/test_core_install.py`, run by CI's `core` job with no extra installed). `import mjswan` imports neither `onnx` nor `rich`: `policy.py`, `scene.py` and `mjlab/runner.py` annotate a `ModelProto` under `TYPE_CHECKING` and read it duck-typed. Someone opening a MuJoCo model with a core install pays for `mujoco` and `numpy` and nothing else.

The exception is deliberate: with the `mjlab` extra installed, `envs/mdp/commands.py` and `envs/mdp/events.py` import torch and mjlab's `sample_uniform` at module level, because the tracer's RNG spy patches a term body's *module globals* and the name has to be one of theirs to be seen ([tracing.md](tracing.md)).

## Naming rules

- **Packages and modules mjswan names are singular**: `build/`, `source/`, `license/`, `document/`, `build/asset.py`, `compile/slot.py`, `mjlab/observation.py`.
- **mjlab-mirror paths keep mjlab's spelling**: `managers/observation_manager.py`, `envs/mdp/observations.py`, `envs/mdp/actions/`. Their import paths are a public contract (`from mjswan.managers.observation_manager import ObservationTermCfg` reads as mjlab's), so they are the one place plural names stay.
- **A file never repeats its package's name**: `mjlab/observation.py`, not `adapters/mjlab_adapter.py`; `build/frontend.py`, not `build/build_client.py`. The same short name recurring under `mjlab/`, `build/mdp/` and `compile/` (`observation`, `termination`, `event`, `command`, `action`) is the intended symmetry: one manager kind, one module per stage, always read qualified by its package.
- **A name shared across modules is public.** Splitting a module exposes helpers to siblings; those lose their underscore (`term_provenance`, `resolved_params`, `is_from_mjlab`) rather than being imported as private names. What one module keeps to itself stays private.

## mjlab in three relations

- **Mirror** (`managers/`, `envs/mdp/`): mjlab's import paths, mjswan's classes. `ActionTermCfg` lives in `managers/action_manager.py`, where mjlab defines its counterpart, and is re-exported from `envs/mdp/actions`. `CommandTermConfig` keeps mjswan's own name: it is not mjlab's `CommandTermCfg`, since it also carries the browser UI and the pending trace ([mdp.md](mdp.md)).
- **Speak** (`mjlab/`): everything that reads mjlab's own objects, one module per concern ([mjlab.md](mjlab.md)). mjlab itself is imported lazily inside the functions that need it.
- **Use** (the root, `build/`): a lazy import of `mjlab/` at the point of need, never at module import time.

## Where things live

- A manifest key: `build/manifest.py` (project, scene, MDP and policy entries, and the `manifest.json` write) or `build/mdp/<kind>.py` (one per traced term kind), nowhere else. `compile/` returns bytes and records (`TermExport`, `EventExport`, `CommandExport`, `GroupExport`) and nothing more.
- A tracing question: `compile/<pass or kind>.py`.
- Anything that reads an mjlab object: `mjlab/`.
- Fetching versus converting: `source/wandb.py` and `source/hf.py` turn a reference into local files and know nothing about what the files mean. The `.pt` → ONNX conversion is `mjlab.runner.export_checkpoint`, fed by `source.wandb.fetch_checkpoints`; the Hub path downloads an `.onnx` and stops. Each source backs one `add_<layer>_<source>()` family and its functions share one vocabulary (`fetch_onnx`, `fetch_motion_npz`) with no `_from_wandb` / `_from_hf` suffix, since the module already says which.
- Ids (`name2id`, `unique_id`, `assign_id`, `assign_name`): `document/ids.py`, since an id is part of the document's naming rules.
- Tests follow the modules they cover (`test_build_*.py`, `test_mjlab_*.py`, `test_source_*.py`, `test_document_ids.py`) and patch the module that owns a name (`mjswan.build.pipeline.ClientBuilder`, `mjswan.source.wandb.fetch_onnx`).

## No compatibility layer

Old module paths are not kept importable, and there are no alias modules: keeping shims would mean keeping the layout they point at. The pre-0.8 aliases (`_compat.py`) and the legacy console scripts (`main`, `simple`, `mjlab`, `serve`) are gone; `mjswan demo` and `mjswan serve` remain. The table below is the migration guide, and CHANGELOG points upgrading users at it.

## Rejected

| Option | Why not |
|---|---|
| Keep a flat root and only rename the private modules | The manifest's shape stayed known in five places and the tracer stayed one 2,457-line file. Names were the symptom. |
| `emit/` for what became `build/` | A compiler's word, and not what an author reads in `Builder.build()`. `bundle` means both the output and the JS bundle here; `serialize` is too narrow for running npm and writing a ZIP; `export` collides with `torch.onnx.export`. |
| `adapter/mjlab/` instead of `mjlab/` | One level deeper for one adapter target, and the package also holds things that are not adapters (the trace envs, the runner, the ONNX metadata reader). |
| Move `envs/mdp/` and `managers/` under `mjlab/` | Their paths are the mirror contract; moving them breaks `from mjswan.managers... import` for no gain. |
| Rename `CommandTermConfig` to `UiCommandTermCfg` | It also carries traced mjlab commands (`pending_trace`), so "UI" would be wrong; and it is not mjlab's `CommandTermCfg`, so that name would mislead. |
| Rename `mdp.py` to avoid the `envs/mdp/` clash | `MdpConfig` is the object model's unit and `envs/mdp/` is mjlab's path; both are correct where they are. |
| Move `template/` under `build/` | npm, Vite, CI and hatch all point at `src/mjswan/template`; its Python side is `build/frontend.py`, which says so. |
| Keep `_compat.py` and old import paths as shims | Every old path would have to stay importable, which is the layout being replaced. |

## Old → new

Every module path from before the package was laid out by layer, and where it now lives. Older code, issues and changelog entries cite the left column.

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
| `compile/tracer.py` | `compile/{slot,proxy,record,replay,export,term,event,command,group}.py` |
