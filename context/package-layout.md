# Package layout

How `src/mjswan/` splits into packages, which way imports point, and how modules are named. Read it before adding or moving a module.

## Layers

```
cli.py → the object model at the root → build/ · cloud/ → compile/
       → mjlab/ · source/ · rig/ → managers/ · envs/mdp/ · document/ · license/
```

A module imports from its own layer or below. An import that would point up (a manager cfg importing `build/`, say) means the code sits in the wrong package.

- **`cli.py`** parses arguments and calls the layers below.
- **The root** is the object model an author holds: `builder.py`, `app.py`, `project.py`, `scene.py`, `policy.py`, `motion.py`, `splat.py`, `skin.py`, `viewer.py`, `mdp.py`, plus `cli.py`. `Builder.build()` delegates to `build.pipeline`, and `add_policy_wandb` / `add_policy_hf` to `source/` and `mjlab/`, both lazily. The only underscore module is `_version.py`, which hatch reads.
- **`app.py`** serves, packages and publishes a *built* document: `Builder.build()` returns its `MjswanApp`, so `builder.py` imports it. It reaches `cloud/` and `build/frontend.py` lazily (the latter to lay the engine beside a `.swn`, building it when no matching one is cached) and never imports `compile/`.
- **`cloud/`** publishes a built document. It reads `document/` and `license/` and imports neither `build/` nor `compile/`.
- **`build/`** writes every file of a document and is the one place that knows the shape of a manifest entry ([build.md](build.md)).
- **`compile/`** traces a term body to ONNX bytes against an env it is handed. It writes no file and imports nothing from `mjlab/`; its only mjswan imports are the `envs/mdp/` registries its parity harness resolves terms through ([tracing.md](tracing.md)).
- **Leaves**: `mjlab/`, `source/`, `rig/`, `managers/`, `envs/mdp/`, `document/`, `license/`. Each depends on the object model at most through `TYPE_CHECKING` imports, but for one: `mjlab/task.py` imports `ViewerConfig`, which `adapt_viewer_config` builds from a task's viewer config.

### What `import mjswan` costs

The root imports the build pipeline and `compile/` lazily, and `mjlab/` imports mjlab only inside its functions, so `import mjswan` never costs a Node build and, in a core install, loads no source backend (`tests/test_core_install.py`, run by CI's `core` job with no extra). It imports neither `onnx` nor `rich`: `policy.py`, `scene.py` and `mjlab/runner.py` annotate a `ModelProto` under `TYPE_CHECKING` and read it duck-typed. Opening a MuJoCo model with a core install costs `mujoco` and `numpy` and nothing else.

One deliberate exception: with the `mjlab` extra installed, `envs/mdp/commands.py` and `envs/mdp/events.py` import torch and mjlab's `sample_uniform` at module level, because the tracer's RNG spy patches a term body's *module globals* and the name has to be one of theirs to be seen ([tracing.md](tracing.md)).

## Naming rules

- **Packages and modules mjswan names are singular**: `build/`, `source/`, `license/`, `document/`, `build/asset.py`, `compile/slot.py`, `mjlab/observation.py`.
- **mjlab-mirror paths keep mjlab's spelling**: `managers/observation_manager.py`, `envs/mdp/observations.py`, `envs/mdp/actions/`. Their import paths are a public contract (`from mjswan.managers.observation_manager import ObservationTermCfg` reads as mjlab's), so they are the one place plural names stay.
- **A file never repeats its package's name**: `mjlab/observation.py`, not `adapters/mjlab_adapter.py`; `build/frontend.py`, not `build/build_client.py`. The same short name under `mjlab/`, `build/mdp/` and `compile/` (`observation`, `termination`, `event`, `command`, `action`, `actuator`) is intended: one module per manager kind per stage, always read qualified by its package.
- **An underscore does not mean module-private.** Some helpers lost theirs when a module split (`term_provenance`, `resolved_params`, `is_from_mjlab`), but many shared names keep one: `compile/`'s tracers import `export.py`'s `_export_onnx` and `_register_consts`, and `mjlab/`, `build/` and `compile/parity.py` read the term registries in `envs/mdp/` and `managers/` (`_custom_registry`, `_action_registry`, `_actuator_registry`).

## mjlab in three relations

- **Mirror** (`managers/`, `envs/mdp/`): mjlab's import paths, mjswan's classes. `ActionTermCfg` lives in `managers/action_manager.py`, where mjlab defines its counterpart, and is re-exported from `envs/mdp/actions`. `CommandTermConfig` keeps mjswan's own name, since it is not mjlab's `CommandTermCfg` ([mdp.md](mdp.md)).
- **Speak** (`mjlab/`): everything that reads mjlab's own objects, one module per concern, importing mjlab lazily inside the functions that need it ([mjlab.md](mjlab.md)).
- **Use** (the root, `build/`): `__init__.py`, `project.py`, `scene.py`, `build/manifest.py` and `build/pipeline.py` import `mjlab/` modules at module level. That costs nothing, since `mjlab/` imports the mjlab library only inside its functions.

## Where things live

- A manifest key: `build/manifest.py` (project, scene, MDP and policy entries, and the `manifest.json` write) or `build/mdp/<kind>.py` (one per traced term kind), nowhere else. `compile/` returns bytes and records (`TermExport`, `EventExport`, `CommandExport`, `GroupExport`, `ActuatorExport`) and nothing more.
- A tracing question: `compile/<pass or kind>.py`.
- Anything that reads an mjlab object: `mjlab/`.
- Fetching versus converting: `source/wandb.py` and `source/hf.py` turn a reference into local files and know nothing about what the files mean. The `.pt` → ONNX conversion is `mjlab.runner.export_checkpoint`, fed by `source.wandb.fetch_checkpoints`; the Hub path downloads an `.onnx` and stops. Each source backs one `add_<layer>_<source>()` family, and its functions share one vocabulary (`fetch_onnx`, `fetch_motion_npz`) with no `_from_wandb` / `_from_hf` suffix, since the module already says which.
- Ids (`name2id`, `unique_id`, `assign_id`, `assign_name`): `document/ids.py`, since an id is part of the document's naming rules.
- Tests follow the modules they cover (`test_build_*.py`, `test_mjlab_*.py`, `test_source_*.py`, `test_document_ids.py`) and patch the module that owns a name (`mjswan.build.pipeline.ClientBuilder`, `mjswan.source.wandb.fetch_onnx`).

## No compatibility layer

Only the current layout is supported. No old module path imports, nothing maps old paths to new ones, and there are no alias modules or legacy console scripts (`mjswan demo` and `mjswan serve` are the commands). A shim would keep the layout it points at.

## Rejected

| Option | Why not |
|---|---|
| Keep a flat root and only rename the private modules | Names were the symptom: the manifest's shape would stay known in five places and the tracer one 2,457-line file. |
| `emit/` instead of `build/` | A compiler's word, not what an author reads in `Builder.build()`. `bundle` means both the output and the JS bundle here; `serialize` is too narrow for running npm and writing a ZIP; `export` collides with `torch.onnx.export`. |
| `adapter/mjlab/` instead of `mjlab/` | One level deeper for one adapter target, and the package holds more than adapters (the trace envs, the runner, the ONNX metadata reader). |
| Move `envs/mdp/` and `managers/` under `mjlab/` | Their paths are the mirror contract; moving them breaks `from mjswan.managers... import` for no gain. |
| Rename `CommandTermConfig` to `UiCommandTermCfg` | It also carries traced mjlab commands (`pending_trace`), so "UI" would be wrong; and it is not mjlab's `CommandTermCfg`, so that name would mislead. |
| Rename `mdp.py` to avoid the `envs/mdp/` clash | `MdpConfig` is the object model's unit and `envs/mdp/` is mjlab's path; both are right where they are. |
| Move `template/` under `build/` | npm, Vite, CI and hatch all point at `src/mjswan/template`; its Python side is `build/frontend.py`. |
| Keep old import paths as shims | See *No compatibility layer*. |
