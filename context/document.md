# The simulation document

`document/` and the rules of what a build writes: the tree, `manifest.json`, ids, the policy slot tables, `format` versus `version`, the `.swn` container, and what a policy switch does to an MDP's events. `build/` writes all of it ([build.md](build.md)); `app.py`, `cli.py` and `cloud/` read it through `document/`, which is torch- and mjlab-free so they need neither. `document/manifest.py` holds the constants both sides share: `DOCUMENT_FORMAT`, `MANIFEST_NAME` and the slot defaults.

## The tree is the document

The built tree (`manifest.json` over `<project-id>/<scene-id>/`) *is* the document; a `.swn` file is that tree as a ZIP. A document needs exactly one thing that describes it, so there is one `manifest.json` at the root and no per-policy or per-scene config file: policy metadata is inlined in the manifest, and four checkpoints share one `mdp` reference rather than four copies of one file.

```
manifest.json
<project-id>/
  LICENSE, NOTICE                 the work's (licenses.md)
  licenses/                       third-party texts, once per project
  <scene-id>/
    scene.mjz | scene.mjb
    mdp/<mdp-id>/{obs,term,command,event,action}/<name>.onnx
    policy/<policy-id>.onnx
    assets/                       <motion>.npz, <splat>.spz, <skin>.vrm | .glb
```

Two rules generate the layout:

- **Every ownership level gets a directory.** Project and scene are the levels at which ids are guaranteed, so each is a directory. A W&B comparison names its checkpoints `model_0`, `model_50`, … in every scene, so a flat `policy/<id>.onnx` would let them overwrite each other.
- **Within a scene, only a kind that can hold more than one file gets a box.** `scene.mjz` is one file per scene and sits directly in the scene directory; `mdp/`, `policy/` and `assets/` can each hold many.

There are no per-project `index.html` copies: the SPA selects a project from `?project=` against its build-time base URL, so a URL like `…/<project-id>/` is a 404 rather than the wrong project.

## Manifest rules

1. **Every key is `snake_case`, at every depth.**
2. **A path resolves against the directory of the level that declares it.** Everything under a scene entry (`scene`, every graph ref, every asset) resolves against `<project-id>/<scene-id>/`; only the top-level `plugins` resolves against the document root. Graph refs are the paths on disk ([build.md](build.md)).
3. **A default is a `"default": true` flag on the entry.** At most one sibling may set it (two is a build error, `_check_defaults`), and when none does the first in document order is the default. Projects and policies both work this way.
4. **A key is omitted when it carries the default**: `in_keys` is absent when it is `["actor"]`, `out_keys` when it is `["action"]`. The reader supplies the default; the writer never emits noise.
5. **`mdp` is always written on a policy**, even when a scene has one MDP: a reader that never needs a fallback cannot get the fallback wrong.
6. **`id` joins `name` on every entry.** `name` is what a person reads; `id` is the directory, the URL parameter and the reference target. A reader never re-derives one from the other.

`actions` lives in the MDP, not on the policy. Two MDPs that name the same action terms (in Python, one object referenced twice) write the same small block twice: the right trade against making actions a fourth ownership level. A term's entry shape is `build/mdp/`'s; `template/src/manifest/index.ts` holds both halves of the contract (the `Manifest*` interfaces mirroring what Python writes, and the `Catalog` the app holds).

## Ids

Every id is `name2id(name)` (`document/ids.py`): lowercased, every run of characters outside `[a-z0-9]` collapsed to one `_`, ends stripped. That keeps an id safe as a file name and a storage key, and readable in a URL. A name with no ASCII letter or digit sanitizes to an empty id and is refused.

| id | unique within |
|---|---|
| project | the document |
| scene | its project (`licenses` is reserved: it is the project's license directory) |
| mdp, policy, splat, skin | its scene |

A collision is **not** an error: two scenes called "Flat Terrain" is a reasonable thing to write. The second becomes "Flat Terrain_1" with id `flat_terrain_1`, with a RuntimeWarning naming both (`assign_name`). The rename reaches the name as well as the id, so every id stays `name2id(name)` and no two entries in the viewer's lists are alike (its select refuses that, and the page renders nothing). A policy's motions, which have no id, are renamed the same way. `unique_id` is the scoped `_N` suffix underneath; MDP ids take the id-only form, `assign_id` ([object-model.md](object-model.md) has how an MDP id is chosen).

The URL parameters `?project=`, `?scene=`, `?policy=` and `?skin=` take ids only (an empty `?skin=` is the Skin menu's None). The frontend's `sanitizeName` mirrors `name2id` character for character, and `src/mjswan/template/src/manifest/name2id_cases.json` pins the pair from both test suites: a case they disagree on (an apostrophe, parentheses, an accent) is a link that opens the wrong scene.

Generated ids (UUIDs) are rejected: a URL is a sharing surface and `?project=g1_locomotion&scene=flat_terrain` has to be readable by a person, and mjswan Cloud simulations are immutable once published, so the link stability UUIDs would buy is stability the system already has.

## The policy slot tables: `in_keys` and `out_keys`

`in_keys` is **the ONNX input slot assignment table**: `in_keys[i]` names the tensor that fills the session's *i*-th input, paired positionally (`core/policy/OnnxModule.ts`). The model's own input names are read from the session and never appear in any config. It is not a list of observation groups: an entry names either an observation group or a tensor the runtime synthesizes, `RUNTIME_INPUT_SLOTS` (`is_init`, `adapt_hx` for a recurrent carry, `time_step`). `out_keys[i]` names the *i*-th output the same way; the actuators are driven from the output named `action`, else the first.

- The defaults are `DEFAULT_IN_KEYS = ("actor",)`, mjlab's own name for the group its actor network reads, and `DEFAULT_OUT_KEYS = ("action",)` (`document/manifest.py`). The TypeScript copy is pinned to the Python one by `core/policy/default_slots.json`: changing either alone breaks every policy that declares no `in_keys`.
- A single-input network needs no `in_keys`: its one input is the default slot. A network with several inputs **must** declare it (`check_slot_tables` raises at `add_policy`), since nothing else records where the runtime-synthesized tensors sit relative to the observation groups. A table whose length differs from the network's inputs or outputs is refused at `add_policy`, not discovered at playback.
- Several outputs with no `out_keys` warn rather than raise, since which one is the action is unknowable; a multi-output network driven from the wrong output does not fail, the robot just falls.
- The tables are declared on `add_policy` and written into the manifest's policy entry. A `config_path` sidecar's tables are ignored with a warning ([build.md](build.md)). A table in the script is verbose where a sidecar hid it, and that is the point: a positional table nobody sees can stay wrong unnoticed.

## An MDP owns its events; a policy switch is an MDP switch

mjlab keeps events in `ManagerBasedRlEnvCfg` beside the other four managers, so an event set belongs to the same unit they do: `MdpConfig.events`. The scene's events are only the default an MDP takes when it declares none. Events therefore switch with the policy, which is what lets a policy switch re-apply domain randomization.

Re-running `mode="startup"` events must not compound. `ModelFieldDefaults` (`core/event/modelWrite.ts`) snapshots each model field on first touch and lives for the model, not for one startup pass. An MDP switch is, in order (`runtime.ts`, `loadPolicyConfig`):

1. restore every snapshotted model field to its compiled value, and `mj_setConst` when any was restored;
2. reseed the term PRNG to the session seed (`termSeed`);
3. build the incoming MDP's `EventManager` and run its startup events.

Clearing the policy still restores, so a scene with no policy is as compiled. The reseed is the rule a scene load already follows, and its consequences are the point: a scene's randomization is a function of (session seed, MDP) alone, not of how long playback ran or of the switch history; switching away and back reproduces the first draw, which is what makes a switch testable; two MDPs with the same randomization terms land in the same world, the right default when comparing policies under one set of conditions.

## `format` and `version`

- **`version`** is the mjswan release that wrote the document. The release workflow sets `_version.py` and the template's `package.json` from one input, so the Python version and the npm engine version are the same string and `https://cdn.jsdelivr.net/npm/mjswan@<version>/dist/mjswan.js` resolves for every released version. A **host** uses `version` to select the engine.
- **`format`** (`DOCUMENT_FORMAT` in `document/manifest.py`) is an integer describing the document's structure, bumped by hand **only** when an engine reading the old structure would misread the new one. The **engine** uses it to protect itself: `parseManifest` refuses a document whose `format` exceeds `MAX_DOCUMENT_FORMAT` (`template/src/manifest/index.ts`), naming both and the document's `version`, and refuses a document with no `format` (the pre-manifest layout, a root `assets/config.json`). Every earlier format still parses.

They are read by different parties for different decisions, so a document carries both. When you add something an older engine would ignore and so run wrong, bump `DOCUMENT_FORMAT` and `MAX_DOCUMENT_FORMAT` together, say beside each what the format adds, and add a row here:

| format | adds | an older engine would |
|---|---|---|
| 1 | the one-manifest layout; a document without `format` predates it and is refused | n/a |
| 2 | `sim` input slots (raw `mjData`) and their `rows` | accept the entry, fail to serve it, freeze the group |
| 3 | `camera.azimuth` as MuJoCo's, the direction the camera looks | show every scene from the opposite side |
| 4 | the `env` input slot (`episode_length_buf`) `time_out` is traced through | never time an episode out |
| 5 | the `action` and `model` input slots, `kind: "model"` writes, a command's `reset_mask`, gated writes, `kind: "root_velocity_b"` | freeze the group, skip the randomization, not run the command |
| 6 | traced action `process` / `reset`, `interpolate`, `hold_s`, `delay`, `joint_position_relative`, the `action_term` input slot | apply the raw action through the native formula |
| 7 | a scene's `skins` | show the bare robot |
| 8 | a motion's `frame_fields` | hold the reading graph's previous value |
| 9 | camera sensors, image-shaped policy inputs, modelled skins, a scene's `graphics` | misread the camera, retarget a modelled skin as a character, draw every scene one way |
| 10 | an action term's DC motor `saturation_effort` / `velocity_limit` | run those motors without the curve |

The frontend still runs a few older documents' native markers (a pre-format-4 `time_out`, a pre-format-5 `prev_action` and model-field descriptor: [frontend.md](frontend.md)).

No `min_viewer` key and no per-feature minimum-viewer table: it would only help the "old engine, new document" pair, which engine selection eliminates for hosted cases, at the cost of a hand-maintained table that must stay correct forever. Adding it later is backward compatible (an old engine ignores an unknown key) but cannot help engines already shipped. A host that selects an engine can keep a version override table (`{"0.9.3": "0.9.7"}`, resolution `overrides[version] ?? version`) for a published engine that turns out broken.

### Traced graphs and stored documents

A graph's body is `torch.onnx.export`'s, but its interface is mjswan's: the input names, the inputs mjswan appends (`rand`, `resample_mask`, previous state), the batch-axis convention, and the manifest's `input_slots` (with `layout` for a fused group), which drop inputs the exporter folded into constants ([tracing.md](tracing.md)). So a stored document survives a change that only *rewraps* what a graph already computes (renaming or reordering inputs and the slots with them, a `Cast` at a boundary, slicing a fused output by `layout`, composing graphs). It does not survive a change that needs computation the graph lacks (a new input, a new meaning for a state tensor, a new way of threading randomness): that needs the Python term body, which never ships. Prefer a format change of the first kind.

## The `.swn` container

A ZIP whose entries are the tree's own paths (`document/container.py`): a directory of files is the authoring model and the container is a reversible packaging step over it, as `.docx` is, so every tool that understands the directory keeps working.

- `document_files()` lists the document off the manifest (the manifest, then everything under each project directory), never off what else is on disk, so the SPA's own `assets/` can never be mistaken for data. A `.swn` never holds the engine (`index.html`, `assets/*.js`, WASM) or the author's custom-term module.
- The manifest is the first entry, so a reader that stops after one file has the one that describes the rest; the others are sorted, for a reproducible archive. `.mjz`, `.npz` and `.spz` are stored (already compressed), the rest deflated.
- `unpack_document()` confines entries to the target and refuses one naming `..` or an absolute path; reading bounds total size, entry count and manifest size, and accepts stored and deflated entries only. A symlink in a built tree is refused rather than followed.
- `as_directory()` gives `publish` and `mjswan info` one code path for a directory or a `.swn`.

Two output modes:

- **Document**: `<name>.swn`, the deliverable (`MjswanApp.save_document()`).
- **App**: a directory holding the engine plus the *expanded* tree, not the zip. This is what `mjswan serve` runs and what GitHub Pages hosts unchanged. Embedding the zip would make the page unpack it before a scene could load, for no benefit where the files can simply be served.

## `app.py`: `MjswanApp`

Wraps a built `dist/`: the engine plus the expanded document. `from_document()` takes either form: a directory is served where it sits; a `.swn` is unpacked to a temporary directory with the packaged engine (`build.frontend.install_spa`) laid over it, building the engine first if no matching SPA is cached, and removed at exit. It refuses a custom-TS document, whose `plugins.js` ships with the engine rather than in the document, so the manifest would point at a module that is not there. `launch()` starts a stdlib HTTP server (COOP/COEP headers, for SharedArrayBuffer and MuJoCo WASM threading) and detects Google Colab, displaying an inline iframe instead. `save_document()` writes the document as one `.swn` via `document/`; `publish()` delegates to `cloud.publish` ([cloud.md](cloud.md)).

## Rejected

- **A config JSON per policy, graphs scoped per policy.** Fixes path collisions but keeps the policy as the MDP's owner, so every checkpoint of a task traces and writes its own copy, and events have no owner.
- **Content-addressed graphs** (`graphs/<sha256>.onnx`, every config pointing in). Deduplicates checkpoints without a new unit, but makes every path opaque, so nothing can be found or diffed by hand and the unzipped tree is unreadable, and it leaves events unowned. Deduplication is a consequence worth having, not a thing to organize the document around.
- **UUIDs** and **`min_viewer`**: above. **Uploading a `.swn` to Cloud as one file**: [cloud.md](cloud.md).
