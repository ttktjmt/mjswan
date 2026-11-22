# The build

`build/`: what `Builder.build()` does. It writes every file of an app (the engine plus the expanded document) and is the one place that knows the shape of a manifest entry. The document it writes is described in [document.md](document.md); the tracing it calls into is [tracing.md](tracing.md).

## `pipeline.py`: `write_app`

The order, and why each step sits there:

1. Refuse two siblings both marked default (`_check_defaults`), then wipe the output directory.
2. Build the frontend (`ClientBuilder`, reusing a cached SPA when it matches) and lay the engine down (`install_spa`). The SPA is project-independent, so it is the same for every build with the same options.
3. Compile author TypeScript terms to `assets/plugins.js` when `uses_custom_js()` is true ([frontend.md](frontend.md) has why it is a separate module).
4. `write_mt_headers` writes the COOP/COEP `_headers` file for `mt=True`.
5. Per project: write the work's `LICENSE` / `NOTICE` and the project's `licenses/`. Then per scene: run pending checkpoint conversions (first, since they create the scene's policies and hand over the trace env); validate muscle action terms against the model's actuators; write the scene as DEFLATE-compressed `scene.mjz` (`mjz.to_zip_deflated`, since `mujoco.to_zip` stores entries uncompressed) or `.mjb`; copy clips, splats and skins into the scene's `assets/` (`asset.py`; motion clips deduplicated by content, and a tracking task's env pointed at the bundled clip before it is built); build or reuse the trace env; trace each `MdpConfig` once into `mdp/<mdp-id>/` (keyed on the config's identity, so a run's checkpoints share one trace); write `policy/<policy-id>.onnx`.
6. Last, the one `manifest.json` at the output root: `{format, version, uses_custom_js, plugins?, projects}`.

A project's `work` and a scene's `credits` (`credit.py`) name the license files, a scene's as `../licenses/…`, each identified (`spdx`, `restriction`, `copyright`), and summarise each VRM skin's `meta`, so the viewer can find them on a host that lists no directory ([licenses.md](licenses.md)).

## `manifest.py`: the entries

`mdp_entry` (calls `build.mdp`), `policy_entry`, `scene_entry`, `graphics_entry` and `splat_entry`, plus `write_manifest`. The manifest's rules (snake_case keys, path resolution, defaults, omitted keys) are in [document.md](document.md).

- `mdp_entry` refuses an MDP with observation or termination terms on a scene with no trace env, naming `set_trace_env(build_single_entity_trace_env(spec_fn))` as the fix ([tracing.md](tracing.md)).
- `graphics_entry` resolves a scene's `graphics` while it still has its model: the `ViewerConfig`'s settings, then the model's `filament.*` flags (MuJoCo 4.0's names; an unknown value warns and takes MuJoCo's default), then MuJoCo's defaults, for every scene, so the viewer can switch a classic one to Filament.
- A `config_path` sidecar (`load_sidecar`) contributes a checkpoint's own defaults to its policy entry (`policy_joint_names`, `default_joint_pos`, `actions`, `clip_actions`). Its slot tables are ignored with a warning: `in_keys` / `out_keys` are declared on `add_policy` and checked against the network there and against the MDP's observation groups here. The sidecar is an authoring input only; the output has one descriptor.
- A motion's `frame_fields` are checked against what the MDP's graphs read off a tracking command ([mdp.md](mdp.md)).

## `mdp/`: one `.onnx` per traced term

One module per term kind (`observation`, `termination`, `event`, `command`, `action`), each tracing plain-callable bodies via `compile/` against the scene's live env, writing `.onnx` bytes under `<scene_dir>/mdp/<mdp-id>/{obs,term,command,event,action}/` and returning the manifest-shaped entry.

- **`graph.py`** owns the path convention and the one write. The ref the manifest carries *is* the path on disk, relative to the scene directory, and is scoped by MDP: term and group names are unique within an MDP but not within a scene, so two MDPs' `actor` groups would otherwise collide. As a backstop, the write refuses to replace a *different* graph already at one path. `stamp_provenance` writes who traced a graph into its own `metadata_props`, since a `.onnx` travels without the manifest that names it.
- **Fusion**: one graph per observation group (clip-then-scale folded in, mjlab's order) and one per termination group (a bool lane per term, so per-term reset reasons survive). A group with a `*Binding` term or per-term history does not fuse; a lone traced termination is deliberately left unfused. Why, and what is never fused: [tracing.md](tracing.md).
- **Events that write `mjModel`** (mjlab's `dr.*`: `geom_friction`, `body_com_offset`, `geom_size`, ...) trace like any other, their model reads and writes keyed by element *names*, since the browser compiles its own model. `event.py` marks one `set_const` when the write leaves MuJoCo's derived constants stale.
- **`provenance.py`** records which function each entry came from (`func` as `module:qualname`, the first line of its `doc`, its resolved `params`), and the `kind` / `term` / `func` stamped into the graph. A `SceneEntityCfg` param travels as its declared name patterns only: its resolved ids are build-env indices and mean nothing to a reader.
- **`sensor.py`** describes the raycast, contact and camera sensors a graph reads (a camera's `rgb` only, fed as float32 and cast back to uint8 in the graph).
- **`debug_vis.py`** describes the Debug Viz drawings that are not command terms (raycast sensors with `debug_vis`, mjlab's `upright` reward), warning for a reward drawing it does not know.
- **`action.py`** merges each action term's `to_dict()` over the sidecar's authored block (a motor robot's PD gains), and for a task's own action class (`TracedActionCfg`) builds the term on the trace env, writes its `action/` process and reset graphs, and names the native apply its binding or mjlab base gives ([tracing.md](tracing.md)). Actions serialize before observations, which may read a traced term's state.
- **`binding.py`**: `require_ts_src`, the check that a `*Binding` names its TypeScript.

## `frontend.py`: the Python side of `template/`

`template/` stays where npm, Vite, CI and hatch expect it; this module drives it.

- Runs the Vite build, with a local `nodeenv` when Node is not installed system-wide.
- Caches the built SPA by a key of the mjswan version, the build options (`base_path`, `gtm_id`, `mt`, `debug`) and a SHA-256 fingerprint of the app's sources, config and declared dependencies (not `node_modules`, `dist`, `package-lock.json` or the tests the wheel leaves out). The SPA carries no project data and no author code, so any build with a matching key can reuse it.
- `install_spa` copies the built engine and its `LICENSE` / `NOTICE` into an app directory, for the builder and for `MjswanApp.from_document` alike.
- `build_plugins_module` bundles author `ts_src` terms into `plugins.js` with esbuild, aliasing every `mjswan/<sub>` subpath of `package.json`'s `exports` to its engine source and `three` to `src/engine/plugin-three-shim.cjs` (why: [frontend.md](frontend.md)). The engine is never rebuilt for it.
- `uses_custom_js()` is what the manifest's flag reads: whether any observation, termination, event or command binding registered a `ts_src`. Actions have no `ts_src` path.

## `mjz.py`

`scene.mjz` is the `MjSpec` plus every asset it references, DEFLATE-compressed, with relative paths rewritten (leading `..` stripped) so the archive is self-contained. Assets are collected from disk, falling back to `spec.assets` by suffix (mjlab stores them under prefixed keys). A spec saved this way carries MJCF the browser compiles itself; an `MjModel` saved as `.mjb` loads only in the MuJoCo that wrote it ([dependencies.md](dependencies.md)).

## Other modules

- `asset.py`: a scene's `assets/`: motion clips (deduplicated by content), splats, skins.
- `skin.py`: a skin's manifest entry: node indices and measured turns, or a modelled skin's joints and bodies; no glTF names ([skins.md](skins.md)).
- `credit.py`: a scene's `credits` and a project's `work`, and the names in `licenses/` ([licenses.md](licenses.md)).
