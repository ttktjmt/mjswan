# The object model

The root of `src/mjswan/`: `Builder` and the `*Handle` / `*Config` pairs an author holds, what each owns and the rules it enforces. CONTEXT.md has the call tree; what the build does with them is [build.md](build.md).

Each `*Handle` wraps a `*Config` dataclass: the handle is the fluent API, the config is the serializable state.

## The public API

`__init__.py` is the canonical public API. It re-exports `Builder` / `MjswanApp`; the six `*Handle` and `*Config` pairs, with the skin's `ModelledSkinConfig`, `MatchByPosition`, `SpringChain`, `SpringCollider`, `ToonShading` and `SolidShading`; mjlab-compatible MDP cfgs (`ObservationGroupCfg`, `ObservationTermCfg`, `ActionTermCfg`, `JointPositionActionCfg`, `JointEffortActionCfg`, `TerminationTermCfg`); command UI (`SliderConfig` / `ButtonConfig` / `CheckboxConfig` and their `Slider` / `Button` / `Checkbox` aliases, `SliderRangeConfig`, `CommandTermConfig`, `CommandBinding`, `CommandUiConfig`, `CommandInput`, `ui_command`, `velocity_command`); `ViewerConfig`, `MdpConfig` and `TerminationDefault`; the `ObservationBinding` / `EventBinding` / `TerminationBinding` / `ActionBinding` escape hatches, the `MdpBinding` umbrella type and the `register_observation` / `register_event` / `register_termination` / `register_command` / `register_action` hooks; and `build_single_entity_trace_env`.

## Ids

Every project / scene / MDP / policy / splat / skin has `id = name2id(name)`, unique within its parent. A collision renames the second `<name>_1` with id `<id>_1` and a RuntimeWarning (`document.ids.assign_name`); a policy's motions, which have no id, are renamed the same way. Rules and reasons: [document.md](document.md).

## `builder.py`: `Builder`

The entry point. Accumulates `ProjectConfig`s; `build()` resolves the output path against the caller's file and hands the projects to `build.pipeline.write_app`. `Builder(license=, copyright=)` is the work's license for every project that sets none ([licenses.md](licenses.md)).

`Builder.from_mjlab(task_id, run_path=...)` is the one-liner for visualizing a single mjlab task. It delegates to `Builder.add_project_mjlab`, which creates a project, adds an mjlab scene and optionally attaches every `model_*.pt` checkpoint of one or more W&B runs (converted to ONNX via mjlab and torch). The manual path is `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb(...)`. `hf_repo_id=` is the same shortcut against a Hugging Face Hub repository, which holds an exported ONNX and so needs no conversion.

## `scene.py`: `SceneConfig` / `SceneHandle`

A scene owns one MuJoCo model (an `MjModel`, saved as binary `.mjb`, or an `MjSpec`, saved as XML in `scene.mjz`), its `control_dt`, its default events (what a policy's MDP gets when it declares none), its trace env, the `MdpConfig`s its policies use (registered on first use, which fixes their ids), zero or more policies, splats and skins, and its `attributions`: the third-party components written to the project's `licenses/` ([licenses.md](licenses.md)).

Two build-time requirements a policy scene has and a model-only scene does not:

- **`control_dt`**: seconds per control step (mjlab's `timestep * decimation`). The model carries only the physics timestep, and a wrong control rate raises nothing at playback, so the builder requires it. `add_scene_mjlab` fills it in from the task.
- **A trace env**: tracing needs a live env to read shapes from and resolve `SceneEntityCfg` regexes against. `add_scene_mjlab` builds one at build time; a plain `add_scene` scene with traced terms needs `set_trace_env(build_single_entity_trace_env(spec_fn))` or the build raises. Why, and what the env may be: [tracing.md](tracing.md).

Events belong to an MDP ([document.md](document.md)), so a scene that sets events and adds no policy writes none, with a RuntimeWarning.

`set_term_defaults` sets, by term name, what the viewer's MDP controls start from instead of mjlab's (a `TerminationDefault` for a termination, a bool for an event), written as the scene's `term_defaults` (format 10). By name rather than per MDP, as the viewer keeps its overrides, so one call covers every checkpoint of a run; a name none of the scene's MDPs has warns at build time ([frontend.md](frontend.md)).

`add_policy_wandb` fetches a run's checkpoints (`source.wandb.fetch_checkpoints`) and converts each with `mjlab.runner.export_checkpoint`; `add_policy_hf` fetches a Hub repository's `.onnx` (`source.hf`) and fills the joint mapping from the task's action terms, else from `mjlab.onnx_meta` ([mjlab.md](mjlab.md)). Each call opens the scene on the highest-step checkpoint it added unless the scene already has a default, and a later `add_policy(default=True)` takes over. `add_policy_wandb(only_latest=False)` converts at build time but checks for wandb, mjlab and torch when called, and skips, with a RuntimeWarning, a checkpoint an earlier run in the list already added (mjlab's runs for one task form a resume chain, so the only name two runs share is that boundary). That skip deduplicates *one* checkpoint seen twice; the id rename handles two *different* objects that sanitize to one id. Neither should grow into the other.

## `mdp.py`: `MdpConfig`

The five term sets (observations, actions, terminations, commands, events) as one unit, as mjlab's `ManagerBasedRlEnvCfg` holds them and trains a checkpoint against all five at once. Each field means what the matching `add_policy` argument means: `None` is "take it from the scene's env config" (or the scene's own events), `{}` is "this MDP genuinely has none". The first policy to use one fills its unset fields and adapts mjlab types in place.

Identity is **by object**, not by content: policies handed the same object share one MDP, traced once into `mdp/<mdp-id>/`, while two equal but separate configs are two MDPs (content-addressed pooling was rejected: [document.md](document.md)). Its id:

- a named MDP is `name2id(name)`;
- one the `add_policy` term-set kwargs built for a single policy takes that policy's id, so `mdp/locomotion/` sits beside `policy/locomotion.onnx`;
- one shared by hand, as `add_policy_wandb` does for a run's checkpoints, is `mdp_<n>`, numbered from zero per scene in first-use order. Per scene, not per document, so a scene's ids stay stable when another scene is added before it.

Policies sharing an MDP must agree on `policy_joint_names`, `policy_num_actions` and the MDP sections of their `config_path` sidecars; the build names a disagreement and asks for a separate `MdpConfig`. Why events belong to the MDP and what a policy switch then does in the browser: [document.md](document.md).

## `policy.py`: `PolicyConfig` / `PolicyHandle`

Holds an `onnx.ModelProto`, its `MdpConfig`, the checkpoint's own metadata (`policy_joint_names`, `default_joint_pos`, `encoder_bias`, `clip_actions`), motion references, and the slot tables `in_keys` / `out_keys`, positional maps onto the network's inputs and outputs ([document.md](document.md) has the rules). Also the checks a network passes on `add_policy`: `check_slot_tables` against its inputs and outputs, `onnx_output_width` against the actuated joints. Serialized as one entry of the scene's `policies` list in `manifest.json`, pointing at `policy/<id>.onnx` and at its MDP by id. `add_motion_wandb` / `add_motion_hf` call `source.*`; `motion.py` attaches a tracking task's clip.

## `splat.py`: `SplatConfig` / `SplatHandle`

A 3D Gaussian Splat (`.spz`) background: scale, position offsets, Euler rotations, an optional collider mesh URL, and a `control` flag exposing live calibration sliders. `add_splat_hf` downloads the file and bundles it as a local one is.

## `skin.py`: `SkinConfig` / `SkinHandle`

A rigged character drawn over the robot, outside the model: [skins.md](skins.md).

## `viewer.py`: `ViewerConfig`

Camera parameters (lookat, distance, fovy, elevation, azimuth) and tracking mode (`OriginType`: AUTO / WORLD / ASSET_ROOT / ASSET_BODY). Elevation and azimuth are MuJoCo's (the direction the camera looks), so an mjlab task's values carry over unchanged. `ViewerConfig.from_position()` computes spherical params from a Cartesian viewer position.

The graphics settings follow mjlab's fields: `color_management` (`"classic"`, MuJoCo's classic renderer's colours, or `"filament"`, its Filament renderer's colour management, tone mapping and exposure), the Filament-only `tone_mapping`, `exposure` (stops), `output_color_space` and `base_color_type`, which a classic config refuses, and `enable_shadows` / `enable_reflections` / `enable_haze`, MuJoCo's `mjRND_*` flags. The manifest carries them as the scene's `graphics`, beside `camera` (format 9); `SceneHandle.viewer` hands the config back for `dataclasses.replace`. How the build resolves them and the browser draws them: [build.md](build.md), [frontend.md](frontend.md).
