# mjlab adapters and sources

`mjlab/` reads mjlab's own objects; `source/` turns a reference (a W&B run, a Hub repository, a URL) into local files. Both are leaves of the layering in [package-layout.md](package-layout.md).

## `mjlab/`: everything that reads mjlab's own objects

mjlab stays a soft dependency: it is imported lazily inside the functions that need it, and `onnx_meta.py` needs it not at all.

- **`observation.py`, `termination.py`, `command.py`, `action.py`, `event.py`**: the per-manager adapters. Observation, termination and event bodies are *not* name-resolved to mirrors: mjlab's own functions are traced directly ([tracing.md](tracing.md)). What remains is config-shape adaptation:
  - reducing mjlab's network-keyed observation dict to the actor's group via the task's runner config (`rl_cfg.obs_groups`): only groups it attributes to a network are renamed or dropped (`critic` is dropped), so an author's extra slots survive. A lone group lands under the default slot, `actor` ([document.md](document.md));
  - `clip_actions`, action-scale, PD-gain, default-velocity and actuator-delay resolution;
  - translating `MyoMuscleActivationActionCfg` to `MuscleActivationActionCfg` ([mdp.md](mdp.md)), and a task's own action class to `TracedActionCfg`;
  - `apply_terrain_spawn`, which swaps mjlab's untraceable `reset_root_state_from_flat_patches` for mjswan's patch-based reset (the browser runs one env where mjlab spreads many across the terrain).
- **`task.py`**: what a scene takes from `env_cfg` (the control rate, terrain data, entity specs, the viewer config).
- **`runner.py`**: what playback takes from `rl_cfg` (`resolve_runner_defaults`), and how a `model_*.pt` becomes ONNX (`create_pt_onnx_export_context`, `export_checkpoint`), aligning the observation normalizer with the checkpoint. It reuses mjlab's own policy export rather than a second mechanism.
- **`env.py`**: `build_single_entity_trace_env(spec_fn)` builds a minimal single-entity `ManagerBasedRlEnv` out of mjlab's own `Entity` / `Scene`, with no reimplemented kinematics. It configures no managers and is never stepped; it is only the tracer's `env.scene[name].data.<field>` read and write target. Joint defaults come from the model's first keyframe, matching what the browser resets to. `build_mjlab_env()` grows `nconmax` / `njmax` until a task's env fits a single-env re-use. `TraceCommandManager` supplies trace-time stand-ins for commands the browser owns (a `ui_command` has no Python side).
- **`gui.py`**: runs mjlab's own `CommandTerm.create_gui` and its viewer's `MjlabViserScene.create_scene_gui` against a recording stand-in, so the browser control panel's command controls and its **Scene** section (Camera, Debug Viz; the manifest's `viewer_gui`) come from mjlab's declaration instead of a hand-copied duplicate that drifts. `ControlPanel/viewer_gui_bindings.json` lists the recorded paths the panel wires up, and `tests/test_mjlab_gui.py` holds each to mjlab's.
- **`sim.py`**: applies a task's `MujocoCfg` sim options to a spec (monkey-patching `MujocoCfg.apply_to_spec()` onto mjlab when needed).
- **`onnx_meta.py`**: parses the `metadata_props` mjlab's `attach_metadata_to_onnx` writes into an exported policy: joint names, rest pose, action scale, and a description of each observation term. Plain strings, so it needs neither mjlab nor torch, which is what lets a Hub-fetched ONNX be added on the light path. Pitfalls:
  - the encoding is lossy: mjlab formats list values with `{:.3f}`, so ints and bools arrive as `"1.000"`;
  - `joint_names` covers *every* joint of the robot while the network emits one action per actuated joint (each action term's joints in joint order, one term after another), so `add_policy_hf` takes the order from the task's action terms where it has them and otherwise pairs the metadata against the scene's own model before using it. An action term's `actuator_names` holds *joint* patterns despite its name (`JointPositionAction` matches them against joint names);
  - observation terms are named but not carried (the functions mjswan traces are not in the file), so an observation group is never reconstructed from this.
- **`detect.py`**: `is_from_mjlab`, the duck-typed check the adapters share, and `field_default` / `differs`, with which the observation and action adapters find a config field set off its default (the first to refuse it, the second to warn).

## `source/`: a reference in, local files out

Each module backs one `add_<layer>_<source>()` family, knows nothing about what the files mean, and carries its own optional dependency: one extra per backend (`mjswan[wandb]`, `mjswan[hf]`), imported inside a function so `import mjswan` touches neither.

- **`wandb.py`**: `resolve_run_path`, `fetch_onnx`, `fetch_motion_npz` (from a run or an artifact) and `fetch_checkpoints`, a context manager yielding a run's `model_*.pt` files for `mjlab.runner` to convert. Policies are named for the file they came from (`model_0`, `model_50`, …).
- **`hf.py`**: the same verbs against the Hugging Face Hub, which holds the finished artifact, so neither mjlab nor torch is involved. With no filename given it takes `policy.onnx`, then `final.onnx`, then the repository's single `.onnx` (the names the Hub's own mjlab download query counts); several unnamed candidates raise rather than pick. `fetch_dir` is `fetch_file` for an asset that is several files: a MuJoCo model is an MJCF plus the meshes it resolves relative to itself, so `add_scene_hf` brings the XML's whole directory down (`snapshot_download` scoped to it) and compiles the spec where it lands, license detection included ([licenses.md](licenses.md)). A splat is the opposite case (one opaque file), so `add_splat_hf` is `fetch_file` handed to `add_splat(source=)`, bundling the `.spz` the way a local one is.
- **`url.py`**: a plain http(s) URL to a local file.
