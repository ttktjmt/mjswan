# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

API-wide rename for consistency: `add_[layer]_[source]` for methods that add an
object, `enable_`/`set_` for toggles, spelled-out MDP binding names. All pre-0.8
names stay importable via `mjswan/_compat.py` until 0.9; renamed methods, modules
and `register_*` functions emit a `DeprecationWarning`, the MDP binding *class*
aliases stay silent (a type alias cannot warn on attribute access). The
velocity-command shortcuts were removed outright, see Removed.

### Added

- `createEngine({ ortWasmPaths })`: where onnxruntime-web fetches its runtime files, for a
  consumer that wants ORT somewhere other than beside the bundle — a shared mirror, say.
  Takes ORT's own `wasmPaths` shapes; a bare prefix string has to serve a full
  `onnxruntime-web/dist/`, since the loader the prefix form asks for is not in this package.

- **Every term entry says which function it is** ([#118](https://github.com/ttktjmt/mjswan/issues/118)):
  `func` (`<module>:<qualname>`), `doc` (the docstring's first line) and `params`
  (JSON-safe: a `SceneEntityCfg` becomes `{entity, joint_names?, …}`) on observation
  terms, fused-group `layout` rows, terminations and their `lanes`, events, and a
  command's reset graph; a traced command names its class. Every written `.onnx` also
  carries `producer_name = "mjswan"`, a `doc_string` and `metadata_props`
  `mjswan.kind` / `mjswan.term` / `mjswan.func`, so Netron's Model Properties name the
  source. Additive — `format` stays 1 and the runtime ignores the keys; they exist for
  readers of the document, mjswan Cloud's inspector first.
- **The build output is a simulation document**
  ([ADR 0006](docs/adr/0006-swn-simulation-document.md)): one `manifest.json` at the
  root — `{format, version, uses_custom_js, plugins?, projects}`, every key `snake_case`
  — over `<project-id>/<scene-id>/{scene.mjz, mdp/<mdp-id>/…, policy/<policy-id>.onnx,
  assets/…}`. Every path under a scene entry resolves against the scene directory. The
  `format` integer versions the layout (an engine refuses a newer document, naming both
  values); `version` is the release that wrote it, and never a gate.
- **`.swn`**: the document as one file. `MjswanApp.save_document()` writes the manifest
  and the project directories — nothing of the engine — as a ZIP of the same tree;
  `mjswan info`, `mjswan serve`, `mjswan publish` and `publish_dist` accept a `.swn`
  wherever they take a directory, and a `.swn` publishes exactly the file set its directory
  would. Serving one expands it beside the packaged engine (`MjswanApp.from_document`) into
  a temporary app; a custom-JS document is refused, since its plugin module ships with the
  engine rather than in the document.
- **`MdpConfig`**: the five term sets a policy runs against — observations, actions,
  terminations, commands and now **events** — as one object. Policies handed the same
  `MdpConfig` share one MDP, traced and written once under `mdp/<mdp-id>/`; the term-set
  kwargs on `add_policy` build an anonymous one, and `add_policy_wandb` builds one per
  call so a run's checkpoints share it. An MDP built for a single policy takes that
  policy's id, so `mdp/locomotion/` sits beside `policy/locomotion.onnx`; a shared one is
  `mdp_0`, `mdp_1`, … per scene in first-use order, and a `name` (`name2id(name)`) wins
  over both.
- `add_policy(events=...)`: events belong to the policy's MDP. A scene's `events` /
  `set_events` are the default for policies that declare none, so an mjlab task's
  `env_cfg.events` lands where it did before. Switching to a policy with a different MDP
  restores the model values the previous MDP's startup randomization changed, reseeds the
  term PRNG from `termSeed`, and runs the new MDP's startup events — so randomization no
  longer compounds across switches and A → B → A reproduces A's draw.
- `add_policy(in_keys=..., out_keys=...)`: the network's slot tables, declared beside the
  network and checked against its input and output counts. A multi-input network without
  `in_keys` is refused at `add_policy` rather than going inert at playback; a slot naming no
  observation group fails the build. A multi-*output* network without `out_keys` is warned
  about instead of refused, naming the output the actuators will be driven from. Written to
  the manifest only when they differ from the defaults (`["actor"]`, `["action"]`).
- `add_project(default=True)` and `add_policy(default=True)` pick what the app opens on;
  two siblings both marked fail the build, none marked means the first added.
- `mjswan info` lists each scene's MDPs with their traced-graph counts, and reads a
  `.swn`.
- **MDP term bodies are traced to ONNX at build time and run by ONNX Runtime Web**
  ([ADR 0005](docs/adr/0005-onnx-traced-terms-superseding-the-declarative-dsl.md)),
  replacing the hand-written TypeScript DSL (see Removed). mjlab's real
  observation / termination / event / command functions are exported as graphs, so
  there is no second copy of the math and no closed primitive set. A term with no
  browser implementation now fails the build instead of being dropped.
- Traced-graph coverage for mjlab's structured sensors: `RayCastSensor` (rays cast
  in the browser, completing `height_scan`) and `ContactSensor`.
- Seeded PRNG behind every term's randomness (`createEngine({ termSeed })`, reported
  back as `MjswanEngineState.termSeed`), so a recorded session replays.
- **WebXR hand tracking as bodies in the simulation** (`createEngine({ handTracking:
  true })`, or `?hands=1` on the bundled app). A headset can bat a scene's objects around,
  rest one on an open palm, and pinch to pick one up: a 2 kg box, lifted by friction
  alone. Opt-in: the bodies are added to every scene the build loads, at about 1.6x per
  physics step.
- **Thumbstick locomotion in VR.** The camera and the tracked hands now hang off an XR
  rig, which is what a session moves: the left stick slides the viewer along its heading,
  and the right stick turns it about the head for as long as it is held, at a rate the
  deflection scales. The rig returns to the origin when the session ends, and the desktop
  camera comes back to the offset it had. Its height comes from a `mj_ray` cast straight
  down under the head, so the viewer stands on the ground at their own height and a
  generated `Rough` terrain no longer buries the view: z = 0 is a generator's base plane,
  not its surface. Body tracking does not carry a viewer in a session: you arrive beside
  the tracked body, where the viewer config points the desktop camera, and move yourself
  from there, so neither the robot's climbing nor an episode reset's teleport to a new
  spawn patch moves you. A desktop chase camera still follows both, as it did.
- **Passthrough AR.** A device that supports `immersive-ar` gets a **Start AR** button
  beside **Enter VR**, and in a see-through session the skybox and the ground planes stop
  being drawn, so the room shows behind the scene. three.js already clears the framebuffer
  transparent for an `alpha-blend` blend mode; what covered the room was mjswan's own
  drawing: a `CubeTexture` background is rendered as a box mesh, and MuJoCo's infinite
  plane is a full-screen quad. The button is mjswan's rather than three's `ARButton`, which
  pins the session to the `local` reference space and would leave the floor at eye height.
- Debug visualisation for command terms, mirroring mjlab's `debug_vis`: `default_viz()`
  emits arrows and markers as data, toggled via `engine.debugVis.set`, on by default.
- **`UniformVelocityCommandCfg` binds to a traced command term in mjswan itself**, so a
  task built on mjlab's locomotion commands migrates with no registration of its own;
  `add_scene_mjlab` picks it up from `env_cfg` like any other term. The binding used to
  live in `examples/mjlab/defaults/commands`, out of reach of other projects. The
  trace-friendly rewrite lives in `mjswan.envs.mdp.commands`; `velocity_command()` stays
  for a manual control panel on a scene with no mjlab task.
- `Builder.add_project_mjlab(task_id, ...)`, the instance-method counterpart to the
  `Builder.from_mjlab` classmethod, for adding an mjlab task to a builder that already
  has other projects. `from_mjlab` now delegates to it.
- `ObservationTermCfg.history_steps`: sparse look-back offsets for a term, e.g.
  `(0, 1, 2, 4, 8, 16)`, where mjlab's `history_length` can only count frames. The
  runtime now stacks per-term history at all (the build emitted `history_length` and
  nothing read it).
- Look-ahead reference slots on the built-in `TrackingCommand`: `ref_root_pos_w`,
  `ref_root_quat_w`, `ref_joint_pos` (the reference trajectory sampled at the command's
  `time_steps` offsets) and `is_ready`, for policies trained on a window of the clip.
- `build_single_entity_trace_env(commands=...)` and `TraceCommandManager`, so a traced
  term can read a command that exists browser-side only.
- `ReferenceJointPositionActionCfg` (`joint_position_reference`): joint targets as a
  motion reference plus a scaled residual, `q_cmd = q_ref(t) + scale * a`. The offset is
  the tracking command's reference pose and moves every control step, where
  `JointPositionActionCfg` offsets from a constant default pose. This is the control law
  a ZEST / BeyondMimic-style tracking policy uses.
- Anchor-frame reference state fields on `TrackingCommand`: `anchor_lin_vel_w`,
  `anchor_ang_vel_w`, `ref_base_height`, `ref_base_lin_vel_b`, `ref_base_ang_vel_b`,
  `ref_gravity_b`, `joint_pos` and `tracked_joint_pos`. Tracking tasks read these off
  mjlab's `MotionCommand` as properties, which the tracer turns into command slots, so
  their own functions trace unmodified.
- **`mode="manual"` event terms, fired from the control panel.** A manual term has no
  schedule: it runs when the operator presses its button. The panel's new `Events`
  section draws one button per manual term and one checkbox per `mode="interval"` term,
  arming or disarming that term's schedule (a disarmed timer stops counting rather than
  banking the wait, so re-arming cannot fire on the next frame). Both are engine verbs
  too (`engine.events.fire(name)` / `engine.events.setArmed(name, armed)`), and the terms
  are reported as `MjswanEngineState.events`; `label` sets the control text, defaulting to
  the term name. This is mjswan's own mode, not one of mjlab's four: mjlab has a viewer to
  bolt a button onto and Python state to gate a term with, where a traced graph has
  neither. A manual term left in an mjlab config is inert there. `disabled_when` names the
  `mode="interval"` term that owns the same job: while that schedule is armed the button
  greys out and `fire()` refuses, so a timer and a button never drive it at once. The
  build refuses a gate that names no interval term of the scene.
- **`dr.geom_size` is described for the browser**, with the broadphase bounds mjlab
  recomputes from it: `geom_rbound` and `geom_aabb` follow the new size in the same pass,
  by geom type (sphere, capsule, ellipsoid, cylinder, box), because a geom that grows
  while its bound stays as compiled stops colliding at its own surface. Any other geom
  type fails the build, naming the geoms and their types; mjlab raises the same refusal,
  but at the first firing.
- The `mjlab-to-mjswan` agent skill ([skills/mjlab-to-mjswan/](skills/mjlab-to-mjswan/)), published from this repo as the `mjswan` Claude Code plugin (`/mjswan:mjlab-to-mjswan`): it ports one mjlab task from any repo into a browser app.

### Changed

- **The engine bundle fetches ONNX Runtime Web's wasm from beside itself**
  ([#123](https://github.com/ttktjmt/mjswan/issues/123)): `dist/mjswan.js` pointed
  `ort.env.wasm.wasmPaths` at `cdn.jsdelivr.net/npm/onnxruntime-web@<version>/dist/`, so
  every policy-driven scene loaded ORT's `.mjs` loader as script from a CDN the host does
  not control — and a host serving `dist/` from its own origin still had to allow
  jsDelivr in `script-src`. The build now emits ORT's wasm as
  `dist/ort-wasm-simd-threaded.jsep.wasm` and names it in `wasmPaths`, resolved against
  `import.meta.url`: a host that serves `dist/` needs no second origin, and no `.mjs` is
  fetched at all, since the bundled ORT build carries its loader inlined. The ORT version
  is still fixed at build time — it is now the bytes in the package rather than a URL. A
  consumer serving `dist/` as published sees only the change of origin; one that mirrored
  `onnxruntime-web` for the old URL can stop.

- **`dist/` ships each WASM once, and `dist/` is 47 MiB instead of 87**
  ([#123](https://github.com/ttktjmt/mjswan/issues/123)): the SPA and library builds write
  into one directory and had been disagreeing about where — the SPA under `assets/`, the
  library build renaming everything to `mjswan-engine-<hash>.wasm` — so 40 MiB of
  byte-identical MuJoCo and ONNX Runtime WASM shipped twice: in the npm package, in the
  wheel (`dist/` is packaged despite being gitignored), and in every built app, which
  copies the whole directory. Both now name each file from its source basename plus a Vite
  content hash, flat in `dist/`, so identical bytes land on one path. The moved files are
  referenced only by the bundles' own `new URL(…, import.meta.url)`, which Vite rewrites,
  so nothing names them from outside; a built app's WASM sits beside `index.html` rather
  than under `assets/`.

- **Every scene carries a `camera`, and the view follows the robot by default.** The
  build used to omit the block for a scene that never called `set_viewer`, and the
  browser filled the gap from defaults generated out of the Python dataclass. The two
  did not agree: a missing block left tracking off, so the camera watched the robot walk
  out of frame, while an explicit `ViewerConfig()` with the same values tracked it. The
  build now resolves the whole block for every scene, `fovy` included, and the browser
  reads what the document says. A scene that set no viewer therefore tracks the first
  non-world body, as one that set the defaults by hand always did.

- **The default observation slot is `actor`** (was `policy`), mjlab's own name for the
  group its actor network reads, in Python (`DEFAULT_OBS_GROUP_KEY`) and the engine
  together. A lone observation group lands there and needs no `in_keys`; the common fused
  graph is `mdp/<mdp-id>/obs/actor.onnx`. The engine's `obs`→`policy` input-name special
  case is gone; the mapping from `in_keys` onto a network's inputs is positional, so what
  the network calls its tensors never matters.
- **Every project, scene, MDP, policy and splat has an id**, `name2id(name)`, unique
  within its parent — the directory it is written to and the value `?project=` /
  `?scene=` / `?policy=` take. Two siblings whose names sanitize alike get `<id>` and
  `<id>_1` with a `RuntimeWarning` naming both. The frontend's `sanitizeName` is the same
  function, pinned to `name2id` by a shared table of cases (apostrophes, parentheses and
  accents included), so `?scene=newton_s_cradle` opens Newton's Cradle.
- mjlab's network-keyed observation dict is reduced by the runner's whole `obs_groups`,
  not by the `actor` name: only groups the runner attributes to a network are renamed or
  dropped, so a multi-input policy's own slots (`command_`) survive beside `actor`.
  `adapt_observations(obs_groups=...)` replaces `policy_groups=`; `MjlabRunnerDefaults.obs_groups`
  replaces `policy_obs_groups`.
- A `config_path` sidecar contributes the checkpoint's own defaults (`policy_joint_names`,
  `default_joint_pos`, an `actions` block) and nothing else: its `onnx` block is dropped
  and any `in_keys` / `out_keys` in it are ignored with a warning pointing at `add_policy`.
- `mjswan/manifest`: `parseManifest` takes the new `Manifest` shape (`AppConfig` remains
  as a deprecated alias), maps its `snake_case` onto the engine's camelCase
  (`ViewerConfig`, `SplatTransform`), orders the default project first, and gives the
  engine a policy config that is the manifest entry merged with its MDP — slot tables at
  the top level, events included. `SceneInput` lost `events` and `graphs`; `PolicyInput.graphs`
  carries the whole MDP's graphs, events included, keyed by scene-relative path.
- Every script under `examples/` is restructured for the document layout: bare groups for
  single-input policies, the Go2 slot tables spelled out in `examples/demo/main.py`, and
  the demo sidecars stripped of their `onnx` blocks.
- **Methods**
  - `ProjectHandle.add_mjlab_scene` → `ProjectHandle.add_scene_mjlab`
  - `SceneHandle.add_policy_from_wandb` → `SceneHandle.add_policy_wandb`
  - `SceneHandle.set_viewer_config` → `SceneHandle.set_viewer`
  - `SceneHandle.add_splat_section` → `SceneHandle.enable_splat_section`
  - `PolicyHandle.add_motion_from_wandb` → `PolicyHandle.add_motion_wandb`
    (parameter `wandb_run_path` → `run_path`)
- **Classes**
  - `mjswanApp` → `MjswanApp` (deprecated alias kept until 0.9)
  - `ObsBinding` / `ObsFunc` → `ObservationBinding`
  - `TermBinding` / `TermFunc` → `TerminationBinding`
  - `EventFunc` → `EventBinding`
  - `CommandTermSpec` → `CommandBinding`
  - `MjlabMdpBinding` → `MdpBinding`
- **Functions**
  - `register_obs_func` → `register_observation`
  - `register_termination_func` → `register_termination`
  - `register_event_func` → `register_event`
  - `register_command_term` → `register_command`
- **Modules**
  - `mjswan.viewer_config` → `mjswan.viewer`
  - `mjswan.wandb_utils` → `mjswan.wandb_io`
- The built `dist/` no longer copies the unused `logo-color.svg` (only `logo.svg`).
- The `examples` extra pins `mjlab==1.5.3` exactly (was `>=1.3.0`), moves to `mujoco`
  3.10 and adds `onnxruntime`. The pin is exact because the tracer reads mjlab's
  internals; a weekly CI parity sweep catches upstream drift.

### Deprecated

All kept as aliases via `_compat.py`, removed in 0.9:

- Renamed methods, modules and `register_*` functions, which emit a
  `DeprecationWarning`.
- The pre-0.8 MDP binding **class aliases** (`ObsBinding`, `ObsFunc`, `TermBinding`,
  `TermFunc`, `EventFunc`, `MjlabMdpBinding`, `CommandTermSpec`), restored as silent
  aliases on their original import paths. Migrate to the spelled-out `*Binding` names.

### Removed

- **`config.json` and the per-policy `<policy>.json`**, replaced by the one root
  `manifest.json` (supersedes ADR 0005 §1). The per-project `index.html` / `logo.svg`
  copies and the `main/` special case for the first project go with them: a project's
  directory is its id, and the app resolves `?project=` against ids only.
- `add_project(id=...)`: a project's id is `name2id(name)`, so the directory and the
  `?project=` value can never disagree. Passing `id=` raises `TypeError`.
- `eventGraphRefs` from `mjswan/engine`, with no alias. Events are part of a policy's MDP
  now, so `policyGraphRefs(config)` already enumerates their graphs; a scene carries no
  event list for a separate helper to scan.
- Slot tables read from a `config_path` sidecar (`onnx.meta.in_keys` / `out_keys`, or
  top-level `in_keys` / `out_keys`): declare them on `add_policy` instead.
- **`mjswan.dsl`**, the declarative composition-graph DSL (ADR 0003) and its TypeScript
  interpreter, with no alias. Term bodies are traced to ONNX instead (see Added), so
  `div` / `sqrt` / `slice_` / `normalize` / `quat_to_rot6d_columns` and the rest have no
  successor: write the term as an ordinary mjlab-style Python function against the live
  env and let the build trace it. `scripts/verify_dsl_migration.py` goes with it.
- `PolicyHandle.add_velocity_command` / `add_command_velocity`, both with no alias. Pass
  `commands={"velocity": mjswan.velocity_command(...)}` to `add_policy()` instead.
- **`viewer_config_defaults.ts` and its generator.** The defaults were code-generated
  from Python's `ViewerConfig` on every build to keep the two in step. The document now
  carries the resolved values instead, so the browser has nothing to keep in step: the
  small view it falls back on is ordinary source, reached only by a manifest mjswan did
  not write, which also logs a warning. Nothing is generated into the template any more,
  so a fresh checkout builds and tests with no Python step in front of it.
- **The generated `custom_*.ts` registries**, and the `Observations` / `Terminations` /
  `Events` modules that held nothing else. A `ts_src` term is collected into the standalone
  `plugins.js` and reaches the engine as `EnginePlugins` (ADR 0004 §10), so nothing had
  written those files since; the build stamped four empty registries out and the runtime
  spread four empty objects into the maps the plugins already filled. The comments claiming
  the registries held `ts_src` terms described the design they replaced.

### Fixed

- **White robots render white.** A `<material>` that declares no `metallic` (every
  material in Menagerie and mjlab, G1's `0.7 0.7 0.7` and Microduck's included) was
  handed MuJoCo's `specular` as its `metalness`. The two are unrelated: `specular` is a
  Blinn-Phong highlight coefficient MuJoCo *adds* to full diffuse albedo, `metalness` is
  three.js' dielectric/conductor switch, so its 0.5 default made every such material half
  metal. three.js scales diffuse by `(1 - metalness)` and the metallic half has no
  `scene.environment` to reflect, so a pure-white surface came out at 62% grey and the
  whole albedo range compressed. Materials that do declare `metallic` are unaffected.
  `reflectance` no longer feeds `reflectivity` either: it is mirror-reflection strength,
  already spent on `envMapIntensity`, and its 0 default forced `ior` to 1.0, flattening
  the dielectric highlight these materials now depend on.
- **The scene is lit in MuJoCo's units, so mjswan matches the MuJoCo viewer.** Measured
  against MuJoCo 3.12's renderer over five robots at a matched camera, mean absolute
  error per pixel drops from **39.5 to 5.5** (8-bit, no exposure fitting). Three fixes,
  none of which a viewer setting could compensate for: `AmbientLight` was built at
  intensity 1.0, but three.js' Lambert BRDF divides irradiance by pi, so ambient landed pi
  times darker than MuJoCo's `rgba * ambient` (the largest term, worst on mjlab scenes
  with their 0.3 headlight ambient); each light's intensity folded `specular` into
  `diffuse` and scaled it by an invented `light_intensity || 0.5`, and is now
  `max(light_diffuse) * pi`, with the `specular/diffuse` ratio moved to the material's
  `specularIntensity`, which is what stops highlights being 4x too hot under MuJoCo's
  defaults; and `ACESFilmicToneMapping` applied a film curve MuJoCo does not have,
  costing up to 56% of the saturation in coloured regions, so the renderer no longer tone
  maps. `outputColorSpace` stays `LinearSRGBColorSpace`, since MuJoCo does no colour
  management either.
- **A material that declares `reflectance="0"` no longer reflects.** `envMapIntensity`
  read it as `mat_reflectance || 0.5`, and MuJoCo's default *is* 0, so every material
  that left the attribute out, or zeroed it deliberately, got a half-strength environment
  reflection.
- **A scene's policies no longer overwrite each other's traced graphs.** A graph's
  bundle path was built from its kind and name alone (`obs/policy.onnx`) under the
  *scene* directory, while its contents are per-MDP, so a second policy in the same
  scene wrote over the first, and every earlier policy's config went on pointing at it.
  Nothing caught it at either end: the build overwrote silently, and the runtime pads or
  truncates a mismatched vector to the `size` the config declares, so a policy loading a
  sibling's graph merely behaved oddly. Graphs now live under their MDP's directory
  (`mdp/<mdp-id>/obs/actor.onnx`, see Added), and a write that would replace a
  *different* graph at one path fails the build rather than shipping it.
- **The control panel is mjviser's, control for control.** Two viewers onto the same
  mjlab task should not look like two products, so the panel now renders what mjlab's
  viser GUI renders: command groups nested in a `Commands` folder, one row per control
  with the label in a `5.975em` column, a slider carrying its two ends as marks *and* a
  `3rem` number box that takes typing (at mjviser's `1.875em` height, which keeps a
  slider row as tall as the checkbox row above it), the "Max" companion above the axis it
  rescales, a checkbox in the control column rather than captioning itself, and a
  full-width filled button carrying the icon its mjlab GUI asked for (`Icon.SQUARE_X` on
  `Zero` being the only one mjlab declares today). The slider thumb is mjviser's bar
  rather than Mantine's dot (`thumbSize: 0` plus a `0.5rem × 0.75rem` box in the slider's
  own colour) on a square-cornered `xs` track. Sizes are `em`-relative, so both panels
  stay identical under their own root font size, and the slider styling lives in the
  theme, so every slider in the app follows, the splat calibration panel included.
  `ButtonConfig` grew an `icon`, recorded by the GUI spy from the term's `create_gui`.
- **The control panel draws button commands.** `button` has been a command input type all
  along, `CommandManager.triggerButton` wired to the term, and `engine.commands.trigger` a
  verb, but the panel filtered its controls down to sliders and checkboxes, so mjlab's own
  `Zero` (declared by every `UniformVelocityCommand` GUI) never existed for a viewer.
  Buttons now render in declaration order, so `Zero` lands under the sliders it zeroes,
  and the values a press moves are re-read from the term rather than left stale in the
  panel's mirror. `UiCommand` answers to `zero` as well, so a hand-written panel gets the
  same button, and a press no term answers to warns once instead of doing nothing.
- A position action term now inherits its PD gains from the entity's actuator configs.
  mjlab's ideal-PD family (`IdealPdActuatorCfg` and subclasses, which is what
  `wbc-mjlab`'s G1 uses) puts a `<motor>` in the model and computes
  `kp·(q* − q) + kd·(0 − q̇)` in torch. The browser mirrors that for a `biastype=none`
  actuator, but read the gains off the *action* term, where mjlab keeps them on the
  *actuator*, so every such task got kp = kd = 0, every `ctrl` zero, and a robot that
  ignored its policy and collapsed. A motor term with no stiffness is now an error rather
  than running limp.
- `register_command` now maps a command config wherever its class lives. The adapter only
  consulted the registry for classes from the `mjlab` package, so a task's own
  `CommandTermCfg` subclass passed through unadapted and failed later in the serializer
  on a missing `pending_trace`.
- The action adapter maps a config wherever its class lives too, and no longer lets a
  rewrite escape into a caller's config. A task's own `ActionTermCfg` subclass passed
  through unconverted, and `resolve_action_scales` then rewrote its `scale` keys in place,
  on the very object the task's live env config holds, leaving mjlab unable to resolve
  them several frames from the cause. An unrecognized term is now copied rather than
  shared.
- The observation and termination adapters convert a config wherever its class lives, as
  the command and action adapters now do. `_is_from_mjlab` read the leaf class's own
  module, so a task's *subclass* of an mjlab `ObservationGroupCfg` passed through
  unadapted, carrying mjlab term objects into the serializer, which failed on the first
  mjswan-only field it read (`AttributeError: 'ObservationTermCfg' object has no
  attribute 'history_steps'`). The whole MRO is consulted now.
- The velocity command's trace-friendly rewrite carries the rest of what mjlab's
  `UniformVelocityCommand` does: forward-only envs (`rel_forward_envs`, which mjlab's own
  velocity tasks set to `0.2` and `play=True` does not clear, so one resample in five
  differed), world-frame envs (`rel_world_envs` and the `vel_command_w` state it reads),
  and the `heading_command` gate, where the rewrite used to track heading unconditionally.
  `init_velocity_prob` is refused with a message rather than silently dropped, since it
  writes the robot's root state during resampling. The parity harness cannot see any of
  this, as it traces the overridden term and compares against that same term, establishing
  "graph == override" and never "override == mjlab"; `tests/test_velocity_command.py` now
  pins the latter directly.
- **`history_length` now stacks oldest frame first**, `[x_{t-n+1} … x_t]`, the order
  mjlab's `CircularBuffer` flattens, where it counted back from the newest frame. Every
  mjlab task carrying per-term or group history was handing its policy a correct-*width*
  observation with time running backwards. **A hand-written config trained on the
  newest-first layout must name its offsets to keep it:** `history_length=3` becomes
  `history_steps=(0, 1, 2)`. The bundled examples are migrated; `history_steps` is
  unchanged and still takes precedence. Neither the parity harness nor a build error could
  have caught this: parity compares term bodies, and history is orchestration around them.
- `PolicyRunner`'s group-level frame stack (the hand-authored `{components,
  history_steps}` shape) stacks oldest-first too, so one rule holds across both history
  paths. `history_interleaved` follows the stack, so its layout is now
  `[a0_{t-n+1}, …, a0_t, a1_…]`.
- A group's `history_length` replaces its terms' whenever it is set, `0` included, which
  switches history off for the group as mjlab's `ObservationManager` does. An explicit `0`
  used to read as "unset" and leave each term's own count standing.
- An event's write now lands on each entity it was made on. The target was read off an
  `asset_cfg` param, which mjlab's own terms carry but a task's term need not (a thrown
  ball's launcher takes a plain `ball_name`), so it serialized as `null` and the runtime
  wrote every root pose to the model's *first* free joint: in a robot-and-ball scene that
  launched the robot across the floor while the ball never moved. The tracer now keys a
  capture by `(entity, kind)`, as mjlab writes per entity, so one term may write several,
  each target naming the graph outputs holding its values. `asset_cfg` stays the fallback
  for a write the proxies could not attribute, and its `joint_ids` scope its own entity.
- The runtime resolves a named entity's free joint through its **joint** name prefix, the
  rule mjlab uses for an entity's own addresses (and the one `entityJointIds` already
  applied). A namespaced model with no such entity now writes nothing rather than falling
  back to the first free joint, which is how a thrown ball launched the robot. An
  unprefixed model is still a single-entity scene, where the one free joint is the
  entity's.
- A root velocity write rotates its angular half into the body frame, as mjlab's
  `write_root_velocity` does. Both halves arrive world-frame, but a free joint's `qvel`
  holds world-frame linear and *body*-frame angular velocity, so the value spun the body
  about the wrong axis for any orientation but identity.
- `write_root_state_to_sim` traces, split into the pose and velocity writes mjlab splits
  it into, and a `write_*_to_sim` the tracer does *not* capture is refused instead of
  forwarded: forwarding it mutated the live tracing env, leaving every later term reading
  a moved sim. A term walking `scene.entities` gets recording stand-ins for the same
  reason.
- mjlab's default `reset_scene_to_default` event no longer fails the build. It restores
  every entity's default root and joint state, which the runtime's own reset already does
  (`mj_resetData` to `qpos0`, or keyframe 0) before any `mode="reset"` event runs, so it is
  emitted as a native no-op with that reason.
- `run_parity` no longer feeds a graph inputs it does not declare, matching the runtime. A
  read the body only *indexes* with (a tracking command's `time_steps`) is recorded as a
  slot but folded in as a constant, so the export prunes it and ORT refused the feed,
  failing the check for every term of a tracking task.
- `OnnxCommand` / `OnnxEvent` no longer feed a graph inputs it does not declare. The export
  prunes an input the body never reads, so a term that draws nothing has no `rand` and a
  state field written without being read has no `prev_<field>`; ORT rejects either, taking
  the whole scene down with `invalid input '...'`. Feeds are now filtered to the session's
  own input names.
- **Switching project pins the project it switched to.** The URL was written at each
  call site from React state. The write that followed a project change is the one for the
  scene that change loads, and it still held the *previous* project in its closure, so it
  put the old one back. `?project=old&scene=new_projects_scene` is a view that cannot be
  reopened: the reload resolves the old project, fails to find that scene in it, and lands
  on the default. The address bar is now a projection of the live selection, written in
  one place from all of it, so a selection that moves two things at once records both.
  Clearing a policy clears `?policy` for the same reason, where the merge used to read
  the cleared `null` as "unchanged" and leave the old name behind, and the parameters
  are normalized on load rather than at the first interaction, so the address bar is
  copyable as soon as the scene is up.
