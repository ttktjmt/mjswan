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

- Raw `mjData` fields as traced-graph slots: a term reading `entity.data.data.<field>`
  (mjlab's `SimData`, for what `EntityData` does not wrap — a muscle model's `act`, the
  sim `time`) now traces, with the browser serving the whole field from `mjData`.
  Replay proxies also forward `physics_dt` / `step_dt` / `cfg` from the real env, and
  `add_policy` fills `policy_num_actions` from the ONNX output width when a policy has
  no joint names (muscle policies).
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
  true })`, or `?hands=1` on the bundled app). Six mocap spheres a hand, each welded to
  a dynamic twin that owns the contact, so a headset can push a scene's objects around
  and pick them up. Opt-in: roughly a third more per physics step, in every scene.
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

- **`mjswan.dsl`**, the declarative composition-graph DSL (ADR 0003) and its TypeScript
  interpreter, with no alias. Term bodies are traced to ONNX instead (see Added), so
  `div` / `sqrt` / `slice_` / `normalize` / `quat_to_rot6d_columns` and the rest have no
  successor: write the term as an ordinary mjlab-style Python function against the live
  env and let the build trace it. `scripts/verify_dsl_migration.py` goes with it.
- `PolicyHandle.add_velocity_command` / `add_command_velocity`, both with no alias. Pass
  `commands={"velocity": mjswan.velocity_command(...)}` to `add_policy()` instead.

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
