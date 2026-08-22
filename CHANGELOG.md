# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

API-wide rename for naming consistency. Methods that add an object now follow
`add_[layer]_[source]`; toggles use `enable_`/`set_`; the MDP binding layer is
spelled out. **All pre-0.8 names remain importable as aliases via
`mjswan/_compat.py` until 0.9; renamed methods, modules, and `register_*`
functions emit a `DeprecationWarning`, the MDP binding *class* aliases stay
silent (a type alias cannot warn on attribute access). The velocity-command
shortcuts were removed outright (no alias) — see Removed.**

### Added

- **MDP term bodies are traced to ONNX at build time and run by ONNX Runtime Web**
  ([ADR 0005](docs/adr/0005-onnx-traced-terms-superseding-the-declarative-dsl.md)),
  replacing the hand-written TypeScript DSL — see Removed. mjlab's real
  observation / termination / event / command functions are exported as graphs, so
  there is no second copy of the math to keep in numeric lockstep and no closed
  primitive set to extend. A term with no browser implementation now fails the
  build instead of being dropped.
- Traced-graph coverage for mjlab's structured sensors: `RayCastSensor` (rays cast
  in the browser, completing `height_scan`) and `ContactSensor`.
- Seeded PRNG behind every term's randomness (`createEngine({ termSeed })`,
  reported back as `MjswanEngineState.termSeed`), so a recorded session replays.
- Debug visualisation for command terms, mirroring mjlab's `debug_vis` — arrows and
  markers are emitted as data by `default_viz()`, toggled via `engine.debugVis.set`,
  and on by default as in mjlab's viewer.
- `Builder.add_project_mjlab(task_id, ...)` — instance-method counterpart to the
  `Builder.from_mjlab` classmethod factory, for adding an mjlab task to a builder
  that already has other projects. `from_mjlab` now delegates to it.
- `ObservationTermCfg.history_steps` — sparse look-back offsets for a term, e.g.
  `(0, 1, 2, 4, 8, 16)`, where mjlab's `history_length` can only count frames.
  The runtime now stacks per-term history at all (previously the build emitted
  `history_length` and nothing read it, so per-term history was dropped).
- Look-ahead reference slots on the built-in `TrackingCommand`: `ref_root_pos_w`,
  `ref_root_quat_w`, `ref_joint_pos` (each the reference trajectory sampled at the
  command's `time_steps` offsets) and `is_ready`, for policies trained on a window
  of the clip rather than the current frame alone.
- `build_single_entity_trace_env(commands=...)` and `TraceCommandManager`, so a
  traced term can read a command that exists browser-side only.
- `ReferenceJointPositionActionCfg` (`joint_position_reference`) — joint position
  targets as a motion reference plus a scaled residual, `q_cmd = q_ref(t) + scale * a`.
  The offset is the tracking command's reference pose and moves every control step,
  where `JointPositionActionCfg` offsets from a constant default pose. This is the
  control law a tracking policy trained ZEST / BeyondMimic-style uses.
- Anchor-frame reference state fields on `TrackingCommand`: `anchor_lin_vel_w`,
  `anchor_ang_vel_w`, `ref_base_height`, `ref_base_lin_vel_b`, `ref_base_ang_vel_b`,
  `ref_gravity_b`, and `joint_pos` / `tracked_joint_pos`. A whole-body tracking task's
  observation terms read these off mjlab's `MotionCommand` as properties, which the
  tracer turns into command slots — so with the browser answering to those names, the
  task's own functions trace unmodified.

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
- The `examples` extra pins `mjlab==1.5.3` exactly (was `>=1.3.0`) and moves to
  `mujoco` 3.10, adding `onnxruntime`. The pin is exact because the tracer reads
  mjlab's internals; a weekly CI parity sweep is what catches upstream drift.

### Deprecated

All kept as aliases via `_compat.py`, removed in 0.9:

- Renamed methods, modules, and `register_*` functions — emit a
  `DeprecationWarning`.
- The pre-0.8 MDP binding **class aliases** — `ObsBinding`, `ObsFunc`,
  `TermBinding`, `TermFunc`, `EventFunc`, `MjlabMdpBinding`, `CommandTermSpec` —
  restored as silent aliases on their original import paths. Migrate to the
  spelled-out `*Binding` names (`ObservationBinding`, `TerminationBinding`,
  `EventBinding`, `MdpBinding`, `CommandBinding`).

### Removed

- **`mjswan.dsl`** — the declarative composition-graph DSL (ADR 0003) and its
  TypeScript interpreter, removed outright with no alias. Term bodies are traced
  to ONNX instead (see Added), so `div` / `sqrt` / `slice_` / `normalize` /
  `quat_to_rot6d_columns` and the rest have no successor: write the term as an
  ordinary mjlab-style Python function against the live env and let the build
  trace it. `scripts/verify_dsl_migration.py` goes with it.
- `PolicyHandle.add_velocity_command` / `add_command_velocity` — both removed
  with no alias. Pass `commands={"velocity": mjswan.velocity_command(...)}` to
  `add_policy()` instead.

### Fixed

- A position action term now inherits its PD gains from the entity's actuator configs.
  mjlab's ideal-PD family (`IdealPdActuatorCfg` and subclasses, which is what
  `wbc-mjlab`'s G1 uses) puts a `<motor>` in the model and computes
  `kp·(q* − q) + kd·(0 − q̇)` in torch. The browser mirrors that for a `biastype=none`
  actuator, but read the gains off the *action* term, where mjlab keeps them on the
  *actuator* — so every such task got kp = kd = 0, every `ctrl` zero, and a robot that
  ignored its policy entirely and collapsed. The runtime also now reports a motor term
  with no stiffness as an error rather than running it limp.
- `register_command` now maps a command config wherever its class lives. The adapter
  only consulted the registry for classes from the `mjlab` package, so a task's own
  `CommandTermCfg` subclass — which is where a downstream project's commands are —
  passed through unadapted and failed later in the serializer on a missing
  `pending_trace`.
- The action adapter maps a config wherever its class lives too, and no longer lets a
  rewrite escape into a caller's config. A task's own `ActionTermCfg` subclass passed
  through unconverted, and `resolve_action_scales` then rewrote its `scale` keys in
  place — on the very object the task's live env config holds, leaving mjlab unable to
  resolve them when the tracing env was built, several frames from the cause. An
  unrecognized term is now copied rather than shared.
- The observation and termination adapters convert a config wherever its class lives,
  as the command and action adapters now do. `_is_from_mjlab` read the class's own
  module, so a task's *subclass* of an mjlab `ObservationGroupCfg` — how a project adds
  a field mjlab has no notion of — passed through unadapted, carrying mjlab term objects
  into the serializer, which failed on the first mjswan-only field it read
  (`AttributeError: 'ObservationTermCfg' object has no attribute 'history_steps'`). The
  whole MRO is consulted now, not the leaf class.
- **`history_length` now stacks oldest frame first** — `[x_{t-n+1} … x_t]`, the order
  mjlab's `CircularBuffer` flattens — where it counted back from the newest frame. Every
  mjlab task carrying per-term or group history was handing its policy a correct-*width*
  observation with time running backwards, and the two sides now mean one thing by a
  count. **A hand-written config trained on the newest-first layout must name its offsets
  to keep it:** `history_length=3` becomes `history_steps=(0, 1, 2)`. The bundled examples
  are migrated; `history_steps` is unchanged and still takes precedence. Neither the
  parity harness nor a build error could have caught the mismatch: parity compares term
  bodies, and history is orchestration around them.
- `PolicyRunner`'s group-level frame stack (the hand-authored `{components,
  history_steps}` config shape) stacks oldest-first too, so one rule holds across both
  history paths. `history_interleaved` follows the stack, so its layout is now
  `[a0_{t-n+1}, …, a0_t, a1_…]`.
- A group's `history_length` replaces its terms' whenever it is set — `0` included, which
  switches history off for the group, as mjlab's `ObservationManager` does. An explicit
  `0` used to read as "unset" and leave each term's own count standing.
- An event's write now lands on the entity it was made on, and on each entity it was made
  on. The write target's entity was read off an `asset_cfg` param, which mjlab's own terms
  carry but a task's term need not — a thrown ball's launcher takes a plain `ball_name` —
  so it serialized as `null`, and the runtime then wrote every root pose to the model's
  *first* free joint. In a two-entity scene (a robot and a ball) that launched the robot
  across the floor while the ball never moved. The tracer keys a capture by `(entity,
  kind)`, as mjlab writes per entity, so one term may now write several; each target
  names the graph outputs holding its values. `asset_cfg` stays the fallback for a write
  the proxies could not attribute, and its `joint_ids` scope its own entity only.
- The runtime resolves a named entity's free joint through its **joint** name prefix, the
  same rule mjlab uses to resolve an entity's own addresses (and the one `entityJointIds`
  already applied). A namespaced model with no such entity now writes nothing rather than
  falling back to the first free joint — mjlab raises on the scene lookup, and the
  fallback is how a thrown ball launches the robot. An unprefixed model is still a
  single-entity scene, where the one free joint is the entity's.
- A root velocity write rotates its angular half into the body frame, as mjlab's
  `write_root_velocity` does. Both halves arrive world-frame, but a free joint's `qvel`
  holds world-frame linear and *body*-frame angular velocity, so the value was spinning
  the body about the wrong axis for any orientation but identity.
- `write_root_state_to_sim` traces, split into the pose and velocity writes mjlab splits
  it into, and a `write_*_to_sim` the tracer does *not* capture is refused instead of
  forwarded — forwarding it mutated the live tracing env, leaving every term traced after
  it reading a moved sim. A term walking `scene.entities` gets recording stand-ins for the
  same reason.
- mjlab's default `reset_scene_to_default` event no longer fails the build. It restores
  every entity's default root and joint state, which is what the runtime's own reset
  already does (`mj_resetData` to `qpos0`, or keyframe 0) before any `mode="reset"` event
  runs, so it is emitted as a native no-op with that reason.
- `run_parity` no longer feeds a graph inputs it does not declare, matching the runtime.
  A read the body only *indexes* with — a tracking command's `time_steps` — is recorded
  as a slot but folded in as a constant, so the export prunes it and ORT refused the
  feed, failing the check for every term of a tracking task.
- `OnnxCommand` / `OnnxEvent` no longer feed a graph inputs it does not declare. The
  export prunes an input the body never reads, so a term that draws nothing has no
  `rand` and a state field written without being read has no `prev_<field>`; ORT
  rejects either, taking the whole scene down with `invalid input '...'`. Feeds are
  now filtered to the session's own input names.
