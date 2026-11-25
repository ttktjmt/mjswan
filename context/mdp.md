# The mjlab-mirror MDP layer

`managers/` and `envs/mdp/`: mjswan's classes at mjlab's import paths, so an mjlab task config imports and adapts cleanly. How terms become graphs is [tracing.md](tracing.md); the `MdpConfig` unit that holds a policy's five term sets is in [object-model.md](object-model.md).

## A deliberate asymmetry

- **`envs/mdp/actions/`** carries *real, directly usable* config classes, because mjlab's own action classes run natively in the browser: `JointPositionActionCfg`, `RelativeJointPositionActionCfg`, `JointVelocityActionCfg`, `JointEffortActionCfg`, `MuscleActivationActionCfg` are supported, as are mjswan's own `ReferenceJointPositionActionCfg` (offset by a tracking command's reference pose) and `CommandJointPositionActionCfg` (targets from a command's value, so it can drive joints the policy does not). `TendonLength/Velocity/EffortActionCfg` and `SiteEffortActionCfg` are exported so mjlab configs import cleanly but raise `NotImplementedError` at build time. `stiffness` / `damping` are mjswan-specific: the browser computes PD externally for motor actuators with `biastype=none`. `delay` carries the actuators' command delays, which `mjlab/action.py`'s `resolve_actuator_delays` reads off the actuator configs.
- **A task's own action class** adapts to `TracedActionCfg`, whose process the build traces; `native_class` / `is_traced` decide, matching by class name (with `_ALIASES` for a class mjswan runs under another name), for the adapter and `run_parity` alike. One of mjlab's own without an mjswan class (`DifferentialIKActionCfg`) is skipped with a warning. `managers/action_manager.py` holds the base `ActionTermCfg`, where mjlab defines its counterpart, and `ActionBinding` / `register_action`, which name what a class's overridden `apply_actions` does (`apply` kind, `interpolate`, `hold_s`, and `trace_override` to adjust the built term before it is traced). The rules are in [tracing.md](tracing.md).
- **`envs/mdp/{observations,terminations,events}.py`** carry only the `*Binding` escape hatch and its `register_*` registry, which `envs/mdp/registry.py`'s `resolve` looks a term's function up in (by function name, else by term name, which also covers a closure), for the adapter and `run_parity` alike. Pass mjlab's own functions instead; a registered override is either a trace-friendly replacement callable or a `*Binding` naming a TypeScript class by `ts_src` (`MdpBinding` is their union). A binding without `ts_src` fails the build: mjswan ships no built-in TS term classes. The one event mjswan owns is `reset_root_state_on_flat_patch`, which replaces mjlab's untraceable `reset_root_state_from_flat_patches` ([tracing.md](tracing.md)); `mjlab/event.py`'s `apply_terrain_spawn` swaps it in.
- **`envs/mdp/commands.py`** is the exception that carries bodies: the `ui_command` presets and the motion command's reset jitter (`motion_rsi_offset`).
- **`managers/`** holds the config-side counterparts (`observation_manager`, `event_manager`, `action_manager`, `termination_manager`, `command_manager`). `ObservationTermCfg` adds `history_steps` (sparse look-back offsets) and `history_interleaved` (Isaac joint-major layout) beyond mjlab's fields; training-only fields (`noise`, `delay_*`, `enable_corruption`) are accepted and ignored.

A term with no browser implementation **fails the build**, naming both ways out (a trace-friendly `register_*` replacement, or a `ts_src` TS class): [tracing.md](tracing.md).

## Actuator models

The browser computes an ideal-PD actuator's torque itself (`core/action/applyAction.ts`), so what an actuator config adds to that PD reaches it only as fields of the action term, which `mjlab/action.py` fills from the entity's actuator configs, matching classes by name in the MRO:

- `DcMotorActuatorCfg`: `saturation_effort` and `velocity_limit`, clipped as mjlab's `dc_motor_clip`.
- WBC-Mjlab's `UnitreeActuatorCfg`: `x1`, `x2`, `y1`, `y2` as `torque_speed_envelope`, clipped as its `clip_unitree_effort`. Matched by name, so mjswan does not depend on wbc-mjlab.
- Every actuator's command delay, as `delay`.

Anything else an `IdealPdActuatorCfg` subclass sets warns at build time (`warn_unrun_actuator_fields`), such as UnitreeActuator's friction, which the demo moves into the joints' `frictionloss` and `damping`.

The Unitree envelope runs although mjlab does not run it: mjlab computes the PD through a fused control law that never calls `_clip_effort`, where wbc-mjlab clips. The envelope is the motor model the config declares and the robot enforces, and the demo's WBC-Mjlab G1 policy was trained under it, on an mjlab that still called `_clip_effort`. A joint without `y2` takes its `y1`, as the class does. Rejected: leaving these motors unlimited to match an mjlab rollout, which drives the G1 with more torque than its motors have.

## Command terms

`managers/command_manager.py` defines the configs policies consume: `SliderConfig` (with `enabled_when`, and an `adjustable_range` → `SliderRangeConfig` companion slider), `CheckboxConfig`, `ButtonConfig`, `CommandUiConfig`, `CommandTermConfig`, `CommandBinding`, the `CommandInput` union, and `register_command`, which binds an mjlab command-cfg *class name* to a `CommandBinding`. `CommandTermConfig` keeps mjswan's own name: it is not mjlab's `CommandTermCfg`, since it also carries the browser UI and the pending trace.

`envs/mdp/commands.py` has the terms mjswan supplies (`velocity_command()` builds the standard locomotion 3-DoF slider command, `ui_command()` a generic UI-driven term) and the bindings for mjlab's:

- `UniformVelocityCommandCfg` is traced from its own body: a real mjlab command class is traced like any other term, its hidden state promoted to explicit graph I/O.
- `MotionCommandCfg` runs as the native `TrackingCommand`, with only its reset jitter traced. `TRACKING_STATE_FIELDS` names the fields it computes; any other a graph reads off it must be in each motion's `frame_fields` (arrays of the clip's `.npz` read at the current frame), which `build/manifest.py` checks. Its debug drawing is mjlab's ghost or body frames (`debug_vis`, `viz.mode`, `viz.ghost_color`). The binding requires mjlab's `motion_file`, `anchor_body_name` and `body_names` (checked in `mjlab/command.py`), so a class that only shares the name (MyoSuite's) raises a `TypeError`; registering a binding under the name replaces it, check included.
- A binding's `viz` restates what mjlab's `_debug_vis_impl` draws as data (`core/command/debugViz.ts` evaluates it), so a `debug_vis=True` velocity task gets its arrows without the author declaring any; a term with no drawing warns at build time.
- A command class only one task uses (`LiftingCommandCfg`) is registered by that task's port, as `examples/demo/main.py` does.

A command is a class in mjlab, as an action is, and stays one here: only its value would fit a pure expression, while its per-step update, UI input, sim writes and drawing would not. Tracing the class body with its state promoted is what gives the browser the task's own code.

## Muscle action term

`MuscleActivationActionCfg` drives MuJoCo muscle actuators (`dyntype=muscle`), resolved per actuator name; the build checks every name resolves to a muscle actuator of the model. It mirrors the contract of myosuite4's `MyoMuscleActivationActionCfg`, the class every myo\* mjlab task instantiates (musclemimic, elbow, leg walk, myouser), and the mjlab adapter runs that class as this one (`envs/mdp/actions._ALIASES`).

`action_mode` (named as mjlab and myosuite name it, so the adapter copies it across) picks the mapping from `raw = scale * a + offset`:

- `sigmoid` (default): the canonical MyoSuite sigmoid `σ(5(raw − 0.5))`, excitation in (0, 1);
- `excitation`: clips `raw` to [0, 1], for models whose actions are already excitations;
- `direct`: clips `raw` to each actuator's own `ctrlrange` and writes it unchanged, for a checkpoint trained against the raw control. A myosuite muscle model declares `ctrlrange="-1 1"` and really uses the negative half, so this is not `excitation` by another name.

`normalize` is the legacy spelling of the first two (`True` is `sigmoid`, `False` is `excitation`), adopted from myosuite4's `MuscleActionTermCfg`; `action_mode` wins when set. `normalize=False` follows myosuite4 (`clip(scale * a + offset, 0, 1)`), not legacy myosuite's raw pass-through: mjswan has no downstream renormalising stage, and the clipped form is the safer default for an arbitrary `ctrlrange`.

myosuite4's `MuscleActionTermCfg` itself is not aliased: it exposes `entity_name` and `normalize` but no `actuator_names`, so translating it needs a path that enumerates the entity's muscles, and no registered task uses it to test against. A second alias is a small addition once one does.
