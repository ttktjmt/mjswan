# Tracing MDP terms to ONNX

`compile/`: how a task's own term functions and actuator laws become ONNX graphs the browser runs, what may and may not be traced, and how parity is held. Read this before touching `compile/`, `build/mdp/` or the frontend's manager layer (`template/src/core/{observation,termination,event,command,action,actuator,onnx}/`).

## The split

mjswan reimplements none of mjlab's MDP term functions. Each manager's **term body** (the author's logic) is traced to ONNX at build time with `torch.onnx.export` and run in the browser by ONNX Runtime Web; each manager's **orchestration** (triggers, scheduling, aggregation, the noise / clip / scale / history pipelines) is native TypeScript.

| Manager | Term body | Native orchestration |
|---|---|---|
| Observation | ONNX, fused per group | concatenate, noise, clip, scale, history |
| Termination | ONNX, fused, awaited every step | OR across terms |
| Action | none for mjlab's own classes; ONNX for a task's own `process_actions` | `raw * scale + offset`, clip; every substep: ramp, hold, command delay, the target into an actuator law (below) or a PD, actuator write |
| Event | ONNX, run only when the trigger fires | interval countdown, startup-once, reset gating |
| Command | ONNX, state threaded explicitly | resample timer, held state, UI override |

Only these five managers exist. An mjlab actuator's torque law is traced too, but evaluated by mjswan's own synchronous evaluator rather than ONNX Runtime (*Actuator laws*, below). Reward, curriculum, metrics and recorders are training concerns and mjswan is a playback tool: a task's config for them is ignored and never emitted (`tests/test_artifact_hygiene.py`). Training-only term fields (`noise`, `delay_*`, `enable_corruption`) are accepted and ignored.

Why ONNX rather than a hand-written interpreter: the traced graph *is* the computation, run by a standard runtime already loaded for the policy, so there is no second copy of mjlab's math to keep numerically in step and no closed op set to grow before a new mjlab field works. An ONNX graph is pure data interpreted by a fixed runtime, so it is effect-free and safe for mjswan Cloud to render ([frontend.md](frontend.md) has the security line).

## The two passes

`compile/` runs each `func(env, **params)` once against a recording proxy to discover its reads, classifies each as time-varying state (a graph input, or "slot") or a model-derived constant (baked in), then exports an `nn.Module` via `torch.onnx.export`. It returns bytes and `*Export` records; writing them is `build/mdp/`'s job ([build.md](build.md)).

- `slot.py`: what a term may read off `env`, and how each read is named (below).
- `proxy.py`: the stand-ins both passes share, which subclass the real objects so a term's `isinstance` checks keep passing.
- `record.py`: the discovery pass. A `sim` slot is narrowed to the rows the term indexes when discovery can tell which, unioned across a group's terms, and shipped whole otherwise. Structured sensors (`RayCastSensor`, `ContactSensor`) contribute one slot per field read. Event and command bodies record under tagged keys and have their `write_*_to_sim` calls captured; an event's `env.sim.model` writes land on a copy (`model.py`) and are captured too.
- `replay.py`: the replay pass, serving the recorded slots back while torch traces; a read discovery never saw raises (see *What cannot be traced*).
- `export.py`: classification, narrowing, constant buffers and the `torch.onnx.export` call (a graph with no inputs and no random op is rewritten as its `Constant` outputs; every graph passes `onnx.checker`'s full check), the single-env guard, the vetted exporter warnings it silences, and two checks an observation, termination or event graph passes before it ships: `check_against_module` runs the source module against it on perturbed inputs, and `no_hidden_reads` fails a refused read the body swallowed (`slot.refuse_read`).
- `term.py` / `event.py` / `command.py` / `action.py` / `actuator.py` / `group.py`: one trace per kind:
  - a value term: a termination that reads nothing fails the build (it would fire every step or never), a baked observation warns by name;
  - an event: writes become outputs, randomness an explicit `rand` input. A class term runs as the instance mjlab built with the env (a new one when the scene's params differ), the env and entities it keeps routed through the proxies, and fails if a call changes its state or it keeps a live object inside a plain object. `EVENTS_WITH_NOTHING_TO_WRITE` names the ones whose trace legitimately captures no write;
  - a stateful command: hidden state promoted to graph I/O, a draw-selected env set gated by `gate.py`, what its `reset` adds gated by `reset_mask`, and the graph checked against the term's own body before it ships;
  - a task's own action term, and an actuator's torque law (below);
  - the fused observation and termination groups.
- `model.py`: an event's `env.sim.model`: reads become slots, writes outputs.
- `gate.py`: env sets a draw selects, kept whole and gated.
- `rng.py`: build-time RNG spy and replay (below).
- `parity.py`: live mjlab env against the exported graphs (below).

## Slots: what a term may read

Eight slot namespaces: a raw `mjData` field (`sim`), an entity `data` field, a named sensor's `sensordata` window, a live command's state field, the episode step counter (`env`: `episode_length_buf`), the policy's last action (`action`), an action term's internals (`action_term`: a traced term's state field, or any term's `_processed_actions`), and the rows of an `mjModel` field an event reads (`model`, live or as compiled).

- An `EntityData` property the browser reads natively (`READER_FIELDS`) is one value slot. Any other property is traced through a copy of the real `EntityData` whose `data` is the sim proxy, so mjlab's property math lands in the graph and its raw reads become `sim` slots.
- Only `_STATIC_DATA_FIELDS` are treated as constant. Anything unrecognized errs toward a graph input, so a missing runtime input fails loudly instead of returning stale values.
- Discovery and replay share one contract: a term may read `env.scene[...]`, `env.command_manager`, `env.sim.data` (the same object as `entity.data.data`), `env.episode_length_buf`, `env.action_manager.action` (or a term's `raw_action`), the constants in `_FORWARDED_ENV_ATTRS` plus `env.scene.env_origins` and an entity's `indexing` and `is_fixed_base` (both passes serve these from the real env), and an event `env.sim.model` and `env.sim.get_default_field(...)`. Any other `env` read raises `UnsupportedEnvRead`, so nothing bakes a constant by reaching the real env.
- An observation may read `get_term(name).<field>` of an action term: a traced term's state field, or any term's `_processed_actions`, fed as float32 whatever the field's dtype. Any other internal raises, as does a property that reads the term's entity: the browser holds no copy of either, so its value would bake.
- Input names are mjswan's, not torch's (`sensor__<name>`, `command__<name>`, `<namespace>__<name>`), and the manifest's `input_slots` tells the runtime what to feed each. `slots_json` drops a slot the exporter folded into a constant (an index tensor baked into the `Gather` it feeds), since ORT rejects a feed that is not a graph input. A graph and its slot list are therefore a matched pair from one export: neither can be derived from the other.

Graphs are float32; MuJoCo's `mjtNum` is float64. The browser's slot reader converts once, at the read, before feeding any graph. float32 is enough for playback.

## The trace env

Tracing needs a live env: to read shapes from, and to resolve every `SceneEntityCfg` regex to static indices at trace time, as mjlab's managers do in `_prepare_terms()`. Those indices are baked into the graph as constants. That is why a plain `add_scene` scene with traced terms must call `set_trace_env(...)`, while `add_scene_mjlab` builds one from the task.

- `mjlab/env.py`'s `build_single_entity_trace_env(spec_fn)` builds a minimal single-entity `ManagerBasedRlEnv` out of mjlab's own `Entity` / `Scene`, with no reimplemented kinematics. It configures no managers and is never stepped: it is only the tracer's `env.scene[name].data.<field>` read and write target. Joint defaults come from the model's first keyframe, matching what the browser resets to.
- **One env, always.** The browser runs a single environment, so the tracer refuses a batched env (`_prepare_single_env_export`): the graph would bake that row count while the runtime feeds one row. The browser's orchestration likewise ports mjlab's trigger *semantics* (interval countdown, resample-time sampling, `min_step_count_between_reset` gating, global versus per-instance timers) with plain numbers and booleans, not mjlab's `(num_envs,)` tensors and `.nonzero()` masks, which exist to batch thousands of envs on a GPU. ORT's tensor shape stays inside `core/onnx/`. A grid of demos is several engine instances, not a batched loop.
- mjlab's command bodies narrow to the envs a draw selected (`env_ids[mask]`, `mask.nonzero()`) and skip an empty set. The tracer keeps such a set whole and carries the mask, so a write through it becomes a `where`, an entity write a gated one, and the emptiness guard is always taken. A trace draws at each range's low end, which every `draw <= p` selection passes, so each guarded branch makes it into the graph.

## Stateful terms

Statefulness does not preclude tracing. A stateful term is exported as an RNN cell is: its hidden state is promoted to explicit inputs and outputs, and the browser's orchestrator holds it across frames. `self.foo` is Python's idiom for recurrent state, not evidence that a term must be native.

- **Command**: `forward(prev_state, resample_mask, rand) → (next_state, command)`. `_resample_command` runs under a `torch.where` gated by `resample_mask`; `_update_command` always runs. A reset is not a separate code path: it is the same graph with `resample_mask = 1`, and with `reset_mask = 1` for what a term's `reset` adds (mjlab's `init_velocity_prob` start). The entry declares each state tensor's name, shape and initial value, captured at export.
- **Event**: a trace carries no state. A class whose call changes its own state fails the build (state inside a plain object it keeps included), as does one that keeps the sim's arrays or a view of them (`env.sim.data.qpos[:]`, a raw warp array), or an entity inside a plain object, from its `__init__`, since reading them would freeze at build time.
- The genuinely native parts are native for other reasons: the UI override of a command is viewer interactivity, and mjlab's metrics dict is training-only logging.

## Randomness

All randomness inside an event or command body arrives as an explicit `rand` input, drawn by one seeded PRNG the browser's orchestrator owns (`core/rng.ts`, xoshiro128\*\* seeded by SplitMix32, spelled out rather than imported so a library change never alters a run). Never use ONNX's `RandomUniform` / `RandomNormal`: the ONNX spec leaves their algorithm open, so they diverge across runtimes and versions and give no seed the orchestrator can persist. Matching mjlab's own draws is not a goal either: torch's RNG is not portable across CPU and CUDA, so mjlab promises only seeded reproducibility within one setup.

`compile/rng.py` is unrelated to the browser's PRNG. At build time it patches the term function's *own* module globals (mjlab binds the sampler's name at import time, which is why `envs/mdp/` imports `sample_uniform` at module level: [package-layout.md](package-layout.md)), records mjlab's real draws, and replays those exact values into the graph's `rand` input, so parity does not diverge on randomness alone.

## Fusion

Terms in a group are traced into **one** graph rather than one per term. The reason is per-call overhead, not arithmetic: a JS-to-WASM crossing, tensor marshalling and a promise round trip cost the same for one node or a hundred. On Velocity-Flat-G1, five one-node observation graphs (three a bare `Identity`) fuse into one 59-node graph and one call per control step, and a slot feeding two terms is read and marshalled once.

> **Fusion changes how many graphs exist, never how often they are called.**

- Observation and termination groups fuse: both run every control step, observation on the policy's critical path. That keeps the per-step ORT cost at about two calls beyond the policy's.
- A group does not fuse when it holds a `*Binding` term (its body exists only in the browser) or terms with their own `history_length` / `history_steps`: mjlab stacks history per term *before* concatenating, and a ring buffer over a fused vector would give step-major order where mjlab gives term-major.
- A lone traced termination is left unfused; two or more fuse, one bool lane per term so per-term reset reasons survive.
- Never fuse the observation graph into the policy graph. It would save one call but break "`policy.onnx` is the trained artifact" and complicate command and joystick injection.

## Action terms: a traced process, a native apply

mjlab splits an action term in two: `process_actions`, once per control step, turns the raw action into `_processed_actions` and updates what the term keeps; `apply_actions`, once per physics substep, turns that into the actuator target. mjswan runs each half where mjlab runs it.

**Apply is always native, synchronous and ONNX-free**: it runs on the hottest loop, every substep, where an awaited graph does not belong (`core/action/applyAction.ts`). Instead of being traced, the entry *names* what apply does, from a small vocabulary the browser implements:

| Field | What apply does each substep |
|---|---|
| `type` | `joint_position` (target minus encoder bias), `joint_position_relative` (measured position plus the action, mjlab's `RelativeJointPositionAction`), `joint_velocity`, `torque`, and the reference, command and muscle kinds |
| `interpolate` | ramps from `processed_from` (the processed action before this step) to `processed`, `alpha = (k + 1) / decimation` (in-hand rotation) |
| `hold_s` | while `float32(episode_length * step_dt) < hold_s`, the target is the joint's measured position (luwu) |
| `delay` | per mjlab delay buffer: its lag in `[min_lag, max_lag]` physics steps with `hold_prob`, `update_period`, `per_env_phase`, read off the actuator cfgs for every kind of term |

A class that inherits `apply_actions` from mjlab's position, relative, velocity or effort action gets that kind with no binding. One that overrides it needs `mjswan.register_action(cfg_name, ActionBinding(apply=..., hold_s=..., interpolate=...))`, and the build refuses it without one. A binding's fields may be callables of the task's cfg (`hold_s=lambda cfg: cfg.hold_duration_s`). `interpolate` and `hold_s` act on a position target, so the build refuses them on a velocity or torque apply. The build looks the binding up itself, so it may be registered before or after `add_policy`. An apply outside the vocabulary (one reading a sensor per substep, writing effort and position together) is refused; widening the vocabulary is an mjswan change.

**Process is traced for a task's own class.** The adapter turns a task's own buildable action cfg mjswan has no class for into a `TracedActionCfg` (`envs/mdp/actions`' `native_class` / `is_traced` decide, for the adapter and `run_parity` alike); one of mjlab's own without an mjswan class (`DifferentialIKActionCfg`) is skipped with a warning. The build constructs the term on the trace env and traces both bodies, state promoted as a stateful command's is:

```
process(raw, slots..., prev_<f>..., [rand]) -> (next_<f>..., processed, [processed_from])
reset(slots..., prev_<f>..., [rand])        -> (next_<f>...)
```

- **State** is every tensor attribute either body writes, found by identity and in-place version, each tensor held so a freed one's id cannot pass for a new one's; its initial value is what `cfg.build(env)` leaves. A reset that writes nothing has no graph. A boolean-mask row write (`x[mask] = ...`) traces as a `where` (`gate.py`'s `MaskedRows`), so it keeps working when the mask changes.
- **Refused**, with the fix in the message: a body that changes Python state (a counter, a new attribute, a deque of past actions), writes the sim, or reads a live mjlab object other than its entity cached at init; two state fields sharing one tensor, which the browser would carry apart; a slot the browser's reader does not serve; a reset that reads the episode clock, which the browser zeroes before its reset terms and mjlab after them; a term whose action width differs from the joints it drives, or that names no joints.
- **Verified before it ships**: both graphs run against the term's own bodies over random actions, resets and episode clocks, and a disagreement fails the build.
- A traced term costs one awaited graph run per control step (`OnnxActionProcess`); its reset graphs run in the reset chain ([frontend.md](frontend.md)).
- mjlab zeroes the targets at a reset and writes them once before the next step, so a delayed actuator acts on zero for its first `lag` substeps; the browser starts each delay's history from that zero frame. Randomized lags come from the browser's seeded RNG (mjlab's distribution, not its stream). The browser keeps one delay per action term, so joints of two terms behind one mjlab delay buffer draw their lags apart, and the build warns when that lag is random. Lags an event sets through `Actuator.set_lags` are not run.

## Actuator laws: traced, evaluated synchronously

mjlab turns an entity's targets into `ctrl` every physics substep (`Entity.write_data_to_sim`): MuJoCo computes the control of the builtin and XML actuators from the target, and mjlab computes every other actuator's in Python, its `compute(ActuatorCmd)` over the joints' targets, positions and velocities (an ideal PD within its effort limit, a DC motor's torque-speed curve, a subclass's friction). The browser runs those laws as traced graphs.

- **One graph per actuator config, per MDP** (`compile/actuator.py`, `build/mdp/actuator.py`): `mjlab/actuator.py`'s `traced_actuator_cfgs` lists the configs of the entities the MDP's action terms drive, from that MDP's own env config, leaving out the ones MuJoCo computes; the build constructs each afresh on the trace env (`build_actuator`) and traces its `compute`, or the law `register_actuator` names for its config class, as `pos, vel, pos_target, vel_target, effort_target -> torque`, one column per joint. The entry is keyed `<Class>_<index>` and lists the model joints its columns follow.
- **Elementwise and stateless, or refused.** The graph may use only `SUPPORTED_OPS`, operators over per-joint vectors and scalars, and is checked against the live `compute` over a run of calls (`check_against_module`), so a law that keeps state between calls disagrees and fails. `LearnedMlpActuatorCfg` fails both ways (a TorchScript network the exporter cannot follow, and an input history), as does a law with a matrix product. The message names the ways out: `register_actuator`, or a builtin actuator. One that drives tendons or sites warns and is left out, since the browser feeds a law joint state only.
- **Evaluated synchronously, every substep** (`core/actuator/`): ONNX Runtime Web answers with a Promise, which has no place in the substep loop. `onnx.ts` reads the graph's protobuf (bounded: bytes, nodes, tensor sizes), and `evaluator.ts` compiles its nodes once into closures over preallocated `Float32Array`s, each step rounded to float32 as ONNX's is. Its operator set is `SUPPORTED_OPS`, and `actuatorParity.test.ts` holds the two in step.
- **Targets as mjlab keeps them** (`ActuatorSet.ts`): a term writes its target, after any command delay, into the law's input for the kind mjlab's term of that kind sets (position, velocity or effort), and `applyAction` then runs every law on the state the last `mj_step` left and writes `ctrl`. Targets persist until an episode's reset zeroes them (mjlab's `EntityData.clear_state`), so a joint no term drives is held toward zero, as mjlab holds it.
- **Why this is not the rejected DSL** (*Rejected*): the evaluator runs ONNX's own operators on the graph the trace produced from the actuator's code, not a hand-written copy of mjlab's math; an actuator law is a few elementwise operators in every actuator mjlab defines but its learned one, and in the reference tasks' own; and a law outside the set is refused at build time rather than approximated.

## What cannot be traced

`torch.onnx.export` records one path through the code. What it cannot record:

- **Input-dependent control flow.** The replay pass serves only the reads discovery saw; a read it never saw (a slot, a command, an action term's field, an `env` buffer, a model field) means the term branches on state, and replay raises rather than trace one branch. Control flow over static values (params, a `for name in body_names:` over a static list) is fine: it unrolls at trace time.
- Ragged structures and string manipulation on the traced path.
- A read that is neither a slot nor a forwarded constant (`UnsupportedEnvRead`), and state the tracer cannot turn into an input (`UntraceableTerm`): baking its trace-time value would freeze a time-varying input.

Such a term **fails the build loudly and names itself**: never force-traced, never silently dropped, and its Python source never ships to the browser. The message names the ways out: a trace-friendly replacement registered with `register_observation` / `register_termination` / `register_event` / `register_command`, a hand-written TypeScript class named by a `*Binding`'s `ts_src` (trusted contexts only: [frontend.md](frontend.md)), or, for an observation, dropping the term and retraining. Four event terms are exempt because there is provably nothing to write: `randomize_terrain`, `encoder_bias`, `reset_scene_to_default` (the runtime's own reset already does it) and a root-state write onto a fixed-base entity (`EVENTS_WITH_NOTHING_TO_WRITE` in `compile/event.py`). mjlab's `reset_root_state_from_flat_patches` (it draws with `torch.randint` and indexes per-env terrain tensors) is swapped for mjswan's traceable `reset_root_state_on_flat_patch` ([mjlab.md](mjlab.md)).

## Parity

Numeric parity is mandatory: every traced graph is checked against the live mjlab term it came from. `compile/parity.py` steps a live env with a seeded action sequence and feeds the same raw state through each exported graph via `onnxruntime` (not torch), asserting `allclose` for every term at every step. The layers each cover something the others cannot:

| Check | What it proves |
|---|---|
| `tests/test_onnx_parity.py` | each traced graph reproduces its mjlab term, every term at every step, over the reference tasks in `SWEEP_TASKS` |
| `tests/test_onnx_command_parity.py` | every reference task's traced command over 16 replayed draws |
| `core/onnx/__tests__/slotReaderParity.test.ts` | the reader hands those graphs mjlab's numbers, for every `EntityData` property and each entity's `EntityIndexing` (fixture from `dump_slot_fixture.py`: two tasks and a synthetic tendon-driven entity) |
| `core/engine/__tests__/rolloutParity.test.ts` | the layers **composed**: state → reader → real ORT session → fused obs → group vector (fixture from `dump_rollout_fixture.py`) |
| `core/action/__tests__/actionParity.test.ts` | task-defined action terms against mjlab, substep by substep, through the real process graph and native apply (fixture from `dump_action_fixture.py`) |
| `core/actuator/__tests__/actuatorParity.test.ts` | traced actuator laws against mjlab, substep by substep, through the real `applyAction` and `ActuatorSet`, and the evaluator's operator set against `SUPPORTED_OPS` (fixture from `dump_actuator_fixture.py`: an ideal PD, a delayed DC motor, a velocity term over a friction law) |
| `tests/test_artifact_hygiene.py` | no training-only manager keys and no Python term source in the emitted bundle |

Rollout parity **replays** mjlab's states rather than co-simulating: mjlab integrates with `mujoco_warp` while the browser runs MuJoCo's own WASM build, so a free-running comparison would measure MuJoCo against itself. The CI side (`parity.yml`, and why it runs weekly) is in [ci.md](ci.md).

## Measured and declined

Do not re-add these as unfinished requirements.

- **Event fusion.** Fusion cuts how many graphs exist, not how often they run, and events already run only when their trigger fires. Across the reference tasks `startup` has at most four traced terms, each run once, and `reset` at most two, so there is nothing to win.
- **Input-tensor pre-allocation** (reusing one input tensor per slot and writing it in place each frame). The per-step cost is the `run()` call, which fusion cut to a few per control step; ORT-Web's wasm backend copies each input into its own heap on every run anyway, so reusing the JS-side tensor saves only the wrapper object.
- **A `source_url` per graph** (a git URL and line for the traced function). Provenance is `func` (`module:qualname`), `doc` and `params` on the entry and `metadata_props` in the graph ([build.md](build.md)), which identify the function wherever it was imported from; a URL needs the author's code in a public repository at a known commit, and a line number drifts.
- **Bit-for-bit session replay.** Traced calls are asynchronous and the runtime skips a call still in flight rather than block the frame; every traced term draws from one shared stream, so a skipped frame shifts every later draw, and whether a frame skips depends on wall-clock time. What does hold: the sequence is a pure function of the seed (`termSeed` in and out of the engine), so startup events and every command's resample schedule (drawn before the in-flight check) reproduce, and terminations are awaited so a reset lands on the step it does in mjlab. `core/__tests__/rng.test.ts` pins the limitation.

## Rejected

- **A declarative JSON DSL of primitive ops with a TypeScript interpreter**: every op is a second, hand-written copy of mjlab's math kept in step by hand, and a closed op set gates each new mjlab field behind an engine change.
- **ONNX random ops**: see *Randomness*.
- **Porting mjlab's batched tensors line for line**: GPU-scale machinery with no payoff at one env.
- **Commands as native, untraced classes**: stateful is not untraceable.
- **Fusing the observation graph into the policy graph**: see *Fusion*.
- **Tracing `apply_actions`**: an awaited run per substep, for behaviour that in every task found reduces to a handful of shapes.
- **Running actuator laws in ONNX Runtime**: an awaited run per substep, as for `apply_actions`. The synchronous evaluator above runs the same graphs.
- **A PD on the action term in place of the actuator's law** (formats 10 and earlier, with a DC motor's curve copied onto the term): a hand-written copy of each actuator's math, which missed every subclass's own (WBC-Mjlab's friction), dropped the position half of a velocity term's PD, and left a joint no term drives limp where mjlab holds it toward zero.
- **Wrapping the policy to hold an action term's state**: it can integrate a delta but not ramp across substeps, hold per substep or delay a command, and it changes the policy's interface.
- **A native class per task's action term**: every task would need an mjswan release, and the term's arithmetic would drift from the task's own code. The trace runs the task's code.
