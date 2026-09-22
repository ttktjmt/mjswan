---
icon: octicons/file-code-16
---

# MDP Terms

A policy attached to a scene needs more than an ONNX file — the browser runtime has to
know which observations to feed in, how to interpret the action, what commands to expose
in the UI, when to reset the episode, and what to randomize. Those five term sets are the
policy's **MDP**, and mjswan takes them as Python kwargs on `add_policy()` — or as one
[`MdpConfig`](../api/core.md#mdpconfig) shared by several policies — using
[mjlab](https://github.com/mujocolab/mjlab){:target="_blank"}'s own config classes.

This page is the practical reference for those kwargs. For *how* they become browser
artifacts, see [How the Build Works](how-it-works.md).

!!! tip "You pass mjlab's functions, not mjswan lookalikes"
    mjswan reimplements none of mjlab's observation, termination, or event functions.
    Pass the real function object — `ObservationTermCfg(func=obs_fns.base_lin_vel)` — and
    the build traces it to ONNX. A function you write yourself gets the same treatment.
    There is no registration step and no name table.

## Top level: `add_policy(...)`

```python
from mjlab.envs.mdp import observations as obs_fns
from mjlab.envs.mdp import terminations as term_fns

scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    policy_joint_names=["FL_hip", "FL_thigh", "FL_calf"],  # …
    default_joint_pos=[0.1, 0.8, -1.5],  # …
    observations=ObservationGroupCfg(terms={...}),
    actions={"joint_pos": JointPositionActionCfg(...)},
    commands={"velocity": mjswan.velocity_command()},
    terminations={
        "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True)
    },
)
```

The relevant kwargs (see the [API reference](../api/core.md#scenehandleadd_policy) for the
full list):

| Kwarg | Purpose |
|---|---|
| `policy_joint_names` | Ordered list of joint names the policy controls. Required for browser-side actuator mapping. |
| `default_joint_pos` | Default pose, one entry per `policy_joint_names`. Used when `use_default_offset=True` on the action term and when an observation subtracts the default pose. |
| `observations` | A single `ObservationGroupCfg`, mjlab's whole `env_cfg.observations` dict, or a dict keyed by the slot names `in_keys` uses. Prefer one of the first two — see [below](#why-not-a-dict-of-groups). |
| `actions` | `dict[str, ActionTermCfg]` keyed by term name (e.g. `"joint_pos"`). |
| `commands` | `dict[str, CommandTermConfig]` keyed by policy-visible command name. |
| `terminations` | `dict[str, TerminationTermCfg]` keyed by termination name. |
| `events` | `dict[str, EventTermCfg]` keyed by event name, in any mode. Defaults to the scene's events — see [Events](#events). |
| `mdp` | An [`MdpConfig`](../api/core.md#mdpconfig) carrying all five term sets, shared by every policy handed the same object. Exclusive with the five kwargs above — see [Sharing one MDP](#sharing-one-mdp-between-policies). |
| `in_keys` / `out_keys` | The network's input and output **slot tables**: `in_keys` for a multi-input network, `out_keys` for a multi-output one whose action is not the first output — see [Multi-input policies](#multi-input-policies). |
| `policy_num_actions` | Output width for policies whose action count cannot be inferred from `policy_joint_names` — muscle-driven ones, which drive actuators rather than joints. |
| `encoder_bias` | Optional per-joint bias; the browser writes `processed_action - encoder_bias` to the actuators (mirrors mjlab). |
| `clip_actions` | Symmetric bound on the raw policy output, applied before any action term, mirroring rsl-rl's `RslRlVecEnvWrapper`. `add_policy_wandb` fills it in from the task's runner config. Not `ActionTermCfg.clip` — see [Actions](#actions). |
| `extras` | Arbitrary JSON payload merged verbatim into the generated policy config. |

## Defaults from an mjlab env config

On a scene from [`add_scene_mjlab`](mjlab.md), `observations` / `commands` / `actions` /
`terminations` / `events` each default to the matching field of the task's `env_cfg`, so a
policy from an mjlab task needs none of them spelled out. Pass one to override that field
only; pass `{}` to say the policy genuinely has none. `clip_actions` likewise defaults to
the task's runner config.

Per-policy `env_cfg=` takes a different config for one policy — in mjlab terms, several
env configs sharing one `scene`, since an env has exactly one observation design. Its
control rate must match the scene's `control_dt`: the runtime derives its physics substep
count and every timer from one value per scene, so a mismatch raises rather than being
silently reinterpreted.

A scene from plain `add_scene` has no config to fall back on, so there every field means
exactly what it says — and it needs a tracing env, see
[below](#scenes-that-are-not-mjlab-tasks).

## Sharing one MDP between policies

The checkpoints of one training run were trained against one MDP, and the build should
trace it once. Build the term sets into an `MdpConfig` and hand the same object to each
policy:

```python
mdp = mjswan.MdpConfig(
    observations=ObservationGroupCfg(terms={...}),
    actions={"joint_pos": JointPositionActionCfg(...)},
    terminations={
        "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True)
    },
    commands={"velocity": mjswan.velocity_command()},
)
for step in (1000, 2000, 3000):
    scene.add_policy(
        name=f"model_{step}", policy=onnx.load(f"model_{step}.onnx"), mdp=mdp
    )
```

Identity is by object: two `MdpConfig`s with equal contents are two MDPs. The build writes
each one once, under `mdp/<mdp-id>/` in the scene directory, and every policy entry points
at its MDP by id. Passing the five term sets straight to `add_policy` builds an anonymous
`MdpConfig` for that policy alone, and the MDP takes **that policy's id** — `mdp/walk/`
beside `policy/walk.onnx`. `add_policy_wandb` builds one per call, so a run's checkpoints
share it; a config shared like that belongs to no single policy and is numbered `mdp_0`,
`mdp_1`, … in first-use order unless it carries a `name` (`name2id(name)` wins over both).

The first policy to use an `MdpConfig` fills its unset fields from the scene's env config
and adapts mjlab types in place; policies sharing one must agree on `policy_joint_names`
and on the sidecar blocks that feed the trace (a disagreement fails the build — give the
odd one its own MDP).

## Observations

A group is an ordered dict of `ObservationTermCfg` — the runtime concatenates term outputs
in declaration order. Pass the group directly:

```python
from mjlab.envs.mdp import observations as obs_fns
from mjswan.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)

obs = ObservationGroupCfg(
    terms={
        "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
        "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel, scale=1.0),
        "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel, scale=0.05),
        "last_action": ObservationTermCfg(func=obs_fns.last_action),
        "velocity_cmd": ObservationTermCfg(
            func=obs_fns.generated_commands,
            params={"command_name": "velocity"},
        ),
    },
)
```

### Fields used at runtime

| Field | Effect |
|---|---|
| `func` | The term body. Any traceable `func(env, **params) -> Tensor`, or an `ObservationBinding` naming a hand-written TypeScript class. |
| `params` | Forwarded to the function at trace time. Regex patterns in a `SceneEntityCfg` resolve to static indices and bake into the graph. |
| `scale` | Element-wise scale, applied after `clip` (mjlab's order). |
| `clip` | `(min, max)`, applied before `scale`. |
| `history_length` | Past frames to stack, **oldest first** (`[x_{t-n+1} … x_t]`), as mjlab's history buffer flattens them. `0` = current frame only. |
| `history_steps` | Look-back offsets to stack, in the order the policy reads them, instead of a count — `(16, 8, 4, 2, 1, 0)` reaches 17 frames back at a width of 6, and `(0, 1, 2)` is `history_length=3` reversed. Takes precedence over `history_length`. |
| `history_interleaved` | Isaac-style element-major layout (`[a0_{t-n+1}, …, a0_t, a1_…]`) instead of frame-major. |

Other mjlab fields (`noise`, `delay_*`, `enable_corruption`, `nan_policy`) are accepted
for config compatibility and ignored — there is no training in the browser.

!!! note "History disables fusion for its group"
    mjlab stacks a term's history *before* concatenating the group, so a group with any
    per-term history falls back to one graph per term rather than one fused graph. This is
    correctness, not an oversight — see [Fusion](how-it-works.md#fusion).

### Why not a dict of groups?

`observations` also accepts `dict[str, ObservationGroupCfg]`, keyed by **slot name** —
the name the policy's `in_keys` table uses for the tensor that fills one of the network's
inputs. A policy exported by mjlab has exactly one input, so the dict form buys nothing:
a lone group lands under `actor`, the default slot, and the network's own tensor name never
matters (the mapping is positional). Passing the group itself is the whole story for a
single-input policy, whatever its input is called.

Groups named for a training-only mjlab network (`"critic"`) are dropped: only the actor is
exported to ONNX, so nothing would consume them, and leaving them in would trace, bundle,
and evaluate them every control step for a value nothing reads.

### Multi-input policies

A network with several inputs — a proprioceptive group and a command group, plus a
recurrent carry — needs a **slot table**: `in_keys[i]` names what fills its *i*-th input,
an observation group or a tensor the runtime synthesizes (`is_init`, `adapt_hx`,
`time_step`). Nothing else records where the synthesized tensors sit relative to the
groups, so `add_policy` refuses a multi-input network without one and checks the table's
length against the network's input count; a slot naming no group fails the build.

```python
scene.add_policy(
    name="Facet",
    policy=onnx.load("facet.onnx"),  # four inputs
    observations={"actor": proprio_group, "command": command_group},
    in_keys=["command", "actor", "is_init", "adapt_hx"],
    out_keys=["command", "policy", ..., "action", ...],  # one label per output
)
```

`out_keys` is the same table for the outputs; the runtime reads `action` and, for a
recurrent policy, `["next", "adapt_hx"]`. Both tables are written to the manifest only
when they differ from the defaults (`["actor"]`, `["action"]`), and a single-input,
single-output network declares neither. A network with **several outputs** does need
`out_keys` unless its action is the first one: the default table names only output 0, and
the actuators would be driven from it. The build warns in that case rather than refusing,
since output 0 may well be the action. `examples/demo/main.py` shows both tables in use on
one policy: the motion-tracking network takes two inputs, so `in_keys` names them, and all
seven outputs are spelled out even though `action` is already the first — the six after it
are the reference pose the clip carries, named so the table lines up.

### Coming from mjlab

An mjlab `env_cfg.observations` is keyed by *network* name, not by ONNX input name — two
namespaces that happen to look alike. Hand the whole thing over and mjswan picks the
policy's group:

```python
observations = env_cfg.observations  # {"actor": ..., "critic": ...}
observations = env_cfg.observations["actor"]  # or just the group; same result
```

Which group that is comes from the task's **runner** config
(`rl_cfg.obs_groups["actor"]`), so a task free to name its groups something else still
resolves correctly; the `"actor"` name is only the fallback when there is no registered
task to ask. If a task's actor reads *several* groups concatenated, mjswan raises — it
feeds one vector per ONNX input and cannot join them, and quietly taking the first would
feed the policy a short observation.

A dict whose keys are not mjlab network names is left exactly as written, which is what
keeps a policy whose input really is called `"observation"` or `"obs_history"` working.

## Actions

Actions are the one manager that is **not** traced: a closed set of term types,
implemented natively in TypeScript because it runs once per physics substep rather than
once per control step. So `mjswan.envs.mdp.actions` carries real, directly-usable config
classes rather than an escape hatch.

```python
from mjswan.envs.mdp.actions import (
    JointPositionActionCfg,
    JointEffortActionCfg,
)

# Joint-position control with browser-side PD
actions = {
    "joint_pos": JointPositionActionCfg(
        actuator_names=(".*",),
        scale=0.25,
        offset=0.0,
        use_default_offset=True,  # action=0 commands the default pose
        stiffness=40.0,  # kp (scalar, per-joint list, or dict by joint name)
        damping=1.0,  # kd
    ),
}

# Direct torque output
actions = {
    "thrust": JointEffortActionCfg(actuator_names=("lift",)),
}
```

| Class | Status |
|---|---|
| `JointPositionActionCfg` | Supported. |
| `JointEffortActionCfg` | Supported. |
| `MuscleActivationActionCfg` | Supported — drives MuJoCo muscle actuators. See [below](#muscle-actuators). |
| `JointVelocityActionCfg`, `TendonLengthActionCfg`, `TendonVelocityActionCfg`, `TendonEffortActionCfg`, `SiteEffortActionCfg` | Exported so mjlab configs import cleanly, but raise `NotImplementedError` at build time. |

`stiffness` and `damping` are mjswan-specific — in mjlab they live on the actuator, but
the browser runtime computes PD externally for motor actuators with `biastype=none`, so
they have to be in the policy config. Both accept a scalar, a per-joint list (aligned with
`policy_joint_names`), or a dict keyed by joint name.

### Muscle actuators

`MuscleActivationActionCfg` drives MuJoCo muscle actuators. `normalize=True` (the default)
applies the canonical MyoSuite sigmoid `σ(5(scale·a + offset − 0.5))` to map policy outputs
into excitation in (0, 1); `normalize=False` clips `scale·a + offset` to [0, 1]. The
semantics mirror `MuscleActionTermCfg.normalize` in myosuite4, and the mjlab adapter
translates `MyoMuscleActivationActionCfg` — the class every `myo*` mjlab task actually
uses — to this one.

Every term's `actuator_names` must resolve to muscle-dyntype actuators in the scene's
model; the build raises on the first that does not, rather than producing a bundle that
misbehaves in the browser.

### Two clips, and they are not the same bound

| | `ActionTermCfg.clip` | `clip_actions` |
|---|---|---|
| Lives on | the action term | the policy (from mjlab's *runner* config) |
| Shape | `dict` of `{pattern: (min, max)}`, per target | one number, symmetric `[-v, +v]` |
| Applied to | `raw * scale + offset` | the policy's raw output, before any term |
| mjlab source | `BaseAction.process_actions` | rsl-rl's `RslRlVecEnvWrapper.step` |

`clip_actions` lands ahead of everything, so a `last_action` observation reads the clamped
vector — matching mjlab, where the wrapper clamps before `env.step` and the action manager
records what it was handed. `add_policy_wandb` reads it from the task automatically; pass
it explicitly only for a hand-built policy or with `only_latest=True`, which skips mjlab
entirely.

`ActionTermCfg.clip` keys are **patterns**, matched with mjlab's anchored `re.fullmatch`.
An unmatched target is left unbounded, and a pattern that matches nothing warns.

## Commands

`commands` is a dict of `CommandTermConfig` values keyed by the policy-visible name your
observations reference (e.g. `params={"command_name": "velocity"}` looks up
`commands["velocity"]`).

```python
commands = {
    "velocity": mjswan.velocity_command(),  # standard 3-DoF locomotion
    "target": mjswan.ui_command(
        [  # arbitrary UI inputs
            mjswan.Slider(
                "target_height", "Target Height (m)", range=(0.3, 1.8), default=1.0
            ),
        ]
    ),
}
```

| Class | Description |
|---|---|
| `mjswan.Slider` | Continuous range slider. Fields: `name`, `label`, `range`, `default`, `step`, `enabled_when`, `adjustable_range`. |
| `mjswan.Button` | Momentary push button. Fields: `name`, `label`. |
| `mjswan.Checkbox` | Boolean toggle. Fields: `name`, `label`, `default`. |

`enabled_when` names a sibling checkbox that gates the slider. `adjustable_range` (a
`SliderRangeConfig`) adds a purely presentational companion "Max &lt;label&gt;" slider that
rescales the value slider's drag range, mirroring mjlab's own play GUI; it carries no
command id, so nothing about it reaches the policy.

An mjlab command class with real logic — a velocity command holding a heading target, a
lifting command sampling a goal pose — is traced like any other term, with its hidden
state promoted to explicit graph I/O. Register the adapter with
`mjswan.register_command(mjlab_name, spec)`; see the
[API reference](../api/core.md#register_command).

## Terminations

Same shape as observations — pass mjlab's own functions:

```python
from mjlab.envs.mdp import terminations as term_fns
from mjswan.managers.termination_manager import TerminationTermCfg

terminations = {
    "time_out": TerminationTermCfg(func=term_fns.time_out, time_out=True),
    "fallen": TerminationTermCfg(
        func=term_fns.bad_orientation,
        params={"limit_angle": 1.0},
    ),
}
```

mjlab's `time_out` is classified **native** by function, as `last_action` is: it compares
the episode step counter the runtime owns, so there is nothing to trace. The build ships
the task's `episode_length_s` alongside the marker, and the runtime compares it against
episode time accumulated from `control_dt`.

Everything else is traced, and a termination that reads no simulation state fails the
build: a constant fires every step or never. Terminations in a group fuse into one graph
emitting a bool *lane* per term, so the manager keeps per-term reset reasons and its
terminated-vs-truncated split. A group with a single traced term is deliberately left
unfused — one graph out of one buys no call and costs a wire shape.

A termination the tracer cannot express fails the build. Registering a hand-written
TypeScript class with `register_termination` is the escape hatch.

## Events

Events are the fifth term set, and the one that most obviously belongs to a policy rather
than a scene: a policy trained with a push expects the push, a policy trained without one
does not. Pass them like the others — mjlab's own `EventTermCfg`s, in any of the four modes
(`startup`, `reset`, `interval`, `manual`) — or leave them to default to the scene's
`events` (`add_scene(events=...)` / `set_events`), which is what an mjlab task's
`env_cfg.events` becomes:

```python
scene.add_policy(
    ...,
    events={"push": EventTermCfg(func=evt_fns.push_by_setting_velocity, mode="interval", ...)},
)
```

Switching between policies with different MDPs restores the model values the previous
MDP's `mode="startup"` randomization changed, reseeds the term PRNG, then runs the new
MDP's startup events, so the randomization never compounds and A → B → A reproduces A.

## Scenes that are not mjlab tasks

Tracing needs a live environment to read shapes from and to resolve `SceneEntityCfg`
patterns against. `add_scene_mjlab` builds one from its task; a plain `add_scene` scene
must be given one:

```python
from mjswan.mjlab.env import build_single_entity_trace_env

scene = project.add_scene(name="Hovering Box", spec=build_spec(), control_dt=0.02)
scene.set_trace_env(build_single_entity_trace_env(build_spec))
```

`build_single_entity_trace_env(spec_fn)` builds a minimal single-entity environment out of
mjlab's own `Entity` and `Scene`. It takes a *zero-argument callable* returning a fresh
`MjSpec` (mjlab's `EntityCfg.spec_fn` contract), configures no managers, and is never
stepped. Pass `entity_name=` to match whatever your terms use as `asset_cfg.name`, and
`commands=` for trace-time stand-ins of commands the browser owns (a `ui_command` has no
Python side).

Without it, a policy with observation or termination terms raises at build time with a
message naming this call.

## End-to-end examples

| Example | What it shows |
|---|---|
| [examples/demo/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/demo/minimum_policy.py){:target="_blank"} | Smallest complete policy — a hand-built two-node ONNX PD controller, one self-authored observation beside two of mjlab's, a `ui_command`, and `set_trace_env`. |
| [examples/demo/main.py](https://github.com/ttktjmt/mjswan/blob/main/examples/demo/main.py){:target="_blank"} | Eight mjlab tasks with their mirrored checkpoints, a motion-tracking policy with its reference clip, two third-party policies with sidecar configs on one scene, and a muscle-driven policy — every asset fetched, none stored. |
| [examples/demo/simple.py](https://github.com/ttktjmt/mjswan/blob/main/examples/demo/simple.py){:target="_blank"} | The same machinery at its smallest: one task, one checkpoint, nothing hand-written. |

## What an mjlab export already carries

mjlab writes the checkpoint's own defaults into the `.onnx` itself — `joint_names`,
`default_joint_pos`, `action_scale`, and a description of each observation term — so an
mjlab policy travels self-describing.
[`add_policy_hf`](../getting-started/examples.md#a-policy-published-on-the-hugging-face-hub)
reads that block and fills `policy_joint_names`, `default_joint_pos` and the
joint-position action term from it, leaving anything you pass explicitly alone. To read
one yourself:

```python
import onnx
from mjswan.mjlab.onnx_meta import read_mjlab_metadata

meta = read_mjlab_metadata(onnx.load("policy.onnx"))
print(meta.joint_names, meta.action_scale, meta.observation_names)
```

`None` means the file carries no mjlab metadata — a hand-built graph, or one from
another framework — which is a normal case, not an error.

Two limits are worth knowing. The encoding is **lossy**: mjlab formats list values with
`{:.3f}`, so numbers come back rounded to three decimals (a scalar written outside a
list keeps full precision). And `joint_names` lists **every joint of the robot in joint
order**, while the network emits one action per actuator in actuator order — the same
list only when the robot has no passive joints and its actuators are declared in joint
order. `add_policy_hf` therefore checks the metadata against your scene's own model
before using any of it, and warns rather than guessing when they disagree.

Observation terms are named but not reconstructable: the metadata carries each term's
scale, clip and history length, but not the function behind the name, and mjswan traces
real functions. So `observations=` remains yours to supply.

## Legacy: passing a JSON file via `config_path=`

`add_policy(config_path="policy.json")` still accepts a JSON sidecar for the checkpoint's
own defaults — `policy_joint_names`, `default_joint_pos`, an `actions` block carrying PD
gains — as an exporter might write it. mjswan merges it into the policy's manifest entry at
build time, Python-side fields winning; the build's output is `manifest.json` alone, and no
per-policy JSON is written. A sidecar's `onnx` block, and any `in_keys` / `out_keys` in it,
are ignored with a warning: the slot tables are declared on `add_policy`, where they are
checked against the network. Prefer the Python kwargs above for new policies.
