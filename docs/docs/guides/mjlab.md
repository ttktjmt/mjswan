---
icon: octicons/arrow-switch-16
---

# Using mjlab

[mjlab](https://github.com/mujocolab/mjlab){:target="_blank"} is a GPU-accelerated
reinforcement learning framework built on MuJoCo Warp. mjswan can visualize mjlab tasks
directly — there is no need to export or convert anything.

An mjlab scene is not just a model: it carries the task's viewer, events, terrain, control
rate, and — for any policy attached to it — the whole MDP layer. It also supplies the live
environment that [ONNX tracing](how-it-works.md) needs, so everything on this page works
without a single `set_trace_env` or `control_dt` of your own.

!!! info "Install"
    mjlab is a soft dependency: `pip install 'mjswan[mjlab]'` installs mjlab 1.6.0 and
    torch. The W&B helpers also need the `wandb` extra (`'mjswan[mjlab,wandb]'`), and the
    Hub helpers the `hf` extra. It is needed at **build time** only, and nothing about it
    ships to the browser.

This page walks through three integration levels, from the one-line shortcut to the full manual form.

## 1. One-liner: `Builder.from_mjlab`

The fastest path. `Builder.from_mjlab(task_id, run_path=...)` creates a project, adds the mjlab scene, and (optionally) attaches every `model_*.pt` checkpoint from one or more W&B runs as ONNX policies.

```python
import mjswan

# Just visualize the scene
app = mjswan.Builder.from_mjlab("Mjlab-Velocity-Flat-Unitree-G1").build()
app.launch()

# Visualize the scene + every checkpoint from a W&B run
app = mjswan.Builder.from_mjlab(
    "Mjlab-Velocity-Flat-Anymal-C",
    run_path="<entity>/<project>/<run_id>",
).build()
app.launch()
```

The W&B form needs `mjswan[wandb,mjlab]` (the `model_*.pt` → ONNX conversion runs locally), and checks for it when called. Or pass `hf_repo_id="<owner>/<name>"` to add a Hugging Face Hub repository's exported ONNX, as `add_policy_hf` does: nothing is converted, so this needs `mjswan[hf,mjlab]` and no W&B account. `examples/demo/simple.py` adds a checkpoint from the public `ttktjmt/mjswan` repository to an mjlab scene the same way.

Each attached policy configures itself from the task: its observations, commands, actions and terminations all come from the task's `env_cfg`, and its raw-action bound from the task's runner config. mjswan binds mjlab's `UniformVelocityCommandCfg` and `MotionCommandCfg` itself; a task with another command class, such as Lift-Cube-Yam's `LiftingCommandCfg`, needs a binding registered with [`mjswan.register_command`](../api/core.md#register_command), as `examples/demo/main.py` does, or the command is skipped with a warning. Under mjlab 1.6, a `trace_override` that replaces `_update_command` takes `env_ids`. For finer control, drop down to the next two patterns.

## 2. Scene helper: `ProjectHandle.add_scene_mjlab`

When you need multiple scenes (one per task) or want to mix mjlab tasks with hand-written scenes, use `add_scene_mjlab(task_id)` on a `ProjectHandle`. It loads the task's MuJoCo spec, applies the task's `viewer` / `events` / terrain data (spawning the single env on a random flat patch when the terrain has one), and returns a normal `SceneHandle`.

mjlab registers two configs per task — `env_cfg` (training) and `play_env_cfg` — and `play` selects between them, as its own `load_env_cfg` does. mjswan defaults to **play**, the opposite of mjlab, because that default serves training scripts and this is a playback tool: the training config sets `episode_length_s` to 10–20 s, which mjswan turns into the browser's `time_out` termination, so a viewer built from it resets the robot every few seconds. Pass `play=False` if you want training-time conditions.

```python
import mjswan
from mjlab.tasks.registry import list_tasks

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Tasks")

for task_id in list_tasks():
    project.add_scene_mjlab(task_id)

builder.build().launch()
```

### Attaching trained policies from W&B

Use `scene.add_policy_wandb(run_path)` to fetch checkpoints from one or more W&B runs and attach them all to the scene:

```python
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="ANYmal C")

scene = project.add_scene_mjlab("Mjlab-Velocity-Flat-Anymal-C")
scene.add_policy_wandb("<entity>/<project>/<run_id>")

builder.build().launch()
```

The scene keeps the `env_cfg` it was built from, and every policy added to it defaults its observations, commands, actions and terminations to that config — so the run path is all you need. `task_id` also defaults to the scene's task.

### Tracking tasks

mjlab registers its tracking tasks with `commands["motion"].motion_file = ""` for the caller to fill in, but you do not have to: `add_policy_wandb` fetches the run's clip, the builder writes it into the bundle, and the tracing env — constructed at build time, not when the scene is added — reads it from there. So a tracking task is the same two lines as any other:

```python
scene = project.add_scene_mjlab("Mjlab-Tracking-Flat-Unitree-G1")
scene.add_policy_wandb("<entity>/<project>/<run_id>")
```

A `motion_file` you set yourself is left alone, as long as it points at a file that exists.

Each distinct clip is written once per scene and shared by every policy that uses it, so the checkpoints of one run do not each get a copy. The file is named after the motion's id, `name2id(name)`; two clips with the same name but different content get a `_1` / `_2` suffix.

### Editing the config first

To change anything else about the task, load the config, edit it, and pass it to `add_scene_mjlab` — the policies then inherit your edited version, because the scene holds the same object:

```python
from mjlab.tasks.registry import load_env_cfg

task_id = "Mjlab-Tracking-Flat-Unitree-G1"
# `play=True` here, not on `add_scene_mjlab`: `env_cfg` *is* one of the two configs, so
# there is nothing left for `play` to select — passing both raises.
env_cfg = load_env_cfg(task_id, play=True)
env_cfg.terminations.pop("anchor_ori")

scene = project.add_scene_mjlab(task_id, env_cfg=env_cfg)
scene.add_policy_wandb("<entity>/<project>/<run_id>")
```

Load it **once**. `load_env_cfg` returns a deepcopy, so a second call gives you an equal but separate config, and edits to one are invisible to the other.

### Overriding one field

Pass any of the four to replace just that one; the rest still come from the config. Pass `{}` to say the policy genuinely has none:

```python
scene.add_policy_wandb(
    "<entity>/<project>/<run_id>",
    terminations={"time_out": TerminationTermCfg(func=term_fns.time_out)},
)
```

mjswan adapts mjlab config classes automatically. `observations` also accepts the task's whole `env_cfg.observations` dict or a single group — see [MDP Terms](policy-config.md), which explains why the key matters.

`add_policy_wandb` accepts a `list[str]` for the run path if you want to bundle checkpoints from multiple runs together, such as a run and its resumptions: a `model_<step>` an earlier run in the list already added is skipped with a warning, since a resumed run first saves the step it stopped at. The latest checkpoint (highest training step) is marked as the default unless the scene already has one, and a later `add_policy(..., default=True)` takes over from it.

If you only want the exported `.onnx` artifact (skipping the `.pt → .onnx` conversion), pass `only_latest=True` — `task_id` is then optional.

## 3. Full manual form

For a model-only viewer of an mjlab task — no policy, no MDP layer — you can go through
mjlab's own `Scene` and hand the `MjSpec` to plain `add_scene`:

```python
import mjswan
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

builder = mjswan.Builder()
project = builder.add_project(name="mjlab Examples")

# `play=True` for the same reason `add_scene_mjlab` defaults to it: this is a viewer.
env_cfg = load_env_cfg("Mjlab-Velocity-Flat-Anymal-C", play=True)
env_cfg.scene.num_envs = 1  # single environment for the viewer
scene_obj = Scene(env_cfg.scene, device="cpu")

project.add_scene(
    spec=scene_obj.spec,  # MjSpec from mjlab
    name="ANYmal C",
)

builder.build().launch()
```

!!! warning "This drops everything except the model"
    A scene built this way has no task attached, so it gets no viewer config, no events, no
    terrain spawning, no `control_dt`, and no tracing environment — and its policies inherit
    no term-set defaults. To customise a task *and* keep all of that, edit the config and
    pass it to `add_scene_mjlab(task_id, env_cfg=...)` as shown
    [above](#editing-the-config-first). Reach for this form only when you genuinely want
    just the geometry.

## What an mjlab scene carries over

| From the task | How it is used |
|---|---|
| `env_cfg.scene` | the MuJoCo spec, compiled and bundled as `scene.mjz` |
| `env_cfg.observations` / `actions` / `commands` / `terminations` | defaults for every policy on the scene, [traced to ONNX](how-it-works.md) |
| `env_cfg.events` | the scene's default events, which every policy that declares none inherits into its MDP; `reset_root_state_uniform` is swapped for a spawn on a random flat terrain patch, since the browser runs one env where mjlab trains many spread across the terrain |
| `env_cfg.sim.timestep × decimation` | `control_dt` |
| `env_cfg.viewer` | the initial camera |
| terrain flat-patch table | baked spawn points |
| the task's **runner** config | which observation group the actor reads, and `clip_actions` |
| the live env | the tracing target for every term body |

Rewards, curricula, metrics and recorders are training-only and simply ignored — mjswan is
a playback tool, so they never reach the bundle.
