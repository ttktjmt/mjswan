---
icon: octicons/code-16
---

# Examples

Copy-paste patterns for common use cases. For a step-by-step first run, see [Quickstart](quickstart.md).

## Scene from an XML string

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Demo")

spec = mujoco.MjSpec.from_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
""")
project.add_scene(spec=spec, name="Sphere")

builder.build().launch()
```

## Scene from a file

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")
project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)
builder.build().launch()
```

## An mjlab task with its trained checkpoints

The shortest path to a policy in a browser, if you trained with
[mjlab](../guides/mjlab.md). Every `model_*.pt` checkpoint in the W&B run is fetched,
converted to ONNX, and attached; observations, actions, commands, terminations and the
control rate all come from the task.

```python
import mjswan

app = mjswan.Builder.from_mjlab(
    "Mjlab-Velocity-Flat-Unitree-G1",
    run_path="<entity>/<project>/<run_id>",
).build()
app.launch()
```

## A policy published on the Hugging Face Hub

Where a W&B run holds training state — which is why the call above needs mjlab and torch
to convert it — a Hub repository holds the exported ONNX. So this path downloads and
stops: `pip install mjswan[hf]` is all it needs.

```python
import mjswan

app = mjswan.Builder.from_mjlab(
    "Mjlab-Velocity-Flat-Unitree-G1",
    hf_repo_id="<owner>/<name>",
).build()
app.launch()
```

On a scene of your own, `add_policy_hf` reads what mjlab baked into the file — the joint
names, the rest pose and the action scale — so they need not be repeated here:

```python
scene.add_policy_hf("<owner>/<name>")
```

With no filename given it takes `policy.onnx`, then `final.onnx`, then the repository's
single `.onnx`; pass `filename=` for anything else, or a list of them to add several
policies at once. The metadata is used only when your scene's model presents the same
joints in actuator order — a mismatch warns and fills nothing rather than misdriving the
actuators. Observation terms are never reconstructed from it: the file names them but
does not carry the functions mjswan traces, so `observations=` is still yours to supply
(or the task's, on a scene from `add_scene_mjlab`).

## Policy with velocity command sliders

```python
import mujoco
import onnx

import mjswan
from mjlab.envs.mdp import observations as obs_fns
from mjswan.envs.mdp.actions import JointPositionActionCfg
from mjswan.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjswan.mjlab.env import build_single_entity_trace_env


def build_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file("robot/scene.xml")


builder = mjswan.Builder()
project = builder.add_project(name="Robot")

scene = project.add_scene(
    spec=build_spec(),
    name="G1",
    control_dt=0.02,  # 50 Hz — required once a scene carries a policy
)
scene.set_trace_env(build_single_entity_trace_env(build_spec))

scene.add_policy(
    name="Locomotion",
    policy=onnx.load("robot/locomotion.onnx"),
    policy_joint_names=JOINTS,
    default_joint_pos=DEFAULT_POSE,
    observations=ObservationGroupCfg(
        terms={
            "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
            "projected_gravity": ObservationTermCfg(func=obs_fns.projected_gravity),
            "joint_pos": ObservationTermCfg(func=obs_fns.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel, scale=0.05),
            "last_action": ObservationTermCfg(func=obs_fns.last_action),
            "velocity_cmd": ObservationTermCfg(
                func=obs_fns.generated_commands,
                params={"command_name": "velocity"},
            ),
        }
    ),
    actions={
        "joint_pos": JointPositionActionCfg(
            actuator_names=(".*",), scale=0.25, use_default_offset=True
        )
    },
    commands={
        "velocity": mjswan.velocity_command(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            default_lin_vel_x=0.5,
        )
    },
)

builder.build().launch()
```

The observation terms are mjlab's own functions; mjswan traces them to ONNX at build time,
so `mjlab` and `torch` must be installed. `set_trace_env` gives the tracer a live
environment to read shapes from — an [mjlab scene](../guides/mjlab.md) builds one itself.
See [MDP Terms](../guides/policy-config.md).

## Multiple policies on one scene

```python
scene = project.add_scene(spec=spec, name="Go2", control_dt=0.02)
scene.set_trace_env(build_single_entity_trace_env(build_spec))

for name, path in [("Policy A", "policy_a.onnx"), ("Policy B", "policy_b.onnx")]:
    scene.add_policy(
        name=name,
        policy=onnx.load(path),
        policy_joint_names=JOINTS,
        observations=obs,
        actions=actions,
        commands={"velocity": mjswan.velocity_command()},
    )
```

The browser UI shows a selector for choosing between policies at runtime. Pass
`default=True` to pick which one loads first.

## Custom command inputs

Each value in `commands={...}` is a `CommandTermConfig` built with
`mjswan.ui_command([...])` (manual UI) or `mjswan.velocity_command(...)` (the locomotion
shortcut). An observation reads it by the dict key:
`params={"command_name": "target"}`.

```python
scene.add_policy(
    name="PD Hover",
    policy=onnx.load("hover.onnx"),
    policy_joint_names=["lift"],
    observations=obs,
    commands={
        "target": mjswan.ui_command(
            [
                mjswan.Slider(
                    "target_height",
                    "Target Height (m)",
                    range=(0.3, 1.8),
                    default=1.0,
                    step=0.05,
                ),
                mjswan.Checkbox("hold", "Hold Position", default=False),
                mjswan.Button("recenter", "Recenter"),
            ]
        ),
    },
)
```

See [examples/demo/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/demo/minimum_policy.py){:target="_blank"} for a complete runnable version — a hand-built two-node ONNX policy, one self-authored observation, and a slider, in one file.

## Multiple projects

One build can hold several projects. The app opens on the one marked `default=True` (or
the first added); a URL picks another by its id, the sanitized name: `?project=humanoids`,
`?scene=g1`.

```python
builder = mjswan.Builder(base_path="/demo/")

quadrupeds = builder.add_project(name="Quadrupeds", default=True)
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go2/scene.xml"), name="Go2")
quadrupeds.add_scene(spec=mujoco.MjSpec.from_file("go1/scene.xml"), name="Go1")

humanoids = builder.add_project(name="Humanoids")
humanoids.add_scene(spec=mujoco.MjSpec.from_file("g1/scene.xml"), name="G1")

builder.build().launch()
```

## Gaussian Splat background (bundled)

Use `source=` to bundle a `.spz` file into the built application. This is the recommended approach: the file is copied into `dist/` at build time, so the app works offline.

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Robot")

scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="G1",
)
scene.add_splat(
    "Lab",
    source="lab.spz",  # bundled into dist/
    scale=1.35,
    z_offset=0.71,
)

builder.build().launch()
```

## Gaussian Splat background (external URL)

Use `url=` to reference a `.spz` file hosted externally. The build stays small, but the browser must fetch the file at runtime.

```python
scene.add_splat(
    "Outdoor",
    url="https://example.com/outdoor.spz",
    scale=3.0,
    z_offset=0.5,
)
```

## Gaussian Splat background (Hugging Face Hub)

`add_splat_hf` downloads the `.spz` and bundles it, so the deployed app is as
self-contained as a local one — the Hub is a build-time source, not a runtime
dependency. Needs `pip install mjswan[hf]`.

```python
scene.add_splat_hf(
    "<owner>/<name>",
    "splats/street.spz",
    scale=3.275,
    z_offset=0.708,
)
```

The splat is named after its file (`street`), or after the repository when the stem
only names a role (`background.spz`); pass `name=` for anything else. The placement
arguments describe how *this* capture lines up with *this* model, which no file on the
Hub knows, so they stay yours to supply. `revision=` pins a branch, tag or commit —
without it the build follows the repository's default branch.

## Multiple splats on one scene

Add several splats to the same scene — the viewer shows a selector to switch between them at runtime.

```python
scene.add_splat("Lab A", source="lab_a.spz", scale=1.35, z_offset=0.71)
scene.add_splat("Lab B", source="lab_b.spz", scale=1.20, z_offset=0.65)
```

## Splat URL input without pre-configured splats

Call `enable_splat_section()` to show the Splat selector in the control panel even when no splats are pre-configured. Users can then paste an arbitrary `.spz` URL at runtime.

```python
scene = project.add_scene(name="Demo", spec=spec)
scene.enable_splat_section()
```

## Calibrating a new splat capture

Set `control=True` to expose live sliders for scale, offset, and rotation while you dial in the alignment. Remove `control=True` once the values are finalised.

```python
scene.add_splat(
    "Lab",
    source="lab.spz",
    scale=1.35,
    z_offset=0.71,
    control=True,  # shows calibration controls in the viewer
)
```

## Headless build (CI-friendly)

`build()` writes `dist/` and returns; `launch()` is the blocking part. Gate it on an
environment variable so the same script works locally and in CI — the convention the
bundled examples follow:

```python
import os

app = builder.build()
if not os.environ.get("MJSWAN_NO_LAUNCH"):
    app.launch()
```

```bash
MJSWAN_NO_LAUNCH=1 MJSWAN_BASE_PATH=/myrepo/ python build.py
```

See [Deployment](../guides/deployment.md) for GitHub Pages and CI/CD setup, or
[Publishing to Cloud](../guides/publishing.md) to skip hosting entirely.

## Setting the camera

```python
scene.set_viewer(
    mjswan.ViewerConfig(
        lookat=(0.0, 0.0, 1.0),
        distance=3.5,
        elevation=-30.0,
        azimuth=45.0,
        origin_type=mjswan.ViewerConfig.OriginType.WORLD,  # or AUTO / ASSET_ROOT / ASSET_BODY
    )
)

# Or from an explicit camera position, letting mjswan compute the spherical params:
scene.set_viewer(mjswan.ViewerConfig.from_position((2.0, 2.0, 1.5), target=(0, 0, 0.8)))
```

`origin_type` decides what the camera tracks: `WORLD` pins it, `ASSET_ROOT` follows the
robot's root, `ASSET_BODY` follows the body named by `body_name`, and `AUTO` picks based on
the scene.
