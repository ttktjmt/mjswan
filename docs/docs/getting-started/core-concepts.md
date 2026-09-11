---
icon: octicons/light-bulb-16
---

# Core Concepts

mjswan uses a four-level hierarchy to describe a browser application: **Builder → Project → Scene → Policy/Splat**. Understanding this structure is the fastest way to get oriented before looking at the API.

```
Builder
  └── Project
        └── Scene
              ├── Splat
              └── Policy
                    └── Motion
```

## Builder

`Builder` is the entry point. It collects everything you define and, when you call `build()`, compiles it into a self-contained web application written to `dist/` (or a directory you choose).

```python
import mjswan

builder = mjswan.Builder()
# … add projects …
app = builder.build()
app.launch()
```

Constructor arguments:

| Argument | Default | Purpose |
|---|---|---|
| `base_path` | `"/"` | URL prefix when hosting at a subdirectory (e.g. `"/mjswan/"` for a GitHub Pages project page) |
| `gtm_id` | `None` | Google Tag Manager container ID; injects the GTM snippet when set |
| `mt` | `False` | Multi-threaded MuJoCo WASM. Requires [Cross-Origin Isolation](../guides/deployment.md#cross-origin-isolation-headers-for-multi-threading) |
| `debug` | `False` | Keep browser console messages in the built bundle |

`build()` returns an [`MjswanApp`](../api/core.md#mjswanapp): `launch()` serves it locally,
`publish()` uploads it to [mjswan Cloud](../guides/publishing.md).

## Project

A project groups related scenes. The app opens on the project marked `default=True` (or
the first added), and a URL selects another with `?project=<id>`:

```python
project = builder.add_project(name="My Robots", default=True)
demo = builder.add_project(name="Demo")  # ?project=demo
```

Every project, scene, policy and splat has an **id** derived from its name — lowercased,
runs of anything but `a-z0-9` collapsed to `_`, edges trimmed, so `"Newton's Cradle"`
becomes `newton_s_cradle`. The id is the directory the object is written to and the value
the URL parameters take. Two siblings whose names sanitize alike get `<id>` and `<id>_1`,
with a warning naming both.

## Scene

A scene contains exactly one MuJoCo model. You supply either an `MjSpec` or an `MjModel`:

```python
import mujoco

# Compressed .mjz — smaller output, recommended for large deployments
scene = project.add_scene(
    spec=mujoco.MjSpec.from_file("robot/scene.xml"),
    name="My Robot",
)

# Binary .mjb — loads slightly faster in the browser, produces larger files
scene = project.add_scene(
    model=mujoco.MjModel.from_xml_path("robot/scene.xml"),
    name="My Robot",
)
```

!!! tip "Which format should I use?"
    Use `spec=` unless you have a specific reason to prefer `model=`. The `.mjz` format uses DEFLATE compression and is significantly smaller — important when approaching GitHub Pages' 1 GB deployment limit.

### `control_dt` — required once a scene carries a policy

The MuJoCo model carries only the physics `timestep`. The rate the *policy* acts at —
mjlab's `timestep * decimation` — is nowhere in the model, and a wrong control rate raises
no error at playback: the policy simply runs at a speed it was never trained for. So the
build asks for it explicitly:

```python
scene = project.add_scene(spec=spec, name="My Robot", control_dt=0.02)  # 50 Hz
```

A model-only scene needs none. [`add_scene_mjlab`](../guides/mjlab.md) fills it in from the
task.

### Events

Events are the reset and randomization terms mjlab attaches to a task — pushing the robot,
perturbing joint positions on reset, randomizing friction at startup. Like observations,
actions, terminations and commands, they belong to the **MDP a policy runs against**
([`MdpConfig`](../api/core.md#mdpconfig)): a policy trained with a push gets its push, and
switching to a policy trained without one switches the push off. A scene's `events` are
the default for any of its policies that declares none:

```python
scene = project.add_scene(spec=spec, name="My Robot", control_dt=0.02, events=events)
scene.set_events(events)  # equivalent, after the fact
scene.add_policy(..., events={"push": push})  # this policy's own instead
```

Switching to a policy with a different MDP first restores the model values the previous
MDP's startup randomization changed, reseeds the term PRNG, and then runs the new MDP's
`mode="startup"` events — so A → B → A reproduces A's first draw, however long the session
has run.

## Splat

A splat is a [Gaussian Splat](https://en.wikipedia.org/wiki/Gaussian_splatting) background rendered behind the MuJoCo simulation. Splats are stored as `.spz` files and give scenes a photorealistic real-world environment without affecting physics.

Add one or more splats to a scene using `add_splat()`. You must supply exactly one of `source` or `url`:

```python
# Recommended: bundle the .spz file into the app
scene.add_splat(
    "Lab Environment",
    source="lab.spz",  # copied into dist/ at build time
    scale=1.35,  # converts splat units → meters
    z_offset=0.71,  # vertical shift to align ground planes
)

# Alternative: reference an external URL (not bundled)
scene.add_splat(
    "Outdoor",
    url="https://example.com/outdoor.spz",
    scale=3.0,
    z_offset=0.5,
)
```

When multiple splats are attached to the same scene, the viewer shows a selector so users can switch between them at runtime.

### Source vs URL

| Option | Effect |
|---|---|
| `source` | Copies the `.spz` into `dist/` at build time — fully self-contained, works offline |
| `url` | Browser fetches the file at runtime — smaller build, requires network access |

### Alignment controls

| Parameter | Description |
|---|---|
| `scale` | Metric scale factor (splat units → metres). Use `metric_scale_factor` from capture metadata if available |
| `x_offset`, `y_offset`, `z_offset` | Position offsets in scaled splat units. `z_offset` aligns ground planes; use `ground_plane_offset` from capture metadata if available |
| `roll`, `pitch`, `yaw` | Rotation in degrees applied on top of the COLMAP → Three.js base rotation |

Set `control=True` to expose these alignment controls as live sliders in the viewer — useful while calibrating a new capture:

```python
scene.add_splat("Lab", source="lab.spz", scale=1.35, control=True)
```

### Splat selector without pre-configured splats

By default, the Splat selector only appears when at least one splat is attached to the scene. Call `enable_splat_section()` to show the selector unconditionally — this lets viewers paste an arbitrary `.spz` URL directly in the control panel at runtime:

```python
scene.enable_splat_section()
```

## Policy

A policy is an ONNX model that runs inference inside the browser. Attach one or more policies to a scene:

```python
import onnx

policy = scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    config_path="locomotion.json",  # optional: observation/action config
)
```

Policies are purely client-side: inference runs in the browser via ONNX Runtime Web, so no
server is needed at runtime.

More usefully, you describe the whole MDP layer from Python by passing `observations=`,
`actions=`, `commands=` and `terminations=` to `add_policy()`. Those take
[mjlab's own config classes and functions](../guides/policy-config.md) — mjswan traces the
term bodies to ONNX at build time, so there is no reimplementation to import and no
`config_path` needed:

```python
from mjlab.envs.mdp import observations as obs_fns
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.trace_env import build_single_entity_trace_env

scene = project.add_scene(spec=build_spec(), name="My Robot", control_dt=0.02)
scene.set_trace_env(
    build_single_entity_trace_env(build_spec)
)  # not needed for mjlab scenes

scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    policy_joint_names=[...],
    observations=ObservationGroupCfg(
        terms={"base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel)}
    ),
)
```

Tracing needs a live environment to read shapes from and resolve entity patterns against.
An [mjlab scene](../guides/mjlab.md) builds one from its task; a plain `add_scene` scene
needs `set_trace_env(...)`. See [How the Build Works](../guides/how-it-works.md) and
[examples/tutorial/minimum_policy.py](https://github.com/ttktjmt/mjswan/blob/main/examples/tutorial/minimum_policy.py){:target="_blank"}.

### Commands

Commands let users interact with a running policy — for example, steering a walking robot with velocity sliders. Pass a `commands=` dict to `add_policy()`:

```python
scene.add_policy(
    name="Locomotion",
    policy=onnx.load("locomotion.onnx"),
    commands={
        "velocity": mjswan.ui_command(
            [
                mjswan.Slider(
                    "lin_vel_x", "Forward Velocity", range=(-1.0, 1.0), default=0.5
                ),
                mjswan.Slider(
                    "lin_vel_y", "Lateral Velocity", range=(-0.5, 0.5), default=0.0
                ),
                mjswan.Slider("ang_vel_z", "Yaw Rate", range=(-1.0, 1.0), default=0.0),
            ]
        ),
    },
)
```

Available command inputs:

| Class | Description |
|---|---|
| `mjswan.Slider` | Continuous range slider. Fields: `name`, `label`, `range`, `default`, `step` |
| `mjswan.Button` | Momentary push button. Fields: `name`, `label` |
| `mjswan.Checkbox` | Boolean toggle. Fields: `name`, `label`, `default` |

### Motion

Motion-tracking policies need one or more reference motions (`.npz` files) loaded alongside the ONNX model. Attach them to the `PolicyHandle`:

```python
policy = scene.add_policy(name="Tracker", policy=onnx.load("tracker.onnx"))

# Local .npz bundled into dist/ at build time
policy.add_motion(
    name="default",
    source="motions/walk.npz",
    fps=50.0,
    anchor_body_name="pelvis",
    body_names=("pelvis",),
    default=True,
    loop=False,
)

# Or fetch from a W&B run
policy.add_motion_wandb(
    run_path="<entity>/<project>/<run_id>",
    anchor_body_name="pelvis",
    body_names=("pelvis",),
)
```

`anchor_body_name` and `body_names` are required — they tell the browser-side tracker which bodies in the MuJoCo model correspond to the dataset. `default=True` marks the motion as the one selected on load; when multiple motions are attached the viewer shows a selector. See the API reference for the full parameter list.

## Licenses

A build carries its license files as files, beside the data they cover
([ADR 0007](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0007-license-files-in-the-build.md)).
There are two kinds, and `manifest.json` never mentions either:

- **The work's own** — `<project-id>/LICENSE` and `NOTICE`. Declare them once on the
  builder, or per project:

  ```python
  builder = mjswan.Builder(license="Apache-2.0", copyright="2026 Example Lab")
  project = builder.add_project("My Robots", license="MIT", copyright="2026 Example Lab")
  project.set_notice("Includes the Go2 model by Unitree Robotics.")
  ```

  A generatable SPDX id — `Apache-2.0`, `MIT`, `BSD-3-Clause`, `BSD-2-Clause`,
  `CC-BY-4.0`, `CC0-1.0` — writes the standard text with an `SPDX-License-Identifier`
  tag on its first line; a path copies your own file verbatim.

- **Third-party components** — `<project-id>/<scene-id>/LICENSE.<component>` and
  `NOTICE.<component>`, one per robot model, motion clip or other asset the scene
  contains. `add_scene(spec=...)` detects them: any `LICENSE*` / `COPYING*` / `NOTICE*`
  beside the model file, or beside the directories its meshes and textures resolve to
  (up to two parents), is copied verbatim and named after that directory — a menagerie
  model yields `LICENSE.unitree_go2`. When nothing is found but the model is one mjswan
  knows (`add_scene_mjlab` also tries the task id), a generated file with the upstream
  holder is written. Anything else — a motion clip, policy weights — is declared:

  ```python
  scene.add_attribution("lafan1", license="CC-BY-4.0", copyright="Ubisoft")
  scene.add_attribution("clip", notice="Retargeted from ...")
  scene.clear_attributions()  # when a detection is wrong
  ```

The root `dist/LICENSE` is the engine's own Apache-2.0 and stays out of every publish.
`mjswan info` lists the files it finds with the license each one is; `mjswan publish`
prints them, warns for a restricted license (CC NC / SA / ND, the GPL family) and refuses
one whose terms forbid redistribution. See [Publishing](../guides/publishing.md).

## Output structure

`builder.build()` writes the engine plus the **simulation document**: one `manifest.json`
describing every project, scene, MDP, policy and splat, and a directory per project with
its data.

```
dist/
├── index.html
├── logo.svg
├── robots.txt
├── manifest.json            ← the document's one descriptor: format, version, projects
├── assets/                  ← compiled JS / CSS / WASM — the engine
├── _headers                 ← only when Builder(mt=True)
├── coi-serviceworker.js     ← only when Builder(mt=True)
├── LICENSE                  ← the engine's Apache-2.0; never the work's
└── <project-id>/            ← name2id(project name), e.g. my_robots/
    ├── LICENSE                ← the work's license, when declared (see Licenses)
    ├── NOTICE                 ← the work's notice, when declared
    └── <scene-id>/          ← name2id(scene name)
        ├── scene.mjz              ← or scene.mjb
        ├── mdp/<mdp-id>/          ← one per MdpConfig: mdp_0, mdp_1, … or its name
        │   ├── obs/<group>.onnx       ← traced observation group (usually one fused graph)
        │   ├── term/<name>.onnx       ← traced termination bodies
        │   ├── command/<name>.onnx
        │   └── event/<name>.onnx      ← the MDP's events
        ├── policy/<policy-id>.onnx    ← the trained network, one per policy
        ├── assets/
        │   ├── <motion-id>.npz        ← one per distinct clip, shared by the scene's policies
        │   └── <splat-id>.spz         ← only when source= is used
        ├── LICENSE.<component>        ← one third-party component this scene contains
        └── NOTICE.<component>         ← its notice, if it has one
```

The result is a fully static site: copy `dist/` to any static host (GitHub Pages, Netlify, S3, …) and it works without a server.

The document is also one file: `app.save_document()` writes `dist.swn`, a ZIP of the
manifest and the project directories with no engine in it, which `mjswan info` and
`mjswan publish` accept wherever they accept a directory.

## Environment variables

| Variable | Effect |
|---|---|
| `MJSWAN_BASE_PATH` | Read by the Vite build (`vite.config.ts`) and used as the asset base. Useful in CI pipelines. |
| `MJSWAN_NO_LAUNCH` | Convention used by the bundled example scripts (e.g. `examples/demo/main.py`) to skip `app.launch()` after building. Honor it in your own build scripts to make them CI-friendly. |
| `MJSWAN_TOKEN` | Access token for [`mjswan publish`](../guides/publishing.md); skips the interactive login. |
| `MJSWAN_API_BASE` / `MJSWAN_WEB_BASE` | Override the mjswan Cloud API and web endpoints. |
