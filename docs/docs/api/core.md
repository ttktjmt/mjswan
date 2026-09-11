---
icon: octicons/package-16
---

# Python API

Every public symbol exported from the `mjswan` package. The TypeScript side is documented
separately in the [Engine API](engine.md).

---

## Builder

```python
class mjswan.Builder(
    base_path: str = "/",
    gtm_id: str | None = None,
    mt: bool = False,
    debug: bool = False,
    *,
    license: str | os.PathLike[str] | None = None,
    copyright: str | None = None,
)
```

Top-level builder that orchestrates projects, scenes, policies, and splats and produces a deployable web application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `base_path` | `str` | `"/"` | URL prefix for subdirectory deployments. Set to e.g. `"/mjswan/"` when the site lives at `https://user.github.io/mjswan/`. |
| `gtm_id` | `str \| None` | `None` | Google Tag Manager container ID (e.g. `"GTM-XXXXXXX"`). When provided, the GTM snippet is injected into the built HTML. |
| `mt` | `bool` | `False` | Enable multi-threaded MuJoCo WASM. Requires Cross-Origin Isolation; mjswan emits a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a `coi-serviceworker.js` (required for GitHub Pages, which cannot set response headers). |
| `debug` | `bool` | `False` | Keep browser console messages in the built application. Defaults to stripping them from the production bundle. |
| `license` | `str \| PathLike \| None` | `None` | The work's license for every project that does not set its own, written as `<project-id>/LICENSE` ([ADR 0007](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0007-license-files-in-the-build.md)). A generatable SPDX id (`"Apache-2.0"`, `"MIT"`, `"BSD-3-Clause"`, `"BSD-2-Clause"`, `"CC-BY-4.0"`, `"CC0-1.0"`) writes the standard text with an `SPDX-License-Identifier` tag on its first line; a path copies the file verbatim. A bad id or a missing file fails here, not at build. |
| `copyright` | `str \| None` | `None` | The holder line of a generated license text, e.g. `"2026 Example Lab"`. |

### Builder.from_mjlab

```python
@classmethod
def from_mjlab(
    task_id: str,
    *,
    run_path: str | list[str] | None = None,
    project_name: str = "mjlab",
    play: bool | None = None,
    env_cfg: Any | None = None,
    base_path: str = "/",
    gtm_id: str | None = None,
    mt: bool = False,
    debug: bool = False,
) -> Builder
```

Convenience factory that creates a `Builder` pre-configured with a single mjlab task. Delegates to the instance method `Builder.add_project_mjlab`. The returned `Builder` already contains one project and one scene; call `build()` directly, or modify it further before building.

`play` and `env_cfg` behave exactly as on `ProjectHandle.add_scene_mjlab`, including being mutually exclusive; both are forwarded unresolved.

When `run_path` is supplied, every `model_*.pt` checkpoint from each W&B run is fetched and converted to ONNX via mjlab + torch (both required). Each attached policy configures itself from the task — observations, commands, actions and terminations from its `env_cfg`, `clip_actions` from its runner config. For finer control, build manually with `add_project` → `ProjectHandle.add_scene_mjlab` → `SceneHandle.add_policy_wandb`.

**Returns** — `Builder`

### Builder.add_project_mjlab

```python
def add_project_mjlab(
    task_id: str,
    *,
    run_path: str | list[str] | None = None,
    project_name: str = "mjlab",
    play: bool | None = None,
    env_cfg: Any | None = None,
) -> ProjectHandle
```

Add a project pre-configured with a single mjlab task (project + mjlab scene + optional W&B policies). The instance-method counterpart to `Builder.from_mjlab`; use it to add an mjlab task to a builder that already has other projects.

**Returns** — `ProjectHandle`

### Builder.add_project

```python
def add_project(
    name: str,
    *,
    default: bool = False,
    license: str | os.PathLike[str] | None = None,
    copyright: str | None = None,
) -> ProjectHandle
```

Add a project to the application. Its id — the directory it is written to and its
`?project=` value — is `name2id(name)`: lowercased, every run of characters other than
`a-z0-9` collapsed to one `_`, edges trimmed (`"Newton's Cradle"` → `newton_s_cradle`). Two
projects whose names sanitize alike get `<id>` and `<id>_1`, with a warning.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI; the id derives from it. |
| `default` | `bool` | `False` | Open on this project when the URL names none. At most one project may set it — two fail the build — and when none does, the first added is the default. |
| `license` | `str \| PathLike \| None` | `None` | This project's license, overriding the builder's; same forms as `Builder(license=)`. |
| `copyright` | `str \| None` | `None` | The holder line of a generated text. |

**Returns** — `ProjectHandle`

**Raises** — `TypeError` for the removed `id=` argument.

### Builder.build

```python
def build(output_dir: str | Path | None = None) -> MjswanApp
```

Compile and save the application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str \| Path \| None` | `None` | Output directory. Defaults to `dist/` next to the calling script. Relative paths are resolved against the caller's directory. |

**Returns** — `MjswanApp`

**Raises** — `ValueError` if no projects have been added.

### Builder.get_projects

```python
def get_projects() -> list[ProjectConfig]
```

Return a copy of all project configurations.

---

## ProjectHandle

Returned by `Builder.add_project()`. Use it to add scenes to a project.

### ProjectHandle.add_scene

```python
def add_scene(
    name: str,
    *,
    model: mujoco.MjModel | None = None,
    spec: mujoco.MjSpec | None = None,
    metadata: dict[str, Any] | None = None,
    control_dt: float | None = None,
    events: Mapping[str, Any] | None = None,
) -> SceneHandle
```

Add a MuJoCo scene. Provide exactly one of `model` or `spec`.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `model` | `mujoco.MjModel \| None` | `None` | Compiled MuJoCo model. Saved as `.mjb` (binary). Loads faster; larger files. |
| `spec` | `mujoco.MjSpec \| None` | `None` | MuJoCo spec. Saved as `.mjz` (DEFLATE-compressed ZIP). Smaller files; slightly slower to load. |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata kept on the scene. |
| `control_dt` | `float \| None` | `None` | Seconds per control step — mjlab's `timestep * decimation`. Required once the scene carries a policy: the model holds only the physics timestep, and a wrong control rate raises nothing at playback. `add_scene_mjlab` fills it in from the task. |
| `events` | `Mapping[str, Any] \| None` | `None` | The scene's default events (`EventTermCfg` instances, mjswan or mjlab). Same as calling `SceneHandle.set_events` afterwards. Events belong to a policy's MDP; these are what a policy gets when it declares none of its own (see [`MdpConfig`](#mdpconfig)). |

**Returns** — `SceneHandle`

**Raises** — `ValueError` if both or neither of `model`/`spec` are provided.

### ProjectHandle.add_scene_mjlab

```python
def add_scene_mjlab(
    task_id: str,
    *,
    play: bool | None = None,
    env_cfg: Any | None = None,
    events: Mapping[str, Any] | None = None,
) -> SceneHandle
```

Load an mjlab task's MuJoCo spec from the task registry and add it as a scene. Requires `mjlab` to be installed. Automatically applies the task's `viewer`, `events`, and any terrain data — including swapping mjlab's `reset_root_state_uniform` for a spawn on a random flat terrain patch, since the browser runs a single env where mjlab trains many spread across the terrain.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `task_id` | `str` | — | mjlab task identifier (e.g. `"go2_flat"`). |
| `play` | `bool \| None` | `None` | Which of the task's two registered configs to load. mjlab keeps them as `env_cfg` (training) and `play_env_cfg`; this selects between them exactly as its `load_env_cfg(task_id, play=...)` does. **Unset means play — the opposite of mjlab's own default, deliberately**: that one serves training scripts, and this is a playback tool. mjlab's training config sets `episode_length_s` to 10–20 s, which mjswan serializes into the browser's `time_out` termination, so a viewer built from it resets the robot every few seconds; it also keeps `push_robot` and the terrain-bounds termination, and lacks `randomize_terrain`. Pass `False` to reproduce training-time conditions. **Mutually exclusive with `env_cfg`** — passing both raises. |
| `env_cfg` | `Any \| None` | `None` | Pre-loaded (and possibly edited) env config to use instead of loading `task_id` fresh. Load it with the `play` you want — `load_env_cfg(task_id, play=True)` — since `play` here then has nothing left to select. The scene keeps whichever config it used, and policies added to it default their term sets to it. A tracking task does not need this: mjlab registers it with `commands["motion"].motion_file = ""`, and the builder points that at the clip it bundles. |
| `events` | `Mapping[str, Any] \| None` | `None` | Scene events, overriding the task's own `env_cfg.events`. Omit to take the task's; pass `{}` for a scene with none. |

**Returns** — `SceneHandle`

**Raises** — `ImportError` if `mjlab` is not installed.

### ProjectHandle.set_license

```python
def set_license(license: str | os.PathLike[str], *, copyright: str | None = None) -> ProjectHandle
```

Set the work's license, written as `<project-id>/LICENSE`: a generatable SPDX id for the
standard text (tagged), or the path to a license text, copied verbatim. Returns `self` for
chaining. **Raises** — `ValueError` for an id with no bundled text or a path that does not
exist.

### ProjectHandle.set_notice

```python
def set_notice(notice: str | os.PathLike[str]) -> ProjectHandle
```

Set the work's notice, written as `<project-id>/NOTICE`: the path to a file, copied
verbatim, or the text itself. Returns `self` for chaining.

### ProjectHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the project. |
| `id` | `str` | `name2id(name)`, unique in the document: the project's directory in the build and its `?project=` value. |

---

## SceneHandle

Returned by `ProjectHandle.add_scene()`. Use it to attach policies, splats, viewer config, and reset events to a scene.

### SceneHandle.add_policy

```python
def add_policy(
    name: str,
    policy: onnx.ModelProto,
    *,
    metadata: dict[str, Any] | None = None,
    config_path: str | None = None,
    source_path: str | None = None,
    env_cfg: Any | None = None,
    task_id: str | None = None,
    mdp: MdpConfig | None = None,
    observations: ObservationGroupCfg | Mapping[str, Any] | None = None,
    commands: Mapping[str, CommandTermConfig] | None = None,
    actions: Mapping[str, ActionTermCfg] | None = None,
    terminations: dict[str, TerminationTermCfg] | None = None,
    events: Mapping[str, Any] | None = None,
    in_keys: Sequence[str] | None = None,
    out_keys: Sequence[str | Sequence[str]] | None = None,
    policy_joint_names: list[str] | None = None,
    policy_num_actions: int | None = None,
    default_joint_pos: list[float] | None = None,
    encoder_bias: list[float] | None = None,
    clip_actions: float | None = None,
    initial_qpos: list[float] | None = None,
    initial_qvel: list[float] | None = None,
    extras: dict[str, Any] | None = None,
    default: bool = False,
) -> PolicyHandle
```

Attach an ONNX policy to the scene. The policy runs against an MDP — `observations`,
`actions`, `terminations`, `commands` and `events` — which it either shares with other
policies (pass one [`MdpConfig`](#mdpconfig) as `mdp`) or gets built from the five term-set
kwargs; not both. The term sets accept mjlab-compatible config classes (mjswan converts
them via the adapter layer; mjlab is a soft dependency), and each defaults to the matching
field of the scene's mjlab env config when it has one — events to the scene's own — so pass
`{}` for a policy that genuinely has none.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `policy` | `onnx.ModelProto` | — | Loaded ONNX model (e.g. from `onnx.load("policy.onnx")`). |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata. |
| `source_path` | `str \| None` | `None` | Path to the source `.onnx` file, kept for reference. |
| `config_path` | `str \| None` | `None` | Path to a JSON sidecar with the checkpoint's own defaults — `policy_joint_names`, `default_joint_pos`, an `actions` block with PD gains. Merged into the policy's manifest entry at build time; Python-side fields win. Its `onnx` block and any `in_keys` / `out_keys` are ignored with a warning: the slot tables are declared here. See [MDP Terms](../guides/policy-config.md#legacy-passing-a-json-file-via-config_path). |
| `env_cfg` | `Any \| None` | `None` | mjlab env config to take this policy's unset term sets from, instead of the scene's. Its control rate must match the scene's `control_dt`. |
| `task_id` | `str \| None` | `None` | mjlab task id used to read the task's runner config (which observation group the actor reads, and `clip_actions`). Defaults to the scene's task. |
| `mdp` | `MdpConfig \| None` | `None` | The MDP to run against. Every policy handed the same object shares it: one set of traced graphs, one `mdp/<id>/` directory. Exclusive with the five term-set kwargs. |
| `observations` | `ObservationGroupCfg \| dict[str, ObservationGroupCfg] \| None` | `None` | A single observation group — mjlab's `env_cfg.observations["actor"]` — mjlab's whole `env_cfg.observations` dict, or a dict keyed by the **slot names** `in_keys` uses. Prefer the first two: a lone group lands under `actor`, the default slot, and needs no `in_keys`. A `"critic"` group is dropped (only the actor is exported to ONNX). Accepts both mjswan and mjlab `ObservationGroupCfg` instances. |
| `commands` | `Mapping[str, CommandTermConfig] \| None` | `None` | Command terms keyed by policy-visible name (e.g. `"velocity"`). Use `mjswan.velocity_command()` or `mjswan.ui_command([...])` to construct values. Accepts mjlab `CommandTermCfg` instances too. |
| `actions` | `Mapping[str, ActionTermCfg] \| None` | `None` | Action term configs keyed by name (e.g. `"joint_pos"`). |
| `terminations` | `dict[str, TerminationTermCfg] \| None` | `None` | Termination term configs keyed by name. |
| `events` | `Mapping[str, EventTermCfg] \| None` | `None` | Event terms keyed by name, in any of the four modes. Defaults to the scene's events; `{}` means none. |
| `in_keys` | `Sequence[str] \| None` | `None` | The network's **input slot table**: `in_keys[i]` names what fills its *i*-th input — an observation group, or a tensor the runtime synthesizes (`is_init`, `adapt_hx`, `time_step`). Positional; the network's own input names never matter. Required when the network has more than one input, checked here against its input count; a single-input network takes its one observation group and needs none. |
| `out_keys` | `Sequence[str \| Sequence[str]] \| None` | `None` | The output slot table, `out_keys[i]` naming the *i*-th output. `action` is the one the runtime drives the actuators from; a recurrent policy also carries `["next", "adapt_hx"]`. Defaults to `["action"]`. |
| `policy_joint_names` | `list[str] \| None` | `None` | Ordered list of joint names the policy controls. Required by the browser runtime to map outputs to actuators. |
| `policy_num_actions` | `int \| None` | `None` | Output width for policies whose action count cannot be inferred from `policy_joint_names` — e.g. muscle-driven ones, which drive actuators rather than joints. |
| `default_joint_pos` | `list[float] \| None` | `None` | Default (resting) joint positions corresponding to `policy_joint_names`. |
| `encoder_bias` | `list[float] \| None` | `None` | Per-joint encoder bias (mirrors mjlab's joint-position action path). |
| `clip_actions` | `float \| None` | `None` | Symmetric bound on the raw policy output, applied before any action term sees it (rsl-rl's `RslRlVecEnvWrapper`). Distinct from `ActionTermCfg.clip`, which bounds `raw * scale + offset` per target. Defaults to the task's runner config; `0.0` is a real bound. |
| `initial_qpos` | `list[float] \| None` | `None` | Optional initial qpos written to the policy's manifest entry for reset logic. |
| `initial_qvel` | `list[float] \| None` | `None` | Optional initial qvel written to the policy's manifest entry for reset logic. |
| `extras` | `dict \| None` | `None` | Extra JSON payload merged verbatim into the policy's manifest entry. |
| `default` | `bool` | `False` | Open on this policy when the URL names none. At most one per scene may set it — two fail the build — and when none does, the first added is the default. |

**Returns** — `PolicyHandle`

**Raises** — `ValueError` when `mdp` is given together with a term set, when `in_keys` /
`out_keys` disagree with the network's input / output count, or when a multi-input network
declares no `in_keys`.

### SceneHandle.add_policy_wandb

```python
def add_policy_wandb(
    run_path: str | list[str],
    *,
    only_latest: bool = False,
    task_id: str | None = None,
    config_path: str | None = None,
    metadata: dict[str, Any] | None = None,
    env_cfg: Any | None = None,
    observations: ObservationGroupCfg | dict[str, ObservationGroupCfg] | None = None,
    commands: Mapping[str, Any] | None = None,
    actions: Mapping[str, ActionTermCfg] | None = None,
    terminations: dict[str, TerminationTermCfg] | None = None,
    clip_actions: float | None = None,
    extras: dict[str, Any] | None = None,
) -> list[PolicyHandle]
```

Fetch ONNX policies from one or more W&B runs and attach them all to the scene. Same `observations` / `commands` / `actions` / `terminations` are applied to every policy.

When `only_latest=False` (the default), all `model_*.pt` checkpoints in each run are downloaded and converted to ONNX via mjlab + torch — `task_id` is required, and comes from the scene unless the scene is a plain one. When `only_latest=True`, only the exported `.onnx` artifact is fetched.

Every term set defaults to the scene's mjlab env config (or to `env_cfg=`, when given), and `task_id` to the scene's task — so for a scene from `add_scene_mjlab` the run path alone is enough. `clip_actions` is read from the task's runner config automatically.

**Returns** — `list[PolicyHandle]` (flat across all runs). The latest checkpoint (highest `_<step>` suffix) is marked as the default.

**Raises** — `ValueError` if `only_latest=False` and `task_id` is missing; `ImportError` if `mjlab`/`torch` are missing.

### SceneHandle.add_splat

```python
def add_splat(
    name: str,
    *,
    source: str | None = None,
    url: str | None = None,
    scale: float = 1.0,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    collider_url: str | None = None,
    control: bool = False,
) -> SplatHandle
```

Add a Gaussian Splat background to the scene. Exactly one of `source` or `url` must be supplied.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the viewer selector. |
| `source` | `str \| None` | `None` | Local path to a `.spz` file. The file is copied into `dist/` during `Builder.build()`. Mutually exclusive with `url`. |
| `url` | `str \| None` | `None` | URL to an external `.spz` file. Fetched by the browser at runtime; not bundled. Mutually exclusive with `source`. |
| `scale` | `float` | `1.0` | Metric scale factor (converts splat units to metres). |
| `x_offset` | `float` | `0.0` | X-axis position offset in scaled splat units. |
| `y_offset` | `float` | `0.0` | Y-axis position offset in scaled splat units. |
| `z_offset` | `float` | `0.0` | Vertical position offset. Use `ground_plane_offset` from capture metadata if available. |
| `roll` | `float` | `0.0` | Roll rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `pitch` | `float` | `0.0` | Pitch rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `yaw` | `float` | `0.0` | Yaw rotation in degrees applied on top of the COLMAP → Three.js base rotation. |
| `collider_url` | `str \| None` | `None` | Optional URL or local path to a `.glb` collision mesh. |
| `control` | `bool` | `False` | Show live scale/offset/rotation controls in the viewer control panel. Useful during calibration. |

**Returns** — `SplatHandle`

**Raises** — `ValueError` if both or neither of `source`/`url` are provided.

### SceneHandle.enable_splat_section

```python
def enable_splat_section() -> SceneHandle
```

Show the Splat selector in the control panel even when no splats are pre-configured. This lets users load a `.spz` file by pasting an external URL at runtime, without requiring any `add_splat()` calls.

Has no effect when at least one splat is already attached (the selector is shown automatically in that case).

Returns `self` for chaining.

### SceneHandle.set_viewer

```python
def set_viewer(config: ViewerConfig) -> SceneHandle
```

Set the camera, tracking mode, and rendering options for the scene. See [`ViewerConfig`](#viewerconfig).

Returns `self` for chaining.

### SceneHandle.set_events

```python
def set_events(events: Mapping[str, Any]) -> SceneHandle
```

Set the scene's default events: a dict of `EventTermCfg` instances (mjswan or mjlab), in
any of the four modes — `"startup"` once when the policy is loaded, `"interval"` on a
countdown timer, `"reset"` on episode reset, `"manual"` from a button. Events belong to a
policy's MDP ([`MdpConfig`](#mdpconfig)); these are what a policy on this scene gets when
it declares none of its own, and a scene with events but no policy writes none.

Equivalent to `add_scene(events=...)`.

Returns `self` for chaining.

### SceneHandle.add_attribution

```python
def add_attribution(
    component: str,
    *,
    license: str | os.PathLike[str] | None = None,
    notice: str | os.PathLike[str] | None = None,
    copyright: str | None = None,
) -> SceneHandle
```

Declare a third-party component this scene contains
([ADR 0007](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0007-license-files-in-the-build.md)).
Written to the scene directory as `LICENSE.<component>` and `NOTICE.<component>`, and shown
on mjswan Cloud beside the work's own license. `add_scene(spec=...)` detects these for a
model with a `LICENSE` beside it or its meshes on disk; `add_scene_mjlab` falls back to a
known-assets table by task id. Motion clips and policy weights have no file to detect and
are only ever declared here. A component already declared, by detection or an earlier
call, is replaced.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `component` | `str` | — | Label for the component, e.g. `"lafan1"`; 1–64 letters, digits, `_`, `-`. It names the file. |
| `license` | `str \| PathLike \| None` | `None` | Its license: a path to the text, copied verbatim, or a generatable SPDX id for the standard text, tagged. |
| `notice` | `str \| PathLike \| None` | `None` | Its notice: a path, copied verbatim, or the text itself. |
| `copyright` | `str \| None` | `None` | The holder line of a generated license text. |

**Returns** — `SceneHandle` (self, for chaining)

**Raises** — `ValueError` when neither `license` nor `notice` is given, for a component
name outside the rule, or for a license id with no bundled text.

### SceneHandle.clear_attributions

```python
def clear_attributions() -> SceneHandle
```

Drop every attribution on the scene, detected or declared. Returns `self` for chaining.

### SceneHandle.set_trace_env

```python
def set_trace_env(env: Any) -> SceneHandle
```

Set the live environment that ONNX tracing runs the scene's term bodies against.

Required for a plain [`add_scene`](#projecthandleadd_scene) scene whose policies carry
observation or termination terms — without it, the build raises. An
[`add_scene_mjlab`](#projecthandleadd_scene_mjlab) scene builds its own at build time;
setting one here pre-empts that.

The env only has to satisfy `env.scene[name].data.<field>` plus the entity write methods
for write-side terms. [`build_single_entity_trace_env`](#build_single_entity_trace_env)
builds a minimal one from a model spec.

Returns `self` for chaining.

---

## build_single_entity_trace_env

```python
mjswan.build_single_entity_trace_env(
    spec_fn: Callable[[], mujoco.MjSpec],
    *,
    entity_name: str = "robot",
    device: str = "cpu",
    zero_geom_margins: bool = True,
    commands: dict[str, Any] | None = None,
) -> Any
```

Build a minimal single-entity mjlab environment for ONNX tracing, out of mjlab's own
`Entity` and `Scene` rather than reimplemented kinematics. It configures no managers and is
never stepped — it is only the tracer's read/write target. Returns it already `reset()`;
pass it to [`SceneHandle.set_trace_env`](#scenehandleset_trace_env).

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `spec_fn` | `Callable[[], MjSpec]` | — | Zero-argument callable returning a **fresh** `MjSpec` (mjlab's `EntityCfg.spec_fn` contract, so it must not share mutable state). |
| `entity_name` | `str` | `"robot"` | Must match whatever the traced functions use as `asset_cfg.name`. |
| `device` | `str` | `"cpu"` | Torch device for the entity's tensors. |
| `zero_geom_margins` | `bool` | `True` | Zero every geom's contact margin before compiling, which mujoco_warp's collision backend requires of some robot XMLs. Safe here since nothing is simulated. |
| `commands` | `dict[str, Any] \| None` | `None` | Trace-time stand-ins for commands the browser owns — a `ui_command` has no Python side, so a term doing arithmetic on its value needs a shape to trace against. |

Joint defaults come from the model's first keyframe, matching what the browser resets to,
so a `*_rel` observation subtracts the same pose on both sides.

**Requires** `mjlab` and `torch` (build-time only).

### SceneHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> SceneHandle
```

Set a metadata entry for the scene. Returns `self` for chaining.

### SceneHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the scene. |
| `id` | `str` | `name2id(name)`, unique in the project: the scene's directory and its `?scene=` value. |

---

## PolicyHandle

Returned by `SceneHandle.add_policy()`. Use it to attach commands, motions, and metadata.

Note that command groups are normally passed to `add_policy(commands=...)` directly. For the standard locomotion case pass `commands={"velocity": mjswan.velocity_command(...)}`.

### PolicyHandle.add_motion

```python
def add_motion(
    *,
    name: str,
    source: str,
    fps: float = 50.0,
    anchor_body_name: str,
    body_names: tuple[str, ...] | list[str],
    dataset_joint_names: list[str] | None = None,
    default: bool = False,
    loop: bool = True,
) -> MotionHandle
```

Attach a bundled `.npz` reference motion to the policy (used by motion-tracking policies).

**Parameters** (all keyword-only)

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name, and the bundled filename stem. Two clips with the same name but different content get a `_1` / `_2` suffix. |
| `source` | `str` | — | Path to a local `.npz`, copied into `dist/` at build time. |
| `fps` | `float` | `50.0` | Frame rate of the clip. |
| `anchor_body_name` | `str` | — | Body the reference trajectory is anchored to. Required. |
| `body_names` | `tuple[str, ...] \| list[str]` | — | Bodies in the MuJoCo model the dataset's bodies correspond to. Required. |
| `dataset_joint_names` | `list[str] \| None` | `None` | Joint order in the dataset. Defaults to the policy's `policy_joint_names`. |
| `default` | `bool` | `False` | Select this motion on load. |
| `loop` | `bool` | `True` | Restart the clip when it ends. |

Each distinct clip is written once per scene and shared by every policy that uses it, so
the checkpoints of one run do not each get a copy.

**Returns** — `MotionHandle`

### PolicyHandle.add_motion_wandb

```python
def add_motion_wandb(
    *,
    name: str | None = None,
    run_path: str | None = None,
    run_id: str | None = None,
    entity: str | None = None,
    project: str | None = None,
    fps: float = 50.0,
    anchor_body_name: str,
    body_names: tuple[str, ...] | list[str],
    dataset_joint_names: list[str] | None = None,
    default: bool = False,
    loop: bool = True,
) -> MotionHandle
```

Download a motion `.npz` artifact from a W&B run and attach it to the policy. Supply either
`run_path="entity/project/run_id"` or `run_id` / `entity` / `project` separately. `name`
defaults to the artifact's own name; every other parameter behaves as on
[`add_motion`](#policyhandleadd_motion).

**Returns** — `MotionHandle`

!!! tip "You usually don't need this for an mjlab tracking task"
    [`add_policy_wandb`](#scenehandleadd_policy_wandb) fetches the run's clip, bundles it,
    and points mjlab's empty `commands["motion"].motion_file` at it. See
    [Using mjlab → Tracking tasks](../guides/mjlab.md#tracking-tasks).

### PolicyHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> PolicyHandle
```

Set a metadata entry for the policy. Returns `self` for chaining.

### PolicyHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the policy. |
| `model` | `onnx.ModelProto` | The attached ONNX model. |

---

## SplatHandle

Returned by `SceneHandle.add_splat()`.

### SplatHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> SplatHandle
```

Set a metadata entry for the splat. Returns `self` for chaining.

### SplatHandle properties

| Property | Type | Description |
|---|---|---|
| `source` | `str \| None` | Local path to the bundled `.spz` file, or `None` if `url` was used. |
| `url` | `str \| None` | External URL to the `.spz` file, or `None` if `source` was used. |
| `scale` | `float` | Metric scale factor. |
| `x_offset` | `float` | X-axis position offset. |
| `y_offset` | `float` | Y-axis position offset. |
| `z_offset` | `float` | Vertical position offset. |
| `roll` | `float` | Roll rotation in degrees. |
| `pitch` | `float` | Pitch rotation in degrees. |
| `yaw` | `float` | Yaw rotation in degrees. |

---

## MotionHandle

Returned by `PolicyHandle.add_motion()` and `add_motion_wandb()`.

### MotionHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> MotionHandle
```

Set a metadata entry for the motion. Returns `self` for chaining.

### MotionHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the motion. |

---

## ViewerConfig

```python
@dataclass
class mjswan.ViewerConfig(
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    distance: float = 4.0,
    fovy: float | None = None,
    elevation: float = -30.0,
    azimuth: float = 45.0,
    origin_type: OriginType = OriginType.AUTO,
    entity_name: str | None = None,
    body_name: str | None = None,
    env_idx: int = 0,
    max_extra_envs: int = 2,
    enable_reflections: bool = True,
    enable_shadows: bool = True,
    height: int = 240,
    width: int = 320,
)
```

Camera and rendering configuration applied to a scene via `SceneHandle.set_viewer()`. Matches the API of `mjlab.viewer.ViewerConfig`.

**Selected fields**

| Field | Description |
|---|---|
| `lookat` | Look-at point in MuJoCo coordinates (x forward, y left, z up). |
| `distance` | Distance from the look-at point to the viewer. |
| `elevation` | Elevation in degrees (negative = viewer above the look-at point). |
| `azimuth` | Azimuth in degrees from the x-axis (forward), CCW. |
| `fovy` | Vertical field of view in degrees (default 45). |
| `origin_type` | One of `ViewerConfig.OriginType.{AUTO, WORLD, ASSET_ROOT, ASSET_BODY}`. Controls how the camera tracks the scene. |
| `body_name` | Body to track when `origin_type` is `ASSET_BODY`. |
| `enable_reflections` / `enable_shadows` | Toggle three.js reflections and shadows. |

### ViewerConfig.from_position

```python
@staticmethod
def from_position(
    position: tuple[float, float, float],
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    fovy: float | None = None,
    origin_type: ViewerConfig.OriginType | None = None,
    body_name: str | None = None,
) -> ViewerConfig
```

Build a `ViewerConfig` from explicit Cartesian viewer/target positions — `lookat`, `distance`, `elevation`, and `azimuth` are computed.

---

## Command inputs

Each entry is a leaf in a command-term UI block. The aliases (`Slider`, `Button`, `Checkbox`) are identical to their `*Config` counterparts; pick whichever reads better.

### Slider

```python
mjswan.Slider(
    name: str,
    label: str,
    range: tuple[float, float] = (-1.0, 1.0),
    default: float = 0.0,
    step: float = 0.01,
    enabled_when: str | None = None,
    adjustable_range: SliderRangeConfig | None = None,
)
```

Continuous range slider.

| Field | Description |
|---|---|
| `name` | Internal key used by the policy observation. |
| `label` | Human-readable label shown in the UI. |
| `range` | `(min, max)` bounds. |
| `default` | Initial value. |
| `step` | Slider increment. |
| `enabled_when` | Optional sibling input name that enables this slider (greys it out when the named input is off). |
| `adjustable_range` | Optional [`SliderRangeConfig`](#sliderrangeconfig) companion slider that rescales this slider's drag range. |

### SliderRangeConfig

```python
mjswan.SliderRangeConfig(
    range: tuple[float, float] = (0.0, 2.0),
    default: float = 1.0,
    step: float = 0.05,
    label: str | None = None,  # defaults browser-side to "Max <label>"
)
```

Bounds for the companion "Max &lt;label&gt;" slider a `Slider` can declare via
`adjustable_range`. The control panel renders it beside the value slider and clamps the
value slider's displayed range to `[-value, value]`, mirroring mjlab's own play GUI.

Purely presentational: it carries no command id, so nothing about it reaches the policy.
Assumes symmetry around zero, which matches the three velocity axes it exists for.

### Button

```python
mjswan.Button(name: str, label: str)
```

Momentary push button. Fields: `name`, `label`.

### Checkbox

```python
mjswan.Checkbox(name: str, label: str, default: bool = False)
```

Boolean toggle. Fields: `name`, `label`, `default`.

---

## Command helpers

### ui_command

```python
mjswan.ui_command(inputs: list[CommandInput]) -> CommandTermConfig
```

Build a `CommandTermConfig` whose value is driven by manual UI inputs (sliders, buttons, checkboxes). Pass the result to `add_policy(commands={...})`.

```python
target_cmd = mjswan.ui_command(
    [
        mjswan.Slider(
            "target_height", "Target Height (m)", range=(0.3, 1.8), default=1.0
        ),
    ]
)
scene.add_policy(name="PD", policy=model, commands={"target": target_cmd})
```

### velocity_command

```python
mjswan.velocity_command(
    *,
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandTermConfig
```

Build a standard `"velocity"` command group (three sliders: `lin_vel_x`, `lin_vel_y`, `ang_vel_z`). Pass it via `commands={"velocity": mjswan.velocity_command(...)}` to `add_policy()`.

This is a manual control panel, not mjlab's `UniformVelocityCommand` — nothing resamples, and the command is whatever the slider says. A scene built on an mjlab task should pass `UniformVelocityCommandCfg` itself, which mjswan binds to a traced term and gives mjlab's own joystick (Enable + per-axis `Max` + `Zero`).

### register_command

```python
mjswan.register_command(mjlab_name: str, spec: CommandBinding) -> None
```

Register an adapter from a custom mjlab `*CommandCfg` class to a browser-side command term. `mjlab_name` should typically be the mjlab config class name (e.g. `"LiftingCommandCfg"`).

---

## MDP extension registries

Every observation / termination / event term is normally traced to ONNX from its own Python function — mjswan ships no built-in TypeScript term classes, so nothing needs registering to work. These three override what one mjlab name resolves to when tracing it as-authored will not do:

```python
mjswan.register_observation(name: str, func: ObservationBinding | Callable) -> None
mjswan.register_termination(name: str, func: TerminationBinding) -> None
mjswan.register_event(name: str, func: EventBinding) -> None
```

Two things to register:

- **A trace-friendly replacement callable** (observations only) — same signature as the original, written so `torch.onnx.export` can follow it. What to reach for when the task's own function is correct but not exportable as written (tensor-method RNG, data-dependent control flow).
- **A `*Binding`** — the escape hatch for a term ONNX tracing cannot express at all. `ts_src` is the absolute path of a `.ts` file exporting the class named by `ts_name`; the builder injects it into the browser bundle. A binding without `ts_src` fails the build: mjswan has no built-in class to fall back on.

A term that fails to trace and has neither of these fails the build, with a message naming both options. It is never silently dropped — a missing observation shortens the vector the policy was trained on, and a missing termination or event leaves the browser without a reset condition the task is configured to have.

Three event terms are exempt, because there is provably nothing for the browser to write:
`randomize_terrain` (one baked terrain, one origin), `encoder_bias` (the runtime applies it
from the policy config), and a root-state write onto a **fixed-base** entity (which cannot
move in mjlab either — mjlab's manipulation tasks configure `reset_base` on their arms
regardless). Startup randomization that perturbs `mjModel` rather than `mjData` — geom
friction, body COM, geom colors — needs no graph either: the build emits a descriptor and
the browser applies it once from the seeded PRNG.

See [How the Build Works](../guides/how-it-works.md#a-term-cannot-be-traced) for the
decision procedure.

---

## Deprecated pre-0.8 names

These aliases still work and warn (class aliases silently), and are scheduled for removal
in **0.9**.

| Pre-0.8 | Current |
|---|---|
| `ProjectHandle.add_mjlab_scene` | [`add_scene_mjlab`](#projecthandleadd_scene_mjlab) |
| `SceneHandle.add_policy_from_wandb` | [`add_policy_wandb`](#scenehandleadd_policy_wandb) |
| `SceneHandle.set_viewer_config` | [`set_viewer`](#scenehandleset_viewer) |
| `SceneHandle.add_splat_section` | [`enable_splat_section`](#scenehandleenable_splat_section) |
| `mjswanApp` | [`MjswanApp`](#mjswanapp) |
| `register_obs_func` | [`register_observation`](#mdp-extension-registries) |
| `register_termination_func` | [`register_termination`](#mdp-extension-registries) |
| `register_event_func` | [`register_event`](#mdp-extension-registries) |
| `register_command_term` | [`register_command`](#register_command) |
| `ObsBinding`, `ObsFunc` | `ObservationBinding` |
| `TermBinding`, `TermFunc` | `TerminationBinding` |
| `EventFunc` | `EventBinding` |
| `MjlabMdpBinding` | `MdpBinding` |
| `CommandTermSpec` | `CommandBinding` |

Two renamed modules keep their old import paths as well: `mjswan.viewer_config` →
`mjswan.viewer`, and `mjswan.wandb_utils` → `mjswan.wandb_io`.

---

## MjswanApp

Returned by `Builder.build()`. The built directory is the engine plus the expanded
[simulation document](#output-structure); `app_dir` is its path.

### MjswanApp.save_document

```python
def save_document(path: str | Path | None = None) -> Path
```

Write the simulation as one `.swn` file and return its path. The document is the build's
data — `manifest.json` and every project directory — packaged as a ZIP of the same tree,
with no engine in it. `path` defaults to the build directory's name with `.swn`, beside
it. `mjswan info`, `mjswan publish` and `MjswanApp.publish` take a `.swn` wherever they
take a directory.

### MjswanApp.launch

```python
def launch(
    *,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
    height: int = 600,
) -> None
```

Start a local HTTP server and (optionally) open the application in a browser. When running inside Google Colab, an inline iframe is rendered instead of starting a blocking server.

The server automatically sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers — required for `SharedArrayBuffer`, which MuJoCo WASM uses for threading.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Bind address (ignored in Colab). |
| `port` | `int` | `8080` | Port. If already in use, the next available port is chosen automatically. |
| `open_browser` | `bool` | `True` | Open the default browser on start (ignored in Colab). |
| `height` | `int` | `600` | Colab iframe height in pixels (ignored outside Colab). |

Blocks until interrupted with `Ctrl-C`.

### MjswanApp.publish

```python
def publish(
    *,
    title: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    token: str | None = None,
    api_base: str | None = None,
) -> PublishResult
```

Upload this build's data files to [mjswan Cloud](../guides/publishing.md) and return the
result, whose `id` gives the hosted page URL. Only the simulation document travels —
`manifest.json`, the scene/policy/motion/splat assets and traced graphs — never the
compiled JavaScript.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `title` | `str \| None` | first project's name | Simulation title. |
| `description` | `str \| None` | `None` | Optional description. |
| `tags` | `list[str] \| None` | `None` | Optional tags. |
| `token` | `str \| None` | `$MJSWAN_TOKEN` | Access token. |
| `api_base` | `str \| None` | `$MJSWAN_API_BASE`, then `https://api.mjswan.com` | Cloud API base URL. |

**Raises** — `mjswan.publish.PublishError` on validation failure or server rejection,
including a build that uses custom-JavaScript MDP terms (`uses_custom_js: true`), which
Cloud cannot render.

Limits: 50 MB per file, 200 MB total, 64 files.

---

## MdpConfig

```python
@dataclass
class mjswan.MdpConfig:
    observations: ObservationGroupCfg | Mapping[str, Any] | None = None
    actions: Mapping[str, ActionTermCfg] | None = None
    terminations: Mapping[str, TerminationTermCfg] | None = None
    commands: Mapping[str, Any] | None = None
    events: Mapping[str, EventTermCfg] | None = None
    name: str | None = None
```

The MDP a policy runs against, as one unit: the five term sets mjlab's managers own. Two
policies handed the **same object** share one MDP — its terms are traced once and written
once, under `mdp/<mdp-id>/` in the scene directory — which is the shape of the common
case, the checkpoints of one training run:

```python
mdp = mjswan.MdpConfig(observations=..., actions=..., terminations=..., commands=...)
scene.add_policy(name="model_1000", policy=onnx.load("model_1000.onnx"), mdp=mdp)
scene.add_policy(name="model_2000", policy=onnx.load("model_2000.onnx"), mdp=mdp)
```

Passing the term sets straight to `add_policy` builds an anonymous `MdpConfig` for that
policy alone. Unset fields are filled from the scene's mjlab env config (events from the
scene's own events) and mjlab types adapted, in place, by the first policy to use it.
`add_policy_wandb` builds one `MdpConfig` per call, so every checkpoint of a run shares it.

`name` fixes the MDP's id (`name2id(name)`). Unnamed, an MDP the `add_policy` term-set
kwargs built takes the id of the policy it was built for; one shared by several policies —
what `add_policy_wandb` does — is numbered `mdp_0`, `mdp_1`, … per scene in first-use
order. Switching policy to a different MDP restores the compiled model, reseeds the term
PRNG, and runs the new MDP's startup events — see
[Core Concepts → Events](../getting-started/core-concepts.md#events).

## Action term configs

`mjswan.envs.mdp.actions` mirrors `mjlab.envs.mdp.actions`, so the import pattern
translates directly. Action is the one manager that is **not** traced to ONNX — it is a
closed set implemented natively in TypeScript, because it runs once per physics substep.

```python
from mjswan.envs.mdp.actions import JointPositionActionCfg
```

| Class | Status |
|---|---|
| `JointPositionActionCfg` | Supported. Adds `use_default_offset`, `stiffness`, `damping`. |
| `JointEffortActionCfg` | Supported. Adds `stiffness`, `damping`. |
| `MuscleActivationActionCfg` | Supported. Adds `normalize` (default `True`). |
| `JointVelocityActionCfg` | Raises `NotImplementedError` at build time. |
| `TendonLengthActionCfg`, `TendonVelocityActionCfg`, `TendonEffortActionCfg` | Raise `NotImplementedError` at build time. |
| `SiteEffortActionCfg` | Raises `NotImplementedError` at build time. |

The unsupported classes are exported so an mjlab config imports cleanly; the failure is at
build time, not import time.

Common `BaseActionCfg` fields: `entity_name`, `actuator_names` (regex patterns, default
`(".*",)`), `scale`, `offset`, `clip` (a `{pattern: (min, max)}` dict), `preserve_order`.

`stiffness` / `damping` are mjswan-specific — the browser computes PD externally for motor
actuators with `biastype=none`. Each accepts a scalar, a per-joint list aligned with
`policy_joint_names`, or a dict keyed by joint name. See
[MDP Terms → Actions](../guides/policy-config.md#actions).

---

## Observation and termination configs

```python
from mjswan.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjswan.managers.termination_manager import TerminationTermCfg
```

API-compatible with their mjlab counterparts, so an mjlab config assigns straight across.

### ObservationTermCfg

| Field | Type | Default | Description |
|---|---|---|---|
| `func` | `Callable \| ObservationBinding` | — | The term body. Any traceable `func(env, **params) -> Tensor` — including mjlab's own — or a binding naming a hand-written TypeScript class. |
| `params` | `dict` | `{}` | Forwarded at trace time. `SceneEntityCfg` patterns resolve to static indices baked into the graph. |
| `scale` | `float \| tuple \| None` | `None` | Element-wise scale, applied after `clip`. |
| `clip` | `tuple[float, float] \| None` | `None` | Applied before `scale`, matching mjlab's order. |
| `history_length` | `int` | `0` | Frames to stack, **oldest first** (`[x_{t-n+1} … x_t]`), as mjlab's history buffer flattens them. `0` = current frame only. |
| `history_steps` | `tuple[int, ...] \| None` | `None` | Look-back offsets in the order the policy reads them, instead of a count — `(16, 8, 4, 2, 1, 0)` reaches 17 frames back with 6 values, and `(0, 1, 2)` is `history_length=3` reversed. Takes precedence over `history_length`. |
| `history_interleaved` | `bool` | `False` | Isaac-style element-major layout instead of frame-major. |

`noise`, `delay_*`, and `flatten_history_dim` are accepted for mjlab compatibility and
ignored — there is no training in the browser.

### ObservationGroupCfg

| Field | Type | Default | Description |
|---|---|---|---|
| `terms` | `dict[str, ObservationTermCfg]` | `{}` | Concatenated in declaration order. |
| `history_length` | `int \| None` | `None` | Group-level override applied to every term whenever it is set — `0` included, which switches the group's history off, as in mjlab. |

`concatenate_terms`, `enable_corruption`, and `flatten_history_dim` are accepted and
ignored.

### TerminationTermCfg

| Field | Type | Default | Description |
|---|---|---|---|
| `func` | `Callable \| TerminationBinding` | — | The term body. `time_out` is classified native automatically — it reads no time-varying state. |
| `params` | `dict` | `{}` | Forwarded at trace time. |
| `time_out` | `bool` | `False` | Marks this as a truncation rather than a terminal failure. |

---

## Output structure

`builder.build()` writes a fully static site: the engine, plus the **simulation document**
— one `manifest.json` describing every project, scene, MDP, policy and splat, and one
directory per project with its data (ADR 0006).

```
dist/
├── index.html
├── logo.svg
├── robots.txt
├── manifest.json            ← the document's one descriptor: format, version, projects
├── assets/                  ← compiled JS / CSS / WASM — the engine, never uploaded
├── LICENSE                  ← the engine's Apache-2.0, never the work's; never uploaded
├── _headers                 ← only when Builder(mt=True)
├── coi-serviceworker.js     ← only when Builder(mt=True)
└── <project-id>/            ← name2id(project name), e.g. my_robots/
    ├── LICENSE                ← the work's license, when declared (Builder / add_project / set_license)
    ├── NOTICE                 ← the work's notice, when declared (set_notice)
    └── <scene-id>/          ← name2id(scene name)
        ├── scene.mjz              ← or scene.mjb (depending on add_scene argument)
        ├── mdp/<mdp-id>/          ← one per MdpConfig: its name, its policy's id, or mdp_0, mdp_1, …
        │   ├── obs/<group>.onnx       ← traced observation group, usually one fused graph (actor.onnx)
        │   ├── term/<name>.onnx       ← traced termination bodies
        │   ├── command/<name>.onnx
        │   └── event/<name>.onnx      ← the MDP's events
        ├── policy/<policy-id>.onnx    ← the trained network, one per policy
        ├── assets/
        │   ├── <motion-id>.npz        ← one per distinct clip, shared by the scene's policies
        │   └── <splat-id>.spz         ← only when source= is used
        ├── LICENSE.<component>        ← a third-party component's license (detected or add_attribution)
        └── NOTICE.<component>         ← its notice
```

Copy `dist/` to any static host (GitHub Pages, Netlify, S3, …) and it works without a server.
License files are not manifest keys ([ADR 0007](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0007-license-files-in-the-build.md)):
`manifest.json` is the same with and without them.

Every key in `manifest.json` is `snake_case`, and every path under a scene entry resolves
against that scene's directory, so a policy entry reads `"onnx": "policy/walk.onnx"` and
its MDP's fused group `"mdp/walk/obs/actor.onnx"`. A key carrying its default is omitted
— `in_keys` when it is `["actor"]`, `out_keys` when it is `["action"]`. The manifest also
stamps `format` (the layout's version, currently 1; an engine refuses a document newer than
it knows) and `version` (the mjswan release that wrote it), independently.

`MjswanApp.save_document()` packs the document — the manifest and the project directories,
nothing of the engine — into one `.swn` file, a ZIP of the same tree. See
[How the Build Works](../guides/how-it-works.md#artifact-layout).
