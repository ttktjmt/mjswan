---
icon: octicons/package-16
---

# Core API

This page documents every public symbol exported from the `mjswan` package.

---

## Builder

```python
class mjswan.Builder(base_path: str = "/", gtm_id: str | None = None)
```

Top-level builder that orchestrates projects, scenes, and policies and produces a deployable web application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `base_path` | `str` | `"/"` | URL prefix for subdirectory deployments. Set to e.g. `"/mjswan/"` when the site lives at `https://user.github.io/mjswan/`. |
| `gtm_id` | `str \| None` | `None` | Google Tag Manager container ID (e.g. `"GTM-XXXXXXX"`). When provided, the GTM snippet is injected into the built HTML. |

### Builder.add_project

```python
def add_project(name: str, *, id: str | None = None) -> ProjectHandle
```

Add a project to the application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `id` | `str \| None` | `None` | URL slug. The first project defaults to `None` (served at `/`). Subsequent projects without an explicit `id` get one derived from `name` (lowercased, spaces/hyphens → underscores). |

**Returns** — `ProjectHandle`

### Builder.build

```python
def build(output_dir: str | Path | None = None) -> mjswanApp
```

Compile and save the application.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str \| Path \| None` | `None` | Output directory. Defaults to `dist/` next to the calling script. Relative paths are resolved against the caller's directory. |

**Returns** — `mjswanApp`

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
) -> SceneHandle
```

Add a MuJoCo scene. Provide exactly one of `model` or `spec`.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Display name shown in the UI. |
| `model` | `mujoco.MjModel \| None` | `None` | Compiled MuJoCo model. Saved as `.mjb` (binary). Loads faster; larger files. |
| `spec` | `mujoco.MjSpec \| None` | `None` | MuJoCo spec. Saved as `.mjz` (DEFLATE-compressed ZIP). Smaller files; slightly slower to load. |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata stored in `config.json`. |

**Returns** — `SceneHandle`

**Raises** — `ValueError` if both or neither of `model`/`spec` are provided.

### ProjectHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the project. |
| `id` | `str \| None` | URL slug of the project. |

---

## SceneHandle

Returned by `ProjectHandle.add_scene()`. Use it to attach policies to a scene.

### SceneHandle.add_policy

```python
def add_policy(
    policy: onnx.ModelProto,
    name: str,
    *,
    metadata: dict[str, Any] | None = None,
    source_path: str | None = None,
    config_path: str | None = None,
) -> PolicyHandle
```

Attach an ONNX policy to the scene.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `policy` | `onnx.ModelProto` | — | Loaded ONNX model (e.g. from `onnx.load("policy.onnx")`). |
| `name` | `str` | — | Display name shown in the UI. |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata. |
| `source_path` | `str \| None` | `None` | Path to the source `.onnx` file. Written to `config.json` for reference. |
| `config_path` | `str \| None` | `None` | Path to a JSON file containing observation/action config. The builder merges this with command definitions and writes it alongside the policy. |

**Returns** — `PolicyHandle`

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

### SceneHandle.set_metadata

```python
def set_metadata(key: str, value: Any) -> SceneHandle
```

Set a metadata entry for the scene. Returns `self` for chaining.

### SceneHandle properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name of the scene. |

---

## PolicyHandle

Returned by `SceneHandle.add_policy()`. Use it to add interactive commands.

### PolicyHandle.add_command

```python
def add_command(name: str, inputs: list[CommandInput]) -> PolicyHandle
```

Add a command group — a named set of inputs (sliders, buttons) that the policy can read as observations.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `name` | `str` | Identifier used by the observation system to retrieve values (e.g. `"velocity"`). |
| `inputs` | `list[CommandInput]` | List of `Slider` or `Button` instances. |

**Returns** — `self` for chaining.

### PolicyHandle.add_velocity_command

```python
def add_velocity_command(
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> PolicyHandle
```

Convenience method: adds a `"velocity"` command group with `lin_vel_x`, `lin_vel_y`, and `ang_vel_z` sliders — the standard pattern for locomotion policies.

**Returns** — `self` for chaining.

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

## Command inputs

### Slider

```python
mjswan.Slider(
    name: str,
    label: str,
    range: tuple[float, float] = (-1.0, 1.0),
    default: float = 0.0,
    step: float = 0.01,
)
```

Alias for `SliderConfig`. Rendered as a range slider in the browser UI.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Internal key used to look up the value in the policy observation. |
| `label` | `str` | Human-readable label shown in the UI. |
| `range` | `tuple[float, float]` | `(min, max)` bounds. |
| `default` | `float` | Initial value. |
| `step` | `float` | Slider increment (default `0.01`). |

### Button

```python
mjswan.Button(name: str, label: str)
```

Alias for `ButtonConfig`. Rendered as a momentary push button in the browser UI.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Internal key. |
| `label` | `str` | Human-readable label shown in the UI. |

---

## mjswanApp

Returned by `Builder.build()`.

### mjswanApp.launch

```python
def launch(
    *,
    host: str = "localhost",
    port: int = 8080,
    open_browser: bool = True,
) -> None
```

Start a local HTTP server and (optionally) open the application in a browser.

The server automatically sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers — required for `SharedArrayBuffer`, which MuJoCo WASM uses for threading.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"localhost"` | Bind address. |
| `port` | `int` | `8080` | Port. If already in use, the next available port is chosen automatically. |
| `open_browser` | `bool` | `True` | Open the default browser on start. |

Blocks until interrupted with `Ctrl-C`.

---

## Helper functions

### velocity_command

```python
mjswan.velocity_command(
    lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    lin_vel_y: tuple[float, float] = (-0.5, 0.5),
    ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    default_lin_vel_x: float = 0.5,
    default_lin_vel_y: float = 0.0,
    default_ang_vel_z: float = 0.0,
) -> CommandGroupConfig
```

Create a standard `"velocity"` `CommandGroupConfig` with three sliders (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`). Equivalent to calling `PolicyHandle.add_velocity_command()` but returns the config object directly.
