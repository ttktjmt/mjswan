---
icon: octicons/browser-16
---

# Engine API (TypeScript)

The `mjswan` npm package is the browser side of mjswan: MuJoCo physics via WebAssembly,
three.js rendering, and ONNX Runtime Web for the policy and traced MDP terms. The Python
`Builder` bundles it for you, so most users never touch this API.

Reach for it in three cases:

- **embedding** a simulation in an app you already own, with your own UI;
- **loading a build's `config.json`** and driving scene/policy switching yourself;
- **authoring custom MDP terms** in TypeScript, with type support.

```bash
npm install mjswan
```

!!! note "Two layers, deliberately separate"
    `createEngine` is **bytes in, snapshot out** — it knows nothing about `config.json`
    and never fetches anything. `mjswan/manifest` parses a build's `config.json` into a
    catalog of loadable things. Keeping them apart is what lets the same engine serve the
    bundled React app, mjswan Cloud, and your own page.

## `createEngine`

```ts
import { createEngine } from 'mjswan';

const engine = await createEngine(container, { multithreaded: false });

const model = await (await fetch('/main/assets/g1/scene.mjz')).arrayBuffer();
await engine.loadScene({ model });
```

```ts
function createEngine(
  element: HTMLElement,
  options?: CreateEngineOptions,
): Promise<MjswanEngine>
```

A headless, instance-scoped engine rendering into `element`. Multiple instances can
coexist; each owns its own MuJoCo module, scene graph, and RNG state.

### `CreateEngineOptions`

| Option | Type | Default | Description |
|---|---|---|---|
| `multithreaded` | `boolean` | `false` | Load the `mujoco/mt` build. Uses `SharedArrayBuffer`, so it requires [COOP/COEP headers](../guides/deployment.md#cross-origin-isolation-headers-for-multi-threading). |
| `termSeed` | `number` | built-in default | Seed for the single PRNG every traced term's `rand` input comes from. Pass back the value read from `MjswanEngineState.termSeed` to re-run a recorded session. |
| `handTracking` | `boolean` | `false` | Put a headset's WebXR-tracked hands in the simulation as mocap-driven capsules, so a VR viewer can push, carry and grasp what it sees ([details](../guides/embedding.md#hand-tracking-in-vr)). Every scene loaded gains the hand bodies, at about 1.6x per physics step. |

### `MjswanEngine`

Verbs are named for their cost: `loadScene` rebuilds the model, everything else is live.

| Member | Signature | Notes |
|---|---|---|
| `loadScene` | `(input: SceneInput) => Promise<void>` | Full model rebuild. |
| `setPolicy` | `(input: PolicyInput \| null) => Promise<void>` | Live; keeps the model loaded. |
| `setSplat` | `(input: SplatInput \| null) => Promise<void>` | Live. |
| `setMotion` | `(name: string \| null) => Promise<boolean>` | Live. Resolves to whether the name was accepted. |
| `setReferenceVisible` | `(visible: boolean) => void` | Motion-tracking ghost toggle. |
| `calibrateSplat` | `(transform: SplatTransform) => void` | Live splat placement, for a calibration UI. |
| `play` / `pause` / `reset` | `() => void` | Playback. |
| `camera` | `CameraControls` | `set(partial)`, `get()`, `frame()`. |
| `commands` | `CommandControls` | `set(id, value)`, `trigger(id)`. |
| `getState` | `() => MjswanEngineState` | Current snapshot. |
| `subscribe` | `(listener) => () => void` | Returns an unsubscribe function. |
| `captureThumbnail` | `(opts?: { maxDim?, quality? }) => Promise<Blob>` | JPEG of the current frame. |
| `dispose` | `() => void` | Tear down and free GPU/WASM resources. |

### `MjswanEngineState`

The immutable snapshot pushed to every `subscribe` listener — build your UI off this.

```ts
interface MjswanEngineState {
  phase: 'running' | 'paused';
  loading: boolean;
  loadingMessage: string | null;
  error: Error | null;
  commands: ReadonlyArray<CommandDescriptor>;
  commandValues: Readonly<Record<string, number>>;
  /** The seed in use, so an app recording a session can persist it. */
  termSeed: number;
}
```

`commands` is the generic description of the policy's UI: each `CommandDescriptor` carries
an `id` (`"group:name"`), a `type` of `'slider' | 'checkbox' | 'button'`, a `label`, and —
for a slider — `min` / `max` / `step`, an optional `enabledWhen` naming a gating checkbox,
and an optional `adjustableRange` companion. Render them however you like and drive them
through `engine.commands`.

### Inputs

The engine never fetches. Every asset arrives as `Bytes` — an `ArrayBuffer`, or a
`() => Promise<ArrayBuffer>` thunk if you want it loaded on demand.

=== "SceneInput"

    ```ts
    interface SceneInput {
      model: Bytes;                        // .mjz (the engine unpacks it)
      policy?: PolicyInput | null;
      splat?: SplatInput | null;
      viewer?: ViewerConfig;
      events?: EventConfig[];              // scene-scoped events
      terrainData?: TerrainData;
      controlDt?: number;                  // mjlab's timestep * decimation
      graphs?: Record<string, Bytes>;      // "event/push_robot.onnx" → bytes
      plugins?: EnginePlugins;             // scene-scoped custom terms
    }
    ```

=== "PolicyInput"

    ```ts
    interface PolicyInput {
      config: object;                      // parsed policy.json
      onnx: Bytes;                         // the trained network
      graphs?: Record<string, Bytes>;      // "obs/policy.onnx" → bytes
      motions?: MotionInput[];
      plugins?: EnginePlugins;             // policy-scoped custom terms
    }
    ```

=== "SplatInput"

    ```ts
    interface SplatInput {
      data: Bytes;                         // .spz
      collider?: Bytes;                    // optional collision mesh
      transform?: SplatTransform;
    }
    ```

`graphs` is keyed by the path the config refers to a graph by — the same
`obs/`, `term/`, `command/`, `event/` layout the
[build emits](../guides/how-it-works.md#artifact-layout). A missing entry warns and skips
that term rather than failing the load. `policyGraphRefs(config)` and
`eventGraphRefs(events)` enumerate what a config needs, for a caller assembling inputs by
hand.

## `mjswan/manifest`

```ts
import { parseManifest, type Catalog } from 'mjswan/manifest';
```

```ts
function parseManifest(config: AppConfig | string, source: ByteSource): Catalog
type ByteSource = (relPath: string) => Bytes
```

Turns a Builder `config.json` plus a byte source into a typed catalog. `source` maps a
build-relative path (`main/assets/g1/scene.mjz`) to bytes — usually a `fetch` wrapper over
the deployment's base URL, but an in-memory resolver works too, which is how a page can
render locally-selected files without an upload round-trip.

The catalog is lazy: nothing is fetched until you call a `build()`.

```ts
const base = 'https://example.com/myapp/';
const bytes = (path: string) => async () =>
  (await fetch(new URL(path, base))).arrayBuffer();

const config = await (await fetch(new URL('assets/config.json', base))).text();
const catalog = parseManifest(config, bytes);

const project = catalog.projects[0];           // the `id: null` "main" project
const scene = project.scenes[0];
await engine.loadScene(await scene.buildScene());          // defaults
await engine.loadScene(await scene.buildScene({ policy: 'Locomotion' }));
```

| Type | Shape |
|---|---|
| `Catalog` | `{ projects: ProjectCatalog[], pluginsPath?: string }` |
| `ProjectCatalog` | `{ name, id, scenes }` |
| `SceneEntry` | `{ name, camera?, splatSection, policies, splats, buildScene(opts?) }` |
| `PolicyEntry` | `{ name, default, motions, build() }` |
| `SplatEntry` | `{ name, control, transform, build() }` |

`sanitizeName(name)` mirrors the Python `name2id` slug helper, for matching a URL fragment
against a display name.

`pluginsPath` is set for a custom-JavaScript build: a trusted app imports that ESM and
passes its exports as `EnginePlugins`. mjswan Cloud ignores it — see
[Publishing](../guides/publishing.md).

Like the engine, the parser ships as a standalone ESM (`dist/manifest.js`), so it too can be
imported straight from a CDN.

## Custom MDP term subpaths

When authoring custom terms in TypeScript, import the base classes from subpath exports
rather than relative paths into `node_modules`:

```ts
import { ObservationBase } from 'mjswan/observation';
import { mjcToThreeCoordinate } from 'mjswan/coordinate';
import type { PolicyState } from 'mjswan/types';
```

| Subpath | Contents |
|---|---|
| `mjswan/observation` | `ObservationBase` and observation helpers |
| `mjswan/command` | command term base classes |
| `mjswan/event` | `EventBase` |
| `mjswan/termination` | `TerminationBase` |
| `mjswan/math` | vector/quaternion helpers used by the built-in terms |
| `mjswan/coordinate` | MuJoCo ↔ three.js coordinate conversion |
| `mjswan/scene` , `mjswan/npz` , `mjswan/bytes` | asset loading helpers |
| `mjswan/types` | shared runtime types |

These export TypeScript sources, so your bundler must handle `.ts` in dependencies (Vite
does). Point `ts_src` on an
[`ObservationBinding`](core.md#mdp-extension-registries) at the compiled class and the
Python builder injects it into the bundle.

!!! tip "Prefer tracing"
    A custom term is only needed when ONNX tracing genuinely cannot express the logic — and
    it makes the build unpublishable to Cloud. Try a trace-friendly replacement callable
    first; see [How the Build Works](../guides/how-it-works.md#a-term-cannot-be-traced).

## Requirements

| Requirement | Version |
|---|---|
| Node.js | 24+ (for building; the runtime is browser-only) |
| Browser | WebAssembly + WebGL2. `SharedArrayBuffer` only for `multithreaded: true`. |

The library build (`dist/mjswan.js`) is a single self-contained ESM: every dependency is
bundled, the MuJoCo and ONNX WASM are co-located next to it, and it runs single-threaded by
default, so it can be loaded cross-origin straight from a CDN without COOP/COEP headers.
