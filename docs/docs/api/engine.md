---
icon: octicons/browser-16
---

# Engine API (TypeScript)

The `mjswan` npm package is the browser side of mjswan: MuJoCo physics via WebAssembly,
three.js rendering, and ONNX Runtime Web for the policy and traced MDP terms. The Python
`Builder` bundles it for you, so most users never touch this API.

Reach for it in three cases:

- **embedding** a simulation in an app you already own, with your own UI;
- **loading a build's `manifest.json`** and driving scene/policy switching yourself;
- **authoring custom MDP terms** in TypeScript, with type support.

```bash
npm install mjswan
```

!!! note "Two layers, deliberately separate"
    `createEngine` is **bytes in, snapshot out** — it knows nothing about `manifest.json`
    and never fetches anything. `mjswan/manifest` parses a build's `manifest.json` into a
    catalog of loadable things. Keeping them apart is what lets the same engine serve the
    bundled React app, mjswan Cloud, and your own page.

## `createEngine`

```ts
import { createEngine } from 'mjswan';

const engine = await createEngine(container, { multithreaded: false });

const model = await (await fetch('/my_robots/g1/scene.mjz')).arrayBuffer();
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
| `handTracking` | `boolean` | `false` | Put a headset's WebXR-tracked hands in the simulation as mocap-driven capsules, so a VR viewer can push and grasp what it sees ([details](../guides/embedding.md#hand-tracking-in-vr)). Every scene loaded gains the hand bodies, at about 1.6x per physics step. |

!!! note "The one thing the engine does fetch"
    `dist/mjswan.js` resolves its own WebAssembly — MuJoCo's and ORT's — relative to itself
    via `import.meta.url`, so serving the published `dist/` from your own origin is enough:
    no CDN, and `script-src 'self'` covers it.

!!! note "Where inference runs"
    Every inference runs on **wasm** (the policy network and the traced MDP term graphs
    alike), and there is nothing to configure. The engine ships ONNX Runtime's CPU build and
    asks for no other provider. Its WebAssembly is 13.3 MiB against 26.5 MiB for the build
    that adds WebGPU: over the 25 MiB per-file limit hosts such as Cloudflare Pages enforce,
    and spent on a GPU round trip per step that networks this size do not win back. Term
    graphs make the case twice over: a step runs many of them, and every inference in the
    page goes through one serialized queue.

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
| `debugVis` | `DebugVisControls` | `set(term, enabled)`: show or hide a command term's drawing, such as the velocity arrows. |
| `events` | `EventControls` | `fire(name)` for a `manual` term, `setArmed(name, armed)` for an `interval` one. |
| `interaction` | `InteractionControls` | `setMode(id)`, `getMode()`, `setParam(mode, name, value)`, `getParams(mode)`, `cancel()`. See [Pointer interaction](#pointer-interaction). |
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
  /** Terms with a debug drawing to toggle; empty when the policy has none. */
  debugVis: ReadonlyArray<DebugVisDescriptor>;
  /** Event terms the operator can drive; empty when the scene has none. */
  events: ReadonlyArray<EventDescriptor>;
  /** Every pointer mode, with whether this scene can run it. */
  interactions: ReadonlyArray<InteractionModeDescriptor>;
  interactionMode: InteractionModeId;
  /** Current parameter values, per mode. */
  interactionParams: Readonly<Record<InteractionModeId, Readonly<Record<string, number>>>>;
  /** The seed in use, so an app recording a session can persist it. */
  termSeed: number;
}
```

`commands` is the generic description of the policy's UI: each `CommandDescriptor` carries
an `id` (`"group:name"`), a `type` of `'slider' | 'checkbox' | 'button'`, a `label`, and —
for a slider — `min` / `max` / `step`, an optional `enabledWhen` naming a gating checkbox,
and an optional `adjustableRange` companion. Render them however you like and drive them
through `engine.commands`.

### Pointer interaction

What a press on the canvas does. The set of modes is closed, and `state.interactions`
describes it, so a host draws the switch and the sliders generically rather than naming
modes itself.

| Mode | `id` | Does | Parameters |
|---|---|---|---|
| Pull | `pull` | Drags a body toward the pointer on a spring for as long as the press is held. The arrow thickens with the force. | `spring` (N/m, default 100) |
| Push | `push` | Taps a body to shove it along the inward normal of the face that was hit. | `impulse` (N s, default 10) |
| Grab | `weld` | Carries a body where you put it, and releasing leaves it with the speed the carry gave it. The cursor shows a closed hand while something is held. | `softness` (s, default 0.02), `torqueScale` (checkbox, default 1) |

Each `InteractionModeDescriptor` carries an `id`, a `label`, an `available` flag with a
`reason` when it is false, and a `params` list. A parameter gives `name`, `label`, `unit`,
`type` (`'slider'` or `'checkbox'`, the latter holding 0 or 1), `step`, `default`, and two
ranges: `min` / `max` is what `setParam` clamps to, and `softMin` / `softMax`, when
present, is the narrower span a slider should drag over. The viewer's number box takes
anything inside `min` / `max`, and a value past the slider pins its thumb at the end.

| Parameter | Slider | Accepted |
|---|---|---|
| `pull.spring` | 0 to 200 | 0 to 2000 |
| `push.impulse` | 0.1 to 50 | 0.1 to 500 |
| `weld.softness` | 0.004 to 0.2 | the same |

`weld.softness` is the weld's time constant (`eq_solref[0]`): how long the held body
takes to catch up with the pointer, so a larger value carries it on a looser spring.
`weld.torqueScale` is `eq_data[10]`; at 1 the body keeps the pose it was grabbed in, at 0
it hangs from the grab point.

**Input bindings are not configurable, by design.** A press that hits a geom belongs to the
active mode and a press that misses belongs to the camera, on a mouse and on a touchscreen
alike, with no hover, no modifier key and no right click. Switching mode is the only input
decision a host makes, through `setMode`. Call `cancel()` before taking the pointer away
(entering an overlay, say) so a held body is let go rather than left under a force.

`available` is false when the loaded scene cannot run a mode, and `reason` says why. Grab
needs the weld the viewer injects into the scene MJCF, which a scene loaded as a compiled
`.mjb` does not have, and which is also withheld from a policy scene whose model does not
namespace its elements: one more body there would change the width of a traced graph's
input.

### Inputs

The engine never fetches. Every asset arrives as `Bytes` — an `ArrayBuffer`, or a
`() => Promise<ArrayBuffer>` thunk if you want it loaded on demand.

=== "SceneInput"

    ```ts
    interface SceneInput {
      model: Bytes;
      modelFormat?: 'mjz' | 'mjb';         // 'mjz' (default) from add_scene(spec=...), 'mjb' from model=
      policy?: PolicyInput | null;
      splat?: SplatInput | null;
      viewer?: ViewerConfig;
      terrainData?: TerrainData;           // spawn patches the policies' events may draw from
      controlDt?: number;                  // mjlab's timestep * decimation
      plugins?: EnginePlugins;             // scene-scoped custom terms
    }
    ```

=== "PolicyInput"

    ```ts
    interface PolicyInput {
      config: object;                      // the manifest's policy entry merged with its MDP
      onnx: Bytes;                         // the trained network
      graphs?: Record<string, Bytes>;      // "mdp/mdp_0/obs/actor.onnx" → bytes
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

`graphs` is keyed by the path the config refers to a graph by — the scene-relative
`mdp/<mdp-id>/{obs,term,command,event}/` layout the
[build emits](../guides/how-it-works.md#artifact-layout). Events travel with the policy:
they are part of its MDP, so `setPolicy` swaps them along with everything else. A missing
entry warns and skips that term rather than failing the load. `policyGraphRefs(config)`
enumerates what a config needs, for a caller assembling inputs by hand.

## `mjswan/manifest`

```ts
import { parseManifest, type Catalog } from 'mjswan/manifest';
```

```ts
function parseManifest(manifest: Manifest | string, source: ByteSource): Catalog
type ByteSource = (relPath: string) => Bytes
```

Turns a build's `manifest.json` plus a byte source into a typed catalog. `source` maps a
build-relative path (`my_robots/g1/scene.mjz`) to bytes — usually a `fetch` wrapper over
the deployment's base URL, but an in-memory resolver works too, which is how a page can
render locally-selected files, or an unpacked `.swn`, without an upload round-trip.

The parser refuses a manifest whose `format` is newer than it understands, naming both
values and the document's `version`, and never gates on `version` itself: that is the
release that wrote the document, for a host to pick a matching engine by.

The catalog is lazy: nothing is fetched until you call a `build()`.

```ts
const base = 'https://example.com/myapp/';
const bytes = (path: string) => async () =>
  (await fetch(new URL(path, base))).arrayBuffer();

const manifest = await (await fetch(new URL('manifest.json', base))).text();
const catalog = parseManifest(manifest, bytes);

const project = catalog.projects[0];           // the default project comes first
const scene = project.scenes[0];
await engine.loadScene(await scene.buildScene());          // defaults
await engine.loadScene(await scene.buildScene({ policy: 'locomotion' }));  // by id
```

| Type | Shape |
|---|---|
| `Catalog` | `{ projects: ProjectCatalog[], pluginsPath?: string }` |
| `ProjectCatalog` | `{ id, name, default, scenes }` |
| `SceneEntry` | `{ id, name, camera?, splatSection, policies, splats, buildScene(opts?) }` |
| `PolicyEntry` | `{ id, name, default, motions, build() }` |
| `SplatEntry` | `{ id, name, control, transform, build() }` |

Every entry carries its `id` — `name2id(name)`, the directory it lives in and the value
the bundled app's `?project=` / `?scene=` / `?policy=` parameters take. `sanitizeName(name)`
is the same function in TypeScript, pinned to the Python one by a shared table of cases,
so a URL written from a display name resolves to the same entry on both sides.

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
| MuJoCo | 3.11.0 (`@ttktjmt/mujoco`, bundled: MuJoCo's own package with the `mjtBool` array fix of [google-deepmind/mujoco#3616](https://github.com/google-deepmind/mujoco/pull/3616)), the same version as the Python package's `mujoco` pin |
| Browser | WebAssembly + WebGL2. `SharedArrayBuffer` only for `multithreaded: true`. |

The library build (`dist/mjswan.js`) is a single self-contained ESM: every dependency is
bundled, the MuJoCo and ONNX WASM are co-located next to it, and it runs single-threaded by
default, so it can be loaded cross-origin straight from a CDN without COOP/COEP headers.
