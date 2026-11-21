# The frontend

`src/mjswan/template/`: TypeScript + React + Vite + three.js + mujoco-wasm, built by `Builder.build()` through `build/frontend.py` ([build.md](build.md)). Skins are in [skins.md](skins.md); what the traced graphs are is in [tracing.md](tracing.md).

## What the browser client does

- Loads the MuJoCo WASM module and steps physics on the main thread against real time, a physics step at a time so the screen gets the poses between two policy steps (`engine/controlStep.ts`, mjlab's viewer budget), and yields to the browser (`engine/yieldToBrowser.ts`).
- Runs the policy **and every traced MDP term body** via onnxruntime-web's CPU build (`onnxruntime-web/wasm`), all on wasm: the build with WebGPU kernels is a 26.5 MiB wasm, over the 25 MiB per file Cloudflare Pages serves, and for graphs this small WebGPU's upload and download round trip loses anyway. Every inference in the page goes through one queue (`core/onnx/runQueue.ts`): ORT-Web's wasm binding keeps a module-global active-run slot, so overlapping `run()` calls throw even across sessions.
- Renders via three.js (reflections, shadows, Gaussian Splat background).
- Supports WebXR (VR and passthrough AR) through `engine.xr`; the host draws the buttons. Tracked hands join the sim as bodies, compiled in at the next model build after the switch is turned on.
- Reads `manifest.json` to discover projects / scenes / MDPs / policies at runtime; `?project=` / `?scene=` / `?policy=` / `?skin=` take ids (`sanitizeName` = `name2id`, pinned by a shared case table: [document.md](document.md)).
- Shows the credits of the scene in view in the About dialog (the panel's logo, or `I`): `ControlPanel/CreditsPanel.tsx` lists the manifest's `credits` and `work`, then the engine's own license, fetching a `NOTICE` when the tab opens and a `LICENSE` when asked.

## Three layers

```
mjswan             the engine: createEngine. No React, no catalog, no manifest, no fetch.
mjswan/manifest    manifest.json + a byte source → a typed, lazy catalog. Framework-free.
React app          ControlPanel + Loader, a client of the two above. In-repo, built by
                   Python for standalone use and demos; not an npm export.
```

mjswan Cloud builds its own overlay on the first two. The React app is the reference client: it is what proves the engine API is sufficient, so a feature the app needs from the engine goes through the public API.

## The engine API (`src/engine/`)

The public boundary is bytes in, snapshot out (`engine/types.ts`):

- **The app owns the catalog; the engine runs one simulation.** MuJoCo holds one live model, so "several scenes, policies, splats, motions" is a catalog of loadable options, and the engine exposes verbs that match each switch's real cost: `loadScene` rebuilds the model; `setPolicy`, `setSplat`, `setSkin`, `setMotion` are live. The app knows whether it is switching scene or policy and picks the verb; the engine guarantees the cost. Next / previous are app one-liners over the catalog it holds.
- **It never fetches.** Content arrives as `Bytes` (an `ArrayBuffer`, or a lazy `() => Promise<ArrayBuffer>` loader, which keeps on-demand motion loading). `mjswan/manifest`'s `parseManifest(manifest, byteSource)` turns a `manifest.json` into a lazy catalog, refusing a `format` newer than it knows, and the app owns the fetching. There is no global `fetch` wrapper and no synthetic origin.
- **Instance-scoped, no module globals.** Command state and every per-scene object belong to the `MjswanEngine` instance, and `dispose()` frees exactly what that instance allocated (WASM module, GL context, loops, ONNX sessions, splat). Several engines on one page are supported, bounded by the browser's live WebGL contexts (about 16) and total WASM memory, a budget the host manages.
- **No engine-side scene cache.** A built scene cannot cross the app/engine boundary as data (`mjModel` / `mjData` are embind-bound to *this* engine's WASM module, the three.js resources to *this* GL context), and keeping several fights the non-shrinkable 2 GB WASM heap. The engine rebuilds on each `loadScene`; warm caching is the host's call, as retained bytes or as another engine instance.
- **OOM is the host's to handle.** The engine surfaces it (`loadScene` rejects with `WasmMemoryLimitError`, `state.error` is set) and never reloads the page, which is a host action like fullscreen.
- **Lifecycle**: `createEngine(element, options?)` is async and loads MuJoCo up front, so a host can subscribe and set the camera before the first scene loads. Options: `multithreaded` (a runtime option, importing `mujoco/mt`; needs COOP/COEP), `termSeed`, `handTracking`, `archive` (limits a host applies to a `.mjz` it did not build).
- **Two channels**: promises for request and result (`loadScene` / `setPolicy` / `setSplat` resolve or reject; `setMotion` returns whether it was accepted, so the app can revert its selection), and one immutable snapshot pushed through `subscribe` for state the engine originates (load progress, command values a reset or termination changed, auto-start, Debug Viz, events, interaction, XR, `termSeed`). A host renders it wholesale (`useSyncExternalStore`) and never demuxes an event taxonomy. The camera pose is deliberately not in it, since it changes every frame: read it with `camera.get()`.
- **Camera**: spherical, in MuJoCo coordinates as `ViewerConfig` is (raw three.js `position` is never exposed). `camera.set()` overwrites the pose, body tracking continues after it, and user drag is always live; `camera.frame()` re-fits to the scene; `camera.setTracking()` is mjviser's "Track camera".
- **Playback**: `play()` / `pause()` / `reset()` (not start / stop: `stop` reads as teardown, which is `dispose()`). `pause()` halts physics while rendering continues; `reset()` is first-class and leaves play / pause alone. Commands are read from the snapshot and written through `engine.commands.set(id, value)` / `trigger(id)`.
- The other namespaces: `graphics`, `debugVis`, `cameraFeeds`, `events`, `interaction`, `xr`, `preview`, and `pick`, plus `captureThumbnail` and `dispose`. `termSeed` goes in and comes back out, to re-run a recorded session (what that does and does not reproduce: [tracing.md](tracing.md)).
- A policy switch is an MDP switch: it restores the model fields the last MDP's events changed, reseeds the term PRNG and runs the new MDP's startup events ([document.md](document.md)).

Rejected for this API: additive controls on a React `mount()` (React stays in the engine's critical path and the CDN bundle); an engine that owns the catalog and next / previous (the app owns the assets, so it owns the catalog); a single opaque `load(everything)` with change detection (a reconciler the app does not need, since it already knows what it is switching).

## The step loop

The step loop follows mjlab's `ManagerBasedRlEnv.step` ordering exactly: forward → command → event → obs → action → physics → term → reset → forward, with **one** `mj_forward` per iteration (`runtime.ts`, `beginControlStep` / `endControlStep`). Do not "optimize" it into a forward after the substeps *and* after the reset. A traced command's update lands asynchronously, so a control step awaits the command manager before reading observations, as mjlab has it before the observation.

The reset chain mirrors `_reset_idx` (`engine/resetChain.ts`): `event(mode="reset")` → observation / action reset → command resample, all awaited in config order, then the termination reset. Writes are last-writer-wins by config order, so concurrency would make the result machine-dependent; events before commands is load-bearing, since both write `qpos`. Terminations are awaited every step, so a reset lands on the step it does in mjlab; other traced calls skip a frame rather than block it while a previous run is in flight.

## `src/core/`

`src/core/` mirrors mjlab's layout (`observation/`, `termination/`, `action/`, `event/`, `command/`) plus:

- `onnx/`: `session.ts` (caches split by lifetime), `runQueue.ts`, `graphRefs.ts`, `slotReader/` serving graph inputs from `mjModel` / `mjData` (`indexing.ts` resolves an entity's elements as mjlab's `EntityIndexing` does, `fields/` holds every `EntityData` property as a reader, one file per section of mjlab's `entity/data.py`, `model.ts` resolves an `mjModel` field's elements by name, and `index.ts` dispatches a slot by namespace), `raycast.ts` casting height-scan rays with `mj_ray`, `contact.ts`, and `camera.ts` serving a camera sensor's image, which `scene/sensorCamera.ts` renders with three.js on the layers `scene/layers.ts` assigns per geom group and `scene/cameraFeeds.ts` keeps for mjlab's Camera Feeds (each image for the host's canvases, and viser's frustum in the scene). Batch-1 tensor shapes stay inside `onnx/`; the managers deal in numbers and short arrays.
- `rng.ts` (xoshiro128\*\*, snapshot-able): the one PRNG behind every `rand` input, never `Math.random()`.
- `policy/`, `scene/`, `xr/` (`handMocap.ts` injects a capsule per hand bone and writes `mocap_pos` / `mocap_quat`; `session.ts` builds each session's options and decides whether a scene can take the hands), and `engine/` (`runtime.ts` step loop and episode step counter, `controlStep.ts`, `yieldToBrowser.ts`, `resetChain.ts`, `viewer_config.ts`).

Each manager is native orchestration around ONNX term bodies:

- `FusedObservation` / `OnnxObservation` / `NativeObservation` / `HistoryObservation`;
- `FusedTermination` / `OnnxTermination` (awaited every step) and `TimeOutTermination` (a pre-format-4 document's native `time_out`);
- `OnnxEvent` + `triggers.ts` + `entityWrite.ts` + `modelWrite.ts` (`ModelFieldDefaults`, one per model; and `modelFieldDr.ts`, a pre-format-5 document's model-field descriptor);
- `OnnxCommand`;
- Action splits as mjlab's does: `ActionManager` runs each term's `process` once per control step (`applyAction.ts`'s native formulas, or a traced term's `OnnxActionProcess`, awaited) and its reset graphs in the reset chain; `applyAction` runs every substep, synchronous, with the ramp, hold, relative target, `commandDelay.ts` and the PD a motor needs. It contains no `await` and no ONNX call ([tracing.md](tracing.md)).

The panel's **Debug Viz** section is mjlab's folder (`core/debugViz/`): **Enabled** (`engine.debugVis.setEnabled`, the URL's `debug`) over one switch per drawing (`engine.debugVis.set`), all on by default as in mjlab's viewers; a drawing shows only while both are on. Command drawings come from each binding's `viz` ([mdp.md](mdp.md)).

## Graphics

`core/scene/graphics.ts` draws a scene as its `graphics` say (`graphicsSettings.ts` holds the settings free of three.js, for the manifest parser): the renderer's tone mapping, exposure and encoding (`applyToRenderer`; MuJoCo's `filmic` is three's custom operator), each geom's `rgba` decoded or as written (`setBaseColor`; the viewer's own fixed colours the same way, in the shader, through `colorAsRgba`), and the lights' shadow switches.

Its `SceneGraphics`, one per model, holds what MuJoCo's classic renderer adds: `mirror.ts` (the reflection, drawn into a texture before the frame and added in the surface's material), `haze.ts`, and the infinite planes of `plane.ts`, wider than the far plane, textured by `GL_OBJECT_PLANE` and moved under the viewer a texture period at a time. Under classic a cube or skybox texture is drawn as stored, as MuJoCo uploads it; under Filament an sRGB sky is decoded on the CPU (`linearCube`), since three tone maps no background it decodes in hardware.

`asSensor` draws a camera sensor's pass as mujoco_warp reads the model: `rgba` as written, sRGB textures raw (a second upload), no reflection or haze, shadows by `use_shadows`, and the whole model, what the viewer hides included (`SeeThrough.asOwn` puts the robot back). `lights.ts` gives shadow maps MuJoCo's frustum, and `scene.ts` gives the model back the `stat.extent` the viewer's parked bodies stretch. The panel's **Graphics** section (`ControlPanel/GraphicsSection.tsx`) drives `engine.graphics` and copies the change as a `set_viewer` call; a host's override lasts until the next `loadScene`.

## Interaction, pick and preview

Two directories under `src/core/` are viewer-only, outside any MDP, and neither reaches Python, the manifest or the `.swn` document:

- `interaction/`: the pointer's view, pull, push and grab modes behind one `InteractionManager`, the only writer of `xfrc_applied`, and `pick.ts`, the `mj_ray` behind `engine.pick`.
- `grab/`: `weldHold.ts`, the weld hold the pointer and the tracked hands share, each with its own trigger.

**`engine.pick(clientX, clientY)`** gives a host what lies under a point of the canvas, in client CSS pixels as a `PointerEvent` gives them, so the host need not copy the engine's camera projection. It casts **MuJoCo's** ray, not three.js's: it answers as the simulation would, for an infinite plane (drawn as a clip-space quad a three.js ray cannot meet) and a height field alike, and it sees nothing that is only drawn (the mode gizmos, the preview, a skin, a splat). It casts against the groups the viewer draws (0 to 2), so it never lands on a collision-only geom no one sees, and unlike the pointer modes it counts the world body's geoms, since the floor is where most things are put. A hit carries `point` and `normal` (MuJoCo world axes, the normal turned to face the camera), `geom`, `body`, and the geom itself: `name`, `type`, `size`, `rgba` as drawn, `pos` and `quat`.

**`engine.preview`** draws one primitive geom that is not in the model and is never simulated: `show({ type, size, pos, quat?, rgba?, handles?, replaces? })` with `type` one of `plane`, `sphere`, `capsule`, `ellipsoid`, `cylinder`, `box` and MuJoCo's `size` for it. It lives in `scene/preview.ts` and is built by the same code as the model's geoms (`scene/primitive.ts`), so what a host previews is what it would add. It is parented to the three.js scene, as the mode gizmos are: a scene load keeps it, `camera.frame()` does not frame it, the pointer modes skip it, and it casts a shadow. `show` again moves and reshapes it, rebuilding geometry only for a new type or size; `hide()` removes it.

- **Handles** (`scene/previewHandles.ts`, the math in MuJoCo's axes): `'size'` puts an arrow on each spatial size entry along the geom's own axes; dragging moves the face under it and the opposite face stays, and where the entry sets the height too (a sphere's or capsule's radius) the bottom stays, so a geom resting on something keeps resting there; a half-size stops at 5 mm. `'turn'` puts a ring about each of the geom's own axes; Shift snaps to 15 degrees. `'move'` puts an arrow along each *world* axis, since the position it changes is written in them. Handles are drawn a fixed size on screen and over everything; a press on a handle is the preview's (taken on the window's capture phase, before the pointer modes and the camera), any other press is theirs, so the camera still orbits while a geom is on show.
- `preview.onChange(listener)` reports the preview as a drag changes it, and once more with `done` on release. The engine changes only the preview: what to write to the model, and when, is the host's, so a geom is not simulated (and the model not rebuilt) while it is being sized.
- `preview.hit(x, y)` says whether the preview or a handle is under a point, since `pick` does not see the preview. `replaces: geom` hides the model's meshes for that geom while the preview stands in for it; the id belongs to the loaded model, so a scene load ends the stand-in.
- Hosts feature-detect by version: `pick` and `preview` arrive in 0.12.4; `hit`, `onChange`, `handles`, `replaces` and a hit's geom fields in 0.12.5; `handles: 'move'` in 0.12.6 (an older engine draws rings for it).
- Not done: a preview of a mesh or of several geoms, moving it by dragging its body, resizing or turning along the world's axes, any of this in XR.

## Builds

Multi-threaded mode (`Builder(mt=True)`) requires COOP/COEP headers; the builder writes a `_headers` file (Netlify / Cloudflare Pages / Vercel) and a service-worker script (required for GitHub Pages).

The template has three Vite build outputs, all written to `template/dist/`; `npm run build` runs all of them.

- **SPA** (`vite.config.ts`, `npm run build:spa`): the standalone app the Python `Builder` assembles. Entry `src/index.tsx`. Without `MJSWAN_MT=1`, `mujoco/mt` resolves to `src/engine/mujocoMtStub.ts`, so it ships one MuJoCo build. It writes `dist/THIRD_PARTY_LICENSES.md`, the license of every package in its bundle: the package's own `LICENSE`, or `licenses/<package>/LICENSE` for one that publishes none, and the build fails when neither exists. The library build bundles a subset of these packages and writes no list of its own, since it runs second into the same `dist/`.
- **Library** (`vite.lib.config.ts`, `npm run build:lib`): a single self-contained ESM `dist/mjswan.js` exposing `createEngine(element, options?)` (entry `src/engine/index.ts`), consumed by mjswan Cloud from a CDN. No React or Mantine. Every dependency is bundled (no bare imports) and the MuJoCo / ONNX WASM is emitted into `dist/assets/`, referenced via `new URL('./assets/x.wasm', import.meta.url)`. Vite lib mode force-inlines those WASM as base64; a `generateBundle` plugin extracts them back out, recognizing each by content digest so it keeps its upstream basename, as mjswan Cloud's loader expects. The SPA build names files the same way into the same directory, so identical bytes land on one path instead of shipping 40 MiB twice; only `mjswan.js` sits at the root, addressed from outside as the npm entry. `src/core/onnx/ortEnv.ts` points `ort.env.wasm.wasmPaths` at ORT's copy, so a host serving `dist/` fetches nothing third-party. `vite.wasm.ts` carries the sources and the ORT assumptions this rests on.
- **Manifest** (`vite.manifest.config.ts`): `dist/manifest.js`, the `mjswan/manifest` catalog parser as a standalone CDN-loadable ESM.

One npm package, `mjswan`, with subpath exports: `mjswan` (the engine), `mjswan/manifest` (the parser), and the TypeScript base classes below. No monorepo split.

## Plugins: custom TypeScript terms

A term that cannot be traced can be written as a TypeScript class and named by a `*Binding`'s `ts_src` ([mdp.md](mdp.md)). That code is author code, and the rules around it are about where it may run.

**Delivery: runtime plugins, never a rebuilt engine.** `build/frontend.py` bundles the author's terms with esbuild into a standalone ESM, `assets/plugins.js` beside the engine, exporting constructors grouped by kind: the `EnginePlugins` shape (`observations`, `terminations`, `events`, `commands`). The app imports it before the first scene and hands it to the engine with the content (`SceneInput.plugins` for scene-scoped events, `PolicyInput.plugins` for the rest), which registers them instance-scoped. The engine bundle is never rebuilt, so the SPA stays project-independent and cacheable, and only a custom-TS build needs Node for this step. The plugin module's `import 'three'` resolves to `src/engine/plugin-three-shim.cjs`, which reads the engine's own three.js instance (`globalThis.__mjswanThree`, set in `src/engine/index.ts`), so `instanceof` and shared scene objects work across the two modules; the engine module must load first. The module ships with the engine, not in the document, which is why a custom-TS `.swn` cannot be served on its own ([document.md](document.md)).

**Imports: subpath exports to TypeScript source.** Author terms import base classes as `mjswan/observation`, `mjswan/event`, `mjswan/termination`, `mjswan/command`, `mjswan/math`, `mjswan/scene`, … (`package.json` `exports`, listed in `template/README.md`). These point at `.ts` source under `src/core/`, not compiled `.d.ts`, so:

- `src/core` must stay in the npm `files` array, or the subpath targets are missing from the published package;
- a consumer needs a bundler that handles TypeScript (Vite and esbuild do; plain `tsc` does not), and gets full IntelliSense from an ordinary `npm install -D mjswan`;
- a new public base class needs a subpath in `exports` and its file under `src/core/`, both in the template.

This is an npm self-reference: author code names the package it extends, never a path into it. An author outside the repository resolves the names through the installed package's `exports`; the plugin build resolves the same names by reading `exports` and passing each subpath to esbuild as an alias, so the map is never duplicated in Python.

Rejected: relative imports into the template (they resolve only inside it and break IntelliSense for authors outside the repository); a separate `@mjswan/types` package (a third package to keep in step with PyPI and npm releases, for nothing the self-reference does not give); a hand-maintained alias table in the Python build (it would drift from `exports`).

**Security: effects, not code.** The danger in author code is its *capabilities* (network, DOM, the host page's origin authority), not that it is code. Pure computation is safe however much of it there is; resource abuse (a loop that never ends) is a separate, budget-shaped problem, and correctness (NaNs) is not a security problem at all. A traced ONNX graph is data run by a fixed runtime, so it falls on the effect-free side. Arbitrary TypeScript does not, and whether a given script is effect-free is undecidable, so:

- Custom TS is a **permanent, trusted-only** capability, kept because it lets mjswan follow new research before mjswan can trace it. Local `mjswan view` / `serve`, mjlab use and the in-repo app load and register it.
- The manifest's top-level `uses_custom_js` lets a consumer enforce a no-author-code policy with one read. **mjswan Cloud rejects such builds** (`publish` refuses them locally first, [cloud.md](cloud.md)): it renders untrusted documents in one shared engine, where author code would run with the platform's origin authority.
- Rejected: sandboxing author code on Cloud as CodePen does. The threat is client-side (XSS-class, in a viewer's browser, never server-side), and CodePen's answer brings a permanent isolation burden (a separate origin, iframe sandbox, strict CSP, resource limits, ongoing vigilance) for something incidental to rendering a trained simulation. Also rejected: a QuickJS-in-WASM interpreter, a heavier trusted surface and slower per step. If Cloud ever needs arbitrary terms without an mjswan change, the structural route is author code compiled to a WASM module instantiated with an empty import object (it cannot name an effect, so one that needs one fails to instantiate) under a fuel and memory budget; nothing like it is built.
