# mjswan web engine

[![npm version](https://img.shields.io/npm/v/mjswan.svg?logo=nodedotjs)](https://www.npmjs.com/package/mjswan)
[![docs](https://img.shields.io/readthedocs/mjswan?logo=readthedocs)](https://mjswan.readthedocs.io)

Browser-side runtime for [mjswan](https://github.com/ttktjmt/mjswan). Interactive MuJoCo
simulations with real-time policy control, running entirely in the browser via WebAssembly.

The package runs MuJoCo physics ([mujoco-wasm](https://github.com/google-deepmind/mujoco/tree/main/javascript)),
renders with [three.js](https://github.com/mrdoob/three.js), and executes policies with
[ONNX Runtime Web](https://github.com/microsoft/onnxruntime).

> **Most users want the Python package.** mjswan is primarily authored in Python
> (`pip install mjswan`), which bundles your models, policies, and UI into a static
> site. This npm package is the browser side, and is useful directly in two cases:
> **embedding** a published simulation in your own page, and **authoring custom MDP
> terms** in TypeScript with full type support. See the
> [documentation](https://mjswan.readthedocs.io) for the full Python workflow.

## Installation

```bash
npm install mjswan
```

Requires Node.js 24+ and a bundler that handles TypeScript sources (Vite recommended.
See [Custom MDP terms](#custom-mdp-terms) below for why).

## Embedding a simulation

The package ships a self-contained library build (`dist/mjswan.js`) that renders an mjswan
simulation into any element. It bundles every dependency and co-locates its WASM, runs
single-threaded by default (no COOP/COEP headers needed), and works cross-origin — so it
can be loaded straight from a CDN. Two layers, deliberately separate: `createEngine` is
**bytes in, snapshot out** and never fetches; `mjswan/manifest` turns a build's
`manifest.json` — the simulation document's one descriptor — into a lazy catalog of
loadable things, and the page owns the fetching.

```js
const { createEngine } = await import('https://cdn.jsdelivr.net/npm/mjswan/dist/mjswan.js');
const { parseManifest } = await import('https://cdn.jsdelivr.net/npm/mjswan/dist/manifest.js');

// Every other asset (scene.mjz, mdp/…/*.onnx, policy/*.onnx, assets/*.npz) resolves
// against the manifest's directory.
const base = 'https://example.com/myapp/';
const bytes = (path) => async () => (await fetch(new URL(path, base))).arrayBuffer();

const engine = await createEngine(container);
const catalog = parseManifest(await (await fetch(new URL('manifest.json', base))).text(), bytes);
await engine.loadScene(await catalog.projects[0].scenes[0].buildScene());
```

Or as a normal bundled import:

```ts
import { createEngine } from 'mjswan';
import { parseManifest } from 'mjswan/manifest';
```

The catalog's `buildScene({ policy, splat })` takes ids — `name2id` of the display names,
the same values the bundled app's `?scene=` / `?policy=` parameters use. A `.swn` document
(the same tree as a ZIP) renders the same way once unpacked into an in-memory resolver. See
the [Engine API](https://mjswan.readthedocs.io/en/latest/api/engine/) for `setPolicy`,
`camera`, `commands`, `subscribe`, `captureThumbnail` and `dispose`.

## Custom MDP terms

When authoring custom observations, commands, events, or terminations for a mjswan
simulation, import the base classes and helpers from subpath exports instead of fragile
relative paths. Install mjswan as a dev dependency and import directly:

```ts
import { ObservationBase } from 'mjswan/observation';
import { mjcToThreeCoordinate } from 'mjswan/coordinate';
import type { PolicyState } from 'mjswan/types';

export class MyObservation extends ObservationBase {
  get size(): number {
    return 3;
  }

  compute(state: PolicyState): Float32Array {
    // ... read MuJoCo state, return the observation vector
  }
}
```

The build step bundles your source into the engine. See
[How the Build Works](https://mjswan.readthedocs.io/en/latest/guides/how-it-works/#a-term-cannot-be-traced)
for when a term needs one, and how a binding's `ts_src` names the file.

> Subpath exports point at TypeScript source (not compiled `.d.ts`), so consumers need a
> bundler that handles TypeScript. Vite does; plain `tsc` does not. This keeps full IDE
> IntelliSense without a separate types package — see
> [ADR 0001](https://github.com/ttktjmt/mjswan/blob/main/docs/adr/0001-npm-self-reference-for-custom-mdp-imports.md).

### Available subpaths

| Import | Provides |
|---|---|
| `mjswan` | `createEngine`, `policyGraphRefs` (the runtime library build) |
| `mjswan/manifest` | `parseManifest`, `sanitizeName`, the `Manifest` types (the catalog parser) |
| `mjswan/observation` | `ObservationBase`, `ObservationConfig` |
| `mjswan/command` | `CommandManager`, command types and helpers |
| `mjswan/event` | `EventBase`, event config and context types |
| `mjswan/termination` | `TerminationBase`, termination config types |
| `mjswan/scene` | Scene helpers (`getPosition`, `getQuaternion`, …) |
| `mjswan/npz` | `.npz` loading (`loadNpz`, `NpzEntry`) |
| `mjswan/coordinate` | MuJoCo ↔ three.js coordinate conversions |
| `mjswan/math` | Quaternion / vector math utilities |
| `mjswan/types` | Shared types (`PolicyState`, `PolicyRunner`, …) |

## Links

- **Documentation**: [mjswan.readthedocs.io](https://mjswan.readthedocs.io)
- **Repository**: [github.com/ttktjmt/mjswan](https://github.com/ttktjmt/mjswan)
- **Python package**: [pypi.org/project/mjswan](https://pypi.org/project/mjswan)
- **Live demo**: [ttktjmt.github.io/mjswan](https://ttktjmt.github.io/mjswan)

## License

Apache-2.0
