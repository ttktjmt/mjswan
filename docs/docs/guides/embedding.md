---
icon: octicons/screen-full-16
---

# Embedding

A built mjswan app is a static site, so embedding it is mostly a question of how much
control you want over the surrounding page.

<div class="grid cards" markdown>

-   :material-application-outline:{ .lg .middle } __iframe__

    ---

    One tag, zero build tooling. The app brings its own UI. Best for articles, blogs, and
    docs pages.

-   :simple-googlecolab:{ .lg .middle } __Colab / Jupyter__

    ---

    `app.launch()` detects Colab and renders an inline iframe instead of blocking.

-   :simple-javascript:{ .lg .middle } __npm engine__

    ---

    `createEngine` in your own page, with your own controls. No mjswan UI at all.

</div>

## iframe

After hosting your app ([Deployment](deployment.md)) or publishing it
([mjswan Cloud](publishing.md)), drop it into any page:

```html
<iframe
  src="https://mjswan-gentlehumanoid.pages.dev?panel=0"
  title="mjswan simulation"
  loading="lazy"
  allow="xr-spatial-tracking; fullscreen"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0; border-radius: 0.4rem;">
</iframe>
```

That exact markup produces this — a real motion-tracking policy running in your browser,
inside this page:

<iframe
  src="https://mjswan-gentlehumanoid.pages.dev?panel=0"
  title="mjswan simulation"
  loading="lazy"
  allow="xr-spatial-tracking; fullscreen"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0; border-radius: 0.4rem;">
</iframe>

/// caption
The [GentleHumanoid](https://mjswan-gentlehumanoid.pages.dev/){:target="_blank"} demo embedded in this page. Try interacting with it.
///

`loading="lazy"` matters more here than for a normal iframe: the app downloads a MuJoCo
WASM module and starts simulating as soon as it loads, so deferring that until it scrolls
into view keeps the host page cheap. `allow="xr-spatial-tracking"` is what lets the
embedded viewer enter VR.

!!! warning "If the frame comes up blank, check `frame-ancestors`"
    Framing is controlled by the **embedded** site, not the host page. If its server sends
    a `Content-Security-Policy: frame-ancestors …` (or a legacy `X-Frame-Options`) that
    does not list your origin, the browser blocks the load and logs:

    ```
    Framing 'https://example.com/' violates the following Content Security Policy
    directive: "frame-ancestors 'self' …". The request has been blocked.
    ```

    mjswan itself never emits that header — a plain `builder.build()` output is frameable
    from anywhere, and so is GitHub Pages. It comes from the host: a Cloudflare Pages
    `_headers` file, an nginx `add_header`, or a dashboard setting. Widen the directive to
    include the origin you are embedding from, or host a copy you control.

### Query parameters

The bundled app reads query parameters, so an embed can open on exactly the scene, policy
and chrome you mean rather than the first thing in the build:

| Parameter | Effect |
|---|---|
| `project` | Select a project by display name or slug. |
| `scene` | Select a scene by display name or slug. |
| `policy` | Select a policy by display name or slug. Defaults to the one marked `default=True`. |
| `panel` | `panel=0` starts with the control panel hidden. Useful in a small iframe. |
| `ref` | `ref=0` starts with the motion-tracking reference ghost hidden. |
| `config` | Load a `config.json` from another URL entirely, relative to the page. |
| `hands` | `hands=1` puts WebXR-tracked hands in the simulation, see below. |

`panel` and `ref` are booleans read as "off only when exactly `0`", and the app writes the
current state back into the URL — so you can arrange a view by hand and copy the address
bar into your `src`.

```html
<iframe src="https://ttktjmt.github.io/mjswan/?scene=G1&policy=Locomotion&panel=0" …></iframe>
```

### Moving around in VR

**Enter VR** stands you in the scene, and the controller sticks take you from there: the
left stick slides you along the way you are looking, and the right stick turns you for as
long as you hold it, as fast as you push it. Turns pivot on your head rather
than on the play area's centre, so turning does not swing you sideways through the scene.
Smooth turning is easier to aim with than snapping to fixed steps and harder on the
stomach, which is why the rate is a modest 90° a second at full deflection.

You arrive where the scene's viewer config points its desktop camera, which is beside the
tracked body by default — not at the world origin, which on a generated terrain is the
generator's base plane rather than a place to stand. From there the view is yours: it does
not follow the body around, so a robot that walks off walks off and you go after it with
the stick. Your floor comes from a ray cast straight down at your feet, so you stand on
the ground at your own height rather than on the headset's idea of z = 0, which is what
keeps you on top of a slope instead of inside it.

### Hand tracking in VR

`hands=1`, or `createEngine(element, { handTracking: true })` when you drive the engine
yourself, makes a headset's tracked hands part of the physics rather than a pair of
floating models. On **Enter VR**, a capsule per bone follows the WebXR joint poses inside
the simulation, so a Quest 3 can bat a scene's objects around, rest one on an open palm,
and pick one up.

A bone enters the model according to what it has to do. The palm's two edges and the five
fingertips carry load, so each is a mocap body welded to a dynamic twin: MuJoCo takes
contact velocity from body velocity, and a body that is teleported every step has none, so
a bare mocap bone can push something but never hold it. The ten bones in between only ever
push, so they stay plain mocap — no degrees of freedom, and almost no cost. Grabs are read
from the contacts themselves: any object two of the hand's own geoms are squeezing from
opposing sides is held, at any thickness and with any number of fingers.

It is opt-in because both halves cost something: every scene the build loads gains the
hand bodies, at about 1.6x per physics step, and the headset must grant the
`hand-tracking` feature — **Meta Quest** is the recommended one, since its browser ships
WebXR hand tracking with nothing to install. Ordinary desktop and mobile viewing is
unaffected: untracked hands sit parked far above the scene.

### Passthrough AR

A headset that supports `immersive-ar` gets a second button, **Start AR**, next to
**Enter VR**. The simulation is the same one — same physics, same policy — but the skybox
and the ground planes stop being drawn, so the room shows through and the robot stands on
your own floor rather than on MuJoCo's checkerboard. The session asks for the
`local-floor` reference space, which is what puts the model's `z = 0` at floor level
instead of at eye level; the sticks move you as they do in VR, so you can walk the scene
to where you want it. Hand tracking works the same here, under the same `hands=1`.

What is not there yet: nothing occludes the scene, so a robot behind your real sofa is
drawn in front of it, and no shadow falls on your floor.

## Google Colab

`app.launch()` detects Colab and renders an inline iframe in the output cell instead of
starting a blocking server, so the notebook version of a build script is the same script:

```python
import mujoco
import mjswan

builder = mjswan.Builder()
project = builder.add_project(name="Demo")
project.add_scene(spec=mujoco.MjSpec.from_file("model.xml"), name="Scene")

app = builder.build()
app.launch(height=600)  # inline iframe in Colab; local server elsewhere
```

If you need to control the server yourself — a different port, or serving a `dist/` you
built earlier — the headers are the only subtle part:

```python
import http.server
import socketserver
import threading

from google.colab import output

PORT = 8000
DIRECTORY = "dist"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Required for SharedArrayBuffer (multi-threaded MuJoCo WASM)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


threading.Thread(
    target=lambda: socketserver.TCPServer(("", PORT), Handler).serve_forever(),
    daemon=True,
).start()

output.serve_kernel_port_as_iframe(PORT, height=600)
```

➔ Full notebooks:
[examples/colab/demo.ipynb](https://github.com/ttktjmt/mjswan/blob/main/examples/colab/demo.ipynb){:target="_blank"}
and
[anymal_c_velocity.ipynb](https://github.com/ttktjmt/mjswan/blob/main/examples/colab/anymal_c_velocity.ipynb){:target="_blank"}.

## npm engine

When you want your own UI — your own play button, your own sliders, your own layout — skip
the iframe and drive the engine directly. It renders into any element and never fetches
anything itself, so you decide where the bytes come from:

```ts
import { createEngine } from 'mjswan';
import { parseManifest } from 'mjswan/manifest';

const base = 'https://example.com/myapp/';
const bytes = (path: string) => async () =>
  (await fetch(new URL(path, base))).arrayBuffer();

const engine = await createEngine(document.getElementById('viewer')!);
const catalog = parseManifest(
  await (await fetch(new URL('assets/config.json', base))).text(),
  bytes,
);

const scene = catalog.projects[0].scenes[0];
await engine.loadScene(await scene.buildScene());

// Your UI, driven off the engine's snapshot.
engine.subscribe((state) => renderControls(state.commands, state.commandValues));
engine.commands.set('velocity:lin_vel_x', 0.8);
```

The library build is a single self-contained ESM, so it also works straight from a CDN
without a bundler:

```js
const { createEngine } = await import(
  'https://cdn.jsdelivr.net/npm/mjswan/dist/mjswan.js'
);
```

See the [Engine API](../api/engine.md) for the full surface.
