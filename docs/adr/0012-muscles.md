# Muscles drawn as MyoSuite draws them

> Status: **Accepted**. Adds `muscles` to a scene's manifest entry, which bumps the document
> `format` from 9 to 10 (ADR 0006 §7).

## Context

A MyoSuite model moves by muscle actuators acting through spatial tendons. mjswan drew
those tendons as every other tendon, thin and one colour, so a policy's muscle activity
did not show. MyoSuite's own renderer (`myosuite/viz/blender_render.py`) draws them two
ways: `--muscles paths`, the tendon paths coloured by activation, and
`--muscles volumetric`, a tube along each path whose belly keeps its volume as the path
shortens (`myosuite/viz/muscle_tubes.py`). A third, `--muscle-mesh atlas`, binds
anatomical muscle meshes to the bones. All of them are drawn only, never simulated.

## Decision

### 1. Built into the engine, drawn only

The muscles are drawn by `core/muscle/`, outside the model as a skin is (ADR 0010), from
the tendon paths MuJoCo leaves after each forward. They are not a render plugin: mjswan
Cloud runs no author code (ADR 0004 §10), and a primitive in the engine reaches its
viewers as data.

### 2. MyoSuite's selection, names and numbers

The muscles are those MyoSuite draws: muscle actuators on a spatial tendon that starts at
a site. A view takes the names of MyoSuite's `RenderConfig`: `style` (its `muscles`),
`color` (`muscle_color`) and `scale` (`muscle_scale`); `detail` is mjswan's own, for a
slower device. `core/muscle/tubes.ts` ports `muscle_tubes.py` and the math of
`_write_muscle_tubes` under MyoSuite's Apache-2.0 license, a tendon over pulleys getting
a tube per branch as MyoSuite #810 draws it. `tests/dump_muscle_fixture.py` records what
MyoSuite's own functions draw, and the commit they came from; `tubes.test.ts` holds the
port to it at 1e-9 m. A change in MyoSuite is taken by dumping again from that change.

### 3. The shape at keyframe 0

MyoSuite fixes each tube's volume at the first frame it records. The engine fixes it at
keyframe 0, or `qpos0` without one: a scene or policy may start anywhere, and a random
first pose changes a tube's thickness by up to 2.2 times. A standing keyframe and
`qpos0` differ by less than 2%.

### 4. The scene holds a default, the viewer changes it

A scene's `muscles` (`SceneHandle.set_muscles`) is only how it starts. The panel's
**Muscles** menu, `?muscles=` and `engine.muscles.set` change the view live, and nothing
is written back: `engine.muscles.reset()` and the next scene load return to the
scene's own.

### 5. Not now: the anatomical atlas

MyoSuite's atlas is a skinned glTF of 416 muscles, bound to bones by rest position.
It would come as another field of a scene's `muscles`, beside `style`, with its file
under `assets/`; nothing here assumes the tubes and paths are the only drawings.

## Consequences

- An older engine refuses every format-10 document, muscles or not, as for any bump,
  rather than drawing plain tendons where the author asked for tubes. mjswan Cloud's
  pinned engine must reach format 10 before it hosts one.
- Every muscle's tubes are one mesh, built once per drawn frame from the wrap points the
  last forward left: the loop forwards far more often than frames are drawn. MyoSuite's
  full body, 416 muscles, at 12 ring points is 160,576 vertices, about 2.4 ms a frame on
  an Apple silicon laptop.
- The port follows MyoSuite. Where mjswan wants another look, it is a field of the
  view, not a change to the port.
