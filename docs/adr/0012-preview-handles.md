# Handles on the preview, and a pick that names the geom

> Status: **Accepted**. Extends `engine.pick` and `engine.preview` (ADR 0011) so a host can
> edit a geom where it stands: resize it and turn it by dragging handles on the preview, and
> pick a geom of the model to edit. mjswan Cloud's Studio is the first host.

## Context

ADR 0011 gave a host a surface under the pointer and a geom drawn there. Studio places a
geom with them, and then edits it with HTML handles positioned by its own copy of the
camera's projection, writing the model on every move of a handle. So a dropped geom falls
and collides while its author is still sizing it, and every move rebuilds the model.

Studio's author wants to size and turn a geom before it joins the simulation, to keep
orbiting the camera while a geom is being placed, and to click any geom of the scene to
edit it. The engine draws the preview and owns the camera, so a drag on the preview's own
handles is cheaper here than in a host, which would have to reproject them every frame.

## Decision

### 1. `pick` says what it hit

Besides the point, the normal and the ids, a `PickHit` carries the geom's `name` (`''`
when it has none), its `type` (MuJoCo's, `mesh`, `hfield` and `sdf` included), its `size`,
its `rgba` as the scene draws it (its material's colour when it has one), and where it is
now: `pos`, and `quat` (w, x, y, z), from `geom_xpos` and `geom_xmat`. A host previews a
geom of the model with these, as it found it.

### 2. `preview.show({ ..., handles })`: arrows that resize it, rings that turn it

`handles: 'size'` puts an arrow on each spatial `size` entry, along the geom's own axes
(x red, y green, z blue): a box's three extents, a radius along x, a half-length along z, a
plane's two extents. Dragging one moves the face under it with the pointer. The opposite
face stays, and so does the bottom (local -z) where the entry sets the height too, as a
sphere's or a capsule's radius does, so a geom resting on something keeps resting there.
A half-size stops at 5 mm.

`handles: 'turn'` puts a ring about each of the geom's own axes. Sweeping the pointer round
the geom's centre turns it by the angle swept, about that axis and through the centre;
Shift snaps to 15 degrees.

Handles are drawn a fixed size on screen and over everything else, as an editor's gizmo is,
and an arrow pointing at the eye (a dot) is hidden. A press on a handle is the handle's
until it is released: the preview takes it on the window's capture phase, before the
pointer modes and the camera see it. Any other press is theirs, as before, so the camera
still orbits while a geom is on show.

The math is in MuJoCo's axes, in `core/scene/previewHandles.ts`.

### 3. `preview.onChange` and `preview.hit`

`onChange(listener)` calls the listener as a drag changes the preview, with the preview as
it now is (`type`, `size`, `pos`, `quat`, `rgba`), and once more with `done` when the
pointer lets go. It returns the unsubscribe. The engine changes only the preview: what to
write, and when, is the host's.

`hit(clientX, clientY)` says whether the preview or one of its handles is under a point of
the canvas. `pick` does not see the preview, so a host tells a click on the geom it is
editing from a click elsewhere with this.

### 4. `replaces`: standing in for a geom of the model

`replaces: geom` hides the model's meshes for that geom while the preview shows, so the
preview is what is seen while it is edited. Showing without it, or hiding the preview,
shows them again. The id belongs to the model loaded, so a scene load ends the stand-in
(the new model's meshes are drawn as built); a host that reloads while editing shows the
preview again with the new id, or hides it. The back face of a finite plane now carries
its geom id too, so it hides with the front.

## Not decided

- A preview of a mesh geom: `type` stays a primitive, so a host edits a mesh's colour
  without one.
- Moving the preview by dragging it, and handles along the world's axes.
- Handles in an XR session.

## Consequences

- Studio drops its HTML handles and their camera math, writes the model once an edit is
  done, and keeps the camera free while it places a geom.
- The preview listens to the pointer: one capture listener on the window, always, and three
  more for the length of a drag.
