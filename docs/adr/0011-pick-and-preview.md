# A pick and a preview for hosts

> Status: **Accepted**. Adds `engine.pick` and `engine.preview` to the headless engine's API
> (ADR 0004): both in 0.12.4, and in 0.12.5 the preview's handles, `onChange`, `hit` and
> `replaces`, with the geom `pick` hit described. mjswan Cloud's Studio, which places and
> edits primitive geoms by pointing at the scene, is the first host.

## Context

The headless engine gives a host a camera pose (`camera.get`) and nothing else about what
is on screen: no scene, no raycaster, no picking. Cloud's Studio puts a box, a sphere or a
capsule where the pointer is, so it re-implements the camera's projection, draws an HTML
outline over the canvas as a stand-in for the geom, and intersects the pointer's ray with
the plane z = 0 on its own. The outline is not the geom (it only approximates a capsule
or a box seen at an angle), it ignores whatever stands on the floor, and the copied math
has to follow the engine's camera conventions from release to release.

Once a geom is placed, Studio edits it with HTML handles positioned by the same copied
projection, writing the model on every move of a handle. So a dropped geom falls and
collides while its author is still sizing it, and every move rebuilds the model. Its
author wants to size and turn a geom before it joins the simulation, to keep orbiting the
camera while placing one, and to click any geom of the scene to edit it.

The engine has what Studio rebuilds: MuJoCo casts rays against the model (`mj_ray`, which
the height scan and the XR grounding already use), `core/scene/scene.ts` builds a mesh for
every geom type, and the engine owns the camera, so handles drawn there need no
reprojection by anyone.

## Decision

### 1. `engine.pick(clientX, clientY)`: what lies under a point of the canvas

Client CSS pixels, as a `PointerEvent` gives them, so a host passes its event's numbers as
they are. The engine casts `mj_ray` from the camera through that point and returns the
first geom it meets, or null:

- `point` and `normal`, in MuJoCo world axes (x forward, y left, z up), as `CameraView`
  is. The normal is the surface's, turned to face the camera.
- `geom` and `body`, the ids of what was hit.
- The geom itself (0.12.5): its `name` (`''` when it has none), its `type` (MuJoCo's,
  `mesh`, `hfield` and `sdf` included), its `size`, its `rgba` as the scene draws it (its
  material's colour when it has one), and where it is now, `pos` and `quat` (w, x, y, z)
  from `geom_xpos` and `geom_xmat`. A host previews a geom of the model with these, as it
  found it.

MuJoCo's ray, not three.js's: it answers as the simulation would, for an infinite plane
(drawn as a quad in clip space, which a three.js ray cannot meet) and a height field
alike, and it sees nothing that is only drawn (the mode gizmos, the preview, a skin, a
splat). It casts against the groups the viewer draws (0 to 2), so it does not land on a
collision-only geom no one sees. Unlike the pointer modes it counts the world body's geoms,
since the floor is where most things are put.

### 2. `engine.preview`: one primitive geom, drawn and never simulated

`preview.show({ type, size, pos, quat?, rgba? })` draws a geom that is not in the model:

- `type` is one of `plane`, `sphere`, `capsule`, `ellipsoid`, `cylinder` and `box`, and
  `size` is MuJoCo's for that type. A plane is a finite one.
- `pos`, and `quat` (w, x, y, z), are in MuJoCo world axes. The default `quat` is the
  identity.
- `rgba` defaults to MuJoCo's geom grey.

It is built by the same geometry and material code as the model's geoms
(`core/scene/primitive.ts`, split out of `scene.ts` for both), so what a host previews is
what it would add. `show` again moves and reshapes the one on show, rebuilding its
geometry only for a new type or size; `preview.hide()` removes it.

It is parented to the three.js scene, as the mode gizmos are: a scene load keeps it,
`camera.frame()` does not frame it, and the pointer modes skip it. It casts a shadow,
which says where it would rest. It is live: the model is never touched.

### 3. Handles on the preview: arrows that resize it, rings that turn it

From 0.12.5, `show` also takes `handles`:

- **`'size'`:** an arrow on each spatial `size` entry, along the geom's own axes (x red,
  y green, z blue). A box gets its three extents, a radius goes along x and a half-length
  along z, and a plane gets its two extents.
  - Dragging an arrow moves the face under it with the pointer, and the opposite face
    stays.
  - The bottom (local -z) stays too where the entry sets the height as well, as a
    sphere's or a capsule's radius does. So a geom resting on something keeps resting
    there.
  - A half-size stops at 5 mm.
- **`'turn'`:** a ring about each of the geom's own axes. Sweeping the pointer round the
  geom's centre turns it by the angle swept, about that axis and through the centre.
  Shift snaps to 15 degrees.

Handles are drawn a fixed size on screen and over everything else, as an editor's gizmo is,
and an arrow pointing at the eye (a dot) is hidden. A press on a handle is the handle's
until it is released: the preview takes it on the window's capture phase, before the
pointer modes and the camera see it. Any other press is theirs, as before, so the camera
still orbits while a geom is on show. The math is in MuJoCo's axes, in
`core/scene/previewHandles.ts`.

### 4. `preview.onChange`, `preview.hit` and `replaces`

`onChange(listener)` calls the listener as a drag changes the preview, with the preview as
it now is (`type`, `size`, `pos`, `quat`, `rgba`), and once more with `done` when the
pointer lets go. It returns the unsubscribe. The engine changes only the preview: what to
write, and when, is the host's.

`hit(clientX, clientY)` says whether the preview or one of its handles is under a point of
the canvas. `pick` does not see the preview, so a host tells a click on the geom it is
editing from a click elsewhere with this.

`replaces: geom` hides the model's meshes for that geom while the preview shows, so the
preview is what is seen while it is edited. Showing without it, or hiding the preview,
shows them again. The id belongs to the model loaded, so a scene load ends the stand-in
(the new model's meshes are drawn as built); a host that reloads while editing shows the
preview again with the new id, or hides it. The back face of a finite plane carries its
geom id too, so it hides with the front.

### 5. Hosts feature-detect them

An engine before 0.12.4 has neither `pick` nor `preview`, so a host checks for `pick` and
keeps a fallback of its own (Studio keeps its outline). One before 0.12.5 has no `hit` or
`onChange`, ignores `handles` and `replaces`, and reports no geom fields in a `PickHit`.

## Not decided

- A preview of a mesh geom, or of more than one geom at a time: `type` stays a primitive,
  so a host edits a mesh's colour without one.
- Moving the preview by dragging it, and handles along the world's axes.
- Picking, previewing and handles in an XR session.

## Consequences

- Studio drops its ray and ground-plane math for placement, and can put a geom on top of
  another one. It drops its HTML handles and their camera math too, writes the model once
  an edit is done, and keeps the camera free while it places a geom.
- The engine owns one more object in its scene and two `mj_ray` output buffers, which it
  frees with itself.
- The preview listens to the pointer: one capture listener on the window, always, and three
  more for the length of a drag.
