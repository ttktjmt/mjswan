# Skins: a rigged character drawn over the robot

> Status: **Accepted**. Adds `skins` to a scene's manifest entry, which bumps the document
> `format` from 6 to 7 (ADR 0006 §7), and a leaf package, `rig/`, under the object model
> (ADR 0008 §1). mjswan Cloud's side is its ADR 0015.

## Context

A robot drawn as its own meshes is the honest picture of what is simulated, and the
wrong one for a demo. A skin draws a rigged character in the robot's place, posed from
the robot's bodies every frame, while physics, collision and the policy stay the
robot's. Rigged characters come as VRM files (humanoids with a declared skeleton,
license and spring bones) and as glTF files (anything with a skeleton, named however
its tool named it). A robot's joints are not a character's: its links are other
lengths, its zero pose is not the character's rest pose, and a four-legged robot has
no shoulders.

## Decision

### 1. Two formats, one file each, bundled as received

A skin is a `.vrm` or a `.glb`, both binary glTF, told apart by the VRM extension in
the JSON chunk rather than by name. The build copies the file byte for byte to
`assets/<skin-id>.vrm` or `.glb`, so a VRM's embedded license travels with it as its
author wrote it. `add_skin` refuses a `.gltf` (its buffers and images are files beside
it), a buffer or image that points outside the file, and the compression extensions
the engine has no decoder for (Draco, meshopt, KTX2/Basis), each with how to export
the file again. Reading a skin's file is `document/skin.py`: what a document may carry.

### 2. A skeleton named three ways

- **A VRM** names its own humanoid. The config maps VRM bone names to robot bodies and
  may not set what the file brings (its humanoid, springs, units, materials, license).
- **A `.glb` given `humanoid`** names its nodes as VRM bones, by a table or a preset
  (`"mixamo"`). A four-legged animal maps its front legs as arms.
- **A `.glb` without it** is keyed by node names, with `limbs` naming its arms and
  legs, for skeletons that are neither.

Every name is resolved to a node index at build time. The browser poses a VRM through
three-vrm's humanoid and every `.glb` through `NodeRig`, three-vrm's normalized rig
nested as the file nests its nodes, so a `.glb` humanoid need not follow VRM's nesting
(a pug's front legs hang from its body, not its shoulders). The topmost mapped node,
a humanoid's hips, carries the position. `reparent` hangs a node from another, its
rest pose kept, for a rig whose feet IK controls carry rather than its legs
(Quaternius's animals): posed from joint angles alone, they would stay behind. The
file is still bundled as received; the move is made where it is read, at build time
and in the browser.

### 3. Fitted to the robot at build time

- **`rest_pose`** names the robot's pose while the character holds its own rest pose:
  a keyframe, or hinge and slide joints. Each bone's rest offset is the turn that
  undoes its body's from the zero pose to that pose, relative to the root body. A VRM
  rests in a T-pose; a humanoid robot's arms hang, so its rest pose holds them out.
- **`root_height`** is the root's height above the robot's soles at that pose, and
  **`hips_offset`** centres the character's limb roots over the robot's across the
  floor. Both are measured, not written.
- **`retarget`** defaults to `"hybrid"`: each bone turns as its body has, keeping the
  character's proportions, then each limb is bent (two-bone IK) so its hand or foot
  lands where the robot's is. `"rotation"` only turns; `"position"` fixes each bone to
  its body and warns when the limbs are more than 10% off the robot's.

The browser receives indices and turns, and reads no names.

### 4. Drawn only, and checkable

Spring bones (hair, a tail) swing by three-vrm, or with `mujoco_springs` in a MuJoCo
model of their own; the scene's model sees neither. With the robot hidden
(`hide_robot`, the default), Debug Viz's **Xray** switch (`?xray=1`) draws it
see-through over the character, to check the fit. The skin itself is not a debug
drawing: Debug Viz's **Enabled** leaves it shown, and the Skin menu's None is how to
see the robot without it.

### 5. Published with its license checked

`mjswan publish` uploads skins as data and refuses what mjswan Cloud would: a file the
engine cannot draw, and a VRM whose license forbids redistribution, since hosting it is
redistributing it. A VRM only its author, or someone they licensed, may use as an avatar
is sent only once the publisher answers that they are; the commit carries the path in
`attested_skins`, and the page says it was published on their word. A VRM for
non-commercial use only publishes with a warning. A `.glb` carries its license as any
component does (ADR 0007), as `LICENSE.skin-<id>` and `NOTICE.skin-<id>`.

### 6. A draft of the maps

`mjswan skin map <file> --robot <xml | mjb | mjlab task>` drafts `humanoid` from the
skeleton's shape, `bones` from the robot's body names, and for a `.glb` its `scale` and
spring chains, and writes a `SkinConfig` for the author to check. It does not guess
`rest_pose`: a robot's link origins sit off the joints they look like, so a solved
T-pose comes out degrees wrong. It lists the keyframes and arm joints instead, and
writes the rest pose of the robots checked by hand.

### 7. Document format 7

A scene may carry `skins`. A format-6 engine would ignore them and show the bare robot,
so `DOCUMENT_FORMAT` is 7.

## Considered options

- **Convert every glTF to a VRM at build time.** The old example did. It rewrites the
  author's file, and a VRM's license fields would then be ours, not theirs.
- **Require VRM's nesting of a `.glb` humanoid.** three-vrm's rig assumes it; most
  animal rigs break it. `NodeRig` poses any nesting.
- **Write rest offsets by hand.** The examples did, in degrees, per bone. `rest_pose`
  reproduces those tables within 0.05° and reads as the pose it is.

## Consequences

- A scene's look is no longer its model's; the Xray switch is how to see the model.
- `rig/` holds what fits a skin to a robot (`humanoid`, `pose`, `guess`); `cloud/`
  reads skins through `document/skin.py` and `license/vrm.py` alone.
- An older engine refuses every format-7 document, skins or not, as for any bump.
