# Skins

A skin is a rigged character drawn over the robot, posed from the robot's bodies every frame. It is **never part of the model**: physics, collision and the policy stay the robot's, and a hidden robot is still simulated, collides and can be grabbed. So a scene's look is not its model's; Debug Viz's Xray switch shows the model. Python side: `skin.py`, `rig/`, `document/skin.py`, `build/skin.py`; browser side: `core/skin/`.

## `skin.py`: `SkinConfig` / `SkinHandle`

One frozen `SkinConfig` is the whole skin: its file, the bone-to-body map, `rest_pose`, placement and, for a `.glb`, its scale, facing, springs, toon material and license. `add_skin` resolves it at once (`resolve_skin`): the file read and checked (`document/skin.py`), the bones found, and the rest offsets, `root_height` and hips measured against the scene's model (`rig/pose.py`), so the manifest entry (`build/skin.py`) holds node indices and turns only, and the browser reads no names. A scene carries any number; the Skin menu shows one at a time. The build copies each file unmodified to `assets/<skin-id>.vrm` or `.glb`.

### The file

A skin is a `.vrm` or a `.glb`, both binary glTF, told apart by the VRM extension in the JSON chunk rather than by name. It is bundled **byte for byte**, so a VRM's embedded license travels as its author wrote it. `add_skin` refuses a `.gltf` (its buffers and images are files beside it), a buffer or image pointing outside the file, and the compression extensions the engine has no decoder for (Draco, meshopt, KTX2 / Basis), each with how to export the file again.

### The skeleton, named three ways

- **A VRM** names its own humanoid. The config maps VRM bone names to robot bodies and may not set what the file brings (its humanoid, springs, units, materials, license).
- **A `.glb` given `humanoid`** names its nodes as VRM bones, by a table or a preset (`"mixamo"`). A four-legged animal maps its front legs as arms.
- **A `.glb` without it** is keyed by node names, with `limbs` naming its arms and legs, for skeletons that are neither.

Every name is resolved to a node index at build time. The browser poses a VRM through three-vrm's humanoid and every `.glb` through `NodeRig` (three-vrm's normalized rig nested as the file nests its nodes), so a `.glb` humanoid need not follow VRM's nesting: a pug's front legs hang from its body, not its shoulders, and most animal rigs break VRM's assumption. The topmost mapped node, a humanoid's hips, carries the position. `reparent` hangs a node from another, its rest pose kept, for a rig whose feet IK controls carry rather than its legs (posed from joint angles alone they would stay behind); the file is still bundled as received, and the move is made where it is read.

### Fitted to the robot at build time

- **`rest_pose`** names the robot's pose while the character holds its own rest pose: a keyframe, or hinge and slide joints. Each bone's rest offset is the turn that undoes its body's from the zero pose to that pose, relative to the root body. A VRM rests in a T-pose; a humanoid robot's arms hang, so its rest pose holds them out. Rejected: hand-written per-bone offsets in degrees, since `rest_pose` reproduces such tables within 0.05° and reads as the pose it is.
- **`root_height`** (the root's height above the robot's soles at that pose) is measured; **`hips_offset`**, unless set, centres the character's limb roots over the robot's across the floor.
- **`retarget`** defaults to `"hybrid"`: each bone turns as its body has, keeping the character's proportions, then each limb is bent (two-bone IK) so its hand or foot lands where the robot's is. `"rotation"` only turns; `"position"` fixes each bone to its body and warns when the limbs are more than 10% off the robot's (`STRETCH_TOLERANCE`).

### Modelled skins

A `ModelledSkinConfig` is the other kind: a `.glb` modelled on the robot itself, at its rest pose and in its world frame, as MyoSuite's muscle atlas is. Nothing is fitted or measured: `MatchByPosition` (`rig/pose.py`'s `match_by_position`) matches each skin joint to the body whose origin or geom centre it rests on at `rest_pose`, first voting a translation when the scene stands the robot elsewhere (a turn it does not find: `rest_pose` then sets the free joint's qpos), and the entry carries `fit: "modelled"`, the joints' nodes and bodies, `align` and `rest_qpos` (document format 9). `SolidShading` draws a file with no materials or normals in one colour, and `hide_tendons` (unset, as `hide_robot`) stops the model's tendons drawing while any skin is shown. `mjswan skin map --robot` drafts a `ModelledSkinConfig` for a file most of whose joints rest on the robot.

### Drafts: `mjswan skin map`

`mjswan skin map <file> --robot <xml | mjb | mjlab task>` (`rig/guess.py`) drafts `humanoid` from the skeleton's shape, `bones` from the robot's body names, and for a `.glb` its `scale` and spring chains, and writes a `SkinConfig` for the author to check; it is never used by a build. It does **not** guess `rest_pose`: a robot's link origins sit off the joints they look like, so a solved T-pose comes out degrees wrong. It lists the keyframes and arm joints instead, and writes the rest pose of the robots checked by hand (`KNOWN_REST_POSES`).

### Publishing

`mjswan publish` uploads skins as data and refuses what mjswan Cloud would ([cloud.md](cloud.md)): a file the engine cannot draw, and a VRM whose license forbids redistribution. A `.glb` carries its license as any component does, as `LICENSE.skin-<id>` and `NOTICE.skin-<id>` ([licenses.md](licenses.md)), and a scene's `credits` summarise each VRM skin's `meta`. Rejected: converting every glTF to a VRM at build time, which rewrites the author's file and makes a VRM's license fields mjswan's, not theirs.

## `core/skin/`: the browser side

Stays outside the scene's model too, and is imported only when a scene has a skin, so it and three-vrm are one lazy chunk a scene without one never fetches.

- `config.ts` holds the manifest entry's types (so the manifest and engine types bring no skin code).
- `skinModel.ts` loads the file: a VRM through `@pixiv/three-vrm`, a `.glb` through `GLTFLoader` (both by `parseSealedGltf`, which refuses a file naming a buffer or image outside itself), scaled to metres and given three-vrm springs and MToon materials from its entry. Its `paintable` switches a skin between drawn as painted (classic, which does not encode its output: sRGB textures read raw, colours encoded) and the file's own colour management (Filament).
- `rig.ts` is the `SkinRig` both are posed through: three-vrm's humanoid for a VRM, `NodeRig` for a `.glb`.
- `retarget.ts` turns each mapped body's rotation since the robot's zero pose into a normalized-bone rotation, places the root above the ground, and holds the two-bone IK (`reachLimb`, `dropToReach`).
- `skin.ts` poses the model in `render()` from the body transforms just drawn, finding the ground under the robot with `mj_ray`: by rotation, by fixing each bone to its body (`retarget: 'position'`), or by default (`'hybrid'`) by rotation, each limb then bent onto where `'position'` puts its end. It divides MToon colours by the scene's summed light (`lightBudget`): a toon shader takes every light it faces in full, so MuJoCo's several lights would clip it to white.
- Spring bones (hair, a tail) are three-vrm's; with `mujoco_springs: true`, `springs.ts` simulates them in a MuJoCo model of their own, its anchor bodies driven onto the posed nodes so the chains feel the motion as inertia. The scene's model sees neither.
- A modelled skin (`fit: 'modelled'`) skips all of that: `bind()` fixes each joint to its body at `body_rest⁻¹ · T(align) · bind`, the bind pose from the skin's inverse bind matrices (`SkinModel.binds`), and leaves undrawn a mesh moved by a joint on no body; a solid material is one `MeshPhysicalMaterial`, normals computed where the file has none, and its drawn meshes are merged into one (`mergeGeometries`).

The runtime holds one skin at a time (`engine.setSkin` swaps it live, the model untouched), hides the model's tendons while one with `hide_tendons` is shown, and, while a skin hides the robot, lists one viewer switch under Debug Viz (`DebugViz.addSwitch`): `xray` (`?xray=1`), which draws the hidden robot see-through over the skin (`scene/seeThrough.ts`), to check the fit. The skin itself is not a debug drawing: Debug Viz's **Enabled** leaves it shown, and the Skin menu's None is how to see the robot without it.
