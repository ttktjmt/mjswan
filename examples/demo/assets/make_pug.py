"""Make ``pug.glb`` from Quaternius's Farm Animals pack.

``pug.glb`` is the Pug of "Farm Animals Animated" by Quaternius (CC0 1.0,
https://quaternius.itch.io/lowpoly-animated-animals), made ready for ``go1_avatar.py``,
which turns it into a VRM. The pack has no stable download URL and only Blender reads
it, so the result is committed and this script is how it was made. With the pack
unzipped, run it under Blender's Python module:

    uv run --no-project --python 3.13 --with bpy==5.2.2 python \\
        examples/demo/assets/make_pug.py Blends/Pug.blend examples/demo/assets/pug.glb
"""

import sys

import bpy

#: Metres between the front and rear shoulders: the Go1's front and rear hips.
SPAN = 0.376


def main() -> None:
    source, out = sys.argv[1:3]
    bpy.ops.wm.open_mainfile(filepath=source, load_ui=False, use_scripts=False)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")

    # The legs are driven by IK, the feet being children of the root: posed from joint
    # angles alone, the feet would stay behind. Each foot goes under its lower leg,
    # which leaves the rest pose and the skinning as they are.
    for bone in rig.pose.bones:
        for constraint in list(bone.constraints):
            bone.constraints.remove(constraint)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    bones = rig.data.edit_bones
    for side in ("L", "R"):
        for end in ("Front", "Back"):
            bones[f"{end}Foot.{side}"].use_connect = False
            bones[f"{end}Foot.{side}"].parent = bones[f"{end}LowLeg.{side}"]
    bpy.ops.object.mode_set(mode="OBJECT")

    # The active outputs are Diffuse BSDFs, which glTF cannot express: the colours
    # would be lost. Each material also has a Principled BSDF of the same colour.
    for material in bpy.data.materials:
        tree = material.node_tree
        if tree is None:
            continue
        principled = next(n for n in tree.nodes if n.type == "BSDF_PRINCIPLED")
        output = next(
            link.to_node for link in tree.links if link.from_node == principled
        )
        for node in [n for n in tree.nodes if n not in (principled, output)]:
            tree.nodes.remove(node)
        output.target = "ALL"
        output.is_active_output = True
        principled.inputs["Roughness"].default_value = 1.0

    world = rig.matrix_world
    front = (world @ rig.data.bones["FrontUpLeg.L"].head_local).y
    rear = (world @ rig.data.bones["BackUpLeg.L"].head_local).y
    rig.scale = [s * SPAN / abs(front - rear) for s in rig.scale]

    bpy.ops.export_scene.gltf(
        filepath=out, export_format="GLB", export_animations=False, export_yup=True
    )


if __name__ == "__main__":
    main()
