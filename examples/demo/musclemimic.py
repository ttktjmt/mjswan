"""MuscleMimic Demo

https://github.com/amathislab/musclemimic
"""

import mujoco
import musclemimic_models as mm

import mjswan

builder = mjswan.Builder()

mm_project = builder.add_project(name="MuscleMimic Demo")

mm_project.add_scene(
    spec=mujoco.MjSpec.from_file(str(mm.get_xml_path("myofullbody"))),
    name="MyoFullBody",
)
mm_project.add_scene(
    spec=mujoco.MjSpec.from_file(str(mm.get_xml_path("bimanual"))),
    name="BimanualMuscle",
)

app = builder.build()
app.launch()
