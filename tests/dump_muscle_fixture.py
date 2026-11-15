"""Dump the muscle-tube fixture for ``tubes.test.ts`` from MyoSuite's own code.

The engine's ``core/muscle/tubes.ts`` ports MyoSuite's volumetric muscles
(``myosuite/viz/muscle_tubes.py`` and the math of ``blender_render._write_muscle_tubes``).
This runs MyoSuite's functions on a small arm whose muscles cover each path through
them: a wrap around a cylinder, two pulleys of divisor 2, and two parts of one muscle
sharing its measured volume. It records the tube points of a second pose, with the
shape fixed at ``qpos0``, and the MyoSuite commit they came from. The length ranges are
given: MuJoCo finds them by simulating, which differs in its last digits between the
WASM and Python builds.

Regenerate with::

    MYOSUITE_DIR=../myosuite .venv/bin/python tests/dump_muscle_fixture.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/mjswan/template/src/core/muscle/__tests__/fixtures/tubes.json"
SCALE = 1.25
POSE = {"elbow": -1.2, "wrist": 0.6}

XML = """<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="upper">
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
      <site name="flexor_origin" pos="0.015 0 -0.05"/>
      <site name="extensor_origin" pos="-0.03 0 -0.05"/>
      <site name="glmax1_origin" pos="0.03 0.02 -0.08"/>
      <site name="glmax2_origin" pos="0.03 -0.02 -0.08"/>
      <body name="lower" pos="0 0 -0.3">
        <joint name="elbow" axis="0 1 0" range="-2 0.3"/>
        <geom name="elbow_wrap" type="cylinder" fromto="0 -0.02 0 0 0.02 0" size="0.025"/>
        <site name="elbow_side" pos="0.05 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"/>
        <site name="flexor_insertion" pos="0.015 0 -0.06"/>
        <site name="glmax1_insertion" pos="0.025 0.02 -0.1"/>
        <site name="glmax2_insertion" pos="0.025 -0.02 -0.1"/>
        <site name="extensor_split" pos="-0.025 0 -0.04"/>
        <body name="hand" pos="0 0 -0.25">
          <joint name="wrist" axis="0 1 0" range="-1 1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.02"/>
          <site name="extensor_a" pos="-0.02 0 -0.05"/>
          <site name="extensor_b" pos="0.02 0 -0.08"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="flexor">
      <site site="flexor_origin"/><geom geom="elbow_wrap" sidesite="elbow_side"/>
      <site site="flexor_insertion"/>
    </spatial>
    <spatial name="extensor">
      <site site="extensor_origin"/><site site="extensor_split"/>
      <pulley divisor="2"/><site site="extensor_split"/><site site="extensor_a"/>
      <pulley divisor="2"/><site site="extensor_split"/><site site="extensor_b"/>
    </spatial>
    <spatial name="glmax1_r"><site site="glmax1_origin"/><site site="glmax1_insertion"/></spatial>
    <spatial name="glmax2_r"><site site="glmax2_origin"/><site site="glmax2_insertion"/></spatial>
  </tendon>
  <actuator>
    <muscle name="flexor" tendon="flexor" lengthrange="0.214 0.3186"/>
    <muscle name="extensor" tendon="extensor" lengthrange="0.5278 0.5762"/>
    <muscle name="glmax1_r" tendon="glmax1_r" force="600" lengthrange="0.1636 0.3249"/>
    <muscle name="glmax2_r" tendon="glmax2_r" force="300" lengthrange="0.1636 0.3249"/>
  </actuator>
</mujoco>"""


def main() -> None:
    myosuite = Path(os.environ["MYOSUITE_DIR"]).resolve()
    spec = importlib.util.spec_from_file_location(
        "muscle_tubes", myosuite / "myosuite/viz/muscle_tubes.py"
    )
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)

    model = mujoco.MjModel.from_xml_string(XML)
    muscles = mt.muscle_actuators(model)
    tendons = model.actuator_trnid[muscles, 0]
    pose = np.zeros(model.nq)
    for joint, angle in POSE.items():
        pose[model.joint(joint).qposadr[0]] = angle
    paths = []
    for qpos in (np.zeros(model.nq), pose):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        points = data.wrap_xpos.reshape(-1, 3)
        objs = data.wrap_obj.reshape(-1)
        paths.append(
            [
                mt.split_branches(points[a : a + n], objs[a : a + n])
                for a, n in zip(data.ten_wrapadr[tendons], data.ten_wrapnum[tendons])
            ]
        )

    # As blender_render._write_muscle_tubes does, the frames being qpos0 and the pose.
    owner = np.repeat(np.arange(len(muscles)), [len(m) for m in paths[0]])
    first = np.r_[True, np.diff(owner) > 0]
    resampled = [[mt.resample_path(b) for m in frame for b in m] for frame in paths]
    centres = np.array([[c for c, _ in frame] for frame in resampled])
    lengths = np.array([[n for _, n in frame] for frame in resampled])
    peak, fraction = mt.muscle_shape(model, lengths[0, first])
    profile = mt.radius_profile(fraction)
    volume = mt.belly_volume(peak, lengths[0, first], profile)
    peak = mt.belly_radius(volume, lengths[:, first], profile) * SCALE
    radii = peak[:, owner, None] * np.where(
        first[:, None], profile[owner], mt.TENDON_RADIUS_RATIO
    )
    points, _, _ = mt.tube_mesh(centres, radii)

    sha = subprocess.run(
        ["git", "-C", str(myosuite), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fixture = {
        "myosuite": sha,
        "xml": XML,
        "scale": SCALE,
        "qpos": pose.tolist(),
        "actuators": muscles.tolist(),
        "volume": volume.tolist(),
        "points": [round(float(v), 12) for v in points[1].ravel()],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {len(owner)} tubes, MyoSuite {sha[:8]}")


if __name__ == "__main__":
    main()
