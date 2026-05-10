<p align="center">
  <img src="https://github.com/ttktjmt/mjswan/raw/main/assets/banner.svg" alt="mjswan" width="60%">
</p>
<p align="center">
  <strong><em>Real-time Interactive RL Simulation in Your Browser</em></strong>
</p>

<p align="center">
  <a href="https://github.com/ttktjmt/mjswan/actions/workflows/deploy.yml"><img src="https://github.com/ttktjmt/mjswan/actions/workflows/deploy.yml/badge.svg" alt="deploy"/></a>
  <a href="https://github.com/ttktjmt/mjswan/actions/workflows/pytest.yml"><img src="https://github.com/ttktjmt/mjswan/actions/workflows/pytest.yml/badge.svg" alt="test"/></a>
  <a href="https://mjswan.readthedocs.io"><img src="https://img.shields.io/readthedocs/mjswan?logo=readthedocs" alt="docs"/></a>
  <a href="https://pypi.org/project/mjswan"><img src="https://img.shields.io/pypi/v/mjswan.svg?logo=pypi" alt="pypi version"></a>
  <a href="https://www.npmjs.com/package/mjswan"><img src="https://img.shields.io/npm/v/mjswan.svg?logo=nodedotjs" alt="npm version"></a>
</p>

<p align="center">
  mjswan is a powerful framework for creating interactive MuJoCo simulations with real-time policy control, running entirely in the browser. Built on top of <a href="https://github.com/google-deepmind/mujoco/tree/main/wasm">mujoco wasm</a>, <a href="https://github.com/microsoft/onnxruntime">onnxruntime</a>, and <a href="https://github.com/mrdoob/three.js/">three.js</a>, it enables easy sharing of RL simulation demos as static sites, perfect for GitHub Pages hosting.
</p>

<p align="center">
  <a href="https://ttktjmt.github.io/mjswan/"><img src="https://github.com/ttktjmt/mjswan/raw/main/assets/demo.gif" width="70%"/></a>
</p>

<p align="center">
  <em>Check out the demo ― <a href="https://ttktjmt.github.io/mjswan/">ttktjmt.github.io/mjswan</a></em>
</p>

<p align="center">
  <a href="https://mjswan-mjlab.pages.dev/"><img src="https://img.shields.io/badge/mjlab-E0E0E0?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAACgUlEQVR4nERSz08TaxQ937zhx2sptK+lEKAtfa/k8SBPFhhI2ADGhRIXgBETjQtduvVPMMaNCS7ZGYwxBIzRmJiIxkSEKChGUIOFCgFatD/SAg30x8x3r/mmGCez+Gbuuefcc7+jN/4/BjCDwCazxNHBAIipyGQQGcwm2ARMlkUdCs4QAIQQGpgJUN/MxL8fMEpI3WIFmC4MtwmNX82tb8d22apJk3p7/unuajJg3B59RlLx/OHwDjBTrbvi4fi5k73BKnvZ9MsIs2SWNQ599ObgyGB3+7/+lbXY+kYcII1Jjdvsq1a6wPCZtjpvpTIFCrU416PxeCJzcHAYOubTy3Uwa8olqNlfA0BKKi/Xr17pslzS5Yt9QX/9ndfhG2OPahtcdpvOzBqzJDKCAdXw9Hn4MFe8NNLp9diCHQ2nekNTb1b3M2mf1zW3EjvI5oFfpoN+J4BwJHmYK54f6rh1fWA2svMtmq7O5U/0dNrttu93pw1DyaqRmM2/Ay4AyVR2fOL9p/B2Jpdr8rin5sKb+9mZhaUXs8tnT3eqbYN1sBTgYOAvAInU/uLHzaVwyuaw9//n4sViWb6mye0siMq+rha3y5ZKFZQHr6eyyl5RUgDkVmSnPeB6Mrs6NTE/ef/tvQfzQ33tmibaWhuZTQ2QzT5naaeJ5B5Jmny8+Hkj3X889PXLVvJHenlhzeVQdK0tDQDpzKbb/efmdhpAPLEHYcaimWx8d2Y1mi8UAC4WjXcfIvV1To/bAbCoDV0DhKVw9DJJsEkkhfotuRRKMhkmk6GrDLJCCWhWo7pHFV4LXcoUWDIkFBHpJPNQWRVkiQiU8qpq1lGxMpuCiaE0fwYAAP//k7+eneLJBrkAAAAASUVORK5CYII=" alt="mjlab Demo"></a>
  &nbsp;
  <a href="https://ttktjmt.github.io/mjswan/robotdesc"><img src="https://img.shields.io/badge/Robot_Descriptions-E0E0E0?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAACYUlEQVR4nHyS208TQRjFv5nd7mW2205JQSExjVwfEEQxoon+q/4DRiKRlhTiLTGxNZK0gA8QLUZpbGtppLvd2es3ZlsfePI8/V7OzDknH5VS9vqXB6/fhGEkrylJkutcr9cnTAHgy/FRJpMJw7DdbjcajUqlAgB+ELx7/qxafpUkMSJqmlatVgEgNXysf7p3d900meM4p6enCwsL/X5/cHlJbyyurK6dn3+nlG5sbBiGAelLvl+r1cb/yv9EklK2Wi0pJRmJ4GroKJQwyyIA3W5XeN7t+XlE6fZ/6aal23lEiZj4QmRtmxJChkM3SSQZy/NGVKETbr7d+7C/O2HfF67rEEJUADANHUDCWLado5ROOF9a4sWZYa+dLc6ZJkviOC2dxqKgaZlJ3NxYE15cfwBaVuczUsogCBDHs2oqLdiMYmhqumXqo85F5A2ZoZm6Gvpi7uY0t5mpqzxn2VnGDI2qqkoV9c/VcBKjfVQ7OdjBJBFCdDqds7OzwWCQ9mkevdjeBoC0A6VUzWQISQ285lcJECCEMVYocM55oVBAxKXlFaKZAEDiOEGJvhCGaVECo5FLAJiVRYlhEKoZVVVUlBhFkfAE51wNovhH5/cMt2kcB0Ls7Lw0DOPh1pZFlff7VRPk8pOnucJU69vXi58Xd9bW0gWFH/hCAEAYRfc3N6WUum70et2DvUq5XK7slhVFOTz8nOd5xhjx/PDk+MRxnK3Hj0DKMAyDILAsq9vpzs7NNpuNUqnE+ZTnjVzXKRanU4PruoiYtW0CQAhBREopokySOIoixhiiBPh3Cn8DAAD//3KKiEEyOiLAAAAAAElFTkSuQmCC" alt="Robot Descriptions Demo"></a>
  &nbsp;
  <a href="https://ttktjmt.github.io/mjswan/playground"><img src="https://img.shields.io/badge/MuJoCo_Playground-E0E0E0?logo=deepmind" alt="MuJoCo Playground Demo"></a>
  &nbsp;
  <a href="https://ttktjmt.github.io/mjswan/myosuite"><img src="https://img.shields.io/badge/MyoSuite-E0E0E0?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzMiAzMiI+PGRlZnM+PHN0eWxlPi5jbHMtMXtmaWxsOiMxNTUyNTc7fS5jbHMtMntmaWxsOiNlNmY1ZjA7fTwvc3R5bGU+PC9kZWZzPjxnIGlkPSJMYXllcl8xMCIgZGF0YS1uYW1lPSJMYXllciAxMCI+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjEsMTNsLTQuMywzLjQ2TDEyLjE3LDkuMzksNy4yOCw2LDEwLjM5LDI2LjJsNi4zMSwxLDYuODctMTAuNDFaTTE1LjY2LDI1LjFsLTMuOS0uNTlMOS4zNCw5LjA2bC41OS40LDYuMjgsOS43NSw0LjkzLTMuOTIuNTQuNzhaIi8+PHBhdGggY2xhc3M9ImNscy0yIiBkPSJNMjEuMTQsMTUuMjlsLTQuOTMsMy45Mkw5LjkzLDkuNDZsLS41OS0uNCwyLjQyLDE1LjQ1LDMuOS41OSw2LTlaIi8+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMTQuNDQsMjQuMzFhMy40OCwzLjQ4LDAsMCwwLTEuMTYtLjE5LDMuNTcsMy41NywwLDAsMC0xLjE1LDYuOTQsMy40NywzLjQ3LDAsMCwwLDEuMTUuMTksMy41OCwzLjU4LDAsMCwwLDMuMzgtMi40MSwzLjU3LDMuNTcsMCwwLDAtMi4yMi00LjUzWiIvPjxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTEzLjA2LDguMjNhMy41NSwzLjU1LDAsMCwwLTEtMi41NEEzLjQ5LDMuNDksMCwwLDAsOS41Niw0LjZIOS40OUEzLjU5LDMuNTksMCwwLDAsNS45Miw4LjFhMy41NywzLjU3LDAsMCwwLDMuNSwzLjYzaC4wN0EzLjU5LDMuNTksMCwwLDAsMTMuMDYsOC4yM1oiLz48cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik0yNi4xOCwxNS4xOEEzLjU1LDMuNTUsMCwwLDAsMjYsMTIuNDVhMy42MiwzLjYyLDAsMCwwLTMuMi0yLDMuNTcsMy41NywwLDAsMC0xLjE2LDcsMy40OCwzLjQ4LDAsMCwwLDEuMTYuMTlBMy41NiwzLjU2LDAsMCwwLDI2LjE4LDE1LjE4WiIvPjxwYXRoIGNsYXNzPSJjbHMtMSIgZD0iTTIyLDUuNTJhMy42MSwzLjYxLDAsMCwwLS4xNy0yLjcyLDMuNTUsMy41NSwwLDAsMC0zLjIxLTJoMEEzLjU3LDMuNTcsMCwxLDAsMjIsNS41MloiLz48L2c+PC9zdmc+" alt="MyoSuite Demo"></a>

</p>

---


## News

- **2026-05-06**: Adopted for the [GentleHumanoid](https://gentle-humanoid.axell.top/#/) live demo [[Demo](https://mjswan-gentlehumanoid.pages.dev/), [X](https://x.com/Axell_wppr/status/2051878574874148953)]
- **2026-04-08**: Featured in the [MuJoCo README](https://github.com/google-deepmind/mujoco#first-party-bindings)


## Features

- **Real-time**: Run mujoco simulations and policy control in real time.
- **Interactive**: Change the state of objects by applying forces.
- **Cross-platform**: Works seamlessly on desktop and mobile devices.
- **VR Support**: Native VR viewer support with WebXR.
- **Client-only**: All computation runs in the browser. No server for simulation is required.
- **Easy Sharing**: Host as a static site for effortless demo distribution (e.g., GitHub Pages).
- **Portable**: Embed the simulation in a web page or Google Colab notebook.
- **Customizable**: Visualize your mujoco models and onnx policies quickly.


## Quick Start

mjswan can be installed with `pip`:
``` sh
pip install mjswan  # or 'mjswan[dev]', 'mjswan[examples]'
```

or with `npm`:
``` sh
npm install mjswan
```

You can run the demo using the `uv` command with the python package `mjswan[examples]`:
``` sh
uv run main
```

The minimum python script for a sanity check:
``` python
import os, mujoco, mjswan

model_path = os.path.join(os.path.dirname(mujoco.__file__), "testdata", "model.xml")
mjspec = mujoco.MjSpec.from_file(model_path)

builder = mjswan.Builder()
builder.add_project(name="Sanity Check").add_scene(name="Test Model", spec=mjspec)
app = builder.build()
app.launch()
```

For detailed instructions, visit the [documentation](https://mjswan.readthedocs.io).


## Third-Party Assets

mjswan incorporates mujoco models from the external sources in its demo. See the respective submodule for full details, including individual model licenses and copyrights. All models are used under their respective licenses. Please review and comply with those terms for any use or redistribution.

[Robot Descriptions License](https://github.com/robot-descriptions/robot_descriptions.py/blob/main/LICENSE) ･ [MuJoCo Playground License](https://github.com/google-deepmind/mujoco_playground/blob/main/LICENSE) ･ [MyoSuite License](https://github.com/MyoHub/myosuite/blob/main/LICENSE)


## Acknowledgments

This project was greatly inspired by the [Facet project demo](https://facet.pages.dev/) from the research group at Tsinghua University.<br>
It is also built upon the excellent work of [zalo/mujoco_wasm](https://github.com/zalo/mujoco_wasm), one of the earliest efforts to run MuJoCo simulations in a browser.


## License

This project is licensed under the [Apache-2.0 License](LICENSE). When using mjswan, please retain attribution notices in the app to help other users discover this project.
