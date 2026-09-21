---
icon: octicons/home-16
---

# Welcome to mjswan!

<p align="center">
  <img src="https://github.com/ttktjmt/mjswan/raw/main/assets/banner.svg" alt="mjswan Header" style="width: 80%;">
  <br/>
  <strong><em>Real-time Interactive AI Robot Simulation in Your Browser</em></strong>
</p>

## What is mjswan?

mjswan turns a MuJoCo model and a trained policy into an interactive web app that runs
entirely in the browser — no server, no backend, no install for whoever you send it to.
Physics comes from [mujoco wasm](https://github.com/google-deepmind/mujoco/tree/main/wasm){:target="_blank" rel="noopener noreferrer"},
inference from [ONNX Runtime Web](https://github.com/microsoft/onnxruntime){:target="_blank" rel="noopener noreferrer"},
and rendering from [three.js](https://github.com/mrdoob/three.js){:target="_blank" rel="noopener noreferrer"}.
The output is a static site, so GitHub Pages is enough to host it.

What makes it more than a model viewer is that the *whole environment* comes along. The
observations your policy reads, the action term it drives, the terminations that end an
episode, the events that randomize a reset — mjswan compiles those from
[mjlab](guides/mjlab.md)'s own Python functions to ONNX at build time and runs them beside
the policy, so the browser reproduces the environment the policy was trained in.

## Try it

<iframe
  src="https://ttktjmt.github.io/mjswan/"
  title="mjswan live demo"
  loading="lazy"
  allow="xr-spatial-tracking; fullscreen"
  style="width: 100%; aspect-ratio: 16 / 9; border: 0; border-radius: 0.4rem;">
</iframe>

/// caption
The live demo, embedded with a single `<iframe>` — see [Embedding](guides/embedding.md).
Drag to orbit, drag the robot to push it, and use the control panel to steer the policy.
///

## Key Features

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Real-time Simulation__

    ---

    Run MuJoCo simulations and policy control in real time

-   :material-cursor-default-click:{ .lg .middle } __Interactive__

    ---

    Change the state of objects by applying forces with intuitive controls

-   :material-server-off:{ .lg .middle } __Client-only__

    ---

    All computation runs in the browser - no server required for simulation

-   :material-share-variant:{ .lg .middle } __Easy Sharing__

    ---

    Host as a static site for effortless demo distribution (e.g., GitHub Pages)

-   :material-devices:{ .lg .middle } __Cross-platform__

    ---

    Works seamlessly on desktop, mobile, and VR devices

-   :material-code-tags:{ .lg .middle } __Portable__

    ---

    Embed the simulation in any web page or Google Colab notebook output cell

</div>

## Use Cases

mjswan is perfect for:

- **Research Demos**: Share your robot learning research with interactive visualizations
- **Education**: Create interactive physics and robotics tutorials
- **Prototyping**: Quickly test and visualize different MuJoCo models and policies
- **Portfolio**: Showcase your robotics projects in an accessible way

## Live Demos

- [Main Demo](https://ttktjmt.github.io/mjswan){:target="_blank" rel="noopener noreferrer"} - mjlab's tasks with their trained checkpoints, plus a splat-backed G1 and muscle actuators
- [GentleHumanoid](https://mjswan-gentlehumanoid.pages.dev/){:target="_blank" rel="noopener noreferrer"} - Motion-tracking humanoid, built on mjswan
- [MuscleMimic](https://mjswan-musclemimic.pages.dev/){:target="_blank" rel="noopener noreferrer"} - Musculoskeletal motion imitation

## Quick Example

=== "A model"

    ```python
    import mujoco

    import mjswan

    builder = mjswan.Builder()
    project = builder.add_project(name="My Robot")

    spec = mujoco.MjSpec.from_string("""
    <mujoco>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
        <body pos="0 0 1">
          <joint type="free"/>
          <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
        </body>
      </worldbody>
    </mujoco>
    """)

    project.add_scene(spec=spec, name="My Scene")

    app = builder.build()
    app.launch()
    ```

=== "A trained mjlab task"

    ```python
    import mjswan

    # Scene, policies, observations, actions, commands, terminations —
    # all from the task and its W&B checkpoints.
    app = mjswan.Builder.from_mjlab(
        "Mjlab-Velocity-Flat-Unitree-G1",
        run_path="<entity>/<project>/<run_id>",
    ).build()
    app.launch()
    ```

## Where to next

<div class="grid cards" markdown>

-   :octicons-play-16:{ .lg .middle } __[Quickstart](getting-started/quickstart.md)__

    ---

    Zero to a running simulation in two minutes

-   :octicons-light-bulb-16:{ .lg .middle } __[Core Concepts](getting-started/core-concepts.md)__

    ---

    Builder → Project → Scene → Policy, and what each one owns

-   :octicons-arrow-switch-16:{ .lg .middle } __[Using mjlab](guides/mjlab.md)__

    ---

    Visualize a trained mjlab task, checkpoints and all

-   :octicons-cpu-16:{ .lg .middle } __[How the Build Works](guides/how-it-works.md)__

    ---

    Why your MDP terms end up as ONNX graphs, and what to do when one won't trace

</div>

## Links

- [GitHub Repository](https://github.com/ttktjmt/mjswan){:target="_blank" rel="noopener noreferrer"}
- [PyPI Package](https://pypi.org/project/mjswan){:target="_blank" rel="noopener noreferrer"}
- [npm Package](https://www.npmjs.com/package/mjswan){:target="_blank" rel="noopener noreferrer"}
- [mjswan - MyoConference 2026 Presentation](https://youtu.be/0B4Ky0hf2Gg){:target="_blank" rel="noopener noreferrer"}

## License

mjswan is licensed under the [Apache-2.0 License](https://github.com/ttktjmt/mjswan/blob/main/LICENSE){:target="_blank" rel="noopener noreferrer"}.
When using mjswan, please retain attribution notices in the app to help other users
discover the project.
