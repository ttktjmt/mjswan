"""mjlab Integration Example - Visualize MuJoCo scenes from all mjlab default tasks

Extracts the MuJoCo model from each mjlab default task and visualizes them
in the browser using mjswan.
"""

from mjlab.tasks.registry import list_tasks

import mjswan


def main():
    builder = mjswan.Builder()
    project = builder.add_project(name="mjlab Examples")

    for task_id in list_tasks():
        project.add_mjlab_scene(task_id)

    app = builder.build()
    app.launch()


if __name__ == "__main__":
    main()
