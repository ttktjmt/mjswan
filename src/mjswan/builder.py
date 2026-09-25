"""Builder class for constructing mjswan applications.

This module provides the main Builder class which serves as the entry point
for programmatically creating interactive MuJoCo simulations.
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from .app import MjswanApp
from .document.ids import assign_name
from .license import resolve_license
from .project import ProjectConfig, ProjectHandle


class Builder:
    """Builder for creating mjswan applications.

    The Builder class provides a fluent API for programmatically constructing
    interactive MuJoCo simulations with ONNX policies. It handles projects, scenes, and policies hierarchically.
    """

    def __init__(
        self,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
        *,
        license: str | os.PathLike[str] | None = None,
        copyright: str | None = None,
    ) -> None:
        """Initialize a new Builder instance.

        Args:
            base_path: Base path for subdirectory deployment (e.g., '/mjswan/').
            gtm_id: Google Tag Manager ID (e.g., 'GTM-XXXXXXX'). Injects GTM snippet if set.
            mt: Enable multi-threaded MuJoCo WASM. Requires COOP/COEP headers — these are
                written as a ``_headers`` file (Netlify/Cloudflare Pages/Vercel) and a
                service worker (required for GitHub Pages hosting). Defaults to False.
            debug: Keep browser console messages in the built app. Defaults to False
                (console messages are stripped from the production bundle).
            license: The work's license, written as ``<project-id>/LICENSE`` for every
                project that does not set its own (ADR 0007 §1): a generatable SPDX id
                (``"Apache-2.0"``, ``"MIT"``, ``"BSD-3-Clause"``, ``"BSD-2-Clause"``,
                ``"CC-BY-4.0"``, ``"CC0-1.0"``) for the standard text, or the path to a
                license text, copied verbatim.
            copyright: The holder line of a generated license text, e.g.
                ``"2026 Example Lab"``.
        """
        self._projects: list[ProjectConfig] = []
        self._base_path = base_path
        self._gtm_id = gtm_id
        self._mt = mt
        self._debug = debug
        # Resolved now so a bad id or a missing file fails here rather than at build.
        self._license = (
            None if license is None else resolve_license(license, copyright=copyright)
        )

    @classmethod
    def from_mjlab(
        cls,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        hf_repo_id: str | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
        base_path: str = "/",
        gtm_id: str | None = None,
        mt: bool = False,
        debug: bool = False,
    ) -> Builder:
        """Create a Builder pre-configured with a single mjlab task.

        This is a convenience factory for the common pattern of visualizing one
        mjlab task. The returned Builder can be further modified before calling
        :meth:`build`.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all ``model_*.pt``
                checkpoints from each run are fetched and converted to ONNX
                via mjlab and torch (the ``wandb`` and ``mjlab`` extras, checked
                when this is called). ``task_id`` above is reused
                for the conversion. Defaults to ``None`` (no policy attached).
                For finer control (e.g. ``only_latest=True``, custom
                observations/actions), build manually with
                :meth:`add_project` → :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`
                → :meth:`~mjswan.scene.SceneHandle.add_policy_wandb`.
            hf_repo_id: Optional Hugging Face Hub repository (``"<owner>/<name>"``)
                whose exported ONNX is added as a policy, with no torch conversion, as
                :meth:`~mjswan.scene.SceneHandle.add_policy_hf` adds it. May be combined
                with ``run_path``. For finer control (a specific file, a pinned
                revision), call that directly.
            project_name: Name for the auto-created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            base_path: Base path for the application (e.g., ``"/mjswan/"``).
            gtm_id: Optional Google Tag Manager container ID.

        Returns:
            Builder with one project and one scene already configured.

        Example:
            ```python
            # Minimal usage
            app = mjswan.Builder.from_mjlab("go2_flat").build()
            app.launch()

            # Load mjlab scene and attach all checkpoints from a W&B run
            app = mjswan.Builder.from_mjlab(
                "Mjlab-Velocity-Flat-Anymal-C",
                run_path="ttktjmt-org/mjlab/dqxvf0eb",
            ).build()

            # Customise before building
            builder = mjswan.Builder.from_mjlab("go2_flat")
            scene = builder.get_projects()[0].scenes[0]  # access SceneConfig
            app = builder.build()
            ```
        """
        builder = cls(base_path=base_path, gtm_id=gtm_id, mt=mt, debug=debug)
        builder.add_project_mjlab(
            task_id,
            run_path=run_path,
            hf_repo_id=hf_repo_id,
            project_name=project_name,
            play=play,
            env_cfg=env_cfg,
        )
        return builder

    def add_project_mjlab(
        self,
        task_id: str,
        *,
        run_path: str | list[str] | None = None,
        hf_repo_id: str | None = None,
        project_name: str = "mjlab",
        play: bool | None = None,
        env_cfg: Any | None = None,
    ) -> ProjectHandle:
        """Add a project pre-configured with a single mjlab task.

        Convenience for the common pattern of visualizing one mjlab task:
        creates a project, adds the mjlab scene, and (optionally) attaches all
        ``model_*.pt`` checkpoints from one or more W&B runs as ONNX policies.

        Args:
            task_id: mjlab task identifier (e.g. ``"go2_flat"``).
            run_path: Optional W&B run path (``"entity/project/run_id"``) or a
                list of such paths. When provided, all checkpoints are fetched
                and converted to ONNX via mjlab and torch (the ``wandb`` and
                ``mjlab`` extras) using ``task_id``. Defaults to ``None`` (no
                policy attached).
            hf_repo_id: Optional Hugging Face Hub repository to take the policy's
                exported ONNX from. See :meth:`from_mjlab`.
            project_name: Name for the created project. Defaults to ``"mjlab"``.
            play: Which of the task's two registered configs to load; unset means play.
                Mutually exclusive with ``env_cfg``. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.
            env_cfg: Pre-loaded (and possibly edited) env config. See
                :meth:`~mjswan.project.ProjectHandle.add_scene_mjlab`.

        Returns:
            ProjectHandle for the created project.
        """
        project = self.add_project(name=project_name)
        # `play` stays unresolved: `add_scene_mjlab` rejects it alongside `env_cfg`, so
        # materialising the default here would trip that guard for every caller.
        scene = project.add_scene_mjlab(task_id, play=play, env_cfg=env_cfg)
        if run_path is not None:
            scene.add_policy_wandb(run_path, task_id=task_id)
        if hf_repo_id is not None:
            scene.add_policy_hf(hf_repo_id, task_id=task_id)
        return project

    def add_project(
        self,
        name: str,
        *,
        default: bool = False,
        license: str | os.PathLike[str] | None = None,
        copyright: str | None = None,
    ) -> ProjectHandle:
        """Add a new project to the builder.

        The project's id (its directory in the build and its ``?project=`` value) is
        ``name2id(name)``, made unique within the document: a second project that
        sanitizes to the same id is renamed ``<name>_1``, id ``<id>_1``, with a warning
        (ADR 0006 §4).

        Args:
            name: Name for the project (displayed in the UI).
            default: Open the app on this project. At most one project may set it;
                when none does, the first added is the default.
            license: This project's license, overriding the builder's: a generatable
                SPDX id or the path to a license text. See
                :meth:`~mjswan.project.ProjectHandle.set_license`.
            copyright: The holder line of a generated license text.

        Returns:
            ProjectHandle for adding scenes and further configuration.
        """
        if default:
            taken = next((p for p in self._projects if p.default), None)
            if taken is not None:
                raise ValueError(
                    f"Project {name!r} cannot be the default: {taken.name!r} already "
                    "is. Exactly one project may set default=True."
                )
        name, ident = assign_name(name, {p.id for p in self._projects}, kind="project")
        project = ProjectConfig(
            name=name,
            id=ident,
            default=default,
            license=(
                self._license
                if license is None
                else resolve_license(license, copyright=copyright)
            ),
        )
        self._projects.append(project)
        return ProjectHandle(project, self)

    def build(
        self,
        output_dir: str | Path | None = None,
        build_frontend: bool | None = None,
    ) -> MjswanApp:
        """Build the application from the configured projects.

        This method finalizes the configuration and creates a MjswanApp
        instance. If output_dir is provided, it also saves the application
        to that directory. If output_dir is not provided, it defaults to
        'dist' in the caller's directory.

        Args:
            output_dir: Optional directory to save the application files.
                       If None, defaults to 'dist' in the caller's directory.

        Returns:
            MjswanApp instance ready to be launched.
        """
        if not self._projects:
            raise ValueError(
                "Cannot build an empty application. "
                "You must add at least one project using builder.add_project() before building.\n"
                "Example:\n"
                "  builder = mwx.Builder()\n"
                "  project = builder.add_project(name='My Project')\n"
                "  scene = project.add_scene(spec=mujoco_spec, name='Scene 1')\n"
                "  app = builder.build()"
            )

        # Get caller's file path
        frame = inspect.stack()[1]
        caller_file = frame.filename
        # Handle REPL or interactive mode where filename might be <stdin> or similar
        if caller_file.startswith("<") and caller_file.endswith(">"):
            base_dir = Path.cwd()
        else:
            base_dir = Path(caller_file).parent

        if output_dir is None:
            output_path = base_dir / "dist"
        else:
            # Resolve relative paths against the caller's directory
            output_path = base_dir / Path(output_dir)

        # TODO: Build with separate function (and then save the web app with _save_web). And set scene.path and policy.path after building.
        self._save_web(output_path, build_frontend=build_frontend)

        return MjswanApp(output_path)

    def _save_web(self, output_path: Path, build_frontend: bool | None = None) -> None:
        """Write the app at exactly ``output_path``; :meth:`build` resolves paths first."""
        from .build.pipeline import write_app

        write_app(
            self._projects,
            output_path,
            base_path=self._base_path,
            gtm_id=self._gtm_id,
            mt=self._mt,
            debug=self._debug,
            build_frontend=build_frontend,
        )

    def get_projects(self) -> list[ProjectConfig]:
        """Get a copy of all project configurations.

        Returns:
            List of ProjectConfig objects.
        """
        return self._projects.copy()


__all__ = ["Builder"]
