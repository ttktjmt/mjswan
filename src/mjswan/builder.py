"""Builder class for constructing mjswan applications.

This module provides the main Builder class which serves as the entry point
for programmatically creating interactive MuJoCo simulations.
"""

from __future__ import annotations

import inspect
import json
import shutil
import warnings
from pathlib import Path

import mujoco
import onnx

from . import __version__
from ._build_client import ClientBuilder
from .app import mjswanApp
from .project import ProjectConfig, ProjectHandle
from .utils import collect_spec_assets, name2id, to_zip_deflated


class Builder:
    """Builder for creating mjswan applications.

    The Builder class provides a fluent API for programmatically constructing
    interactive MuJoCo simulations with ONNX policies. It handles projects, scenes, and policies hierarchically.
    """

    def __init__(self, base_path: str = "/") -> None:
        """Initialize a new Builder instance.

        Args:
            base_path: Base path for the application (e.g., '/mjswan/').
                      This is used for deployment to subdirectories.
        """
        self._projects: list[ProjectConfig] = []
        self._base_path = base_path

    def add_project(self, name: str, *, id: str | None = None) -> ProjectHandle:
        """Add a new project to the builder.

        Args:
            name: Name for the project (displayed in the UI).
            id: Optional ID for URL routing. If not provided, the first project
                defaults to None (main route), and subsequent projects default to sanitized name.

        Returns:
            ProjectHandle for adding scenes and further configuration.
        """
        # Determine project ID:
        # - If id is explicitly provided, use it
        # - First project without id defaults to None (main route)
        # - Subsequent projects without id default to sanitized name
        if id is not None:
            project_id = id
        elif not self._projects:
            project_id = None
        else:
            project_id = name2id(name)

        project = ProjectConfig(name=name, id=project_id)
        self._projects.append(project)
        return ProjectHandle(project, self)

    def build(self, output_dir: str | Path | None = None) -> mjswanApp:
        """Build the application from the configured projects.

        This method finalizes the configuration and creates a mjswanApp
        instance. If output_dir is provided, it also saves the application
        to that directory. If output_dir is not provided, it defaults to
        'dist' in the caller's directory.

        Args:
            output_dir: Optional directory to save the application files.
                       If None, defaults to 'dist' in the caller's directory.

        Returns:
            mjswanApp instance ready to be launched.
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
        self._save_web(output_path)

        return mjswanApp(output_path)

    def _save_config_json(self, output_path: Path) -> None:
        """Save configuration as JSON.

        Creates root assets/config.json with project metadata and structure information.
        Individual project assets (scenes/policies) are saved under project-id/assets/.
        """
        # Create root config with project metadata and structure info
        root_config = {
            "version": __version__,
            "projects": [
                {
                    "name": project.name,
                    "id": project.id,
                    "scenes": [
                        {
                            "name": scene.name,
                            "path": f"{name2id(scene.name)}/{scene.scene_filename}",
                            **({"splat": scene.splat.to_dict()} if scene.splat is not None else {}),
                            "policies": [
                                (
                                    {
                                        "name": policy.name,
                                        **(
                                            {
                                                "config": f"{name2id(scene.name)}/"
                                                f"{name2id(policy.name)}.json"
                                            }
                                            if getattr(policy, "config_path", None)
                                            or getattr(policy, "commands", None)
                                            else {}
                                        ),
                                        **(
                                            {"source": policy.source_path}
                                            if getattr(policy, "source_path", None)
                                            else {}
                                        ),
                                    }
                                )
                                for policy in scene.policies
                            ],
                        }
                        for scene in project.scenes
                    ],
                }
                for project in self._projects
            ],
        }

        # Save root config.json in assets directory
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)
        root_config_file = assets_dir / "config.json"
        with open(root_config_file, "w") as f:
            json.dump(root_config, f, indent=2)

    def _policy_filename(self, name: str) -> str:
        if not name or name.strip() == "":
            raise ValueError("Policy name must be a non-empty string.")
        if "/" in name or "\\" in name:
            raise ValueError(
                "Policy name cannot contain path separators ('/' or '\\')."
            )
        return name

    def _save_web(self, output_path: Path) -> None:
        """Save as a complete web application with hybrid structure.

        Output Structure:
            dist/
            ├── index.html
            ├── logo.svg
            ├── manifest.json
            ├── robots.txt
            ├── assets/
            │   ├── config.json
            │   └── (compiled js/css files)
            └── <project-id>/ (or 'main')
                ├── index.html
                ├── logo.svg
                ├── manifest.json
                └── assets/
                    └── <scene-id>/
                        ├── scene.mjz/.mjb
                        ├── <policy-id>.onnx
                        └── <policy-id>.json
        """
        if output_path.exists():
            shutil.rmtree(output_path)

        output_path.mkdir(parents=True, exist_ok=True)

        # Copy template directory
        template_dir = Path(__file__).parent / "template"
        if template_dir.exists():
            # Build client first
            package_json = template_dir / "package.json"
            if package_json.exists():
                print("Building the mjswan application...")
                builder = ClientBuilder(template_dir)
                builder.build(base_path=self._base_path)

            # Copy all files from template to output_path
            shutil.copytree(
                template_dir,
                output_path,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".nodeenv", "__pycache__", "*.pyc", ".md"
                ),
            )

            # Move built files from nested dist/ to output_path root
            built_dist = output_path / "dist"
            if built_dist.exists() and built_dist.is_dir():
                # Move all files from dist/ to output_path
                for item in built_dist.iterdir():
                    dest = output_path / item.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(item), str(output_path))
                # Remove the now-empty dist directory
                built_dist.rmdir()

                # Clean up development files that shouldn't be in production
                dev_files = [
                    "src",
                    "node_modules",
                    ".nodeenv",
                    "package.json",
                    "package-lock.json",
                    "tsconfig.json",
                    "vite.config.ts",
                    "eslint.config.cjs",
                    ".browserslistrc",
                    ".gitignore",
                    "README.md",
                ]
                for dev_file in dev_files:
                    dev_path = output_path / dev_file
                    if dev_path.exists():
                        if dev_path.is_dir():
                            shutil.rmtree(dev_path)
                        else:
                            dev_path.unlink()

                # Remove public directory after build
                public_dir = output_path / "public"
                if public_dir.exists():
                    shutil.rmtree(public_dir)
        else:
            warnings.warn(
                f"Template directory not found at {template_dir}.",
                category=RuntimeWarning,
            )

        # Create root assets directory for shared config
        assets_dir = output_path / "assets"
        assets_dir.mkdir(exist_ok=True)

        # Save root configuration (project metadata and structure)
        self._save_config_json(output_path)

        # Save MuJoCo models and ONNX policies per project
        for project in self._projects:
            # Use 'main' for projects without ID, otherwise use the project ID
            project_dir_name = project.id if project.id else "main"
            project_dir = output_path / project_dir_name
            project_assets_dir = project_dir / "assets"

            # Create directories
            project_assets_dir.mkdir(parents=True, exist_ok=True)

            # Copy index.html to each project directory so direct navigation works
            root_index = output_path / "index.html"
            if root_index.exists():
                shutil.copy(str(root_index), str(project_dir / "index.html"))

            # Copy static root assets
            for static_name in ["manifest.json", "logo.svg"]:
                src_static = output_path / static_name
                if src_static.exists():
                    shutil.copy(str(src_static), str(project_dir / static_name))

            # Save scenes and policies
            for scene in project.scenes:
                scene_id = name2id(scene.name)
                scene_dir = project_assets_dir / scene_id
                scene_dir.mkdir(parents=True, exist_ok=True)
                scene_path = scene_dir / scene.scene_filename
                if scene.spec is not None:
                    scene.spec.assets.update(collect_spec_assets(scene.spec))
                    to_zip_deflated(scene.spec, str(scene_path))  # Saves as .mjz
                else:
                    if scene.model is None:
                        raise RuntimeError(
                            f"Scene '{scene.name}' has no model to save as .mjb"
                        )
                    mujoco.mj_saveModel(scene.model, str(scene_path))  # Saves as .mjb

                # Save policies
                for policy in scene.policies:
                    policy_id = name2id(policy.name)
                    policy_path = scene_dir / f"{policy_id}.onnx"
                    onnx.save(policy.model, str(policy_path))

                    config_path = getattr(policy, "config_path", None)
                    if config_path:
                        config_src = Path(config_path).expanduser()
                        if not config_src.is_absolute():
                            config_src = (Path.cwd() / config_src).resolve()
                        if config_src.exists():
                            target = policy_path.with_suffix(".json")
                            try:
                                with open(config_src, "r") as f:
                                    data = json.load(f)
                                data.setdefault("onnx", {})
                                if isinstance(data["onnx"], dict):
                                    data["onnx"]["path"] = policy_path.name
                                # Serialize commands if any are defined
                                if policy.commands:
                                    data["commands"] = {
                                        name: cmd.to_dict()
                                        for name, cmd in policy.commands.items()
                                    }
                                with open(target, "w") as f:
                                    json.dump(data, f, indent=2)
                            except Exception:
                                shutil.copy(str(config_src), str(target))
                        else:
                            warnings.warn(
                                f"Policy config path not found: {config_src}",
                                category=RuntimeWarning,
                                stacklevel=2,
                            )
                    elif policy.commands:
                        # No config_path but commands defined - create config with commands only
                        target = policy_path.with_suffix(".json")
                        data = {
                            "onnx": {"path": policy_path.name},
                            "commands": {
                                name: cmd.to_dict()
                                for name, cmd in policy.commands.items()
                            },
                        }
                        with open(target, "w") as f:
                            json.dump(data, f, indent=2)

        print(f"✓ Saved mjswan application to: {output_path}")

    def get_projects(self) -> list[ProjectConfig]:
        """Get a copy of all project configurations.

        Returns:
            List of ProjectConfig objects.
        """
        return self._projects.copy()


__all__ = ["Builder"]
