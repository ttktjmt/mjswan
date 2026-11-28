"""What ``Builder.build()`` does: write the engine and the expanded document as one app.

Output structure::

    dist/
    ├── index.html, logo.svg, robots.txt
    ├── LICENSE            (the engine's, from the SPA; never the work's)
    ├── assets/            (compiled js/css/wasm, plugins.js)
    ├── manifest.json      (the one descriptor; every key snake_case)
    └── <project-id>/
        ├── LICENSE, NOTICE            (the work's, when declared)
        ├── licenses/
        │   └── LICENSE.<component>, NOTICE.<component>  (third-party, once per text)
        └── <scene-id>/
            ├── scene.mjz | scene.mjb
            ├── mdp/<mdp-id>/{obs,term,command,event,action}/<name>.onnx
            ├── policy/<policy-id>.onnx
            └── assets/    (<motion>.npz, <splat>.spz, <skin>.vrm | .glb)
"""

from __future__ import annotations

import gc
import os
import secrets
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import onnx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from ..document import MANIFEST_NAME, read_manifest
from ..envs.mdp.actions.actions import (
    MuscleActivationActionCfg,
    validate_muscle_actuators,
)
from ..license import LICENSES_DIR
from ..mjlab.env import mdp_episode_length
from .asset import (
    point_env_cfg_at_bundled_motion,
    write_scene_motions,
    write_scene_skins,
    write_scene_splats,
)
from .credit import license_names
from .frontend import TEMPLATE_DIR, ClientBuilder, install_spa, uses_custom_js
from .manifest import (
    graphics_entry,
    load_sidecar,
    mdp_entry,
    policy_entry,
    scene_entry,
    write_manifest,
)
from .mjz import files_outside, to_zip_deflated

if TYPE_CHECKING:
    from ..project import ProjectConfig
    from ..scene import SceneConfig


def write_app(
    projects: list[ProjectConfig],
    output_path: Path,
    *,
    base_path: str = "/",
    gtm_id: str | None = None,
    mt: bool = False,
    debug: bool = False,
    build_frontend: bool | None = None,
) -> None:
    """Write ``projects`` as a complete web application at ``output_path``.

    ``output_path`` must be missing, empty or a previous build, which is replaced; any
    other directory is refused before anything is deleted. The app is written to a
    directory beside it and moved into place once complete, so a build that fails
    leaves the previous one. ``base_path``, ``gtm_id``, ``mt`` and ``debug`` are the
    frontend build's options (see :meth:`.frontend.ClientBuilder.build`);
    ``build_frontend`` forces or skips the Node build.
    """
    _check_defaults(projects)
    output_path = Path(os.path.abspath(output_path))
    _check_output(output_path)
    staging = output_path.with_name(f".{output_path.name}.{secrets.token_hex(4)}.tmp")
    staging.mkdir(parents=True)
    try:
        _write_app(
            projects,
            staging,
            base_path=base_path,
            gtm_id=gtm_id,
            mt=mt,
            debug=debug,
            build_frontend=build_frontend,
        )
        _swap_in(staging, output_path)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"✓ Saved mjswan application to: {output_path}")


def _check_output(output_path: Path) -> None:
    """Refuse an output path ``write_app`` may not replace: a link, a file, or a
    directory holding anything but a previous build. ``build(".")`` would otherwise
    delete the working directory."""
    if output_path.is_symlink():
        raise ValueError(
            f"{output_path} is a symbolic link. Build into the directory it names."
        )
    if not output_path.exists():
        return
    if not output_path.is_dir():
        raise ValueError(f"{output_path} is a file, not a directory to build into.")
    if any(output_path.iterdir()) and not _is_build(output_path):
        raise ValueError(
            f"{output_path} is not empty and holds no mjswan build ({MANIFEST_NAME}), "
            "so build() will not delete it. Build into a new or empty directory, or "
            "delete this one first."
        )


def _is_build(path: Path) -> bool:
    """Whether *path* holds a build: a ``manifest.json`` that lists projects."""
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and isinstance(manifest.get("projects"), list)


def _swap_in(staging: Path, output_path: Path) -> None:
    """Move *staging* to *output_path*, the previous build out of the way first and
    deleted once the new one is in place."""
    old = staging.with_name(f"{staging.name}.old") if output_path.exists() else None
    if old is not None:
        output_path.rename(old)
    try:
        staging.rename(output_path)
    except OSError:
        if old is not None:
            old.rename(output_path)
        raise
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def _write_app(
    projects: list[ProjectConfig],
    output_path: Path,
    *,
    base_path: str,
    gtm_id: str | None,
    mt: bool,
    debug: bool,
    build_frontend: bool | None,
) -> None:
    """Write every file of the app into the empty directory ``output_path``."""
    template_dir = TEMPLATE_DIR
    client_builder: ClientBuilder | None = None
    if template_dir.exists():
        package_json = template_dir / "package.json"
        if package_json.exists():
            print("Building the mjswan application...")
            client_builder = ClientBuilder(template_dir)
            client_builder.build(
                base_path=base_path,
                gtm_id=gtm_id,
                mt=mt,
                debug=debug,
                build_frontend=build_frontend,
            )

        if not install_spa(output_path, template_dir):
            warnings.warn(
                f"No built SPA found at {template_dir / 'dist'}; the output will be "
                "missing the web application.",
                category=RuntimeWarning,
            )
    else:
        warnings.warn(
            f"Template directory not found at {template_dir}.",
            category=RuntimeWarning,
        )

    # The SPA's own directory; `plugins.js` lives beside the bundle it extends.
    assets_dir = output_path / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Custom-MDP terms compile to the runtime ESM the manifest's `plugins` points at.
    if uses_custom_js() and client_builder is not None:
        print("Compiling custom-MDP term module (plugins.js)...")
        client_builder.build_plugins_module(assets_dir / "plugins.js")

    if mt:
        write_mt_headers(output_path)

    scene_entries: dict[tuple[str, str], dict] = {}
    max_name_len = max(len(p.name) for p in projects)
    for project in projects:
        # markup=False: these show the author's names, and rich would drop a "[...]".
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn(
                "{task.description}", style="progress.description", markup=False
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.fields[scene]}", style="dim", markup=False),
        ) as progress:
            task = progress.add_task(
                project.name.ljust(max_name_len),
                total=len(project.scenes),
                scene="",
            )
            project_dir = output_path / project.id
            project_dir.mkdir(parents=True, exist_ok=True)
            _write_license_files(
                project_dir,
                {
                    name: data
                    for name, data in (
                        ("LICENSE", project.license),
                        ("NOTICE", project.notice),
                    )
                    if data is not None
                },
            )
            licenses = license_names(project)
            if licenses:
                (project_dir / LICENSES_DIR).mkdir()
                _write_license_files(
                    project_dir / LICENSES_DIR,
                    {name: text for (_, text), name in licenses.items()},
                )
            steps = _SceneSteps(progress)
            for scene in project.scenes:
                progress.update(task, scene=scene.name)
                scene_dir = project_dir / scene.id
                scene_dir.mkdir(parents=True, exist_ok=True)
                scene_path = scene_dir / scene.scene_filename

                # First: the conversion is what creates this scene's policies (and
                # hands over its export env as the trace env).
                for pending in scene.pending_conversions:
                    steps.begin("converting checkpoints", total=len(pending.run_paths))
                    pending.run(steps.on)
                scene.pending_conversions.clear()

                _validate_muscle_action_terms(scene)

                steps.begin("building scene", total=3)
                steps.on("packaging scene")
                if scene.spec is not None:
                    graphics = graphics_entry(scene.viewer, scene.spec)
                    if scene.asset_roots is not None:
                        outside = []
                    elif scene.outside_files is not None:
                        outside = scene.outside_files
                    else:
                        outside = files_outside(scene.spec)
                    try:
                        to_zip_deflated(
                            scene.spec, str(scene_path), roots=scene.asset_roots
                        )
                    except (ValueError, FileNotFoundError) as e:
                        e.add_note(f"Packing scene {scene.name!r} of {project.name!r}.")
                        raise
                    if outside:
                        warnings.warn(
                            _outside_message(scene.name, outside),
                            category=RuntimeWarning,
                            stacklevel=2,
                        )
                    scene.spec = None
                else:
                    if scene.model is None:
                        raise RuntimeError(
                            f"Scene '{scene.name}' has no model to save as .mjb"
                        )
                    graphics = graphics_entry(scene.viewer, scene.model)
                    mujoco.mj_saveModel(scene.model, str(scene_path))
                    scene.model = None
                gc.collect()

                # Before anything needing the trace env: a tracking task's env loads
                # its clip from disk, and the bundled copy is that file.
                steps.on("bundling clips")
                scene_assets_dir = scene_dir / "assets"
                scene_assets_dir.mkdir(exist_ok=True)
                motion_files = write_scene_motions(scene, scene_assets_dir)
                point_env_cfg_at_bundled_motion(scene, scene_assets_dir, motion_files)

                steps.on("copying splats")
                write_scene_splats(scene, scene_assets_dir)
                write_scene_skins(scene, scene_assets_dir)
                if not any(scene_assets_dir.iterdir()):
                    scene_assets_dir.rmdir()

                if scene.events and not scene.policies and scene.events_explicit:
                    warnings.warn(
                        f"Scene {scene.name!r} has events but no policy. Events "
                        "belong to a policy's MDP, so with no policy "
                        "to carry them they are not written.",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )

                steps.begin("tracing mdps", total=len(scene.mdps) + len(scene.policies))
                sidecars = {p.id: load_sidecar(p) for p in scene.policies}
                mdp_entries = []
                for mdp, mdp_id in zip(scene.mdps, scene.mdp_ids):
                    owners = [p for p in scene.policies if p.mdp is mdp]
                    steps.on(f"mdp/{mdp_id}")
                    env = _scene_trace_env(scene)
                    with mdp_episode_length(env, mdp.episode_length_s):
                        mdp_entries.append(
                            mdp_entry(
                                mdp,
                                mdp_id,
                                owners,
                                sidecars=sidecars,
                                env=env,
                                scene_dir=scene_dir,
                                on_term=lambda name, mdp_id=mdp_id: steps.on(
                                    f"mdp/{mdp_id}/{name}"
                                ),
                            )
                        )

                policy_entries = []
                if scene.policies:
                    (scene_dir / "policy").mkdir(exist_ok=True)
                mdp_entry_by_id = {entry["id"]: entry for entry in mdp_entries}
                for policy in scene.policies:
                    steps.on(f"policy/{policy.name}")
                    onnx.save(
                        policy.model,
                        str(scene_dir / "policy" / f"{policy.id}.onnx"),
                    )
                    policy_entries.append(
                        policy_entry(
                            policy,
                            mdp_entry_by_id[scene.mdp_id(policy.mdp)],
                            sidecars[policy.id],
                            motion_files,
                        )
                    )

                scene_entries[(project.id, scene.id)] = scene_entry(
                    scene, mdp_entries, policy_entries, licenses, graphics
                )
                progress.advance(task)
            steps.close()

    write_manifest(output_path, projects, scene_entries)


def write_mt_headers(output_path: Path) -> None:
    """Write the COOP/COEP ``_headers`` multi-threaded MuJoCo needs (SharedArrayBuffer).

    Netlify, Cloudflare Pages and Vercel honor ``_headers``. GitHub Pages cannot set
    headers, so there the ``coi-serviceworker.js`` the Vite build emits for mt=True
    stands in.
    """
    headers_content = (
        "/*\n"
        "  Cross-Origin-Opener-Policy: same-origin\n"
        "  Cross-Origin-Embedder-Policy: require-corp\n"
        "\n"
    )
    (output_path / "_headers").write_text(headers_content)


class _SceneSteps:
    """The sub-step line under a project's progress bar: one phase at a time, counting
    its own units and naming the one it is on."""

    def __init__(self, progress: Progress):
        self._progress = progress
        self._task = progress.add_task("", total=1, scene="")
        self._started = False

    def begin(self, label: str, total: int) -> None:
        self._started = False
        # `total=None` means "keep the current total" to rich, so it is always passed.
        self._progress.reset(
            self._task, total=total, description=f"   └ {label}", scene=""
        )

    def on(self, name: str) -> None:
        """Name the unit now starting; the previous one counts as done."""
        self._progress.update(self._task, scene=name, advance=1 if self._started else 0)
        self._started = True

    def close(self) -> None:
        self._progress.remove_task(self._task)


def _outside_message(scene: str, files: list[str]) -> str:
    """The warning for files packed from outside a scene's model folder."""
    folders: dict[str, int] = {}
    for file in sorted(set(files)):
        folder = str(Path(file).parent)
        folders[folder] = folders.get(folder, 0) + 1
    listed = "; ".join(f"{n} in {folder}" for folder, n in list(folders.items())[:3])
    if len(folders) > 3:
        listed += f"; and files in {len(folders) - 3} more folders"
    return (
        f"Scene {scene!r} packs files from outside its model folder: {listed}. A model "
        "can name any file on disk, so check that these belong in the build, or pass "
        "asset_root= to keep it to the folders you choose."
    )


def _check_defaults(projects: list[ProjectConfig]) -> None:
    """Refuse two siblings both marked default.

    None set is fine: the first in document order is then the default.
    """
    defaults = [p.name for p in projects if p.default]
    if len(defaults) > 1:
        raise ValueError(
            f"Projects {defaults!r} are all marked default=True; at most one may be."
        )
    for project in projects:
        for scene in project.scenes:
            names = [p.name for p in scene.policies if p.default]
            if len(names) > 1:
                raise ValueError(
                    f"Scene {scene.name!r} has policies {names!r} all marked "
                    "default=True; at most one may be."
                )


def _validate_muscle_action_terms(scene: SceneConfig) -> None:
    """Check every ``MuscleActivationActionCfg`` names muscle actuators of the model.

    Raises ``ValueError`` on the first violation: at build time, not in the browser.
    """
    muscle_terms: list[tuple[str, MuscleActivationActionCfg]] = []
    for policy in scene.policies:
        actions = getattr(policy, "actions", None) or {}
        for term_name, cfg in actions.items():
            if isinstance(cfg, MuscleActivationActionCfg):
                muscle_terms.append((term_name, cfg))
    if not muscle_terms:
        return

    model = scene.model
    if model is None and scene.spec is not None:
        model = scene.spec.compile()
    if model is None:
        return

    for term_name, cfg in muscle_terms:
        validate_muscle_actuators(model, cfg, term_name=term_name)


def _scene_trace_env(scene: SceneConfig) -> Any | None:
    """The env ONNX tracing runs term bodies against, built on first use.

    Deferred to build time so a tracking task's env is constructed only once its clip is
    on disk, and so a scene that traces nothing never pays for one at all.
    """
    if scene.mjlab_env is not None:
        return scene.mjlab_env
    if scene.mjlab_env_cfg is None:
        return None
    from ..mjlab.env import build_mjlab_env

    scene.mjlab_env = build_mjlab_env(scene.mjlab_env_cfg)
    scene.mjlab_env.reset()
    return scene.mjlab_env


def _write_license_files(directory: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (directory / name).write_bytes(data)


__all__ = ["write_app", "write_mt_headers"]
