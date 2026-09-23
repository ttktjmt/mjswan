"""What ``Builder.build()`` does: write the engine and the expanded document as one app.

Output structure (ADR 0006 §2, license files ADR 0007 §1)::

    dist/
    ├── index.html, logo.svg, robots.txt
    ├── LICENSE            (the engine's, from the SPA; never the work's)
    ├── assets/            (compiled js/css/wasm, plugins.js)
    ├── manifest.json      (the one descriptor; every key snake_case)
    └── <project-id>/
        ├── LICENSE, NOTICE            (the work's, when declared)
        └── <scene-id>/
            ├── scene.mjz | scene.mjb
            ├── mdp/<mdp-id>/{obs,term,command,event}/<name>.onnx
            ├── policy/<policy-id>.onnx
            ├── assets/    (<motion>.npz, <splat>.spz)
            └── LICENSE.<component>, NOTICE.<component>  (third-party)
"""

from __future__ import annotations

import gc
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

from ..envs.mdp.actions.actions import (
    MuscleActivationActionCfg,
    validate_muscle_actuators,
)
from .asset import (
    point_env_cfg_at_bundled_motion,
    write_scene_motions,
    write_scene_splats,
)
from .frontend import TEMPLATE_DIR, ClientBuilder, install_spa, uses_custom_js
from .manifest import load_sidecar, mdp_entry, policy_entry, scene_entry, write_manifest
from .mjz import collect_spec_assets, to_zip_deflated

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

    ``output_path`` is wiped first. ``base_path``, ``gtm_id``, ``mt`` and ``debug`` are
    the frontend build's options (see :meth:`.frontend.ClientBuilder.build`);
    ``build_frontend`` forces or skips the Node build.
    """
    _check_defaults(projects)
    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

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
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[scene]}"),
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
            steps = _SceneSteps(progress)
            for scene in project.scenes:
                progress.update(task, scene=scene.name)
                scene_dir = project_dir / scene.id
                scene_dir.mkdir(parents=True, exist_ok=True)
                scene_path = scene_dir / scene.scene_filename
                for attribution in scene.attributions:
                    _write_license_files(scene_dir, attribution.files())

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
                    scene.spec.assets.update(collect_spec_assets(scene.spec))
                    to_zip_deflated(scene.spec, str(scene_path))
                    scene.spec = None
                else:
                    if scene.model is None:
                        raise RuntimeError(
                            f"Scene '{scene.name}' has no model to save as .mjb"
                        )
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
                if not any(scene_assets_dir.iterdir()):
                    scene_assets_dir.rmdir()

                if scene.events and not scene.policies and scene.events_explicit:
                    warnings.warn(
                        f"Scene {scene.name!r} has events but no policy. Events "
                        "belong to a policy's MDP (ADR 0006 §3), so with no policy "
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
                    mdp_entries.append(
                        mdp_entry(
                            mdp,
                            mdp_id,
                            owners,
                            sidecars=sidecars,
                            env=_scene_trace_env(scene),
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
                    mdp_id = scene.mdp_id(policy.mdp)
                    policy_entries.append(
                        policy_entry(
                            policy,
                            mdp_id,
                            sidecars[policy.id],
                            motion_files,
                            obs_keys=list(
                                mdp_entry_by_id[mdp_id].get("observations") or {}
                            ),
                        )
                    )

                scene_entries[(project.id, scene.id)] = scene_entry(
                    scene, mdp_entries, policy_entries
                )
                progress.advance(task)
            steps.close()

    write_manifest(output_path, projects, scene_entries)

    print(f"✓ Saved mjswan application to: {output_path}")


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


def _check_defaults(projects: list[ProjectConfig]) -> None:
    """Refuse two siblings both marked default (ADR 0006, manifest rule 3).

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
