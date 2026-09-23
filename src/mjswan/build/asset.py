"""What lands in a scene's ``assets/``: the clips its policies carry, and its splats."""

from __future__ import annotations

import hashlib
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..document.ids import name2id, unique_id

if TYPE_CHECKING:
    from ..scene import SceneConfig


def motion_key(motion: Any) -> str:
    """Identity of a clip's content, so two policies sharing one share its file."""
    if motion.data is not None:
        return hashlib.sha256(motion.data).hexdigest()
    if motion.source is not None:
        src = _resolve_motion_source(motion.source)
        try:
            return hashlib.sha256(src.read_bytes()).hexdigest()
        except OSError:
            return f"src:{src}"  # missing; the copy below warns about it
    return f"empty:{motion.name}"


def _resolve_motion_source(source: str) -> Path:
    src = Path(source).expanduser()
    return src if src.is_absolute() else (Path.cwd() / src).resolve()


def write_scene_motions(scene: SceneConfig, scene_dir: Path) -> dict[str, str]:
    """Write each distinct clip in the scene once; return content key -> filename.

    Scene-scoped, since the checkpoints of one run share a clip. Equal content collapses
    to one file whatever its name; a name clash with different content gets a
    ``_1``/``_2`` suffix.
    """
    files: dict[str, str] = {}
    used: set[str] = set()
    for policy in scene.policies:
        for motion in policy.motions:
            key = motion_key(motion)
            if key in files:
                continue
            stem = unique_id(name2id(motion.name), used)
            used.add(stem)
            filename = f"{stem}.npz"
            files[key] = filename

            target = scene_dir / filename
            if motion.data is not None:
                target.write_bytes(motion.data)
            elif motion.source is not None:
                src = _resolve_motion_source(motion.source)
                if src.exists():
                    shutil.copy2(str(src), str(target))
                else:
                    warnings.warn(
                        f"Motion source file not found: {src}",
                        category=RuntimeWarning,
                        stacklevel=2,
                    )
    return files


def point_env_cfg_at_bundled_motion(
    scene: SceneConfig, scene_dir: Path, motion_files: dict[str, str]
) -> None:
    """Aim a tracking task's ``motion_file`` at the clip just written to the bundle.

    mjlab registers tracking tasks with ``motion_file=""`` and ``MotionLoader`` reads
    the path when the env is constructed, so the trace env loads the bundled copy.
    """
    env_cfg = scene.mjlab_env_cfg
    if env_cfg is None or scene.mjlab_env is not None or not motion_files:
        return
    for term in (getattr(env_cfg, "commands", None) or {}).values():
        if not hasattr(term, "motion_file"):
            continue
        existing = getattr(term, "motion_file", "") or ""
        if existing and Path(existing).expanduser().is_file():
            continue
        term.motion_file = str(scene_dir / next(iter(motion_files.values())))
        return


def write_scene_splats(scene: SceneConfig, scene_dir: Path) -> None:
    """Copy each file-backed splat to ``<scene_dir>/<id>.spz``; URL splats stay put."""
    for splat in scene.splats:
        if splat.source is None:
            continue
        src = Path(splat.source).expanduser()
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        if src.exists():
            shutil.copy2(str(src), str(scene_dir / f"{splat.id}.spz"))
        else:
            warnings.warn(
                f"Splat source file not found: {src}",
                category=RuntimeWarning,
                stacklevel=2,
            )


__all__ = [
    "motion_key",
    "point_env_cfg_at_bundled_motion",
    "write_scene_motions",
    "write_scene_splats",
]
