"""The Hugging Face Hub as a source: a repository id in, files out.

A Hub repository holds exported artifacts, so this module only downloads and loads (no
mjlab, no torch). ``huggingface_hub`` is optional (the ``hf`` extra), imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx

#: Tried in order when a caller names no file: the names mjlab policy repositories use,
#: and the ones the Hub's mjlab download query counts at the repository root
#: (huggingface/huggingface.js#2473).
DEFAULT_POLICY_FILENAMES = ("policy.onnx", "final.onnx")

#: Stems that name the file's role rather than the policy, so the repository names it.
_GENERIC_STEMS = frozenset({"policy", "final", "model", "actor"})

#: The same for a scene, whose identity is its directory rather than its XML's name.
_GENERIC_SCENE_STEMS = frozenset({"scene", "model", "robot", "main"})

#: The same for a splat, which is one file, so the repository is what is left to name it.
_GENERIC_SPLAT_STEMS = frozenset({"splat", "background", "scene"})


def _hub() -> Any:
    """The ``huggingface_hub`` module, or an ImportError naming the extra to install."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to fetch assets from the Hugging Face Hub. "
            "Install it with: pip install 'mjswan[hf]'"
        ) from exc
    return huggingface_hub


def list_repo_onnx(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> list[str]:
    """Every ``.onnx`` file in the repository, as repository-root-relative paths."""
    api = _hub().HfApi(token=token)
    files = api.list_repo_files(repo_id, revision=revision, repo_type=repo_type)
    return sorted(name for name in files if name.endswith(".onnx"))


def resolve_policy_filename(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> str:
    """The one ``.onnx`` to take from a repository the caller named no file in.

    :data:`DEFAULT_POLICY_FILENAMES` first, then the single ``.onnx`` if that is all
    there is. Several candidates raise rather than pick one, since a wrong pick is
    silent: the wrong policy loads and the robot merely misbehaves.
    """
    candidates = list_repo_onnx(
        repo_id, revision=revision, repo_type=repo_type, token=token
    )
    return choose_policy_filename(candidates, repo_id=repo_id)


def choose_policy_filename(candidates: list[str], *, repo_id: str = "") -> str:
    """The name :func:`resolve_policy_filename` would pick out of ``candidates``."""
    where = f" in {repo_id!r}" if repo_id else ""
    if not candidates:
        raise ValueError(
            f"No .onnx file{where}. A Hub repository mjswan can load a policy from "
            "carries its exported ONNX; pass filename= if it lives under another "
            "extension."
        )
    for name in DEFAULT_POLICY_FILENAMES:
        if name in candidates:
            return name
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"{len(candidates)} .onnx files{where} and none is named "
        f"{' or '.join(DEFAULT_POLICY_FILENAMES)}: {candidates}. Pass filename= to say "
        "which one (or a list of them to add several)."
    )


def _named_by_file_or_repo(filename: str, generic: frozenset[str], repo_id: str) -> str:
    """A file's stem, or the repository's own name when the stem only names a role."""
    stem = Path(filename).stem
    if stem.lower() in generic:
        return repo_id.rsplit("/", 1)[-1] or stem
    return stem


#: Folders that hold policies rather than name one, skipped when naming by directory.
_POLICY_CONTAINER_DIRS = frozenset(
    {"checkpoints", "exported", "models", "onnx", "policies", "policy"}
)


def policy_name_for(repo_id: str, filename: str) -> str:
    """A fetched policy's display name: its stem, or for a generic stem its directory.

    ``policies/walk/policy.onnx`` is ``walk`` and ``run1/exported/policy.onnx`` is
    ``run1``: a folder that only holds policies is passed over. With none left, the
    repository's own name.
    """
    stem = Path(filename).stem
    if stem.lower() not in _GENERIC_STEMS:
        return stem
    for parent in Path(filename).parents:
        if parent.name and parent.name.lower() not in _POLICY_CONTAINER_DIRS:
            return parent.name
    return repo_id.rsplit("/", 1)[-1] or stem


def splat_name_for(repo_id: str, filename: str) -> str:
    """A fetched splat's display name, by the rule :func:`policy_name_for` uses."""
    return _named_by_file_or_repo(filename, _GENERIC_SPLAT_STEMS, repo_id)


def scene_name_for(repo_id: str, path: str) -> str:
    """A fetched scene's display name: its stem, or for a generic stem its directory.

    ``scenes/unitree_g1/scene.xml`` is ``unitree_g1``; at the repository root, the
    repository's name.
    """
    stem = Path(path).stem
    if stem.lower() not in _GENERIC_SCENE_STEMS:
        return stem
    parent = Path(path).parent.name
    return parent or repo_id.rsplit("/", 1)[-1] or stem


def fetch_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> Path:
    """Download one file and return its local path.

    ``huggingface_hub`` caches under ``~/.cache/huggingface``, so a rebuild of the same
    revision downloads nothing.
    """
    return Path(
        _hub().hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            repo_type=repo_type,
            token=token,
        )
    )


def fetch_dir(
    repo_id: str,
    path: str | None = None,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download a directory of the repository and return its local path.

    For a multi-file asset such as an MJCF and its meshes, which MuJoCo resolves
    relative to the XML and so must land beside it.

    Args:
        repo_id: Hub repository, ``"<owner>/<name>"``.
        path: Directory within the repository. ``None`` or ``"."`` takes the whole
            repository.
        revision: Branch, tag or commit. ``None`` takes the default branch.
        repo_type: ``"model"`` (default), ``"dataset"`` or ``"space"``.
        token: Hub token for a gated or private repository.
        allow_patterns: Repository-root-relative ``fnmatch`` patterns replacing the
            default of everything under ``path``. Pass them when the asset reaches
            outside its directory (an MJCF whose ``meshdir`` points at a shared folder),
            since nothing here parses the XML.

    Returns:
        The local directory for ``path``, inside the Hub cache.
    """
    if path in ("", "."):
        # The root, which as a pattern (`./*`) would match nothing.
        path = None
    if allow_patterns is None and path is not None:
        # `fnmatch`'s `*` crosses `/`, so one pattern takes the whole subtree.
        allow_patterns = [f"{path.rstrip('/')}/*"]
    root = Path(
        _hub().snapshot_download(
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            token=token,
            allow_patterns=allow_patterns,
        )
    )
    local = root / path if path else root
    if not local.is_dir():
        raise ValueError(
            f"{path!r} is not a directory in {repo_id!r} (nothing matched "
            f"{allow_patterns}). Pass allow_patterns= if the asset's files live "
            "elsewhere in the repository."
        )
    return local


def fetch_onnx(
    repo_id: str,
    filename: str | None = None,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> tuple[str, onnx.ModelProto]:
    """Download one ONNX policy from the Hub.

    Args:
        repo_id: Hub repository, ``"<owner>/<name>"``.
        filename: Path within the repository. ``None`` resolves it via
            :func:`resolve_policy_filename`.
        revision: Branch, tag or commit. ``None`` takes the repository's default branch,
            so a build pins nothing; pass a commit to make one reproducible.
        repo_type: ``"model"`` (default), ``"dataset"`` or ``"space"``.
        token: Hub token for a gated or private repository. ``None`` uses the locally
            stored login, then anonymous access.

    Returns:
        A ``(policy_name, onnx_model)`` tuple; see :func:`policy_name_for` for the name.
    """
    if filename is None:
        filename = resolve_policy_filename(
            repo_id, revision=revision, repo_type=repo_type, token=token
        )
    local_path = fetch_file(
        repo_id, filename, revision=revision, repo_type=repo_type, token=token
    )
    return policy_name_for(repo_id, filename), onnx.load(str(local_path))


def fetch_motion_npz(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    repo_type: str = "dataset",
    token: str | None = None,
) -> tuple[str, bytes]:
    """Download a ``.npz`` reference motion from the Hub.

    ``repo_type`` defaults to ``"dataset"``, where retargeted motion sets are published.
    Returns ``(motion_name, payload)``, the name being the file's stem.
    """
    local_path = fetch_file(
        repo_id, filename, revision=revision, repo_type=repo_type, token=token
    )
    return Path(filename).stem, local_path.read_bytes()


__all__ = [
    "DEFAULT_POLICY_FILENAMES",
    "choose_policy_filename",
    "fetch_dir",
    "fetch_file",
    "fetch_motion_npz",
    "fetch_onnx",
    "list_repo_onnx",
    "policy_name_for",
    "resolve_policy_filename",
    "scene_name_for",
    "splat_name_for",
]
