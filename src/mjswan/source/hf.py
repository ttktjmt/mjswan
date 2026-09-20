"""The Hugging Face Hub as a source: a repository id in, files out.

A W&B run holds training state, so its ``model_*.pt`` checkpoints (:mod:`.wandb`) go
through a live mjlab env and torch to become ONNX (:mod:`mjswan.mjlab.runner`). A Hub
repository holds the published artifact instead, so this module only downloads and
loads — no mjlab, no torch. ``huggingface_hub`` is its one dependency, and it is
optional: nothing here is imported until a caller asks for the Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx

#: Tried in order when a caller names no file. These are the two names Hub repositories
#: publishing mjlab policies converged on, and the ones the Hub's own mjlab download
#: query counts at the repository root (huggingface/huggingface.js#2473).
DEFAULT_POLICY_FILENAMES = ("policy.onnx", "final.onnx")

#: Stems that name the file's role rather than the policy. A repository whose policy is
#: called one of these is better labelled by the repository itself.
_GENERIC_STEMS = frozenset({"policy", "final", "model", "actor"})


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
    there is. Several unnamed candidates raise rather than pick one: picking wrong is
    silent — the wrong policy loads and the robot merely misbehaves.
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


def policy_name_for(repo_id: str, filename: str) -> str:
    """The display name a fetched policy gets: its stem, or the repository's own name.

    ``policy.onnx`` names the file's role, not the policy, so a repository whose file is
    called that is labelled by its own last path segment instead.
    """
    stem = Path(filename).stem
    if stem.lower() in _GENERIC_STEMS:
        return repo_id.rsplit("/", 1)[-1] or stem
    return stem


def fetch_file(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> Path:
    """Download one file and return its local path.

    ``huggingface_hub`` caches under ``~/.cache/huggingface``, so a repeated build of
    the same revision re-uses the download rather than fetching it again.
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

    ``repo_type`` defaults to ``"dataset"``: a clip is data, and the retargeted sets
    published so far are dataset repositories. Returns ``(motion_name, payload)``, the
    name being the file's stem.
    """
    local_path = fetch_file(
        repo_id, filename, revision=revision, repo_type=repo_type, token=token
    )
    return Path(filename).stem, local_path.read_bytes()


__all__ = [
    "DEFAULT_POLICY_FILENAMES",
    "choose_policy_filename",
    "fetch_file",
    "fetch_motion_npz",
    "fetch_onnx",
    "list_repo_onnx",
    "policy_name_for",
    "resolve_policy_filename",
]
