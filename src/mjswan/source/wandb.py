"""Weights & Biases as a source: a run path or an artifact path in, files out.

A run's ``model_*.pt`` checkpoints are training state; turning one into ONNX is
:func:`mjswan.mjlab.runner.export_checkpoint`'s business, not this module's. A run's
latest ``.onnx`` and the motion clip it used or logged come back as they are.
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import onnx


def resolve_run_path(
    *,
    run_path: str | None = None,
    run_id: str | None = None,
    entity: str | None = None,
    project: str | None = None,
) -> str:
    """Resolve either a fully qualified run path or ``run_id`` shorthand."""
    if run_path:
        return run_path
    if run_id and entity and project:
        return f"{entity}/{project}/{run_id}"
    raise ValueError(
        "Provide either run_path='entity/project/run_id' or "
        "run_id together with entity and project."
    )


def resolve_artifact_path(
    artifact_path: str,
) -> tuple[str, str, str]:
    """Resolve a W&B artifact reference to ``(artifact_name, type, file_path)``.

    Accepts either:
    - a fully-qualified artifact name like ``entity/project/name:v0``
    - a W&B artifact URL like
      ``https://wandb.ai/entity/project/artifacts/motions/name/v0/files/motion.npz``
    """
    parsed = urlparse(artifact_path)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.strip("/")
        match = re.match(
            r"^(?P<entity>[^/]+)/(?P<project>[^/]+)/artifacts/"
            r"(?P<artifact_type>[^/]+)/(?P<name>[^/]+)/(?P<version>[^/]+)"
            r"(?:/files/(?P<file_path>.+))?$",
            path,
        )
        if match is None:
            raise ValueError(
                "Unsupported W&B artifact URL format. "
                "Expected something like "
                "'https://wandb.ai/entity/project/artifacts/motions/name/v0/files/motion.npz'."
            )
        artifact_name = (
            f"{match.group('entity')}/{match.group('project')}/"
            f"{match.group('name')}:{match.group('version')}"
        )
        artifact_type = match.group("artifact_type")
        file_path = match.group("file_path") or "motion.npz"
        return artifact_name, artifact_type, file_path

    return artifact_path, "motions", "motion.npz"


def fetch_onnx(run_path: str) -> tuple[str, onnx.ModelProto]:
    """Download the latest ONNX policy file from a W&B run.

    Finds the most recently updated ``.onnx`` file attached to the run and
    loads it into memory as an :class:`onnx.ModelProto`.  The policy name is
    the filename with its extension removed (e.g. ``"2026-02-25_04-30-08.onnx"``
    becomes ``"2026-02-25_04-30-08"``).

    Args:
        run_path: W&B run path in the format ``"entity/project/run_id"``.

    Returns:
        A ``(policy_name, onnx_model)`` tuple for the latest ``.onnx`` file.

    Raises:
        ValueError: If no ``.onnx`` files are found in the run.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]

    if not onnx_files:
        raise ValueError(f"No .onnx files found in W&B run: {run_path}")

    latest = max(onnx_files, key=lambda f: f.updated_at)

    with tempfile.TemporaryDirectory() as tmp_dir:
        latest.download(root=tmp_dir, replace=True)
        local_path = Path(tmp_dir) / latest.name
        name = local_path.stem
        model = onnx.load(str(local_path))

    return name, model


def fetch_motion_npz(run_path: str) -> tuple[str, bytes]:
    """Download the ``motion.npz`` artifact used or logged by a W&B run."""
    import wandb

    api = wandb.Api()
    run = api.run(run_path)
    artifact = next((a for a in run.used_artifacts() if a.type == "motions"), None)
    if artifact is None:
        raise ValueError(f"No motion artifact found in W&B run: {run_path}")

    artifact_name = artifact.name.split("/")[-1]
    motion_name = artifact_name.split(":", 1)[0] or "motion"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(artifact.download(root=tmp_dir))
        motion_path = root / "motion.npz"
        if not motion_path.exists():
            raise ValueError(
                f"Motion artifact for run '{run_path}' did not contain motion.npz"
            )
        return motion_name, motion_path.read_bytes()


def fetch_motion_npz_from_artifact(
    artifact_path: str,
) -> tuple[str, bytes]:
    """Download ``motion.npz`` directly from a W&B motion artifact."""
    import wandb

    artifact_name, artifact_type, file_path = resolve_artifact_path(artifact_path)

    api = wandb.Api()
    artifact = api.artifact(artifact_name, type=artifact_type)
    motion_name = artifact_name.split("/")[-1].split(":", 1)[0] or "motion"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(artifact.download(root=tmp_dir))
        motion_path = root / file_path
        if not motion_path.exists():
            raise ValueError(
                f"Motion artifact '{artifact_name}' did not contain '{file_path}'"
            )
        return motion_name, motion_path.read_bytes()


@contextlib.contextmanager
def fetch_checkpoints(run_path: str) -> Iterator[list[tuple[str, Path]]]:
    """Every ``model_*.pt`` of a run as ``(name, path)`` pairs, sorted by training step.

    The files sit in a temporary directory that lives for the ``with`` block. Sorted so
    the caller sees ``model_0``, ``model_50``, ``model_100``, … in that order.

    Raises:
        ValueError: If the run holds no ``model_*.pt`` file.
    """
    import wandb

    api = wandb.Api()
    run = api.run(run_path)

    pt_files = [f for f in run.files() if re.match(r"^model_\d+\.pt$", f.name)]
    if not pt_files:
        raise ValueError(f"No model_*.pt files found in W&B run: {run_path}")
    pt_files.sort(key=lambda f: int(re.search(r"\d+", f.name).group()))  # type: ignore[union-attr]

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoints: list[tuple[str, Path]] = []
        for wandb_file in pt_files:
            wandb_file.download(root=tmp_dir, replace=True)
            pt_path = Path(tmp_dir) / wandb_file.name
            checkpoints.append((pt_path.stem, pt_path))
        yield checkpoints


__all__ = [
    "fetch_checkpoints",
    "fetch_motion_npz",
    "fetch_motion_npz_from_artifact",
    "fetch_onnx",
    "resolve_artifact_path",
    "resolve_run_path",
]
