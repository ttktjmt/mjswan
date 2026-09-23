"""Where a license file may sit in a build, and how big it may be (ADR 0007 §1).

Files sit under a project directory, never at the build root, whose ``LICENSE`` is the
engine's:

- ``<project-id>/LICENSE`` and ``NOTICE``: the work's;
- ``<project-id>/<scene-id>/LICENSE.<component>`` / ``NOTICE.<component>``: one
  third-party component the scene contains (a bare scene-level ``LICENSE`` is labelled
  with the scene id).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

#: The platform refuses a license or notice file larger than this.
LICENSE_FILE_MAX_BYTES: int = 64 * 1024

LICENSE_CONTENT_TYPE: str = "text/plain; charset=utf-8"

_BASENAME = re.compile(r"^(LICENSE|NOTICE)(?:\.([A-Za-z0-9_-]{1,64}))?$")
_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

Kind = Literal["LICENSE", "NOTICE"]
Scope = Literal["project", "scene"]


@dataclass(frozen=True)
class LicenseLocation:
    """Where a build-relative path puts a license file."""

    kind: Kind
    scope: Scope
    project: str
    scene: str | None = None
    #: The component the file covers: its suffix, or the scene id when it has none.
    component: str | None = None


def parse_license_path(path: str) -> LicenseLocation | None:
    """The location a build-relative POSIX path names, or ``None`` when it is not a
    license file where the platform reads one (:func:`license_path_problem` says why)."""
    segments = path.split("/")
    m = _BASENAME.match(segments[-1])
    if m is None:
        return None
    kind: Kind = "LICENSE" if m.group(1) == "LICENSE" else "NOTICE"
    suffix = m.group(2)
    if len(segments) == 2 and suffix is None:
        return LicenseLocation(kind=kind, scope="project", project=segments[0])
    if len(segments) == 3:
        scene = segments[1]
        return LicenseLocation(
            kind=kind,
            scope="scene",
            project=segments[0],
            scene=scene,
            component=suffix or scene,
        )
    return None


def has_license_basename(path: str) -> bool:
    """Whether the basename is a license file's, wherever it sits."""
    return _BASENAME.match(path.rsplit("/", maxsplit=1)[-1]) is not None


def license_path_problem(path: str) -> str | None:
    """Why a path with a license basename is refused, or ``None`` when it is fine."""
    if not has_license_basename(path) or parse_license_path(path) is not None:
        return None
    depth = len(path.split("/"))
    if depth == 1:
        return (
            "LICENSE and NOTICE belong in the project directory, not at the build "
            "root, which holds the engine's own LICENSE"
        )
    if depth == 2:
        return "a project-level license file is LICENSE or NOTICE, without a component suffix"
    return (
        "license files sit in the project directory or in a scene directory, not deeper"
    )


def is_license_file_path(path: str) -> bool:
    """Whether ``path`` is a license file where the platform reads one."""
    return parse_license_path(path) is not None


def component_id(name: str) -> str:
    """``name`` made fit for a ``LICENSE.<component>`` suffix."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:64]
    return cleaned or "component"


__all__ = [
    "LICENSE_CONTENT_TYPE",
    "LICENSE_FILE_MAX_BYTES",
    "Kind",
    "LicenseLocation",
    "Scope",
    "component_id",
    "has_license_basename",
    "is_license_file_path",
    "license_path_problem",
    "parse_license_path",
]
