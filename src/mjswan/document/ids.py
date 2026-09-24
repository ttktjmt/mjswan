"""Identifiers in the document: sanitized names, unique within the level that owns the
directory (ADR 0006 §4)."""

from __future__ import annotations

import re
import warnings
from collections.abc import Collection


def name2id(name: str) -> str:
    """Convert a name to a URL-friendly identifier.

    After lowercasing, every run of characters outside ``[a-z0-9]`` becomes one
    underscore, so the id is safe as a storage object key: an unescaped ``'`` in an
    R2/S3 key breaks the presigned SigV4 upload signature.

    Examples:
        >>> name2id("My Project")
        'my_project'
        >>> name2id("Test-Scene")
        'test_scene'
        >>> name2id("Newton's Cradle")
        'newton_s_cradle'
    """
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def unique_id(base: str, taken: Collection[str]) -> str:
    """``base`` if free, else the first of ``base_1``, ``base_2``, … not in ``taken``.

    ``taken`` is the level that owns the directory (ADR 0006 §4): a project id is unique
    in the document, a scene id in its project, a policy/mdp/asset id in its scene. Two
    siblings that sanitize to one id are both kept, the second renamed rather than
    refused, since two scenes called "Flat Terrain" is a reasonable thing to write.
    """
    if base not in taken:
        return base
    n = 1
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _base_id(name: str, kind: str) -> str:
    """``name2id(name)``, refused when empty: it would have no directory and no URL."""
    base = name2id(name)
    if not base:
        raise ValueError(
            f"{kind.capitalize()} name {name!r} has no ASCII letter or digit, so it "
            "sanitizes to an empty id. Give it a name with at least one."
        )
    return base


def assign_id(
    name: str, taken: Collection[str], *, kind: str, stacklevel: int = 3
) -> str:
    """``name2id(name)``, made unique among ``taken``; warns when it had to be suffixed.

    ``kind`` names the level in the warning. For an object the viewer lists, use
    :func:`assign_name`, which renames the name along with the id.
    """
    base = _base_id(name, kind)
    ident = unique_id(base, taken)
    if ident != base:
        warnings.warn(
            f"{kind.capitalize()} {name!r} sanitizes to {base!r}, which another {kind} "
            f"already uses; it is stored as {ident!r} instead.",
            category=RuntimeWarning,
            stacklevel=stacklevel,
        )
    return ident


def assign_name(
    name: str, taken: Collection[str], *, kind: str, stacklevel: int = 3
) -> tuple[str, str]:
    """``(name, id)`` for a new sibling: the id is ``name2id(name)``, and both are unique.

    A name whose id a sibling in ``taken`` already holds takes the same ``_1``, ``_2``, …
    suffix as its id, with a warning. So the viewer never lists two entries alike, and
    each entry's name still sanitizes to its ``?scene=`` / ``?policy=`` value.
    """
    base = _base_id(name, kind)
    ident = unique_id(base, taken)
    if ident == base:
        return name, ident
    renamed = name + ident[len(base) :]
    warnings.warn(
        f"{kind.capitalize()} {name!r} sanitizes to {base!r}, which another {kind} "
        f"already uses, so it is renamed {renamed!r} (id {ident!r}).",
        category=RuntimeWarning,
        stacklevel=stacklevel,
    )
    return renamed, ident


__all__ = ["assign_id", "assign_name", "name2id", "unique_id"]
