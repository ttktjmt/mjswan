"""The ``.swn`` simulation document: a build's data as one file (ADR 0006 §8).

A build is a directory, ``manifest.json`` over ``<project-id>/<scene-id>/``, and that
directory is the document. Packaging it is a ZIP whose entries are the tree's paths, so
unpacking one gives back exactly what a build wrote and nothing is described twice.

In go the manifest and every file under a project directory. The engine
(``index.html``, ``assets/*.js``, WASM) and the author's custom-term module do not: an
app is the engine plus the expanded tree, and mjswan Cloud supplies its own engine.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

DOCUMENT_SUFFIX = ".swn"
MANIFEST_NAME = "manifest.json"

#: The structure a build writes: what files exist, where they sit, and what the manifest
#: says about them. Bumped by hand, only when an engine reading the old structure would
#: misread the new one. Distinct from ``version``, the mjswan release: a host picks an
#: engine by ``version``, an engine protects itself by ``format`` (ADR 0006 §7).
#:
#: 2: ``input_slots`` may carry a raw ``mjData`` field (``{"sim": ...}``), narrowed to
#: ``rows``. A format-1 engine accepts the entry and then cannot serve it, freezing the
#: observation group at its previous value — exactly the misread the number exists for.
DOCUMENT_FORMAT = 2

#: Already-compressed containers: deflating them again costs time for nothing.
_STORED_SUFFIXES = frozenset({".mjz", ".npz", ".spz"})


class DocumentError(ValueError):
    """A ``.swn`` that cannot be read as a simulation document."""


def _not_a_document(source: Path, why: str) -> DocumentError:
    return DocumentError(f"{source} is not a simulation document: {why}.")


def is_document(path: str | Path) -> bool:
    """Whether ``path`` is a ``.swn`` file (as opposed to a built directory)."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() == DOCUMENT_SUFFIX


def read_manifest(source: str | Path) -> dict:
    """The manifest of a built directory or a ``.swn`` document."""
    source = Path(source)
    if is_document(source):
        # Same refusals as unpacking one, so a caller that wants only the manifest gets
        # "not a document" rather than a zipfile error.
        try:
            with zipfile.ZipFile(source) as zf:
                return json.loads(zf.read(MANIFEST_NAME))
        except zipfile.BadZipFile as exc:
            raise _not_a_document(source, "not a ZIP archive") from exc
        except KeyError as exc:
            raise _not_a_document(source, f"no {MANIFEST_NAME}") from exc
    return json.loads((source / MANIFEST_NAME).read_text())


def document_files(dist_dir: str | Path) -> list[Path]:
    """The files a document consists of, relative to ``dist_dir``, manifest first.

    Everything under a project directory belongs to the document, nothing else does. The
    project directories come from the manifest rather than from what is on disk, so the
    SPA's own ``assets/`` can never be mistaken for one.
    """
    dist_dir = Path(dist_dir)
    if not (dist_dir / MANIFEST_NAME).is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {dist_dir}; pass the directory builder.build() wrote."
        )
    manifest = read_manifest(dist_dir)
    files = [Path(MANIFEST_NAME)]
    for project in manifest.get("projects", []):
        project_dir = dist_dir / project["id"]
        if not project_dir.is_dir():
            continue
        files.extend(
            sorted(
                p.relative_to(dist_dir) for p in project_dir.rglob("*") if p.is_file()
            )
        )
    return files


def write_document(dist_dir: str | Path, target: str | Path | None = None) -> Path:
    """Package a built directory as ``<target>.swn`` and return the path written.

    ``target`` defaults to the directory's own name with the ``.swn`` suffix, beside it.
    The manifest is the first entry, so a reader that stops after one file has the one
    that describes the rest; the others are sorted, for a reproducible archive.
    """
    dist_dir = Path(dist_dir)
    if target is None:
        target = dist_dir.with_suffix(DOCUMENT_SUFFIX)
    target = Path(target)
    if target.suffix.lower() != DOCUMENT_SUFFIX:
        target = target.with_suffix(DOCUMENT_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as zf:
        for rel in document_files(dist_dir):
            compress = (
                zipfile.ZIP_STORED
                if rel.suffix.lower() in _STORED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            zf.write(dist_dir / rel, rel.as_posix(), compress_type=compress)
    return target


def unpack_document(document: str | Path, target_dir: str | Path) -> Path:
    """Expand a ``.swn`` into ``target_dir``, the tree the build wrote, and return it.

    Entries are confined to the target: an archive naming ``../x`` or an absolute path is
    refused rather than written outside it.
    """
    document = Path(document)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(document)
    except zipfile.BadZipFile as exc:
        raise _not_a_document(document, "not a ZIP archive") from exc
    with zf:
        names = zf.namelist()
        if MANIFEST_NAME not in names:
            raise _not_a_document(document, f"no {MANIFEST_NAME}")
        # `extractall` confines entries itself, by dropping the parts that would escape.
        # Checked first anyway: an entry that needs dropping means a tampered document,
        # better refused than unpacked as a silently mangled tree.
        for name in names:
            rel = PurePosixPath(name)
            if rel.is_absolute() or ".." in rel.parts:
                raise DocumentError(
                    f"{document} names {name!r}, which would land outside the "
                    "directory it is unpacked into."
                )
        zf.extractall(target_dir)
    return target_dir


@contextlib.contextmanager
def as_directory(source: str | Path) -> Iterator[Path]:
    """``source`` itself when it is a built directory; a ``.swn`` unpacked to a temp dir.

    Lets a consumer that walks the tree (``publish``, ``mjswan info``) take either form
    with one code path. The temporary directory is removed on exit.
    """
    source = Path(source)
    if is_document(source):
        with tempfile.TemporaryDirectory(prefix="mjswan-swn-") as tmp:
            yield unpack_document(source, Path(tmp))
    else:
        yield source


__all__ = [
    "DOCUMENT_FORMAT",
    "DOCUMENT_SUFFIX",
    "MANIFEST_NAME",
    "DocumentError",
    "as_directory",
    "document_files",
    "is_document",
    "read_manifest",
    "unpack_document",
    "write_document",
]
