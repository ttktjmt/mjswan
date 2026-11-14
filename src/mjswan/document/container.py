"""The ``.swn`` container: a build's data as one file (ADR 0006 §8).

A ZIP whose entries are the build tree's own paths, so unpacking one gives back exactly
what the build wrote. It holds the manifest and every file under a project directory,
never the engine (``index.html``, ``assets/*.js``, WASM) or the author's custom-term
module: an app is the engine plus the expanded tree, and mjswan Cloud brings its own.
"""

from __future__ import annotations

import contextlib
import json
import re
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from .manifest import MANIFEST_NAME

DOCUMENT_SUFFIX = ".swn"

#: Already-compressed containers: deflating them again costs time for nothing.
_STORED_SUFFIXES = frozenset({".mjz", ".npz", ".spz"})

#: Every project id a build writes comes from ``name2id``.
_BUILT_ID = re.compile(r"[a-z0-9_]+")

#: Summed declared sizes bound the real ones: zipfile stops each entry at its own.
MAX_UNPACKED_BYTES = 2 * 2**30
MAX_ENTRIES = 20_000
#: json.loads builds about 25 times a manifest's bytes in objects; real ones are a few MB.
MAX_MANIFEST_BYTES = 16 * 2**20
#: The methods ``write_document`` uses: bzip2 and LZMA inflate a whole read before
#: zipfile cuts it at the entry's size.
_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


class DocumentError(ValueError):
    """A ``.swn`` that cannot be read as a simulation document."""


def _not_a_document(source: Path, why: str) -> DocumentError:
    return DocumentError(f"{source} is not a simulation document: {why}.")


def _symlinked(path: Path) -> DocumentError:
    return DocumentError(
        f"{path} is a symlink; a document holds only the build's own files."
    )


def _open(document: Path) -> zipfile.ZipFile:
    """``document`` opened, if it is one a build could have written."""
    try:
        zf = zipfile.ZipFile(document)
    except zipfile.BadZipFile as exc:
        raise _not_a_document(document, "not a ZIP archive") from exc
    infos = zf.infolist()
    if MANIFEST_NAME not in zf.namelist():
        why = f"no {MANIFEST_NAME}"
    elif (
        len(infos) > MAX_ENTRIES or sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES
    ):
        why = f"it unpacks to over {MAX_ENTRIES} files or {MAX_UNPACKED_BYTES} bytes"
    elif any(i.compress_type not in _METHODS for i in infos):
        why = "it uses a compression no build writes"
    else:
        return zf
    zf.close()
    raise _not_a_document(document, why)


def is_document(path: str | Path) -> bool:
    """Whether ``path`` is a ``.swn`` file (as opposed to a built directory)."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() == DOCUMENT_SUFFIX


def read_manifest(source: str | Path) -> dict:
    """The manifest of a built directory or a ``.swn`` document."""
    source = Path(source)
    # read(n) inflates at most n bytes; read() inflates up to 2**30 before cutting.
    if is_document(source):
        with _open(source) as zf, zf.open(MANIFEST_NAME) as f:
            data = f.read(MAX_MANIFEST_BYTES + 1)
    else:
        with open(source / MANIFEST_NAME, "rb") as f:
            data = f.read(MAX_MANIFEST_BYTES + 1)
    if len(data) > MAX_MANIFEST_BYTES:
        raise _not_a_document(
            source, f"its {MANIFEST_NAME} is over {MAX_MANIFEST_BYTES} bytes"
        )
    return json.loads(data)


def document_files(dist_dir: str | Path) -> list[Path]:
    """The files a document consists of, relative to ``dist_dir``, manifest first.

    Everything under a project directory belongs to the document, nothing else does. The
    project directories come from the manifest rather than from what is on disk, so the
    SPA's own ``assets/`` can never be mistaken for one.
    """
    dist_dir = Path(dist_dir)
    manifest_path = dist_dir / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise _symlinked(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {dist_dir}; pass the directory builder.build() wrote."
        )
    manifest = read_manifest(dist_dir)
    files = [Path(MANIFEST_NAME)]
    for project in manifest.get("projects", []):
        project_id = str(project["id"])
        if not _BUILT_ID.fullmatch(project_id):
            raise DocumentError(f"Project id {project_id!r} is not one a build writes.")
        project_dir = dist_dir / project_id
        if project_dir.is_symlink():
            raise _symlinked(project_dir)
        if not project_dir.is_dir():
            continue
        # rglob yields a symlinked directory but does not descend into it.
        for p in sorted(project_dir.rglob("*")):
            if p.is_symlink():
                raise _symlinked(p)
            if p.is_file():
                files.append(p.relative_to(dist_dir))
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
    # Listed before the archive exists, so a refusal leaves no partial .swn behind.
    files = document_files(dist_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as zf:
        for rel in files:
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
    with _open(document) as zf:
        names = zf.namelist()
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
    "DOCUMENT_SUFFIX",
    "DocumentError",
    "as_directory",
    "document_files",
    "is_document",
    "read_manifest",
    "unpack_document",
    "write_document",
]
