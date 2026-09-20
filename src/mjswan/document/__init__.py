"""The ``.swn`` simulation document (ADR 0006): what the manifest says, how the tree is
named, and the container that packages it as one file.

This package knows nothing of torch or mjlab: it is the side of the format that
``app``, ``cli`` and ``cloud`` read, and that ``build`` writes.
"""

from .container import (
    DOCUMENT_SUFFIX,
    DocumentError,
    as_directory,
    document_files,
    is_document,
    read_manifest,
    unpack_document,
    write_document,
)
from .ids import assign_id, name2id, unique_id
from .manifest import (
    DEFAULT_IN_KEYS,
    DEFAULT_OUT_KEYS,
    DOCUMENT_FORMAT,
    MANIFEST_NAME,
    RUNTIME_INPUT_SLOTS,
)

__all__ = [
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "DOCUMENT_FORMAT",
    "DOCUMENT_SUFFIX",
    "MANIFEST_NAME",
    "RUNTIME_INPUT_SLOTS",
    "DocumentError",
    "as_directory",
    "assign_id",
    "document_files",
    "is_document",
    "name2id",
    "read_manifest",
    "unique_id",
    "unpack_document",
    "write_document",
]
