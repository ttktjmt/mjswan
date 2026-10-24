"""The ``.swn`` simulation document (ADR 0006): what the manifest says, how the tree is
named, the container that packages it as one file, and the skin files it may carry.

Torch- and mjlab-free, so ``app``, ``cli`` and ``cloud`` can read what ``build`` writes
without them (ADR 0008 §1).
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
from .skin import COMPRESSION_EXTENSIONS, SkinFile, read_skin_file

__all__ = [
    "COMPRESSION_EXTENSIONS",
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "DOCUMENT_FORMAT",
    "DOCUMENT_SUFFIX",
    "MANIFEST_NAME",
    "RUNTIME_INPUT_SLOTS",
    "DocumentError",
    "SkinFile",
    "as_directory",
    "assign_id",
    "document_files",
    "is_document",
    "name2id",
    "read_manifest",
    "read_skin_file",
    "unique_id",
    "unpack_document",
    "write_document",
]
