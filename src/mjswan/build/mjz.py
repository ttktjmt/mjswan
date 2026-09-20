"""Pack a scene into one ``.mjz``: its ``MjSpec`` plus every asset it references.

``mujoco.to_zip`` stores entries uncompressed, so this is the DEFLATE-compressed
equivalent, with the model's mesh and texture paths rewritten to the flat layout inside
the archive. Buffer-only textures, which have no file to copy, are encoded to PNG.
"""

from __future__ import annotations

import os
import posixpath
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import IO, Union

import mujoco


def _strip_leading_dotdot(path: str) -> str:
    """Normalize a POSIX path and strip any leading ``..`` components."""
    normalized = posixpath.normpath(path)
    if normalized == ".":
        return ""
    parts = normalized.split("/")
    while parts and parts[0] == "..":
        parts.pop(0)
    return "/".join(parts) if parts else ""


def _make_zip_safe_path(path: str) -> str:
    """Return a ZIP-safe relative path.

    Absolute paths (e.g. those that MuJoCo resolves when loading a spec) are
    reduced to their basename so the ZIP entry and the rewritten XML reference
    only the filename.  Relative paths have leading ``..`` components stripped
    by the :func:`_strip_leading_dotdot` logic.
    """
    posix_path = path.replace("\\", "/")
    # Absolute POSIX path (/foo/...) or Windows drive path (C:/foo/...)
    is_absolute = posixpath.isabs(posix_path) or (
        len(posix_path) >= 3 and posix_path[1] == ":" and posix_path[2] == "/"
    )
    if is_absolute:
        return posixpath.basename(posix_path)
    return _strip_leading_dotdot(posix_path)


def _iter_asset_lookup_candidates(path: str) -> list[str]:
    """Return candidate spec.assets keys for a referenced asset path.

    ``spec.assets`` may contain any of:
    - the original relative path,
    - a normalized ZIP-safe path (leading ``..`` stripped), or
    - only the basename when the source spec used an absolute meshdir/texturedir.

    MyoSuite scenes hit the third case after mjlab attaches child specs into the
    final scene spec, so we try progressively looser matches in that order.
    """
    posix_path = path.replace("\\", "/")
    candidates: list[str] = []
    for candidate in (
        posix_path,
        _make_zip_safe_path(posix_path),
        posixpath.basename(posix_path),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def collect_spec_assets(spec: mujoco.MjSpec) -> dict[str, bytes]:
    """Collect all asset files referenced by a MjSpec from disk.

    Uses ``spec.modelfiledir``, ``spec.meshdir``, and ``spec.texturedir``
    to resolve file paths.  Returns a dictionary mapping POSIX-style relative
    paths (suitable for ZIP entries) to file contents.
    """
    base_dir = spec.modelfiledir or ""
    mesh_dir = spec.meshdir or ""
    texture_dir = spec.texturedir or ""

    assets: dict[str, bytes] = {}

    def _read(dir_hint: str, filename: str) -> None:
        if not filename:
            return
        # POSIX-style key for ZIP entries / MuJoCo asset references
        rel = posixpath.join(dir_hint, filename) if dir_hint else filename
        # OS-native path for filesystem reads
        full = os.path.join(base_dir, rel)
        if os.path.isfile(full):
            assets[_make_zip_safe_path(rel)] = Path(full).read_bytes()

    # Meshes
    for mesh in spec.meshes:
        _read(mesh_dir, mesh.file)

    # Textures (single file and cube-map faces)
    for texture in spec.textures:
        _read(texture_dir, texture.file)
        for i in range(len(texture.cubefiles)):
            _read(texture_dir, texture.cubefiles[i])

    # Heightfields
    for hfield in spec.hfields:
        _read("", hfield.file)

    # Skins
    for skin in spec.skins:
        _read("", skin.file)

    return assets


def _rewrite_xml_paths(xml_str: str, mesh_dir: str, texture_dir: str) -> str:
    """Rewrite asset paths in MuJoCo XML so the file is self-contained.

    Resolves ``meshdir``/``texturedir`` + ``file`` into a single normalised
    path (with leading ``..`` stripped), then removes the directory hints
    from ``<compiler>`` elements.
    """
    root = ET.fromstring(xml_str)

    # Rewrite <compiler> meshdir / texturedir
    for compiler in root.iter("compiler"):
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    # Rewrite <mesh file="...">
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f:
            rel = posixpath.join(mesh_dir, f) if mesh_dir else f
            mesh.set("file", _make_zip_safe_path(rel))

    # Rewrite <texture file/cube-map="...">
    _TEX_FILE_ATTRS = (
        "file",
        "fileup",
        "fileback",
        "filedown",
        "filefront",
        "fileleft",
        "fileright",
    )
    for tex in root.iter("texture"):
        for attr in _TEX_FILE_ATTRS:
            f = tex.get(attr)
            if f:
                rel = posixpath.join(texture_dir, f) if texture_dir else f
                tex.set(attr, _make_zip_safe_path(rel))

    # Rewrite <hfield file="..."> and <skin file="...">
    for tag in ("hfield", "skin"):
        for elem in root.iter(tag):
            f = elem.get("file")
            if f:
                elem.set("file", _make_zip_safe_path(f))

    # Fix <default> hierarchy errors from spec.to_xml():
    # 1. Remove classless nested defaults ("empty class name" error).
    # 2. Merge same-named nested defaults ("repeated default class name" error).
    def _fix_default_tree(elem: ET.Element) -> None:
        # Don't remove classless root <default> (direct child of <mujoco>).
        parent_is_default = elem.tag == "default"
        for child in list(elem):
            if child.tag != "default":
                continue
            cls = child.get("class")
            if not cls and parent_is_default:
                # Rule 1: remove classless nested defaults
                elem.remove(child)
                continue
            if cls:
                # Rule 2: merge same-named direct children into this element
                same_named = [
                    gc for gc in child if gc.tag == "default" and gc.get("class") == cls
                ]
                for dup in same_named:
                    insert_pos = list(child).index(dup)
                    for grandchild in list(dup):
                        child.insert(insert_pos, grandchild)
                        insert_pos += 1
                    child.remove(dup)
            # Recurse after fixing this level
            _fix_default_tree(child)

    _fix_default_tree(root)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def _buffer_texture_to_png(data: bytes | bytearray, width: int, height: int) -> bytes:
    """Encode raw RGB/RGBA texture buffer data (from MuJoCo) as PNG bytes.

    Args:
        data: Raw pixel bytes (RGB or RGBA, uint8, row-major).
        width: Texture width in pixels.
        height: Texture height in pixels.

    Returns:
        PNG-encoded bytes.

    Raises:
        ValueError: If dimensions are zero or channel count is unsupported.
    """
    raw = bytes(data) if isinstance(data, (bytes, bytearray)) else data.tobytes()

    n_pixels = width * height
    if n_pixels == 0:
        raise ValueError("zero-area texture")
    total = len(raw)
    if total % n_pixels != 0:
        raise ValueError(f"data length {total} is not divisible by {n_pixels} pixels")
    nchannel = total // n_pixels
    if nchannel not in (3, 4):
        raise ValueError(f"unsupported channel count {nchannel} (expected 3 or 4)")

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    color_type = 2 if nchannel == 3 else 6  # 2 = RGB, 6 = RGBA
    ihdr = _chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    )

    stride = width * nchannel
    scanlines = bytearray()
    for y in range(height):
        scanlines += b"\x00"  # filter type: None
        scanlines += raw[y * stride : (y + 1) * stride]
    idat = _chunk(b"IDAT", zlib.compress(bytes(scanlines)))
    iend = _chunk(b"IEND", b"")

    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def to_zip_deflated(spec: mujoco.MjSpec, file: Union[str, IO[bytes]]) -> None:
    """Save an MjSpec as a ZIP file with DEFLATE compression.

    This is a compressed alternative to ``mujoco.to_zip`` which stores
    entries uncompressed.  The output is a standard ZIP that JSZip (and
    other readers) can decompress transparently.

    The function collects asset files from disk, normalises all relative
    paths (stripping leading ``..`` components) so that the resulting ZIP
    is self-contained.
    """
    base_dir = spec.modelfiledir or ""
    mesh_dir = spec.meshdir or ""
    texture_dir = spec.texturedir or ""

    # Collect assets from disk, falling back to spec.assets with suffix matching
    # (mjlab stores assets under prefixed keys like "assets/robot/model.stl").
    files_to_zip: dict[str, bytes | str] = {}

    def _read_asset(rel: str) -> bytes | None:
        """Return asset bytes from disk or spec.assets, or None if not found."""
        full = os.path.join(base_dir, rel)
        if os.path.isfile(full):
            return Path(full).read_bytes()
        assets = spec.assets
        if not assets:
            return None
        candidates = _iter_asset_lookup_candidates(rel)
        for candidate in candidates:
            if candidate in assets:
                return assets[candidate]
        # Suffix match: spec.assets key ends with "/" + candidate
        # (e.g. "assets/robot/model.stl" ends with "robot/model.stl")
        for candidate in candidates:
            for key, data in assets.items():
                key_posix = key.replace("\\", "/")
                if key_posix.endswith("/" + candidate):
                    return data
        return None

    for mesh in spec.meshes:
        if not mesh.file:
            continue
        rel = posixpath.join(mesh_dir, mesh.file) if mesh_dir else mesh.file
        data = _read_asset(rel)
        if data is not None:
            files_to_zip[_make_zip_safe_path(rel)] = data

    for texture in spec.textures:
        for fname in [texture.file] + [
            texture.cubefiles[i] for i in range(len(texture.cubefiles))
        ]:
            if not fname:
                continue
            rel = posixpath.join(texture_dir, fname) if texture_dir else fname
            data = _read_asset(rel)
            if data is not None:
                files_to_zip[_make_zip_safe_path(rel)] = data

    for hfield in spec.hfields:
        if not hfield.file:
            continue
        data = _read_asset(hfield.file)
        if data is not None:
            files_to_zip[_make_zip_safe_path(hfield.file)] = data

    for skin in spec.skins:
        if not skin.file:
            continue
        data = _read_asset(skin.file)
        if data is not None:
            files_to_zip[_make_zip_safe_path(skin.file)] = data

    # Buffer textures have no backing file and cause spec.to_xml() to raise FatalError.
    # Temporarily assign each one a filename and restore after to_xml().
    _buffer_tex_restore: list[tuple[mujoco.MjsTexture, str]] = []
    for i, texture in enumerate(spec.textures):
        if texture.file:
            continue  # file-backed texture already collected above
        try:
            data = texture.data
            if data is None or len(data) == 0:
                continue
        except AttributeError:
            continue
        w, h = texture.width, texture.height
        if w <= 0 or h <= 0:
            continue
        tex_label = texture.name if texture.name else str(i)
        png_filename = f"_buf_{tex_label}.png"
        try:
            png_bytes = _buffer_texture_to_png(data, w, h)
        except ValueError:
            continue
        rel = posixpath.join(texture_dir, png_filename) if texture_dir else png_filename
        files_to_zip[_make_zip_safe_path(rel)] = png_bytes
        _buffer_tex_restore.append((texture, texture.file))
        texture.file = png_filename

    # Generate XML and rewrite paths
    xml_str = spec.to_xml()

    # Restore spec to avoid mutating the caller's object
    for texture, orig_file in _buffer_tex_restore:
        texture.file = orig_file

    xml_str = _rewrite_xml_paths(xml_str, mesh_dir, texture_dir)
    files_to_zip[spec.modelname + ".xml"] = xml_str

    # Write the ZIP
    if isinstance(file, str):
        directory = os.path.dirname(file)
        os.makedirs(directory, exist_ok=True)
        file = open(file, "wb")
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, contents in files_to_zip.items():
            zf.writestr(filename, contents)
