"""Pack a scene into one ``.mjz``: its ``MjSpec`` plus every asset it references.

``mujoco.to_zip`` stores entries uncompressed, so this is the DEFLATE-compressed
equivalent, with the model's mesh and texture paths rewritten to the flat layout inside
the archive. Buffer-only textures, which have no file to copy, are encoded to PNG, and
default classes MuJoCo could not read back are written out on each element.
"""

from __future__ import annotations

import os
import posixpath
import struct
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import zlib
from pathlib import Path
from typing import IO, Any, Union

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


def _is_absolute(posix_path: str) -> bool:
    """An absolute POSIX path (/foo/...) or Windows drive path (C:/foo/...)."""
    return posixpath.isabs(posix_path) or (
        len(posix_path) >= 3 and posix_path[1] == ":" and posix_path[2] == "/"
    )


def _make_zip_safe_path(path: str) -> str:
    """Return a ZIP-safe relative path.

    Absolute paths (e.g. those that MuJoCo resolves when loading a spec) are
    reduced to their basename so the ZIP entry and the rewritten XML reference
    only the filename.  Relative paths have leading ``..`` components stripped
    by the :func:`_strip_leading_dotdot` logic.
    """
    posix_path = path.replace("\\", "/")
    if _is_absolute(posix_path):
        return posixpath.basename(posix_path)
    return _strip_leading_dotdot(posix_path)


def _absolute_entry_names(paths: list[str]) -> dict[str, str]:
    """ZIP entries for the absolute paths among ``paths``, one per distinct file.

    An absolute path goes in under its basename unless another one with that basename
    is a different file, as when two attached models each ship a ``textured.obj``:
    then each takes the shortest tail of its path that tells them apart.
    """
    groups: dict[str, set[str]] = {}
    for path in paths:
        posix_path = path.replace("\\", "/")
        if _is_absolute(posix_path):
            normalized = posixpath.normpath(posix_path)
            groups.setdefault(posixpath.basename(normalized), set()).add(normalized)
    names: dict[str, str] = {}
    for group in groups.values():
        parts = {path: path.split("/") for path in group}
        depth = 1
        while len({"/".join(p[-depth:]) for p in parts.values()}) < len(group):
            depth += 1
        for path, split in parts.items():
            names[path] = "/".join(split[-depth:]).lstrip("/")
    return names


def _entry_name(path: str, absolute_names: dict[str, str]) -> str:
    """The ZIP entry for one referenced path: its unique name if absolute."""
    posix_path = path.replace("\\", "/")
    if _is_absolute(posix_path):
        named = absolute_names.get(posixpath.normpath(posix_path))
        if named:
            return named
    return _make_zip_safe_path(path)


def _referenced_paths(
    spec: mujoco.MjSpec, mesh_dir: str, texture_dir: str
) -> list[str]:
    """Every asset path the spec names, joined to its directory hint."""

    def joined(dir_hint: str, filename: str) -> str:
        return posixpath.join(dir_hint, filename) if dir_hint else filename

    paths = [joined(mesh_dir, mesh.file) for mesh in spec.meshes if mesh.file]
    for texture in spec.textures:
        cubefiles = [texture.cubefiles[i] for i in range(len(texture.cubefiles))]
        for name in [texture.file, *cubefiles]:
            if name:
                paths.append(joined(texture_dir, name))
    paths.extend(hfield.file for hfield in spec.hfields if hfield.file)
    paths.extend(skin.file for skin in spec.skins if skin.file)
    return paths


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
    absolute_names = _absolute_entry_names(
        _referenced_paths(spec, mesh_dir, texture_dir)
    )

    assets: dict[str, bytes] = {}

    def _read(dir_hint: str, filename: str) -> None:
        if not filename:
            return
        # POSIX-style key for ZIP entries / MuJoCo asset references
        rel = posixpath.join(dir_hint, filename) if dir_hint else filename
        # OS-native path for filesystem reads
        full = os.path.join(base_dir, rel)
        if os.path.isfile(full):
            assets[_entry_name(rel, absolute_names)] = Path(full).read_bytes()

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


_FILE_ATTRS = {
    "mesh": ("file",),
    "texture": (
        "file",
        "fileup",
        "fileback",
        "filedown",
        "filefront",
        "fileleft",
        "fileright",
    ),
    "hfield": ("file",),
    "skin": ("file",),
}


def _file_elements(root: ET.Element) -> list[ET.Element]:
    """The elements of a model XML that name an asset file, in document order."""
    return [
        elem
        for elem in root.iter()
        if any(elem.get(attr) for attr in _FILE_ATTRS.get(elem.tag, ()))
    ]


def _referenced_path(tag: str, file: str, mesh_dir: str, texture_dir: str) -> str:
    """A file attribute joined to the directory hint its element takes."""
    dir_hint = {"mesh": mesh_dir, "texture": texture_dir}.get(tag, "")
    return posixpath.join(dir_hint, file) if dir_hint else file


def _rewrite_xml_paths(
    xml_str: str,
    mesh_dir: str,
    texture_dir: str,
    absolute_names: dict[str, str] | None = None,
    recovered: dict[tuple[int, str], str] | None = None,
) -> str:
    """Rewrite asset paths in MuJoCo XML so the file is self-contained.

    Resolves ``meshdir``/``texturedir`` + ``file`` into a single normalised
    path (with leading ``..`` stripped), then removes the directory hints
    from ``<compiler>`` elements. ``absolute_names`` is what
    :func:`_absolute_entry_names` gave the archive's entries. ``recovered``
    names the entry of each file MuJoCo's encoder found, by the element's
    index in :func:`_file_elements` and the attribute.
    """
    names = absolute_names or {}
    recovered = recovered or {}
    root = ET.fromstring(xml_str)

    for compiler in root.iter("compiler"):
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    for index, elem in enumerate(_file_elements(root)):
        for attr in _FILE_ATTRS[elem.tag]:
            file = elem.get(attr)
            if not file:
                continue
            entry = recovered.get((index, attr))
            if entry is None:
                path = _referenced_path(elem.tag, file, mesh_dir, texture_dir)
                entry = _entry_name(path, names)
            elem.set(attr, entry)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


# MuJoCo 3.11's writer never writes class="main", so a root-class element inside another
# childclass takes that class when the XML is loaded back.
_ROOT_CLASS = "main"
_CLASSED = (
    "bodies",
    "frames",
    "geoms",
    "joints",
    "sites",
    "cameras",
    "lights",
    "actuators",
    "meshes",
    "materials",
    "equalities",
    "pairs",
    "flexes",
)
_LIMITED = {0: "false", 1: "true", 2: "auto"}


def _drop_empty_nameless_defaults(root: ET.Element) -> None:
    """Remove nested classes with neither a name nor contents. The writer leaves one for
    each model attached with an empty prefix (mjlab's terrain), and it sets nothing."""
    for default in list(root.iter("default")):
        for child in default.findall("default"):
            if not child.get("class") and len(child) == 0:
                default.remove(child)


def _defaults_unreadable(root: ET.Element) -> bool:
    """Whether MuJoCo would refuse these defaults: a nested class without a name, or one
    named twice (models attached with an empty prefix, as MyoSuite attaches its arms)."""
    seen: set[str] = set()
    for default in root.iter("default"):
        for child in default.findall("default"):
            name = child.get("class")
            if not name or name in seen:
                return True
            seen.add(name)
    return False


def _enclosing_class(element: Any, world: Any) -> str:
    """The childclass an element of the spec takes from the frames and bodies around it."""
    frame, body = element.frame, element.parent
    while True:
        while frame is not None:
            if frame.childclass:
                return frame.childclass
            frame = frame.frame
        if body is world or body.childclass:
            return body.childclass
        frame, body = body.frame, body.parent


def _root_class_dropped(spec: mujoco.MjSpec) -> bool:
    """Whether the spec puts a root-class element inside another childclass."""
    scopes = [*spec.bodies, *spec.frames]
    if all(s.childclass in ("", _ROOT_CLASS) for s in scopes):
        return False
    world = spec.worldbody

    def dropped(element: Any) -> bool:
        return _enclosing_class(element, world) not in ("", _ROOT_CLASS)

    if any(
        s is not world and s.childclass == _ROOT_CLASS and dropped(s) for s in scopes
    ):
        return True
    return any(
        e.classname.name == _ROOT_CLASS and dropped(e)
        for kind in ("geoms", "joints", "sites", "cameras", "lights")
        for e in getattr(spec, kind)
    )


def _numbers(value: Any) -> str:
    try:
        return " ".join(repr(float(x)) for x in value)
    except TypeError:
        return repr(float(value))


def _spell_out_tendons(root: ET.Element, tendons: list[Any]) -> None:
    """Write what a class could set on each tendon on the tendon itself: Python cannot
    move a tendon to another class."""
    elems = [
        e for s in root.findall("tendon") for e in s if e.tag in ("spatial", "fixed")
    ]
    if len(elems) != len(tendons):
        raise RuntimeError("the written tendons do not match the spec's")
    for elem, tendon in zip(elems, tendons):
        if tendon.name and elem.get("name") != tendon.name:
            raise RuntimeError(f"tendon {tendon.name!r} was written out of order")
        elem.attrib.pop("class", None)
        values: dict[str, str] = {
            "group": str(int(tendon.group)),
            "limited": _LIMITED[int(tendon.limited)],
            "actuatorfrclimited": _LIMITED[int(tendon.actfrclimited)],
            "range": _numbers(tendon.range),
            "actuatorfrcrange": _numbers(tendon.actfrcrange),
            "solreflimit": _numbers(tendon.solref_limit),
            "solimplimit": _numbers(tendon.solimp_limit),
            "solreffriction": _numbers(tendon.solref_friction),
            "solimpfriction": _numbers(tendon.solimp_friction),
            "frictionloss": _numbers(tendon.frictionloss),
            "springlength": _numbers(tendon.springlength),
            "width": _numbers(tendon.width),
            "margin": _numbers(tendon.margin),
            "stiffness": _numbers(tendon.stiffness),
            "damping": _numbers(tendon.damping),
            "armature": _numbers(tendon.armature),
            "rgba": _numbers(tendon.rgba),
        }
        if tendon.material:
            values["material"] = tendon.material
        if len(tendon.userdata):
            values["user"] = _numbers(tendon.userdata)
        if elem.tag == "fixed":  # a fixed tendon has no path to draw
            for attr in ("width", "rgba", "material"):
                values.pop(attr, None)
        for attr, value in values.items():
            elem.set(attr, value)


def _flat_xml(spec: mujoco.MjSpec) -> str:
    """The spec's XML with every element on the root class and its values written out.

    For defaults the writer cannot write back. Each element keeps the values it took from
    its class, so moving it to the root class makes the writer spell them out; the spec
    is restored afterwards.
    """
    root_default = spec.default
    undo: list[tuple[Any, str, Any]] = []
    try:
        for kind in _CLASSED:
            for element in getattr(spec, kind):
                if isinstance(getattr(element, "classname", None), mujoco.MjsDefault):
                    undo.append((element, "classname", element.classname))
                    element.classname = root_default
                if kind in ("bodies", "frames") and element.childclass:
                    undo.append((element, "childclass", element.childclass))
                    element.childclass = ""
        xml_str = spec.to_xml()
    finally:
        for element, attr, value in reversed(undo):
            setattr(element, attr, value)

    root = ET.fromstring(xml_str)
    _spell_out_tendons(root, list(spec.tendons))
    for default in root.findall("default"):
        for nested in default.findall("default"):
            default.remove(nested)
    left = next(
        (
            e
            for e in root.iter()
            if e.tag != "default" and ({"class", "childclass"} & e.attrib.keys())
        ),
        None,
    )
    if left is not None:
        raise RuntimeError(
            f"cannot write the default class of <{left.tag} name={left.get('name')!r}>"
        )
    return ET.tostring(root, encoding="unicode")


def _encode(
    spec: mujoco.MjSpec, extra_assets: dict[str, bytes]
) -> tuple[ET.Element, dict[str, bytes]]:
    """The model XML and the files MuJoCo's own ``.mjz`` encoder packs for the spec.

    The encoder resolves each element's file against the spec it was attached from,
    which ``spec.modelfiledir`` cannot. ``extra_assets`` are lent to it for files only
    this module has, the PNGs of buffer textures.
    """
    lent = [name for name in extra_assets if name not in spec.assets]
    for name in lent:
        spec.assets[name] = extra_assets[name]
    try:
        spec.compile()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.mjz")
            spec.encode(path)
            with zipfile.ZipFile(path) as zf:
                files = {name: zf.read(name) for name in zf.namelist()}
    finally:
        for name in lent:
            del spec.assets[name]
    xml_names = [name for name in files if name.endswith(".xml")]
    if len(xml_names) != 1:
        raise RuntimeError(f"MuJoCo's encoder wrote {len(xml_names)} XML files")
    return ET.fromstring(files.pop(xml_names[0])), files


def _unused_entry(name: str, data: bytes, files: dict[str, bytes | str]) -> str:
    stem, ext = posixpath.splitext(name)
    entry, n = name, 0
    while entry in files and files[entry] != data:
        n += 1
        entry = f"{stem}_{n}{ext}"
    return entry


def _recover_missing(
    spec: mujoco.MjSpec,
    xml_str: str,
    missing: set[str],
    *,
    mesh_dir: str,
    texture_dir: str,
    files_to_zip: dict[str, bytes | str],
    extra_assets: dict[str, bytes],
) -> dict[tuple[int, str], str]:
    """Add the files this module could not find, as MuJoCo's encoder finds them.

    Both XMLs come from the same writer, so their file elements pair up in order.
    Returns the entries by element index and attribute, for :func:`_rewrite_xml_paths`.
    """
    try:
        encoded_root, encoded = _encode(spec, extra_assets)
    except (ValueError, RuntimeError, mujoco.FatalError) as e:
        raise FileNotFoundError(_missing_message(missing)) from e
    ours = _file_elements(ET.fromstring(xml_str))
    theirs = _file_elements(encoded_root)
    recovered: dict[tuple[int, str], str] = {}
    found: set[str] = set()
    if [e.tag for e in ours] == [e.tag for e in theirs]:
        for index, (mine, packed) in enumerate(zip(ours, theirs)):
            for attr in _FILE_ATTRS[mine.tag]:
                file, entry = mine.get(attr), packed.get(attr)
                if not file or entry not in encoded:
                    continue
                path = _referenced_path(mine.tag, file, mesh_dir, texture_dir)
                if path not in missing:
                    continue
                data = encoded[entry]
                entry = _unused_entry(entry, data, files_to_zip)
                files_to_zip[entry] = data
                recovered[(index, attr)] = entry
                found.add(path)
    if missing - found:
        raise FileNotFoundError(_missing_message(missing - found))
    return recovered


def _missing_message(missing: set[str]) -> str:
    listed = ", ".join(sorted(missing)[:5])
    more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
    return f"cannot find the scene's asset files {listed}{more}"


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
    is self-contained. Files of specs attached from another directory are
    found with MuJoCo's own encoder; a file it cannot find either raises
    ``FileNotFoundError``.
    """
    base_dir = spec.modelfiledir or ""
    mesh_dir = spec.meshdir or ""
    texture_dir = spec.texturedir or ""
    absolute_names = _absolute_entry_names(
        _referenced_paths(spec, mesh_dir, texture_dir)
    )

    # Collect assets from disk, falling back to spec.assets with suffix matching
    # (mjlab stores assets under prefixed keys like "assets/robot/model.stl").
    files_to_zip: dict[str, bytes | str] = {}
    missing: set[str] = set()

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

    def _add_asset(rel: str) -> None:
        data = _read_asset(rel)
        if data is None:
            missing.add(rel)
        else:
            files_to_zip[_entry_name(rel, absolute_names)] = data

    for mesh in spec.meshes:
        if mesh.file:
            _add_asset(_referenced_path("mesh", mesh.file, mesh_dir, texture_dir))

    for texture in spec.textures:
        for fname in [texture.file] + [
            texture.cubefiles[i] for i in range(len(texture.cubefiles))
        ]:
            if fname:
                _add_asset(_referenced_path("texture", fname, mesh_dir, texture_dir))

    for hfield in spec.hfields:
        if hfield.file:
            _add_asset(hfield.file)

    for skin in spec.skins:
        if skin.file:
            _add_asset(skin.file)

    # Buffer textures have no backing file and cause spec.to_xml() to raise FatalError.
    # Temporarily assign each one a filename and restore after to_xml().
    _buffer_tex_restore: list[tuple[mujoco.MjsTexture, str]] = []
    buffer_pngs: dict[str, bytes] = {}
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
        buffer_pngs[rel] = buffer_pngs[png_filename] = png_bytes
        _buffer_tex_restore.append((texture, texture.file))
        texture.file = png_filename

    try:
        root = ET.fromstring(spec.to_xml())
        _drop_empty_nameless_defaults(root)
        if _defaults_unreadable(root) or _root_class_dropped(spec):
            xml_str = _flat_xml(spec)
        else:
            xml_str = ET.tostring(root, encoding="unicode")
        recovered = (
            _recover_missing(
                spec,
                xml_str,
                missing,
                mesh_dir=mesh_dir,
                texture_dir=texture_dir,
                files_to_zip=files_to_zip,
                extra_assets=buffer_pngs,
            )
            if missing
            else {}
        )
    finally:
        # Restore spec to avoid mutating the caller's object
        for texture, orig_file in _buffer_tex_restore:
            texture.file = orig_file

    xml_str = _rewrite_xml_paths(
        xml_str, mesh_dir, texture_dir, absolute_names, recovered
    )
    files_to_zip[spec.modelname + ".xml"] = xml_str

    # Write the ZIP
    if isinstance(file, str):
        directory = os.path.dirname(file)
        os.makedirs(directory, exist_ok=True)
        file = open(file, "wb")
    with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for filename, contents in files_to_zip.items():
            zf.writestr(filename, contents)
