"""The third-party components a scene carries, and how a build declares them (ADR 0007
§2): files detected beside a model, a table of models that ship without one, and what
``publish`` and ``info`` read back from a built tree."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .path import LicenseLocation, component_id, parse_license_path
from .spdx import Tier, display_name, generate_license_text, identify_license


@dataclass
class Attribution:
    """One third-party component of a scene and the files that travel with it.

    Written by the build as ``<project>/<scene>/LICENSE.<component>`` and
    ``NOTICE.<component>``; either may be absent, not both.
    """

    component: str
    license: bytes | None = None
    notice: bytes | None = None
    #: Where it came from, for messages: the file's path, "known asset", or "author".
    origin: str = ""

    @property
    def spdx(self) -> str | None:
        if self.license is None:
            return None
        return identify_license(self.license.decode("utf-8", "replace")).spdx

    def files(self) -> dict[str, bytes]:
        """What the scene directory gets, by basename."""
        out: dict[str, bytes] = {}
        if self.license is not None:
            out[f"LICENSE.{self.component}"] = self.license
        if self.notice is not None:
            out[f"NOTICE.{self.component}"] = self.notice
        return out


@dataclass(frozen=True)
class KnownAsset:
    """A model whose license is known, for when no file is found beside it."""

    component: str
    spdx: str
    holder: str
    url: str
    #: Alternative token sets; a name (a menagerie directory, a model name, an mjlab
    #: task id) matches when every token of one set appears in it.
    tokens: tuple[frozenset[str], ...]

    def license_text(self) -> bytes:
        return generate_license_text(self.spdx, self.holder).encode("utf-8")


_UNITREE = 'HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")'
_MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie/tree/main/"


def _unitree(model: str, *, bare: bool = False) -> KnownAsset:
    """``bare``: the model name alone matches too (``go2``, ``g1_29dof``), for names
    nothing else is called."""
    tokens = [frozenset({"unitree", model})]
    if bare:
        tokens.append(frozenset({model}))
    return KnownAsset(
        component=f"unitree_{model}",
        spdx="BSD-3-Clause",
        holder=_UNITREE,
        url=f"{_MENAGERIE}unitree_{model}",
        tokens=tuple(tokens),
    )


#: Matched on the tokens of a menagerie directory (``unitree_g1``), model name or mjlab
#: task id (``Mjlab-Velocity-Flat-Unitree-G1``). Extend when an example adds a model.
KNOWN_ASSETS: tuple[KnownAsset, ...] = (
    _unitree("a1"),
    _unitree("go1", bare=True),
    _unitree("go2", bare=True),
    _unitree("g1", bare=True),
    _unitree("h1", bare=True),
    _unitree("z1"),
    KnownAsset(
        component="anybotics_anymal_c",
        spdx="BSD-3-Clause",
        holder="ANYbotics AG",
        url=f"{_MENAGERIE}anybotics_anymal_c",
        tokens=(frozenset({"anymal", "c"}),),
    ),
    KnownAsset(
        component="anybotics_anymal_b",
        spdx="BSD-3-Clause",
        holder="ANYbotics AG",
        url=f"{_MENAGERIE}anybotics_anymal_b",
        tokens=(frozenset({"anymal", "b"}),),
    ),
)


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def known_asset(*names: str | None) -> KnownAsset | None:
    """The first known asset one of ``names`` matches, or ``None``."""
    for name in names:
        if not name:
            continue
        tokens = _tokens(name)
        for asset in KNOWN_ASSETS:
            if any(alternative <= tokens for alternative in asset.tokens):
                return asset
    return None


def known_attribution(*names: str | None) -> Attribution | None:
    asset = known_asset(*names)
    if asset is None:
        return None
    return Attribution(
        component=asset.component,
        license=asset.license_text(),
        origin=f"known asset ({asset.url})",
    )


# `LICENSE`, `LICENSE.txt`, `LICENSE-APACHE`, `COPYING.LESSER`; not `LICENSING.md`.
_LICENSE_NAMES = re.compile(r"^(LICENSE|LICENCE|COPYING)([.-].*)?$", re.IGNORECASE)
_NOTICE_NAMES = re.compile(r"^NOTICE([.-].*)?$", re.IGNORECASE)


def _matching(directory: Path, names: re.Pattern[str]) -> list[Path]:
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_file() and names.match(p.name)]


def _absolute(path: str) -> Path:
    """``path`` made absolute with its symlinks kept: the Hub cache links each file
    into ``blobs/``, away from the license beside the link."""
    return Path(os.path.abspath(os.path.expanduser(path)))


def spec_asset_directories(spec: Any) -> list[Path]:
    """The model's own directory, then those its meshes, textures, heightfields and
    skins resolve to (as :func:`mjswan.build.mjz.collect_spec_assets` resolves them).

    Empty for a spec parsed from a string: resolving its asset paths against the
    working directory would find whatever happens to be there.
    """
    base = spec.modelfiledir or ""
    if not base:
        return []
    base_dir = _absolute(base)
    found: list[Path] = [base_dir]

    def add(dir_hint: str, filename: str) -> None:
        if not filename:
            return
        full = _absolute(os.path.join(base, dir_hint, filename))
        if full.parent not in found:
            found.append(full.parent)

    for mesh in spec.meshes:
        add(spec.meshdir or "", mesh.file)
    for texture in spec.textures:
        add(spec.texturedir or "", texture.file)
        for cube in texture.cubefiles:
            add(spec.texturedir or "", cube)
    for hfield in spec.hfields:
        add("", hfield.file)
    for skin in spec.skins:
        add("", skin.file)
    return found


def _component_name(directory: Path) -> str:
    """The directory's name, or at a Hub snapshot's root the repository's: the
    snapshot is named for its commit (``models--<owner>--<name>/snapshots/<sha>``)."""
    if directory.parent.name == "snapshots" and "--" in directory.parent.parent.name:
        return directory.parent.parent.name.rsplit("--", 1)[-1]
    return directory.name


def detect_attributions(
    directories: Iterable[Path], *, max_parents: int = 2
) -> list[Attribution]:
    """The license and notice files found beside a model, copied verbatim.

    For each directory, the nearest level (itself, then up to ``max_parents`` parents)
    holding any ``LICENSE*`` / ``COPYING*`` / ``NOTICE*`` wins. Two parents reach a
    model's directory from its ``assets/``; a third would reach the repository root and
    take the repository's license instead. Files are de-duplicated by content, and the
    component is the directory's name.
    """
    attributions: list[Attribution] = []
    seen_dirs: set[Path] = set()
    seen_content: set[bytes] = set()

    def fresh(data: bytes | None) -> bytes | None:
        """``data`` unless the same bytes were already taken from another directory."""
        if data is None:
            return None
        digest = hashlib.sha256(data).digest()
        if digest in seen_content:
            return None
        seen_content.add(digest)
        return data

    for directory in directories:
        for level, candidate in enumerate([directory, *directory.parents]):
            if level > max_parents:
                break
            if candidate in seen_dirs:
                break
            licenses = _matching(candidate, _LICENSE_NAMES)
            notices = _matching(candidate, _NOTICE_NAMES)
            if not licenses and not notices:
                continue
            seen_dirs.add(candidate)
            component = component_id(_component_name(candidate))
            paired = min(len(licenses), len(notices))
            pairs: list[tuple[Path | None, Path | None]] = [
                *zip(licenses[:paired], notices[:paired]),
                *((p, None) for p in licenses[paired:]),
                *((None, p) for p in notices[paired:]),
            ]
            for n, (lic, note) in enumerate(pairs):
                lic_bytes = fresh(lic.read_bytes() if lic else None)
                note_bytes = fresh(note.read_bytes() if note else None)
                if lic_bytes is None and note_bytes is None:
                    continue
                attributions.append(
                    Attribution(
                        component=component if n == 0 else f"{component}_{n + 1}",
                        license=lic_bytes,
                        notice=note_bytes,
                        origin=str(lic or note),
                    )
                )
            break
    return attributions


@dataclass(frozen=True)
class LicenseDeclaration:
    """One license file in a built tree and what it says."""

    path: str
    location: LicenseLocation
    spdx: str | None
    tier: Tier | None

    def describe(self) -> str:
        """``demo/go2/LICENSE.unitree_go2 (BSD-3-Clause)``; a NOTICE has no identifier."""
        return f"{self.path} ({display_name(self.spdx)})" if self.spdx else self.path


def declare(path: str, data: bytes) -> LicenseDeclaration | None:
    """What a file at a build-relative path declares, or ``None`` if it is not a license
    file where the platform reads one."""
    location = parse_license_path(path)
    if location is None:
        return None
    if location.kind == "NOTICE":
        return LicenseDeclaration(path, location, None, None)
    found = identify_license(data.decode("utf-8", "replace"))
    return LicenseDeclaration(path, location, found.spdx, found.tier)


__all__ = [
    "KNOWN_ASSETS",
    "Attribution",
    "KnownAsset",
    "LicenseDeclaration",
    "declare",
    "detect_attributions",
    "known_asset",
    "known_attribution",
    "spec_asset_directories",
]
