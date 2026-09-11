"""License files in a build (ADR 0007): where they sit, what they say, and what
``publish`` does about them.

Files sit under a project directory, never at the build root, whose ``LICENSE`` is the
engine's:

- ``<project-id>/LICENSE`` and ``NOTICE``: the work's;
- ``<project-id>/<scene-id>/LICENSE.<component>`` / ``NOTICE.<component>``: one
  third-party component the scene contains (a bare scene-level ``LICENSE`` is labelled
  with the scene id).

The naming rule, identifier and tier table mirror mjswan Cloud's ``@mjswan/licenses``
and are kept in step by hand; the platform's copy decides what a publish is accepted
with. Nothing here verifies a claim: a text is classified when it can be and otherwise
left alone.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Literal

#: The platform refuses a license or notice file larger than this.
LICENSE_FILE_MAX_BYTES: int = 64 * 1024

#: The Content-Type a license file is uploaded with.
LICENSE_CONTENT_TYPE: str = "text/plain; charset=utf-8"

#: The identifier given to a text that cannot be classified.
CUSTOM: str = "LicenseRef-custom"

_MPG_NONCOMMERCIAL = "LicenseRef-MPG-NonCommercial"
_UR_GRAPHICAL = "LicenseRef-UR-Graphical-Documentation"

#: Licenses whose terms forbid redistribution. The first two are recognised from their
#: text, the last two only from an SPDX tag.
BLOCKED: frozenset[str] = frozenset(
    {_MPG_NONCOMMERCIAL, _UR_GRAPHICAL, "LicenseRef-AMASS", "LicenseRef-SMPL"}
)

_RESTRICTED_PREFIXES = ("CC-BY-NC", "CC-BY-SA", "CC-BY-ND", "GPL-", "LGPL-", "AGPL-")

_BASENAME = re.compile(r"^(LICENSE|NOTICE)(?:\.([A-Za-z0-9_-]{1,64}))?$")
_COMPONENT = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# An identifier is at most 64 characters; a longer run is not a tag.
_TAG = re.compile(
    r"^\s*SPDX-License-Identifier:\s*([A-Za-z0-9.+-]{1,64})(?![A-Za-z0-9.+-])"
)
_SPDX_ID = re.compile(r"^[A-Za-z0-9.+-]+$")

Kind = Literal["LICENSE", "NOTICE"]
Scope = Literal["project", "scene"]
Tier = Literal["notice", "restricted", "blocked"]


# ── The naming rule ───────────────────────────────────────────────────────────


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


# ── Identification ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Identification:
    spdx: str
    tier: Tier


def tier_of(spdx: str) -> Tier:
    if spdx in BLOCKED:
        return "blocked"
    if spdx.startswith(_RESTRICTED_PREFIXES):
        return "restricted"
    return "notice"


def identify_license(text: str) -> Identification:
    """Which license a ``LICENSE`` text is.

    An SPDX tag on the first line wins; otherwise the text is matched on phrases only
    one license contains, and what nothing matches is :data:`CUSTOM`.
    """
    first_line = text.split("\n", 1)[0]
    tagged = _TAG.match(first_line)
    if tagged is not None:
        spdx = tagged.group(1)
        return Identification(spdx, tier_of(spdx))
    spdx = _identify_text(re.sub(r"\s+", " ", text))
    return Identification(spdx, tier_of(spdx))


def _search(pattern: str, t: str) -> re.Match[str] | None:
    return re.search(pattern, t, re.IGNORECASE)


def _identify_text(t: str) -> str:
    if _search(r"Apache License,? Version 2\.0", t):
        return "Apache-2.0"
    if _search(
        r"Permission is hereby granted, free of charge, to any person obtaining a copy",
        t,
    ):
        return "MIT"
    if _search(
        r"Clear BSD License|NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED",
        t,
    ):
        return "BSD-3-Clause-Clear"
    if _search(
        r"Redistributions of source code must retain the above copyright notice", t
    ):
        endorse = _search(
            r"Neither the name of .{0,200}?may be used to endorse or promote products",
            t,
        )
        return "BSD-3-Clause" if endorse else "BSD-2-Clause"
    if _search(
        r"Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted",
        t,
    ):
        return "ISC"
    if _search(r"Mozilla Public License,? Version 2\.0", t):
        return "MPL-2.0"
    if _search(r"The origin of this software must not be misrepresented", t):
        return "Zlib"
    if _search(
        r"This is free and unencumbered software released into the public domain", t
    ):
        return "Unlicense"
    if _search(r"CC0 1\.0 Universal", t):
        return "CC0-1.0"

    cc = _search(
        r"Attribution(-NonCommercial)?(-NoDerivatives|-NoDerivs|-ShareAlike)? ([34]\.0) International",
        t,
    )
    if cc:
        nc = "-NC" if cc.group(1) else ""
        variant = ""
        if cc.group(2):
            variant = "-SA" if _search(r"ShareAlike", cc.group(2)) else "-ND"
        return f"CC-BY{nc}{variant}-{cc.group(3)}"
    plain_cc = _search(r"Creative Commons Attribution ([34]\.0) International", t)
    if plain_cc:
        return f"CC-BY-{plain_cc.group(1)}"

    if _search(r"GNU AFFERO GENERAL PUBLIC LICENSE", t):
        return "AGPL-3.0-only"
    lgpl = _search(r"GNU LESSER GENERAL PUBLIC LICENSE Version (3|2\.1)", t)
    if lgpl:
        return f"LGPL-{lgpl.group(1)}-only"
    gpl = _search(r"GNU GENERAL PUBLIC LICENSE Version (3|2)", t)
    if gpl:
        return f"GPL-{gpl.group(1)}.0-only"

    # Max Planck's license behind SMPL and AMASS: research use only, no redistribution.
    if _search(r"non-commercial scientific research purposes", t):
        return _MPG_NONCOMMERCIAL
    if _search(r"Universal Robots", t) and _search(r"Graphical Documentation", t):
        return _UR_GRAPHICAL
    return CUSTOM


# ── What publish says ─────────────────────────────────────────────────────────


def display_name(spdx: str) -> str:
    return "Custom" if spdx == CUSTOM else spdx


def restriction_label(spdx: str) -> str | None:
    """A short label for a restricted license, or ``None``."""
    if tier_of(spdx) != "restricted":
        return None
    if "-NC" in spdx:
        return "Non-commercial"
    if "-SA" in spdx:
        return "Share-alike"
    if "-ND" in spdx:
        return "No derivatives"
    return "Copyleft"


def restricted_warning(path: str, spdx: str) -> str:
    """The one-line warning a restricted file earns at publish."""
    clauses: list[str] = []
    if "-NC" in spdx:
        clauses.append("non-commercial use only")
    if "-ND" in spdx:
        clauses.append("no sharing of adapted material")
    if "-SA" in spdx:
        clauses.append("derivatives must carry the same license")
    if re.match(r"^(A?GPL|LGPL)-", spdx):
        clauses.append("copyleft: derivatives must be licensed under the same terms")
    return (
        f"{path} is {spdx} ({', '.join(clauses)}). Publishing it is your "
        "responsibility under those terms."
    )


def _describe_blocked(spdx: str) -> str:
    return {
        _MPG_NONCOMMERCIAL: "the Max Planck non-commercial research license (SMPL, AMASS)",
        _UR_GRAPHICAL: "Universal Robots' Terms for Graphical Documentation",
        "LicenseRef-AMASS": "the AMASS license",
        "LicenseRef-SMPL": "the SMPL license",
    }.get(spdx, spdx)


def blocked_refusal(path: str, spdx: str) -> str:
    """The publish refusal for a blocked file."""
    return (
        f"{path} is {_describe_blocked(spdx)}, whose terms do not permit "
        "redistribution; it cannot be published to mjswan Cloud"
    )


# ── Generation ────────────────────────────────────────────────────────────────

#: Licenses whose standard text is bundled, so an SPDX id alone produces the file.
GENERATABLE_LICENSES: tuple[str, ...] = (
    "Apache-2.0",
    "MIT",
    "BSD-3-Clause",
    "BSD-2-Clause",
    "CC-BY-4.0",
    "CC0-1.0",
)


def license_template(spdx: str) -> str:
    """The SPDX text of a generatable license, placeholders and all."""
    if spdx not in GENERATABLE_LICENSES:
        raise ValueError(
            f"No standard text is bundled for {spdx!r}; the bundled ones are "
            f"{', '.join(GENERATABLE_LICENSES)}. Pass the path to its text instead."
        )
    return (
        resources.files(__name__)
        .joinpath("templates")
        .joinpath(f"{spdx}.txt")
        .read_text(encoding="utf-8")
    )


def generate_license_text(spdx: str, holder: str = "", year: int | None = None) -> str:
    """The standard text of a generatable license, SPDX tag first, holder filled in.

    ``holder`` is everything after ``Copyright (c)``, year included unless ``year`` is
    passed. The MIT and BSD texts carry their own copyright line; the others get one
    prepended when a holder is given.
    """
    template = license_template(spdx)
    name = holder.strip()
    line = " ".join(part for part in (str(year) if year else "", name) if part)
    body = template.replace("<year> <copyright holders>", line).replace(
        "<year> <owner>", line
    )
    if body == template and name:
        body = f"Copyright (c) {line}\n\n{template}"
    return f"SPDX-License-Identifier: {spdx}\n\n{body}"


def _is_file_reference(value: str | os.PathLike[str]) -> bool:
    if isinstance(value, os.PathLike):
        return True
    # A multi-line string is a text, never a path.
    return "\n" not in value and Path(value).expanduser().is_file()


def resolve_license(
    license: str | os.PathLike[str], *, copyright: str | None = None
) -> bytes:
    """The bytes of a ``LICENSE``: a path is copied verbatim; a generatable SPDX id
    becomes the standard text, with ``copyright`` (``"2026 Example Lab"``) as its
    holder line."""
    if _is_file_reference(license):
        return Path(license).expanduser().read_bytes()
    text = str(license)
    if _SPDX_ID.match(text):
        return generate_license_text(text, copyright or "").encode("utf-8")
    raise ValueError(
        f"license={text!r} is neither an SPDX identifier nor a file that exists. Pass "
        f"one of {', '.join(GENERATABLE_LICENSES)} or the path to a license text."
    )


def resolve_notice(notice: str | os.PathLike[str]) -> bytes:
    """The bytes of a ``NOTICE``: a path is copied verbatim, anything else is the text."""
    if _is_file_reference(notice):
        return Path(notice).expanduser().read_bytes()
    text = str(notice)
    return (text if text.endswith("\n") else text + "\n").encode("utf-8")


# ── Attributions: what a scene carries ────────────────────────────────────────


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


# ── Known assets: models that ship without a license file beside them ─────────


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


# ── Detection: license files beside a model on disk ───────────────────────────

# `LICENSE`, `LICENSE.txt`, `LICENSE-APACHE`, `COPYING.LESSER`; not `LICENSING.md`.
_LICENSE_NAMES = re.compile(r"^(LICENSE|LICENCE|COPYING)([.-].*)?$", re.IGNORECASE)
_NOTICE_NAMES = re.compile(r"^NOTICE([.-].*)?$", re.IGNORECASE)


def _matching(directory: Path, names: re.Pattern[str]) -> list[Path]:
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [p for p in entries if p.is_file() and names.match(p.name)]


def spec_asset_directories(spec: Any) -> list[Path]:
    """The model's own directory, then those its meshes, textures, heightfields and
    skins resolve to (as :func:`mjswan.utils.collect_spec_assets` resolves them).

    Empty for a spec parsed from a string: resolving its asset paths against the
    working directory would find whatever happens to be there.
    """
    base = spec.modelfiledir or ""
    if not base:
        return []
    base_dir = Path(base).expanduser().resolve()
    found: list[Path] = [base_dir]

    def add(dir_hint: str, filename: str) -> None:
        if not filename:
            return
        full = Path(os.path.join(base, dir_hint, filename)).expanduser().resolve()
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
            component = component_id(candidate.name)
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


# ── The build's declaration, as `publish` and `info` read it back ─────────────


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
    "BLOCKED",
    "CUSTOM",
    "GENERATABLE_LICENSES",
    "KNOWN_ASSETS",
    "LICENSE_CONTENT_TYPE",
    "LICENSE_FILE_MAX_BYTES",
    "Attribution",
    "Identification",
    "KnownAsset",
    "LicenseDeclaration",
    "LicenseLocation",
    "blocked_refusal",
    "component_id",
    "declare",
    "detect_attributions",
    "display_name",
    "generate_license_text",
    "has_license_basename",
    "identify_license",
    "is_license_file_path",
    "known_asset",
    "known_attribution",
    "license_path_problem",
    "license_template",
    "parse_license_path",
    "resolve_license",
    "resolve_notice",
    "restricted_warning",
    "restriction_label",
    "spec_asset_directories",
    "tier_of",
]
