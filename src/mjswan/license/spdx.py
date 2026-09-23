"""Which license a text is, what tier that puts it in, and the standard texts mjswan can
write itself (ADR 0007).

The identifier and tier table mirror mjswan Cloud's ``@mjswan/licenses`` and are kept in
step by hand; the platform's copy decides what a publish is accepted with. Nothing here
verifies a claim: a text is classified when it can be and otherwise left alone.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

# An identifier is at most 64 characters; a longer run is not a tag.
_TAG = re.compile(
    r"^\s*SPDX-License-Identifier:\s*([A-Za-z0-9.+-]{1,64})(?![A-Za-z0-9.+-])"
)
_SPDX_ID = re.compile(r"^[A-Za-z0-9.+-]+$")

Tier = Literal["notice", "restricted", "blocked"]


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
    return (Path(__file__).parent / "templates" / f"{spdx}.txt").read_text(
        encoding="utf-8"
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


__all__ = [
    "BLOCKED",
    "CUSTOM",
    "GENERATABLE_LICENSES",
    "Identification",
    "Tier",
    "blocked_refusal",
    "display_name",
    "generate_license_text",
    "identify_license",
    "license_template",
    "resolve_license",
    "resolve_notice",
    "restricted_warning",
    "restriction_label",
    "tier_of",
]
