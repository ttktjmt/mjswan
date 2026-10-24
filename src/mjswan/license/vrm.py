"""A VRM's license as its own ``meta`` declares it, VRM 1.0 or 0.x, read the way mjswan
Cloud reads it: hosting the file is redistributing it, so a VRM that forbids that is not
published, and one only its author may use is published on the uploader's word.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

AvatarUse = Literal["author", "licensed", "everyone"]

_AVATAR_USE_1: dict[str, AvatarUse] = {
    "onlyAuthor": "author",
    "onlySeparatelyLicensedPerson": "licensed",
    "everyone": "everyone",
}
_AVATAR_USE_0: dict[str, AvatarUse] = {
    "OnlyAuthor": "author",
    "ExplicitlyLicensedPerson": "licensed",
    "Everyone": "everyone",
}


@dataclass(frozen=True)
class VrmLicense:
    """The terms one VRM's ``meta`` states, with VRM 1.0's defaults where it is silent."""

    version: Literal["0", "1"]
    name: str | None
    authors: tuple[str, ...]
    license: str | None
    """VRM 1.0's ``licenseUrl``; VRM 0.x's ``licenseName`` as written."""
    other_license_url: str | None
    redistribution: bool
    commercial: bool
    avatar_use: AvatarUse
    credit_required: bool

    @property
    def needs_attestation(self) -> bool:
        """Only its author, or someone they licensed, may use it as an avatar."""
        return self.avatar_use != "everyone"


def _text(value: Any) -> str | None:
    """A meta string as mjswan Cloud reads it: anything but a non-empty string is unset."""
    return value if isinstance(value, str) and value else None


def vrm_license(meta: Mapping[str, Any], version: Literal["0", "1"]) -> VrmLicense:
    if version == "1":
        authors = meta.get("authors")
        return VrmLicense(
            version="1",
            name=_text(meta.get("name")),
            authors=tuple(filter(_text, authors)) if isinstance(authors, list) else (),
            license=_text(meta.get("licenseUrl")),
            other_license_url=_text(meta.get("otherLicenseUrl")),
            redistribution=meta.get("allowRedistribution") is True,
            commercial=(_text(meta.get("commercialUsage")) or "personalNonProfit")
            != "personalNonProfit",
            avatar_use=_AVATAR_USE_1.get(
                _text(meta.get("avatarPermission")) or "", "author"
            ),
            credit_required=meta.get("creditNotation") != "unnecessary",
        )
    name = _text(meta.get("licenseName"))
    author = _text(meta.get("author"))
    return VrmLicense(
        version="0",
        name=_text(meta.get("title")),
        authors=(author,) if author else (),
        license=name,
        other_license_url=_text(meta.get("otherLicenseUrl")),
        redistribution=name != "Redistribution_Prohibited",
        commercial=meta.get("commercialUssageName") == "Allow"
        and "NC" not in (name or ""),
        avatar_use=_AVATAR_USE_0.get(
            _text(meta.get("allowedUserName")) or "", "author"
        ),
        credit_required=(name or "").startswith("CC_BY") or name == "Other",
    )


def describe(terms: VrmLicense) -> str:
    """One line for the CLI: whose avatar it is and under what."""
    who = ", ".join(terms.authors) or "an unnamed author"
    license = terms.license or terms.other_license_url or "no license named"
    return f"{terms.name or 'unnamed avatar'} by {who} ({license})"


__all__ = ["AvatarUse", "VrmLicense", "describe", "vrm_license"]
