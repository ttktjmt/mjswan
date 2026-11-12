"""A scene's ``credits`` and a project's ``work``: the license files written beside them
(ADR 0007), named in the manifest because a static host lists no directory. The texts
stay in the files.

A project's third-party texts sit together in ``<project>/licenses/``, each written once
(mjswan-cloud ADR 0017 §2.7), so a component every scene carries is one file."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..license import (
    LICENSES_DIR,
    copyright_line,
    identify_license,
    is_license_file_path,
    restriction_label,
)
from ..license.vrm import vrm_license

if TYPE_CHECKING:
    from ..project import ProjectConfig
    from ..scene import SceneConfig

#: ``(LICENSE.<component> or NOTICE.<component>, text)`` -> the file's name in
#: ``licenses/``.
LicenseNames = dict[tuple[str, bytes], str]


def license_names(project: ProjectConfig) -> LicenseNames:
    """The name each third-party text is written under: its component's, once per text.
    When two texts share a name, the first in scene order keeps it and the others take
    ``-2``, ``-3``, … in scene order, passing over a name another component has. A name
    equal to an earlier one but for case (one file on a case-insensitive disk), or one
    publish would not read as a license file, is numbered too."""
    variants: dict[str, list[bytes]] = {}
    for scene in project.scenes:
        for attribution in scene.attributions:
            for name, text in attribution.files().items():
                texts = variants.setdefault(name, [])
                if text not in texts:
                    texts.append(text)
    reserved = {name.casefold() for name in variants}
    used: set[str] = set()
    names: LicenseNames = {}
    for name, texts in variants.items():
        n = 2
        for text in texts:
            out = name
            if out.casefold() in used or not is_license_file_path(
                f"p/{LICENSES_DIR}/{out}"
            ):
                while (out := _numbered(name, n)).casefold() in reserved | used:
                    n += 1
            used.add(out.casefold())
            names[(name, text)] = out
    return names


def _numbered(name: str, n: int) -> str:
    """``name`` with ``-n``, its component shortened to stay within 64 characters."""
    kind, component = name.split(".", 1)
    suffix = f"-{n}"
    return f"{kind}.{component[: 64 - len(suffix)]}{suffix}"


def _license_fields(path: str, data: bytes) -> dict[str, str]:
    text = data.decode("utf-8", "replace")
    spdx = identify_license(text).spdx
    restriction = restriction_label(spdx)
    holder = copyright_line(text)
    return {
        "license": path,
        "spdx": spdx,
        **({"restriction": restriction} if restriction else {}),
        **({"copyright": holder} if holder else {}),
    }


def scene_credits(scene: SceneConfig, names: LicenseNames) -> list[dict[str, Any]]:
    """One entry per third-party component, paths relative to the scene directory. A
    ``skin-<id>`` component belongs to that skin; a VRM skin adds a summary of its
    ``meta``."""
    skins = {f"skin-{skin.config.id}": skin for skin in scene.skins}
    entries: dict[str, dict[str, Any]] = {}
    for attribution in scene.attributions:
        component = attribution.component
        entry: dict[str, Any] = {"component": component}
        if component in skins:
            entry["skin"] = skins[component].config.id
        files = {
            name: f"../{LICENSES_DIR}/{names[(name, text)]}"
            for name, text in attribution.files().items()
        }
        if attribution.license is not None:
            entry.update(
                _license_fields(files[f"LICENSE.{component}"], attribution.license)
            )
        if attribution.notice is not None:
            entry["notice"] = files[f"NOTICE.{component}"]
        entries[component] = entry
    for component, skin in skins.items():
        if skin.file.vrm_version is None:
            continue
        terms = vrm_license(skin.file.meta, skin.file.vrm_version)
        entry = entries.setdefault(
            component, {"component": component, "skin": skin.config.id}
        )
        entry["vrm"] = {
            **({"name": terms.name} if terms.name else {}),
            "authors": list(terms.authors),
            **({"license": terms.license} if terms.license else {}),
            **(
                {"other_license_url": terms.other_license_url}
                if terms.other_license_url
                else {}
            ),
        }
    return list(entries.values())


def work_entry(project: ProjectConfig) -> dict[str, Any] | None:
    """The work's own ``LICENSE`` and ``NOTICE``, relative to the project directory."""
    if project.license is None and project.notice is None:
        return None
    return {
        **(
            _license_fields("LICENSE", project.license)
            if project.license is not None
            else {}
        ),
        **({"notice": "NOTICE"} if project.notice is not None else {}),
    }


__all__ = [
    "LicenseNames",
    "license_names",
    "scene_credits",
    "work_entry",
]
