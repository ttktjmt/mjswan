"""License files in a build (ADR 0007): where they sit, what they say, and what
``publish`` does about them.

The naming rule, identifier and tier table mirror mjswan Cloud's ``@mjswan/licenses``
and are kept in step by hand; the platform's copy decides what a publish is accepted
with.
"""

from .attribution import (
    KNOWN_ASSETS,
    Attribution,
    KnownAsset,
    LicenseDeclaration,
    declare,
    detect_attributions,
    known_asset,
    known_attribution,
    spec_asset_directories,
)
from .path import (
    LICENSE_CONTENT_TYPE,
    LICENSE_FILE_MAX_BYTES,
    LicenseLocation,
    component_id,
    has_license_basename,
    is_license_file_path,
    license_path_problem,
    parse_license_path,
)
from .spdx import (
    BLOCKED,
    CUSTOM,
    GENERATABLE_LICENSES,
    Identification,
    blocked_refusal,
    display_name,
    generate_license_text,
    identify_license,
    license_template,
    resolve_license,
    resolve_notice,
    restricted_warning,
    restriction_label,
    tier_of,
)

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
