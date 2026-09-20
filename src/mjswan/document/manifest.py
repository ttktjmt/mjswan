"""What the manifest is: its file name, the document format it declares, and the slot
defaults a policy entry may leave out (ADR 0006 §5, §7)."""

from __future__ import annotations

MANIFEST_NAME = "manifest.json"

#: The structure a build writes: what files exist, where they sit, and what the manifest
#: says about them. Bumped by hand, only when an engine reading the old structure would
#: misread the new one. Distinct from ``version``, the mjswan release: a host picks an
#: engine by ``version``, an engine protects itself by ``format`` (ADR 0006 §7).
#:
#: 2: ``input_slots`` may carry a raw ``mjData`` field (``{"sim": ...}``), narrowed to
#: ``rows``. A format-1 engine accepts the entry, cannot serve it, and freezes the
#: observation group at its previous value.
DOCUMENT_FORMAT = 2

#: Input slots the runtime fills itself rather than from an observation group: the
#: recurrent carry (``is_init``, ``adapt_hx``) and the step counter (``time_step``).
RUNTIME_INPUT_SLOTS = frozenset({"is_init", "adapt_hx", "time_step"})

#: What the runtime assumes when a policy declares no slot table (ADR 0006 §5).
DEFAULT_IN_KEYS = ("actor",)
DEFAULT_OUT_KEYS = ("action",)

__all__ = [
    "DEFAULT_IN_KEYS",
    "DEFAULT_OUT_KEYS",
    "DOCUMENT_FORMAT",
    "MANIFEST_NAME",
    "RUNTIME_INPUT_SLOTS",
]
