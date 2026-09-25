"""Whether an object is mjlab's, without importing mjlab."""

from __future__ import annotations

from typing import Any


def is_from_mjlab(obj: Any) -> bool:
    """Whether any class in *obj*'s MRO comes from ``mjlab``: a task's subclass of an
    mjlab config reports its own module, so the leaf class alone is not enough."""
    return any(
        (getattr(klass, "__module__", "") or "").startswith("mjlab")
        for klass in type(obj).__mro__
    )


__all__ = ["is_from_mjlab"]
