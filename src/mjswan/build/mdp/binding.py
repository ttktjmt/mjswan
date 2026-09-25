"""The one check a custom-TS binding passes before its ``to_dict()`` is written."""

from __future__ import annotations

from typing import Any


def require_ts_src(kind: str, name: str, binding: Any) -> None:
    """A ``*Binding`` without ``ts_src`` names a class the browser does not have.

    mjswan ships no built-in TS term classes, so a binding is only ever the custom-TS
    escape hatch: without the file the term goes missing from a bundle that reports
    itself complete.
    """
    if binding.ts_src:
        return
    raise ValueError(
        f"{kind} term {name!r} is bound to TS class {binding.ts_name or '(unnamed)'!r} "
        "but no `ts_src` was given, so the browser has no implementation to run: "
        "mjswan ships no built-in TS term classes. Either let the build trace the "
        f"term's own function, or point `ts_src` at a `.ts` file exporting "
        f"{binding.ts_name or 'the class'!r}."
    )
