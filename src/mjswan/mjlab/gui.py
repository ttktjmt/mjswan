"""Record an mjlab command term's viser GUI as a control-panel descriptor.

Running ``CommandTerm.create_gui`` against a recording stand-in makes mjlab's own
declaration the control panel's only definition, so its slider ranges cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Handle:
    """One recorded control, doubling as the handle ``create_gui`` keeps.

    mjlab stores handles to read ``.value`` later and its ``on_update`` callbacks
    assign ``.min``/``.max``; plain attributes cover both. Callbacks are never
    replayed — they only re-derive what the recording already holds.
    """

    kind: str
    label: str
    value: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    icon: str | None = None

    def on_update(self, fn: Any) -> Any:
        return fn

    def on_click(self, fn: Any) -> Any:
        return fn


@dataclass
class _Folder:
    """Context manager only. The title is dropped: mjlab uses
    ``name.capitalize()``, which the browser already derives from the term name
    (``ControlPanel.formatGroupName``)."""

    label: str

    def __enter__(self) -> _Folder:
        return self

    def __exit__(self, *_: Any) -> None:
        pass


@dataclass
class _GuiRecorder:
    """``server.gui``, recording in call order.

    Only the ``add_*`` mjlab's command terms use — an unknown one raises
    ``AttributeError`` rather than silently dropping a control the viewer shows.
    """

    recorded: list[_Handle] = field(default_factory=list)

    def _record(self, handle: _Handle) -> _Handle:
        self.recorded.append(handle)
        return handle

    def add_folder(self, label: str, **_: Any) -> _Folder:
        return _Folder(label)

    def add_checkbox(
        self, label: str, initial_value: bool = False, **_: Any
    ) -> _Handle:
        return self._record(_Handle("checkbox", label, value=bool(initial_value)))

    def add_slider(
        self,
        label: str,
        min: float | None = None,
        max: float | None = None,
        step: float | None = None,
        initial_value: float | None = None,
        **_: Any,
    ) -> _Handle:
        return self._record(
            _Handle("slider", label, value=initial_value, min=min, max=max, step=step)
        )

    def add_button(self, label: str, icon: Any = None, **_: Any) -> _Handle:
        # viser's `Icon` members are plain tabler names (`Icon.SQUARE_X` == "square-x").
        return self._record(
            _Handle("button", label, icon=str(icon) if icon is not None else None)
        )


@dataclass
class _ServerRecorder:
    gui: _GuiRecorder = field(default_factory=_GuiRecorder)


def _slug(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


def to_ui_descriptor(handles: list[_Handle]) -> dict[str, Any] | None:
    """Recorded controls -> a ``commands.<term>.ui`` descriptor, ``None`` if empty.

    mjlab identifies handles by variable reference, so the conventions here are
    structural rather than label-based:

    - The first checkbox is named ``enabled``, which is what ``OnnxCommand.isUiEnabled``
      looks for.
    - A one-sided slider is a "Max <label>" companion rescaling the next axis, never an
      axis itself — an axis straddles zero.
    - Order is the contract: axis sliders map onto the command vector positionally.
    """
    inputs: list[dict[str, Any]] = []
    enable_name: str | None = None
    companion: dict[str, Any] | None = None

    for handle in handles:
        if handle.kind == "checkbox":
            name = "enabled" if enable_name is None else _slug(handle.label)
            enable_name = enable_name or name
            inputs.append(
                {
                    "type": "checkbox",
                    "name": name,
                    "label": handle.label,
                    "default": bool(handle.value),
                }
            )
        elif handle.kind == "slider":
            if handle.min is not None and handle.min >= 0:
                companion = {
                    "min": handle.min,
                    "max": handle.max,
                    "step": handle.step,
                    "default": handle.value,
                }
                continue
            entry: dict[str, Any] = {
                "type": "slider",
                "name": _slug(handle.label),
                "label": handle.label,
                "min": handle.min,
                "max": handle.max,
                "step": handle.step,
                "default": handle.value,
            }
            if enable_name is not None:
                entry["enabled_when"] = enable_name
            if companion is not None:
                # No `label`: the browser synthesizes `Max <label>`, matching mjlab.
                entry["adjustable_range"] = companion
                companion = None
            inputs.append(entry)
        elif handle.kind == "button":
            entry = {
                "type": "button",
                "name": _slug(handle.label),
                "label": handle.label,
            }
            if handle.icon is not None:
                entry["icon"] = handle.icon
            inputs.append(entry)

    return {"inputs": inputs} if inputs else None


def record_gui(term: Any, name: str) -> dict[str, Any] | None:
    """A built term's ``create_gui`` recorded as a UI descriptor.

    ``None`` when it declares no controls — ``create_gui`` is a base-class no-op, so an
    empty recording is the only signal that a term does not override it.
    """
    server = _ServerRecorder()
    term.create_gui(name, server, lambda: 0)
    return to_ui_descriptor(server.gui.recorded)


__all__ = ["record_gui", "to_ui_descriptor"]
