"""Command-term configuration and registration.

Named after ``mjlab.managers.command_manager``. As in mjlab, each policy owns a
dictionary of command terms, each producing a vector observations consume. The browser
UI is metadata on top: manual slider/button/checkbox controls are one built-in
``UiCommand`` term (:func:`mjswan.envs.mdp.commands.ui_command`), and an mjlab command
class is bound by cfg-class name through :func:`register_command`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

CommandType = Literal["slider", "button", "checkbox"]


@dataclass
class SliderRangeConfig:
    """A companion slider that rescales another slider's drag range.

    Mirrors the "Max <label>" slider mjlab's play GUI pairs with each velocity axis.
    Purely presentational: the browser clamps the value slider's displayed range to
    ``[-value, value]`` locally and sends nothing to the engine.
    """

    range: tuple[float, float] = (0.0, 2.0)
    default: float = 1.0
    step: float = 0.05
    label: str | None = None
    """Companion slider's label; defaults browser-side to ``Max <label>``."""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "min": self.range[0],
            "max": self.range[1],
            "step": self.step,
            "default": self.default,
        }
        if self.label is not None:
            data["label"] = self.label
        return data


@dataclass
class SliderConfig:
    """Configuration for a slider input exposed by a command term."""

    name: str
    label: str
    range: tuple[float, float] = (-1.0, 1.0)
    default: float = 0.0
    step: float = 0.01
    enabled_when: str | None = None
    """Optional input name in the same command group that enables this slider."""
    adjustable_range: SliderRangeConfig | None = None
    """Optional companion slider rescaling this one's reach — see
    :class:`SliderRangeConfig`. Symmetric around zero."""

    @property
    def min(self) -> float:
        return self.range[0]

    @property
    def max(self) -> float:
        return self.range[1]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "slider",
            "name": self.name,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "default": self.default,
        }
        if self.enabled_when is not None:
            data["enabled_when"] = self.enabled_when
        if self.adjustable_range is not None:
            data["adjustable_range"] = self.adjustable_range.to_dict()
        return data


Slider = SliderConfig


@dataclass
class CheckboxConfig:
    """Configuration for a checkbox input exposed by a command term."""

    name: str
    label: str
    default: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "checkbox",
            "name": self.name,
            "label": self.label,
            "default": self.default,
        }


Checkbox = CheckboxConfig


@dataclass
class ButtonConfig:
    """Configuration for a button input exposed by a command term."""

    name: str
    label: str
    icon: str | None = None
    """Tabler icon name, as viser's ``Icon`` spells it (``Icon.SQUARE_X`` is
    ``"square-x"``); the browser draws only the ones it bundles."""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "button",
            "name": self.name,
            "label": self.label,
        }
        if self.icon is not None:
            data["icon"] = self.icon
        return data


Button = ButtonConfig

CommandInput: TypeAlias = SliderConfig | ButtonConfig | CheckboxConfig


@dataclass
class CommandUiConfig:
    """Optional UI metadata attached to a command term."""

    inputs: list[CommandInput] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": [inp.to_dict() for inp in self.inputs]}


@dataclass
class PendingCommandTrace:
    """An mjlab ``CommandTermCfg`` not yet traced to ONNX.

    Tracing needs a live env and an output directory, neither available at
    ``add_policy()`` time, so it is deferred to build time as for the other term kinds.
    """

    mjlab_cfg: Any
    """The raw mjlab ``CommandTermCfg``."""

    state_fields: list[str]
    """Attribute names on the built term constituting its hidden state."""

    command_field: str
    """Which state field is the command value."""

    trace_override: Callable[[Any], None] | None = None
    """Hook mutating a freshly ``build()``-constructed term in place, e.g. rebinding
    ``_resample_command`` to a trace-friendly implementation."""

    ui: dict[str, Any] | None = None
    """Author-authored control-panel descriptor, already resolved to a concrete dict."""

    viz: list[dict[str, Any]] | None = None
    """Debug-vis primitives; :func:`mjswan.mjlab.command.default_viz` fills these in
    when none are given."""


@dataclass
class CommandTermConfig:
    """Serialized browser-side command-term configuration."""

    term_name: str
    params: dict[str, Any] = field(default_factory=dict)
    ui: CommandUiConfig | None = None
    pending_trace: PendingCommandTrace | None = None
    """When set, this term is not yet resolved — see :class:`PendingCommandTrace`."""
    pending_reset_trace: PendingResetTrace | None = None
    """A native term's reset-time graph, not yet traced — see :class:`PendingResetTrace`."""

    def to_dict(self) -> dict[str, Any]:
        if self.pending_trace is not None or self.pending_reset_trace is not None:
            raise TypeError(
                f"CommandTermConfig({self.term_name!r}) is pending ONNX trace — "
                "use mjswan.build.mdp.serialize_command(name, cfg, env, out_dir) "
                "instead of to_dict() directly (the Builder does this automatically)."
            )
        data = {"name": self.term_name, **self.params}
        if self.ui is not None:
            data["ui"] = self.ui.to_dict()
        return data


@dataclass(frozen=True)
class PendingResetTrace:
    """A reset-time graph a *native* command term applies.

    A native command can still have randomization that is term math rather than a data
    lookup — ``MotionCommand``'s reference-state jitter, say — which traces exactly like
    a reset Event while the class itself stays native.
    """

    func: Callable[..., None]
    """Event-shaped body, ``func(env, env_ids, **params)``, ending in ``write_*_to_sim``."""

    params: dict[str, Any]
    """Resolved params for *func* (``SceneEntityCfg``s included)."""


@dataclass(frozen=True)
class CommandBinding:
    """Binding from an mjlab command-cfg class name to its browser command term.

    Three mutually-exclusive shapes:

    - **Native**: ``ts_name`` names a permanently-native TS class and ``serializer``
      builds its params from the mjlab cfg. It may still declare ``reset_trace``, a
      ``(mjlab_cfg) -> (func, params) | None`` hook naming one reset-time body to trace
      (see :class:`PendingResetTrace`): the class stays native, its randomization does
      not.
    - **ONNX-traced**: ``state_fields``/``command_field`` set, so the term is built and
      traced at build time and served by the shared ``OnnxCommand`` handler. Set
      ``trace_override`` when it needs a trace-friendly rewrite first. ``ui`` and ``viz``
      may each be a value or a ``(mjlab_cfg) -> value`` callable; an omitted ``viz``
      falls back to :func:`mjswan.mjlab.command.default_viz`.
    - **``ts_src`` escape hatch**: a hand-written TS command term.
    """

    ts_name: str = ""
    serializer: Callable[[Any], Mapping[str, Any]] | None = None
    ts_src: str | None = None
    state_fields: list[str] | None = None
    command_field: str | None = None
    trace_override: Callable[[Any], None] | None = None
    ui: dict[str, Any] | Callable[[Any], dict[str, Any]] | None = None
    viz: list[dict[str, Any]] | Callable[[Any], list[dict[str, Any]]] | None = None
    reset_trace: (
        Callable[[Any], tuple[Callable[..., None], dict[str, Any]] | None] | None
    ) = None

    @property
    def is_onnx_traced(self) -> bool:
        return self.state_fields is not None and self.command_field is not None


_custom_registry: dict[str, CommandBinding] = {}


def register_command(mjlab_name: str, spec: CommandBinding) -> None:
    """Register a custom command term adapter.

    ``mjlab_name`` should typically be the mjlab config class name, e.g.
    ``"LiftingCommandCfg"``.
    """

    _custom_registry[mjlab_name] = spec


__all__ = [
    "Button",
    "ButtonConfig",
    "Checkbox",
    "CheckboxConfig",
    "CommandBinding",
    "CommandInput",
    "CommandTermConfig",
    "CommandType",
    "CommandUiConfig",
    "PendingCommandTrace",
    "PendingResetTrace",
    "Slider",
    "SliderConfig",
    "_custom_registry",
    "register_command",
]
