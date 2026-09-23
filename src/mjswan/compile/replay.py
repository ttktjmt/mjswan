"""Replay pass: serve the recorded slots back to the term while torch traces it.

The term runs a second time inside an ``nn.Module``'s ``forward``, against stand-ins
that answer every read from the dynamic inputs and baked constants the discovery pass
classified. A read the discovery pass never saw raises: the term's control flow would
be input-dependent, which is not traceable (ADR 0005 §Consequences).
"""

from __future__ import annotations

import warnings
from typing import Any, Collection, cast

import torch
from torch.utils._pytree import tree_map

from .proxy import (
    _command_proxy,
    _FieldProxy,
    _plain,
    _sensor_proxy,
    _SimStandIn,
    _traces_through,
    _with_sim_data,
)
from .record import _index_rows, _WriteCaptureMixin
from .slot import (
    _COMMAND_NS,
    _EVENT_ENV_READS,
    _SENSOR_NS,
    _SIM_NS,
    _TERM_ENV_READS,
    READER_FIELDS,
    SlotKey,
    TaggedKey,
    _forward_env_attr,
)


class _NarrowedField(_FieldProxy):
    """A raw field the graph takes already narrowed to ``rows``.

    The term still indexes it by the field's own ids; those are remapped here to
    positions in the input. A read of exactly the rows, in order, is the input itself,
    so no Gather enters the graph.
    """

    _rows: list[int]
    _size: int

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__ and isinstance(args[0], cls):
            return args[0]._remapped(args[1])
        return func(*tree_map(_plain, args), **tree_map(_plain, kwargs))

    def _remapped(self, index: Any) -> torch.Tensor:
        with warnings.catch_warnings():
            # The index is a model constant (`indexing.body_ids`), the same one
            # discovery read; the tracer cannot know that and warns on `tolist()`.
            warnings.filterwarnings(
                "ignore", message="Converting a tensor to a Python list"
            )
            rows = _index_rows(index, self._size)
        lookup = {row: at for at, row in enumerate(self._rows)}
        if rows is None or any(row not in lookup for row in rows):
            raise ValueError(
                f"A term indexed a narrowed sim field by rows the discovery pass did "
                f"not see ({index!r}); the index depends on the input, which is not "
                "traceable."
            )
        positions = [lookup[row] for row in rows]
        parts = cast("tuple[Any, ...]", index)
        sel = parts[1]
        new: Any
        if isinstance(sel, int) or (isinstance(sel, torch.Tensor) and sel.dim() == 0):
            new = positions[0]
        elif positions == list(range(len(self._rows))):
            new = slice(None)
        else:
            new = torch.tensor(positions, dtype=torch.long)
        return _plain(self)[(parts[0], new, *parts[2:])]


def _narrowed_field(tensor: torch.Tensor, rows: list[int], size: int) -> _NarrowedField:
    field = tensor.as_subclass(_NarrowedField)
    field._rows = rows
    field._size = size
    return field


class _ReplaySimData:
    """Serves recorded raw ``SimData`` fields during the replay pass.

    A narrowed one arrives as a :class:`_NarrowedField`, which remaps the term's
    indexing to the rows the input carries.
    """

    def __init__(self, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (_SIM_NS, name)
        if key not in self._slots:
            raise AttributeError(
                f"sim field {name!r} was not recorded during discovery"
            )
        return self._slots[key]


class _ReplayData:
    """Serves recorded slots; a traced-through property recomputes off the sim proxy."""

    def __init__(
        self,
        entity: str,
        slots: dict[SlotKey, torch.Tensor],
        real_env: Any,
        reader_fields: Collection[str],
    ):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_slots", slots)
        object.__setattr__(self, "_real_env", real_env)
        object.__setattr__(self, "_reader_fields", reader_fields)
        object.__setattr__(self, "_traced", None)

    def _through(self) -> Any:
        """A copy of the live ``EntityData`` (its indexing and constants) over the
        replay sim proxy, built on first use."""
        if self._traced is None:
            real = self._real_env.scene[self._entity].data
            object.__setattr__(
                self, "_traced", _with_sim_data(real, _ReplaySimData(self._slots))
            )
        return self._traced

    def __getattr__(self, name: str) -> Any:
        if name == "data":
            return self._through().data
        key = (self._entity, name)
        if key in self._slots:
            return self._slots[key]
        if _traces_through(
            self._real_env.scene[self._entity].data, name, self._reader_fields
        ):
            return getattr(self._through(), name)
        raise AttributeError(
            f"Term read undeclared slot {key!r} during tracing. This field "
            "was not seen in the discovery pass: the term's control flow is "
            "input-dependent, which is not traceable (ADR 0005 §Consequences)."
        )


class _ReplayEntity:
    def __init__(
        self,
        entity: str,
        slots: dict[SlotKey, torch.Tensor],
        real_env: Any,
        reader_fields: Collection[str],
    ):
        self.data = _ReplayData(entity, slots, real_env, reader_fields)


class _ReplaySensorData:
    """Serves a structured sensor's recorded fields during the replay pass."""

    def __init__(self, sensor: str, slots: dict[SlotKey, torch.Tensor]):
        object.__setattr__(self, "_sensor", sensor)
        object.__setattr__(self, "_slots", slots)

    def __getattr__(self, name: str) -> torch.Tensor:
        key = (_SENSOR_NS, f"{self._sensor}.{name}")
        if key not in self._slots:
            raise AttributeError(
                f"sensor field {self._sensor}.{name} was not recorded during discovery"
            )
        return self._slots[key]


class _ReplayScene:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
        *,
        real_env: Any,
        reader_fields: Collection[str],
    ):
        self._slots = slots
        self._sensors = sensors or {}
        self._real_env = real_env
        self._reader_fields = reader_fields

    def __getitem__(self, name: str) -> Any:
        real = self._sensors.get(name)
        if real is not None:
            whole = (_SENSOR_NS, name)
            if whole in self._slots:
                return _sensor_proxy(real, lambda: self._slots[whole])
            # Structured sensor: the discovery pass recorded its fields separately.
            return _sensor_proxy(real, lambda: _ReplaySensorData(name, self._slots))
        return _ReplayEntity(name, self._slots, self._real_env, self._reader_fields)


class _ReplayCommandManager:
    """Serves recorded command-state slots back during tracing."""

    def __init__(self, slots: dict[SlotKey, torch.Tensor], commands: dict[str, Any]):
        self._slots = slots
        self._commands = commands

    def get_term(self, name: str) -> Any:
        real = self._commands.get(name)
        if real is None:
            raise AttributeError(
                f"Term read command {name!r} during tracing that the discovery pass "
                "never saw: the term's control flow is input-dependent, which is "
                "not traceable (ADR 0005 §Consequences)."
            )
        return _command_proxy(
            real, lambda attr, _v: self._slots[(_COMMAND_NS, f"{name}.{attr}")]
        )

    def get_command(self, name: str) -> torch.Tensor:
        return self._slots[(_COMMAND_NS, f"{name}.command")]


class _ReplayEnv:
    def __init__(
        self,
        slots: dict[SlotKey, torch.Tensor],
        sensors: dict[str, Any] | None = None,
        commands: dict[str, Any] | None = None,
        *,
        real_env: Any,
        reader_fields: Collection[str] = READER_FIELDS,
    ):
        self.scene = _ReplayScene(
            slots, sensors, real_env=real_env, reader_fields=reader_fields
        )
        self.command_manager = _ReplayCommandManager(slots, commands or {})
        self.sim = _SimStandIn(_ReplaySimData(slots))
        self._real_env = real_env

    def __getattr__(self, name: str) -> Any:
        # Forwarded, not copied, so nothing drifts from the real env.
        return _forward_env_attr(self._real_env, name, _TERM_ENV_READS)


class _EvReplayData:
    def __init__(self, entity: str, served: dict[TaggedKey, Any]):
        object.__setattr__(self, "_entity", entity)
        object.__setattr__(self, "_served", served)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("data", self._entity, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared data slot ('data', {self._entity!r}, "
                f"{name!r}) during tracing (input-dependent read?)."
            ) from None


class _EvReplayEntity(_WriteCaptureMixin):
    def __init__(self, entity, served, captures):
        object.__setattr__(self, "_name", entity)
        object.__setattr__(self, "_served", served)
        object.__setattr__(self, "data", _EvReplayData(entity, served))
        object.__setattr__(self, "_captures", captures)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._served[("attr", self._name, name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared attr ('attr', {self._name!r}, {name!r})."
            ) from None


class _EvReplayScene:
    def __init__(self, served, captures, real_env: Any):
        self._served = served
        self._captures = captures
        self._real_env = real_env

    def __getitem__(self, name: str) -> _EvReplayEntity:
        return _EvReplayEntity(name, self._served, self._captures)

    def __getattr__(self, name: str) -> Any:
        if name == "entities":
            # Static structure, so the keys come from the real env; only the
            # entities' tensors are recorded slots.
            return {
                key: _EvReplayEntity(key, self._served, self._captures)
                for key in self._real_env.scene.entities
            }
        try:
            return self._served[("scene", name)]
        except KeyError:
            raise AttributeError(
                f"Event term read undeclared scene attr ('scene', {name!r})."
            ) from None


class _EventReplayEnv:
    def __init__(self, served, captures, *, real_env: Any):
        self.scene = _EvReplayScene(served, captures, real_env)
        self._real_env = real_env

    def __getattr__(self, name: str) -> Any:
        # Forwarded, so replay sees the same `num_envs` discovery ran against.
        return _forward_env_attr(self._real_env, name, _EVENT_ENV_READS)
