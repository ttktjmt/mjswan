"""Stand-ins for live mjlab objects, shared by the discovery and replay passes.

A proxy subclasses the real object's class, so the ``isinstance`` checks in term bodies
keep passing, and replaces only what a pass needs to see: a sensor's ``.data``, a
command's tensor attributes, ``env.sim``. A raw sim field travels as a
:class:`_FieldProxy`, a tensor subclass whose indexing the tracer watches.
"""

from __future__ import annotations

import collections
import copy
import functools
import types
from typing import Any, Callable, Collection

import torch

from .slot import _TERM_ENV_READS, _forward_env_attr, refuse_read

#: Torch calls that read a tensor's shape, never its values: a proxy treats them as
#: no read at all.
_SHAPE_QUERIES = frozenset({"__get__", "size", "dim", "ndimension", "numel", "__len__"})


def _traces_through(data: Any, name: str, reader_fields: Collection[str]) -> bool:
    """Whether ``data.<name>`` is a property to run against the sim proxy.

    Only a property can be: a plain tensor field has no math to trace, so it stays a
    slot of its own.
    """
    if name in reader_fields:
        return False
    return isinstance(getattr(type(data), name, None), property)


def _with_sim_data(data: Any, sim: Any) -> Any:
    """A shallow copy of a real ``EntityData`` with ``data`` swapped for ``sim``.

    ``EntityData`` is a plain dataclass and ``data`` an ordinary field, so its
    properties run unchanged against the proxy; ``indexing``, ``model`` and the tensor
    fields (``encoder_bias``, ``gravity_vec_w``) stay real and bake as constants.
    """
    replay = copy.copy(data)
    object.__setattr__(replay, "data", sim)
    return replay


def _class_proxy(real: Any, overrides: dict[str, Any]) -> Any:
    """A stand-in for a live mjlab object that still satisfies ``isinstance`` checks.

    Terms assert on concrete classes (``builtin_sensor`` on ``BuiltinSensor``, say),
    so subclass the real object's class and share its ``__dict__``, replacing only
    what ``overrides`` names.
    """
    cls = type(real)
    proxy_cls = type(f"_Proxy{cls.__name__}", (cls,), overrides)
    proxy = object.__new__(proxy_cls)
    proxy.__dict__ = real.__dict__
    return proxy


def _sensor_proxy(real: Any, get_data: Callable[[], Any]) -> Any:
    """A sensor stand-in whose ``.data`` comes from ``get_data``."""
    return _class_proxy(real, {"data": property(lambda _self: get_data())})


def _command_proxy(real: Any, on_tensor: Callable[[str, Any], Any]) -> Any:
    """A command-term stand-in routing every tensor attribute through ``on_tensor``.

    A command's state lives in plain instance attributes, so ``__getattr__`` never
    fires and ``__getattribute__`` is the only hook that sees the read.
    """

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            return on_tensor(attr, value)
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def action_term_window(manager: Any, name: str) -> tuple[int, int]:
    """Where action term *name*'s slice of the policy's action vector starts, and its
    width, as ``ActionManager.process_action`` splits the vector in config order.

    Raises for a name the manager does not hold, as mjlab's ``get_term`` does.
    """
    names = list(manager.active_terms)
    offset = 0
    for term_name, dim in zip(names, manager.action_term_dim, strict=True):
        if term_name == name:
            return offset, int(dim)
        offset += int(dim)
    raise ValueError(
        f"Term read action term {name!r}, which the scene does not define. "
        f"Available: {', '.join(names) if names else '(none)'}."
    )


def _forward_action_attr(manager: Any, name: str) -> Any:
    """``manager.<name>`` for a non-tensor (a width, the term names); raise for a tensor
    the runtime does not hold, which would otherwise bake as a constant."""
    value = getattr(manager, name)
    if isinstance(value, torch.Tensor):
        raise refuse_read(manager, name, _TERM_ENV_READS, "action_manager.")
    return value


@functools.cache
def _live_classes() -> tuple[type, ...]:
    try:
        from mjlab.entity import Entity
        from mjlab.entity.data import EntityData
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
        from mjlab.scene import Scene
        from mjlab.sensor import Sensor
        from mjlab.sim.sim import Simulation
        from mjlab.sim.sim_data import TorchArray, WarpBridge
    except ImportError:
        return ()
    return (
        Entity,
        EntityData,
        ManagerBasedRlEnv,
        ManagerBase,
        ManagerTermBase,
        Scene,
        Sensor,
        Simulation,
        TorchArray,
        WarpBridge,
    )


def _is_live_object(value: Any, depth: int = 0) -> bool:
    """Whether *value* holds sim state that changes under a term: an entity or its
    data, a sensor, the scene, the sim or one of its arrays, the env, a manager or its
    term, or a container of one. A read through it bypasses the tracer and would bake
    as a constant."""
    if isinstance(value, (torch.Tensor, str, bytes, int, float, type(None))):
        return False
    if isinstance(value, (list, tuple, set, frozenset)):
        return depth < 3 and any(_is_live_object(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return depth < 3 and any(_is_live_object(v, depth + 1) for v in value.values())
    if isinstance(value, _live_classes()):
        return True
    if (type(value).__module__ or "").split(".")[0] == "mujoco_warp":
        return True
    return hasattr(value, "data") and (
        hasattr(value, "write_joint_state_to_sim") or hasattr(value, "update")
    )


#: Values compared by identity alone: their contents are code, not state.
_OPAQUE = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.ModuleType,
    type,
    functools.partial,
)

_CONTAINERS = (list, tuple, set, frozenset, collections.deque)


def _attrs(value: Any) -> dict[str, Any] | None:
    """A plain object's attributes, from its ``__dict__`` and its ``__slots__``, or
    ``None`` for code and anything else without either."""
    if isinstance(value, _OPAQUE):
        return None
    state = getattr(value, "__dict__", None)
    found = dict(state) if isinstance(state, dict) else None
    for cls in type(value).__mro__:
        slots = cls.__dict__.get("__slots__", ())
        for slot in (slots,) if isinstance(slots, str) else slots:
            if slot in ("__dict__", "__weakref__"):
                continue
            found = {} if found is None else found
            mangled = slot.startswith("__") and not slot.endswith("__")
            try:
                found[slot] = getattr(
                    value, f"_{cls.__name__.lstrip('_')}{slot}" if mangled else slot
                )
            except AttributeError:  # an unset slot
                pass
    return found


def _holds_live(value: Any, within: frozenset[int] = frozenset()) -> bool:
    """Whether *value* is a live object or holds one at any depth, inside containers
    and plain objects. A reference back to a value in *within* (ids of the values
    enclosing it) is passed over."""
    if id(value) in within:
        return False
    if _is_live_object(value):
        return True
    within = within | {id(value)}
    if isinstance(value, dict):
        members = value.values()
    elif isinstance(value, _CONTAINERS):
        members = value
    else:
        members = (_attrs(value) or {}).values()
    return any(_holds_live(v, within) for v in members)


class _EnvConstants:
    """A term's ``_env`` read off a proxy: the forwarded constants, nothing live."""

    def __init__(self, real: Any):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        return _forward_env_attr(self._real, name, _TERM_ENV_READS)


#: The processed action's backing field, which the browser serves for every term.
_PROCESSED_FIELD = "_processed_actions"


def _action_term_proxy(
    real: Any,
    raw_action: Callable[[], torch.Tensor],
    read_state: Callable[[str, torch.Tensor], torch.Tensor],
    traced: frozenset[str] | None,
) -> Any:
    """An action-term stand-in.

    A native term (*traced* ``None``) serves ``raw_action`` from ``raw_action()``, a
    traced one its state fields (*traced*) through ``read_state``, and either its
    processed action through ``read_state`` (ADR 0009 §5). Any other tensor, or a live
    object such as its entity, raises: the browser holds neither, so its value would
    bake. A property runs on the stand-in, so what it reads comes back through here.
    """

    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        if attr == "raw_action" and traced is None:
            return raw_action()
        if attr == "_env":
            return _EnvConstants(object.__getattribute__(self, attr))
        if isinstance(getattr(type(real), attr, None), property):
            return object.__getattribute__(self, attr)
        value = object.__getattribute__(self, attr)
        if isinstance(value, torch.Tensor):
            if attr == _PROCESSED_FIELD or (traced is not None and attr in traced):
                return read_state(attr, value)
        elif not _holds_live(value, frozenset({id(real)})):
            return value
        raise refuse_read(real, attr, _TERM_ENV_READS, "action_manager.get_term(...).")

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def entity_static(name: str, value: Any) -> Any:
    """An entity attribute that is neither a tensor nor a scalar, as a term may use it.

    Static structure (``indexing``, name lists, methods) passes through; an actuator
    refuses a ``set_*`` call, which would change Python-side state the browser does not
    run, mutating the trace env and capturing nothing.
    """
    if name == "actuators":
        return [_actuator_guard(actuator) for actuator in value]
    return value


def _actuator_guard(real: Any) -> Any:
    def __getattribute__(self: Any, attr: str) -> Any:  # noqa: N807
        value = object.__getattribute__(self, attr)
        if attr.startswith("set_") and callable(value):
            raise ValueError(
                f"Event term calls {type(real).__name__}.{attr}(), which sets actuator "
                "state in Python that the browser does not run, so the change would be "
                "lost. The browser takes PD gains from the policy's action config."
            )
        return value

    return _class_proxy(real, {"__getattribute__": __getattribute__})


def _is_sensor(scene: Any, name: str) -> bool:
    """Whether ``scene[name]`` resolves to a sensor rather than an entity."""
    sensors = getattr(scene, "sensors", None)
    return bool(sensors) and name in sensors


class _FieldProxy(torch.Tensor):
    """A raw sim field with ``__getitem__`` in the tracer's hands; every other torch
    function sees the plain tensor. Never constructed directly: ``as_subclass``."""


def _plain(value: Any) -> Any:
    return value.as_subclass(torch.Tensor) if isinstance(value, _FieldProxy) else value


class _SimStandIn:
    """``env.sim`` for a term: ``.data`` is the pass's sim proxy, and nothing else.

    The rest of ``Simulation`` (``mj_model``, ``forward()``) would act on the real env
    mid-trace, so it raises.
    """

    def __init__(self, data: Any, real: Any = None):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name: str) -> Any:
        raise refuse_read(self._real, name, _TERM_ENV_READS, "sim.")
