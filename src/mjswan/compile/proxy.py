"""Stand-ins for live mjlab objects, shared by the discovery and replay passes.

A proxy subclasses the real object's class, so the ``isinstance`` checks in term bodies
keep passing, and replaces only what a pass needs to see: a sensor's ``.data``, a
command's tensor attributes, ``env.sim``. A raw sim field travels as a
:class:`_FieldProxy`, a tensor subclass whose indexing the tracer watches.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Collection

import torch

from .slot import _TERM_ENV_READS, UnsupportedEnvRead


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

    def __init__(self, data: Any):
        object.__setattr__(self, "data", data)

    def __getattr__(self, name: str) -> Any:
        raise UnsupportedEnvRead(f"sim.{name}", _TERM_ENV_READS)
