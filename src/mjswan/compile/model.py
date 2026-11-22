"""``env.sim.model`` for an event body: reads become graph inputs, writes outputs.

mjlab's domain randomization writes per-world model fields (``geom_friction``,
``body_mass``, ...) through ``env.sim.model`` and reads compiled values through
``env.sim.get_default_field``. Neither touches the real model here: discovery works on
a copy, and replay on a field rebuilt from the graph's inputs. The document names
elements rather than numbering them, since a plain scene's trace env numbers them apart
from the browser's model.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_leaves, tree_map

from .proxy import _SHAPE_QUERIES, _FieldProxy, _plain
from .slot import _EVENT_ENV_READS, _MODEL_NS, SlotKey, _sim_tensor, refuse_read

#: A model field's element kind by name prefix, the ``mjtObj`` naming it, and the
#: ``MjModel`` count of that kind.
_ELEMENT_KINDS: tuple[tuple[str, str, Any, str], ...] = (
    ("body_", "body", mujoco.mjtObj.mjOBJ_BODY, "nbody"),
    ("geom_", "geom", mujoco.mjtObj.mjOBJ_GEOM, "ngeom"),
    ("site_", "site", mujoco.mjtObj.mjOBJ_SITE, "nsite"),
    ("jnt_", "joint", mujoco.mjtObj.mjOBJ_JOINT, "njnt"),
    ("dof_", "dof", None, "nv"),
    ("actuator_", "actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, "nu"),
    ("tendon_", "tendon", mujoco.mjtObj.mjOBJ_TENDON, "ntendon"),
    ("cam_", "camera", mujoco.mjtObj.mjOBJ_CAMERA, "ncam"),
    ("light_", "light", mujoco.mjtObj.mjOBJ_LIGHT, "nlight"),
    ("mat_", "material", mujoco.mjtObj.mjOBJ_MATERIAL, "nmat"),
    ("tex_", "texture", mujoco.mjtObj.mjOBJ_TEXTURE, "ntex"),
    ("pair_", "pair", mujoco.mjtObj.mjOBJ_PAIR, "npair"),
    ("eq_", "equality", mujoco.mjtObj.mjOBJ_EQUALITY, "neq"),
)
#: Fields indexed by ``qpos`` address, named like a dof: by joint and offset.
_QPOS_FIELDS = frozenset({"qpos0", "qpos_spring"})

#: Fields whose change leaves MuJoCo's precomputed constants stale, for a body that
#: does not say so through mjlab's ``requires_model_fields(..., recompute=...)``.
SET_CONST_FIELDS = frozenset({"body_ipos", "body_mass", "body_inertia", "dof_armature"})


def _element_kind(name: str) -> tuple[str, Any, str]:
    if name in _QPOS_FIELDS:
        return "qpos", None, "nq"
    for prefix, kind, obj, count in _ELEMENT_KINDS:
        if name.startswith(prefix):
            return kind, obj, count
    raise ValueError(
        f"Event term reads model field {name!r}, whose elements the browser cannot "
        "name: only body, geom, site, joint, dof, qpos, actuator, tendon, camera, "
        "light, material, texture, pair and equality fields are served."
    )


def element_refs(mj_model: Any, name: str, ids: list[int]) -> dict[str, Any]:
    """``{"element", "names"[, "offsets"]}`` naming elements *ids* of field *name*.

    A dof or ``qpos`` address is its joint's name and the offset within that joint.
    """
    kind, obj, _count = _element_kind(name)
    if kind in ("dof", "qpos"):
        if kind == "dof":
            joints = [int(mj_model.dof_jntid[i]) for i in ids]
            starts = mj_model.jnt_dofadr
        else:
            joints = [bisect.bisect_right(mj_model.jnt_qposadr, i) - 1 for i in ids]
            starts = mj_model.jnt_qposadr
        names = [
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints
        ]
        refs: dict[str, Any] = {
            "element": kind,
            "names": names,
            "offsets": [int(i - starts[j]) for i, j in zip(ids, joints)],
        }
    else:
        names = [mujoco.mj_id2name(mj_model, obj, i) for i in ids]
        refs = {"element": kind, "names": names}
    unnamed = [i for i, n in zip(ids, names) if not n]
    if unnamed:
        raise ValueError(
            f"Event term touches {kind} {unnamed} of model field {name!r}, which have "
            "no name in the model; the browser finds elements by name, so name them."
        )
    return refs


@dataclass(frozen=True)
class Layout:
    """A model field's tensor shape, and which axis runs over its elements."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    axis: int
    """1 for a per-world field, whose axis 0 is the world; else 0."""
    count: int

    @property
    def inner(self) -> tuple[int, ...]:
        return self.shape[self.axis + 1 :]

    def rows(self, value: torch.Tensor, ids: list[int]) -> torch.Tensor:
        """``(1, len(ids), *inner)``: elements *ids* of *value*, world 0's."""
        return (value[0] if self.axis == 1 else value)[ids].unsqueeze(0)

    def element_index(self, ids: list[int]) -> tuple[Any, ...]:
        return (0, ids) if self.axis == 1 else (ids,)


def _layout(mj_model: Any, name: str, value: torch.Tensor) -> Layout:
    count = int(getattr(mj_model, _element_kind(name)[2]))
    shape = tuple(value.shape)
    if len(shape) >= 2 and shape[0] == 1 and shape[1] == count:
        axis = 1
    elif shape and shape[0] == count:
        axis = 0
    else:
        raise ValueError(
            f"Model field {name!r} is shaped {shape}, with no axis over its {count} "
            "elements."
        )
    return Layout(shape, value.dtype, axis, count)


def _touched(layout: Layout, index: Any) -> set[int]:
    """The elements an index into the field reaches."""
    view = [1] * len(layout.shape)
    view[layout.axis] = layout.count
    ids = torch.arange(layout.count).view(view).expand(layout.shape)
    return set(ids[index].reshape(-1).tolist())


def _cells(layout: Layout, index: Any) -> torch.Tensor:
    """``(m, ndim)`` positions an index writes, in row-major order."""
    mask = torch.zeros(layout.shape, dtype=torch.bool)
    mask[index] = True
    return mask.nonzero()


def _int64_index(index: Any) -> Any:
    """*index* with its integer tensors widened to int64, which ONNX's Gather and
    ScatterND require; mjlab indexes with int32."""

    def widen(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor) and leaf.dtype in (torch.int16, torch.int32):
            return leaf.long()
        return leaf

    return tree_map(widen, index)


@dataclass
class _Field:
    layout: Layout
    original: torch.Tensor
    shadow: torch.Tensor
    read: set[int] = field(default_factory=set)


@dataclass
class ModelWrite:
    """One captured write: the field, the positions it set, and the values set there."""

    name: str
    positions: torch.Tensor
    values: torch.Tensor


class ModelRecorder:
    """Discovery's model: reads note the elements they reach, writes land on a copy."""

    def __init__(self, sim: Any):
        self._sim = sim
        self.fields: dict[tuple[str, bool], _Field] = {}
        self.writes: list[ModelWrite] = []

    def field(self, name: str, default: bool = False) -> torch.Tensor:
        key = (name, default)
        state = self.fields.get(key)
        if state is None:
            # Before the read, so a field with no elements (`opt`) is refused by name.
            _element_kind(name)
            if default:
                value = self._sim.get_default_field(name)
            else:
                value = _sim_tensor(getattr(self._sim.model, name))
            value = value.detach()
            state = _Field(
                _layout(self._sim.mj_model, name, value), value.clone(), value.clone()
            )
            self.fields[key] = state
        proxy = state.shadow.as_subclass(_ModelField)
        proxy._model = (self, key)
        return proxy

    def read(self, key: tuple[str, bool], index: Any) -> torch.Tensor:
        state = self.fields[key]
        state.read |= _touched(state.layout, index)
        return state.shadow[index]

    def read_whole(self, key: tuple[str, bool]) -> None:
        state = self.fields[key]
        state.read |= set(range(state.layout.count))

    def write(self, key: tuple[str, bool], index: Any, value: Any) -> None:
        name, default = key
        if default:
            raise ValueError(
                f"Event term writes the compiled default of model field {name!r}."
            )
        state = self.fields[key]
        state.shadow[index] = _plain(value)
        positions = _cells(state.layout, index)
        values = state.shadow[tuple(positions.T)].to(torch.float32).clone()
        self.writes.append(ModelWrite(name, positions, values))

    def slots(self) -> dict[SlotKey, tuple[torch.Tensor, list[int]]]:
        """Per field read: the read elements' rows, the graph input, and their ids."""
        out = {}
        for (name, default), state in self.fields.items():
            if state.read:
                ids = sorted(state.read)
                rows = state.layout.rows(state.original, ids).to(torch.float32)
                out[(_MODEL_NS, f"{name}.default" if default else name)] = (rows, ids)
        return out

    def slot_json(self, key: SlotKey, ids: list[int]) -> dict[str, Any]:
        name, _, source = key[1].partition(".")
        entry = {"model": name, **element_refs(self._sim.mj_model, name, ids)}
        if source == "default":
            entry["default"] = True
        return entry

    def write_target(self, write: ModelWrite, output: str) -> dict[str, Any]:
        """The browser's description of one write: elements by name, cells within."""
        layout = self.fields[(write.name, False)].layout
        elements = write.positions[:, layout.axis].tolist()
        ids = sorted(set(elements))
        local = {i: at for at, i in enumerate(ids)}
        inner = write.positions[:, layout.axis + 1 :]
        strides = [math.prod(layout.inner[d + 1 :]) for d in range(len(layout.inner))]
        offsets = (inner * torch.tensor(strides, dtype=torch.long)).sum(dim=1).tolist()
        return {
            "kind": "model",
            "field": write.name,
            **element_refs(self._sim.mj_model, write.name, ids),
            "cells": [[local[e], int(o)] for e, o in zip(elements, offsets)],
            "outputs": [output],
        }

    def plan(self) -> ModelPlan:
        """What replay needs to rebuild every field and re-capture every write."""
        return ModelPlan(
            layouts={key: s.layout for key, s in self.fields.items()},
            read={key: sorted(s.read) for key, s in self.fields.items() if s.read},
            writes=[
                np.ravel_multi_index(
                    tuple(w.positions.T.numpy()),
                    self.fields[(w.name, False)].layout.shape,
                ).tolist()
                for w in self.writes
            ],
        )


@dataclass(frozen=True)
class ModelPlan:
    layouts: dict[tuple[str, bool], Layout]
    read: dict[tuple[str, bool], list[int]]
    writes: list[list[int]]
    """Per write, in order: the flat positions it sets."""


class ModelReplay:
    """Replay's model: each field rebuilt from the rows discovery read, which arrive as
    graph inputs, with every write's cells gathered into an output."""

    def __init__(self, plan: ModelPlan, served: dict[tuple[str, bool], torch.Tensor]):
        self._plan = plan
        self._served = served
        self._fields: dict[tuple[str, bool], torch.Tensor] = {}
        self.outputs: list[torch.Tensor] = []

    def field(self, name: str, default: bool = False) -> torch.Tensor:
        key = (name, default)
        value = self._fields.get(key)
        if value is None:
            layout = self._plan.layouts.get(key)
            if layout is None:
                raise AttributeError(
                    f"Event term read model field {name!r} during tracing, which the "
                    "discovery pass never saw: the term's control flow is "
                    "input-dependent, which is not traceable."
                )
            value = torch.zeros(layout.shape, dtype=layout.dtype)
            ids = self._plan.read.get(key)
            if ids:
                value[layout.element_index(ids)] = self._served[key][0].to(layout.dtype)
            self._fields[key] = value
        proxy = value.as_subclass(_ModelField)
        proxy._model = (self, key)
        return proxy

    def read(self, key: tuple[str, bool], index: Any) -> torch.Tensor:
        return self._fields[key][_int64_index(index)]

    def read_whole(self, key: tuple[str, bool]) -> None:
        pass

    def write(self, key: tuple[str, bool], index: Any, value: Any) -> None:
        field_value = self._fields[key]
        field_value[_int64_index(index)] = _plain(value)
        positions = torch.tensor(self._plan.writes[len(self.outputs)], dtype=torch.long)
        self.outputs.append(field_value.reshape(-1)[positions].to(torch.float32))


class _ModelField(_FieldProxy):
    """A model field: indexing goes to the pass's model, any other use reads it whole."""

    _model: tuple[Any, tuple[str, bool]]

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
            model, key = args[0]._model
            return model.read(key, args[1])
        if func is torch.Tensor.__setitem__ and isinstance(args[0], cls):
            model, key = args[0]._model
            model.write(key, args[1], args[2])
            return None
        if getattr(func, "__name__", None) not in _SHAPE_QUERIES:
            for leaf in tree_leaves((args, kwargs)):
                if isinstance(leaf, cls):
                    model, key = leaf._model
                    model.read_whole(key)
        return func(*tree_map(_plain, args), **tree_map(_plain, kwargs))


class ModelSim:
    """``env.sim`` for an event body: ``model`` and ``get_default_field`` only."""

    def __init__(self, model: ModelRecorder | ModelReplay, real_sim: Any):
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_real", real_sim)

    @property
    def model(self) -> _ModelView:
        return _ModelView(self._model)

    def get_default_field(self, name: str) -> torch.Tensor:
        return self._model.field(name, default=True)

    @property
    def per_world_default_fields(self) -> Any:
        return self._real.per_world_default_fields

    def __getattr__(self, name: str) -> Any:
        raise refuse_read(self._real, name, _EVENT_ENV_READS, "sim.")


class _ModelView:
    _model: ModelRecorder | ModelReplay

    def __init__(self, model: ModelRecorder | ModelReplay):
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name: str) -> torch.Tensor:
        return self._model.field(name)


def read_model_slot(env: Any, key: SlotKey, ids: list[int]) -> torch.Tensor:
    """A model slot's current value off *env*: the rows of elements *ids*."""
    name, _, source = key[1].partition(".")
    if source == "default":
        value = env.sim.get_default_field(name)
    else:
        value = _sim_tensor(getattr(env.sim.model, name))
    value = value.detach()
    return _layout(env.sim.mj_model, name, value).rows(value, ids).to(torch.float32)


__all__ = [
    "SET_CONST_FIELDS",
    "Layout",
    "ModelPlan",
    "ModelRecorder",
    "ModelReplay",
    "ModelSim",
    "ModelWrite",
    "element_refs",
    "read_model_slot",
]
