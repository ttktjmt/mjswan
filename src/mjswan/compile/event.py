"""Trace an event term body to ONNX.

An event returns ``None`` and writes via ``entity.write_*_to_sim`` or into
``env.sim.model``, so the tensors it would write become the graph outputs, and its
randomness is threaded in as an explicit ``rand`` input replayed by
:class:`.rng.ReplayRng`.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Iterator

import numpy as np
import torch
from torch import nn

from .export import (
    DivergentGraph,
    _classify_tagged,
    _const_values,
    _export_onnx,
    _prepare_single_env_export,
    _register_consts,
    check_against_module,
    no_hidden_reads,
)
from .model import SET_CONST_FIELDS, ModelPlan, ModelRecorder, ModelReplay
from .proxy import _is_live_object
from .record import (
    WriteCaptures,
    _EventCaptureEnv,
    _flatten_captures,
    entity_write_target,
)
from .replay import _EventReplayEnv
from .rng import DrawRecorder, ReplayRng
from .slot import _MODEL_NS, SlotKey, TaggedKey, _slot_input_name


def _state_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _state_snapshot(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {key: _state_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_state_snapshot(item) for item in value)
    if isinstance(value, (str, bytes, bool, int, float, type(None))):
        return value
    return ("opaque", id(value))


def _same_state(before: Any, after: Any) -> bool:
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        return (
            before.dtype == after.dtype
            and before.shape == after.shape
            and torch.equal(before, after)
        )
    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        return before.dtype == after.dtype and np.array_equal(
            before, after, equal_nan=True
        )
    if type(before) is not type(after):
        return False
    if isinstance(before, dict):
        return before.keys() == after.keys() and all(
            _same_state(before[k], after[k]) for k in before
        )
    if isinstance(before, tuple):
        return len(before) == len(after) and all(
            _same_state(a, b) for a, b in zip(before, after)
        )
    return before == after


@contextmanager
def _event_instance(
    func: Callable[..., None], proxy: Any, real_env: Any
) -> Iterator[None]:
    """Route a bound event's cached env/entities through the same write capture."""
    owner = getattr(func, "__self__", None)
    if owner is None or not hasattr(owner, "__dict__"):
        yield
        return
    replacements = {id(real_env): proxy, id(real_env.scene): proxy.scene}
    entities = getattr(real_env.scene, "entities", {})
    for name, entity in entities.items():
        if not hasattr(entity, "data"):
            continue
        replacements[id(entity)] = proxy.scene[name]
        replacements[id(entity.data)] = proxy.scene[name].data
    original = {}
    for name, value in vars(owner).items():
        if id(value) in replacements:
            original[name] = value
        elif _is_live_object(value):
            raise ValueError(
                f"Event instance caches unsupported live state in {name!r}; read it through the env argument"
            )
    try:
        for name, value in original.items():
            setattr(owner, name, replacements[id(value)])
        before = _state_snapshot(
            {k: v for k, v in vars(owner).items() if k not in original}
        )
        yield
        after = _state_snapshot(
            {k: v for k, v in vars(owner).items() if k not in original}
        )
        if not _same_state(before, after):
            raise DivergentGraph(
                "Event instance mutates state between calls; stateful event history is not supported"
            )
    finally:
        for name, value in original.items():
            setattr(owner, name, value)


class _EventModule(nn.Module):
    """Wraps a side-effecting event ``func`` so ``forward(*dynamic, rand)`` returns
    the tensors the term would write.

    Dynamic reads arrive as ``forward`` args, constants as buffers or plain Python
    values; all are served back through the replay env.
    """

    def __init__(
        self,
        func: Callable[..., None],
        params: dict[str, Any],
        dynamic_keys: list[SlotKey],
        tensor_consts: dict[TaggedKey, torch.Tensor],
        scalar_consts: dict[TaggedKey, Any],
        *,
        env_ids: torch.Tensor | None,
        real_env: Any,
        model_plan: ModelPlan | None = None,
    ):
        super().__init__()
        self._func = func
        self._params = params
        self._env_ids = env_ids
        self._dynamic_keys = dynamic_keys
        self._scalar_consts = scalar_consts
        self._real_env = real_env
        self._model_plan = model_plan
        self._const_buffers = _register_consts(self, tensor_consts)

    def forward(self, *args: torch.Tensor):
        *dynamic, rand = args
        served: dict[TaggedKey, Any] = dict(self._scalar_consts)
        served.update(_const_values(self, self._const_buffers))
        rows: dict[tuple[str, bool], torch.Tensor] = {}
        for key, tensor in zip(self._dynamic_keys, dynamic):
            if key[0] == _MODEL_NS:
                name, _, source = key[1].partition(".")
                rows[(name, source == "default")] = tensor
            else:
                served[("data", *key)] = tensor
        captures: WriteCaptures = {}
        model = ModelReplay(self._model_plan, rows) if self._model_plan else None
        env = _EventReplayEnv(served, captures, real_env=self._real_env, model=model)
        with (
            ReplayRng(self._func, rand=rand),
            _event_instance(self._func, env, self._real_env),
        ):
            self._func(env, self._env_ids, **self._params)
        _, tensors = _flatten_captures(captures)
        return (*tensors, *(model.outputs if model else ()))


@dataclass
class EventExport:
    """The result of tracing one event term body to ONNX."""

    name: str
    mode: str
    onnx_bytes: bytes
    input_slots: list[SlotKey]
    input_names: list[str]
    rand_dim: int
    rand_ranges: list[list[float]]
    """Per-element ``[low, high]`` the runtime draws ``rand`` from."""
    output_names: list[str]
    write_targets: list[dict[str, Any]]
    """Per write-kind descriptor: what the outputs target (entity, kind, fields)."""
    reference_outputs: tuple[torch.Tensor, ...]
    reference_rand: torch.Tensor
    constant_slots: list[str] = field(default_factory=list)
    input_shapes: list[list[int]] = field(default_factory=list)
    """Traced shape of each input slot, parallel to ``input_slots``."""
    input_rows: list[list[int] | None] = field(default_factory=list)
    """Per input slot, the model element ids it carries, or None."""
    slot_extras: dict[SlotKey, dict[str, Any]] = field(default_factory=dict)
    """A model slot's naming of its elements, for the manifest."""
    set_const: bool = False
    """Whether the browser owes ``mj_setConst`` after the model writes."""


def trace_event_term(
    func: Callable[..., None],
    params: dict[str, Any],
    env: Any,
    *,
    name: str,
    mode: str,
    is_global_time: bool = False,
    opset: int = 17,
) -> EventExport:
    """Trace a side-effecting (write-to-sim) event term body to ONNX.

    The written tensors become the graph outputs and randomness arrives as ``rand``.
    Time-varying ``entity.data`` fields become graph inputs; everything else the term
    reads (scene tensors, control-flow scalars) is baked in. The body gets the
    ``env_ids`` mjlab would pass it in ``mode``.
    """
    env_ids = _env_ids(mode, is_global_time=is_global_time)
    log: list[tuple[TaggedKey, Any]] = []
    captures: WriteCaptures = {}
    proxy = _EventCaptureEnv(env, log, captures, model=True)
    with (
        DrawRecorder(func) as rec,
        no_hidden_reads(f"Event term {name!r}"),
        _event_instance(func, proxy, env),
    ):
        func(proxy, env_ids, **params)
    recorder: ModelRecorder | None = proxy.model
    model_writes = recorder.writes if recorder else []

    if not captures and not model_writes:
        raise ValueError(
            f"Event term {name!r} wrote nothing traceable (no write_joint_state / "
            "write_root_state / write_root_link_pose / write_root_link_velocity call, "
            "and no env.sim.model write); handle it natively or extend _WRITE_FIELDS."
        )
    output_names, ref_tensors = _event_outputs(captures, recorder)
    ref_rand = rec.rand_vector
    rand_dim = rec.rand_dim
    rand_ranges = rec.rand_ranges

    dynamic, tensor_consts, scalar_consts = _classify_tagged(log)
    model_rows = recorder.slots() if recorder else {}
    for key, (rows, _ids) in model_rows.items():
        dynamic[key] = rows

    dynamic_keys = sorted(dynamic)
    dyn_input_names = [_slot_input_name(k) for k in dynamic_keys]
    example = tuple(dynamic[k] for k in dynamic_keys) + (ref_rand,)
    input_names = [*dyn_input_names, "rand"]

    module = _EventModule(
        func,
        params,
        dynamic_keys,
        tensor_consts,
        scalar_consts,
        env_ids=env_ids,
        real_env=env,
        model_plan=recorder.plan() if recorder else None,
    ).eval()
    _prepare_single_env_export(env.num_envs)
    # `rand` keeps its traced length: it is one flat draw vector, not a batch of rows.
    onnx_bytes = _export_onnx(
        module,
        example,
        input_names=input_names,
        output_names=output_names,
        batch_axis=[*dyn_input_names, *output_names],
        opset=opset,
    )
    check_against_module(
        module,
        onnx_bytes,
        example,
        input_names=input_names,
        output_names=output_names,
        what=f"Event term {name!r}",
        rand_ranges=rand_ranges,
        fixed=[n for k, n in zip(dynamic_keys, dyn_input_names) if k[0] == _MODEL_NS],
    )

    # The write says which entity it landed on; `asset_cfg` is the fallback.
    asset_cfg = params.get("asset_cfg")
    asset_name = getattr(asset_cfg, "name", None)
    write_targets = []
    for key, values in captures.items():
        entity, kind = key
        target = entity_write_target(key, values, asset_name)
        # `asset_cfg`'s ids scope its own entity; any other one gets all of its joints.
        joint_ids = _static_ids(getattr(asset_cfg, "joint_ids", None))
        scoped = entity is None or entity == asset_name
        if kind == "joint_state" and scoped and joint_ids is not None:
            target["joint_ids"] = joint_ids
        write_targets.append(target)
    if recorder:
        model_outputs = output_names[len(output_names) - len(model_writes) :]
        write_targets += [
            recorder.write_target(write, output)
            for write, output in zip(model_writes, model_outputs)
        ]

    return EventExport(
        name=name,
        mode=mode,
        onnx_bytes=onnx_bytes,
        input_slots=dynamic_keys,
        input_names=dyn_input_names,
        rand_dim=rand_dim,
        rand_ranges=rand_ranges,
        output_names=output_names,
        write_targets=write_targets,
        reference_outputs=tuple(t.detach() for t in ref_tensors),
        reference_rand=ref_rand.detach(),
        constant_slots=[":".join(str(p) for p in k) for k in sorted(tensor_consts)],
        input_shapes=[list(dynamic[k].shape) for k in dynamic_keys],
        input_rows=[
            model_rows[k][1] if k in model_rows else None for k in dynamic_keys
        ],
        slot_extras={
            k: recorder.slot_json(k, ids)
            for k, (_rows, ids) in model_rows.items()
            if recorder
        },
        set_const=bool(model_writes)
        and (
            int(getattr(func, "recompute", 0)) > 0
            or any(w.name in SET_CONST_FIELDS for w in model_writes)
        ),
    )


def _event_outputs(
    captures: WriteCaptures, model: ModelRecorder | None
) -> tuple[list[str], list[torch.Tensor]]:
    """Every output an event writes: entity writes, then model writes in call order."""
    names, tensors = _flatten_captures(captures)
    for i, write in enumerate(model.writes if model else ()):
        names.append(f"model__{write.name}__{i}")
        tensors.append(write.values)
    return names, tensors


def _env_ids(mode: str, *, is_global_time: bool = False) -> torch.Tensor | None:
    """The ``env_ids`` mjlab's ``EventManager.apply`` passes a term: ``None`` at startup
    and for a global-time interval, else the firing env's id (0 in a single-env trace).
    A body that indexes ``data[env_ids]`` traces an extra axis on ``None``."""
    if mode == "startup" or (mode == "interval" and is_global_time):
        return None
    return torch.arange(1)


def _static_ids(ids: Any) -> Any:
    if isinstance(ids, slice):
        return "all"
    if hasattr(ids, "tolist"):
        return ids.tolist()
    return ids


#: Events whose trace legitimately captures no write, and why. The build fails on any
#: other untraceable event: a dropped reset randomization is invisible in the browser.
EVENTS_WITH_NOTHING_TO_WRITE: dict[str, str] = {
    "randomize_terrain": (
        "it re-draws each env's sub-terrain origin, and the browser has one baked terrain "
        "with one origin"
    ),
    "encoder_bias": (
        "it writes `Entity.data.encoder_bias`, which the runtime applies from the policy "
        "config's `encoder_bias` rather than from an event graph"
    ),
    "reset_scene_to_default": (
        "it restores every entity's default root and joint state, which is what the "
        "runtime's own reset already does (`mj_resetData` to `qpos0`, or keyframe 0) "
        'before any `mode="reset"` event runs'
    ),
}


def _term_arg(func: Any, params: dict[str, Any], key: str) -> Any:
    """A keyword as the body sees it: the term's value, else *func*'s default."""
    if key in params:
        return params[key]
    param = inspect.signature(func).parameters.get(key)
    return (
        None
        if param is None or param.default is inspect.Parameter.empty
        else param.default
    )


def event_writes_nothing_reason(
    func: Callable[..., None], env: Any, params: dict[str, Any]
) -> str | None:
    """Why *func*'s trace legitimately captured no write, or ``None``."""
    func_name = getattr(func, "__name__", "")
    reason = EVENTS_WITH_NOTHING_TO_WRITE.get(func_name)
    if reason is not None:
        return reason
    if not func_name.startswith("reset_root_state"):
        return None
    # A root write cannot move a fixed-base entity, in mjlab either, and its
    # manipulation tasks still configure `reset_base` on their arms, leaving
    # `asset_cfg` to the signature default, hence `_term_arg` and not `params`.
    entity_name = getattr(_term_arg(func, params, "asset_cfg"), "name", None)
    if entity_name is None:
        return None
    try:
        entity = env.scene[entity_name]
    except (KeyError, TypeError):
        return None
    if getattr(entity, "is_fixed_base", False):
        return f"entity {entity_name!r} is fixed-base, so a root write cannot move it"
    return None
