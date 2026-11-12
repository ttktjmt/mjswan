"""Shared trace mechanics: classifying reads, narrowing inputs, baking constants, the
``torch.onnx.export`` call itself, and the checks a graph passes before it ships."""

from __future__ import annotations

import contextlib
import io
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Protocol, Sequence
from unittest import mock

import numpy as np
import torch
from torch import nn

from .record import _merge_narrowing, _RecordingSimData
from .replay import _narrowed_field
from .slot import (
    _ACTION_NS,
    _ACTION_TERM_NS,
    _COMMAND_NS,
    _ENV_NS,
    _SENSOR_NS,
    _SIM_NS,
    SimRows,
    SlotKey,
    TaggedKey,
    _is_dynamic_field,
    watch_refused_reads,
)


def _register_consts(
    module: nn.Module, constants: dict[Any, torch.Tensor], prefix: str = "_const"
) -> dict[Any, str]:
    """Register each constant as a buffer, returning ``slot key -> buffer name``.

    Buffers rather than plain attributes so ``torch.onnx.export`` folds them into the
    graph instead of tracing them as free-floating tensors.
    """
    names: dict[Any, str] = {}
    for i, (key, value) in enumerate(constants.items()):
        buffer_name = f"{prefix}_{i}"
        module.register_buffer(buffer_name, value.detach().clone())
        names[key] = buffer_name
    return names


def _const_values(module: nn.Module, names: dict[Any, str]) -> dict[Any, torch.Tensor]:
    """The registered constants, keyed by the slot each one serves."""
    return {key: getattr(module, name) for key, name in names.items()}


def _export_onnx(
    module: nn.Module,
    example: tuple[torch.Tensor, ...],
    *,
    input_names: list[str],
    output_names: list[str],
    batch_axis: list[str],
    opset: int,
) -> bytes:
    """Export ``module`` to ONNX bytes with ``batch_axis`` names given a dynamic axis 0.

    ``dynamo=False``: the TorchScript tracer records the concrete tensor ops we want,
    while torch.export traces Python control flow and trips on the proxies.
    """
    buffer = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module,
            example,
            buffer,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={n: {0: "batch"} for n in batch_axis},
            opset_version=opset,
            dynamo=False,
        )
    import onnx

    model = onnx.load_from_string(
        _restore_output_names(buffer.getvalue(), output_names)
    )
    if not model.graph.input and not any(
        node.op_type in _RANDOM_OPS for node in model.graph.node
    ):
        # TorchScript can emit invalid index_put graphs for empty index tensors,
        # even though the entire no-input reset is a constant computation.
        with torch.no_grad():
            produced = module(*(value.clone() for value in example))
        outputs = produced if isinstance(produced, tuple) else (produced,)
        if len(outputs) != len(output_names):
            raise ValueError("Constant graph output count differs from requested names")
        nodes, values = [], []
        for name, value in zip(output_names, outputs):
            tensor = onnx.numpy_helper.from_array(value.detach().cpu().numpy())
            nodes.append(onnx.helper.make_node("Constant", [], [name], value=tensor))
            values.append(
                onnx.helper.make_tensor_value_info(name, tensor.data_type, tensor.dims)
            )
        model.graph.CopyFrom(
            onnx.helper.make_graph(nodes, model.graph.name, [], values)
        )
    _check_structure(model, "Exported graph")
    return model.SerializeToString()


def _check_structure(model: Any, what: str) -> None:
    """Check even graphs numeric parity cannot run or deliberately skips."""
    import onnx

    try:
        onnx.checker.check_model(model, full_check=True)
    except (onnx.checker.ValidationError, onnx.shape_inference.InferenceError) as exc:
        raise DivergentGraph(
            f"{what} failed ONNX structural validation: {exc}"
        ) from exc


def _restore_output_names(onnx_bytes: bytes, output_names: list[str]) -> bytes:
    """Give every graph output the name it was exported under.

    An output that constant-folds to an initializer is named after it (``"14"``), but
    consumers look outputs up by the requested name, so an ``Identity`` restores it.
    """
    import onnx
    from onnx import helper

    model = onnx.load_from_string(onnx_bytes)
    graph = model.graph
    renamed = False
    for output, wanted in zip(graph.output, output_names):
        if output.name == wanted:
            continue
        graph.node.append(helper.make_node("Identity", [output.name], [wanted]))
        output.name = wanted
        renamed = True
    return model.SerializeToString() if renamed else onnx_bytes


def _narrow_slots(slots: dict[SlotKey, torch.Tensor], sim_rows: SimRows) -> None:
    """Hand the replay the narrowed sim inputs as fields that remap the term's ids."""
    for key, (rows, size) in sim_rows.items():
        slots[key] = _narrowed_field(slots[key], rows, size)


def _narrow_inputs(
    dynamic: dict[SlotKey, torch.Tensor], sims: Sequence[_RecordingSimData]
) -> SimRows:
    """Narrow each sim example input to the rows the reads touched (see
    :func:`_merge_narrowing`); returns what was narrowed, for the module and export."""
    sim_rows = _merge_narrowing(sims)
    for key, (rows, _size) in sim_rows.items():
        dynamic[key] = dynamic[key][:, rows]
    return sim_rows


def _input_rows(
    dynamic_keys: Sequence[SlotKey], sim_rows: SimRows
) -> list[list[int] | None]:
    """Per input slot, the rows it carries, or None for a whole field."""
    return [sim_rows[k][0] if k in sim_rows else None for k in dynamic_keys]


def _classify_slots(
    log: list[tuple[SlotKey, Any]],
    dynamic: dict[SlotKey, torch.Tensor],
    constants: dict[SlotKey, torch.Tensor],
) -> bool:
    """Split a recorded read log into graph inputs and baked constants.

    Sensor, command-state, raw sim-data, env-counter and last-action reads are live
    state by definition; an entity data field is dynamic unless it is a model-derived
    constant. Returns whether *this* log contributed a dynamic slot, which a group's
    caller needs per term.
    """
    saw_dynamic = False
    for key, value in log:
        if not isinstance(value, torch.Tensor):
            continue  # non-tensor attribute access, not a graph slot
        namespace, field_name = key
        live = namespace in (
            _SENSOR_NS,
            _COMMAND_NS,
            _SIM_NS,
            _ENV_NS,
            _ACTION_NS,
            _ACTION_TERM_NS,
        )
        if live or _is_dynamic_field(field_name):
            dynamic.setdefault(key, value)
            saw_dynamic = True
        else:
            constants.setdefault(key, value)
    return saw_dynamic


def _classify_tagged(
    log: list[tuple[TaggedKey, Any]],
) -> tuple[
    dict[SlotKey, torch.Tensor], dict[TaggedKey, torch.Tensor], dict[TaggedKey, Any]
]:
    """Split an event/command read log into dynamic inputs, tensor and scalar constants.

    Only a time-varying ``entity.data`` field or another command's state becomes a
    graph input; scene tensors and control-flow scalars are baked.
    """
    dynamic: dict[SlotKey, torch.Tensor] = {}
    tensor_consts: dict[TaggedKey, torch.Tensor] = {}
    scalar_consts: dict[TaggedKey, Any] = {}
    for key, value in log:
        is_tensor = isinstance(value, torch.Tensor)
        if key[0] == _COMMAND_NS and is_tensor:
            dynamic.setdefault(key, value)
        elif key[0] == "data" and _is_dynamic_field(key[2]) and is_tensor:
            dynamic.setdefault((key[1], key[2]), value)
        elif is_tensor:
            tensor_consts.setdefault(key, value)
        else:
            scalar_consts.setdefault(key, value)
    return dynamic, tensor_consts, scalar_consts


_EXPORT_FILTERS_INSTALLED = False


def _prepare_single_env_export(num_envs: int) -> None:
    """Refuse a batched trace, and silence the three warnings a single-env one raises.

    All three are safe here: the `index_put_` mutation is read back as a graph output,
    `len(env_ids)` bakes the row count this guard pins to 1, and the `torch.tensor`
    constants are config that cannot vary.
    """
    if num_envs != 1:
        raise ValueError(
            f"tracing with num_envs={num_envs}: the graph would bake that row count "
            "while the runtime feeds one row (session.ts pins the batch axis to 1). "
            "Trace single-env, as Project.add_mjlab_task does."
        )
    global _EXPORT_FILTERS_INSTALLED
    if _EXPORT_FILTERS_INSTALLED:
        return
    warnings.filterwarnings(
        "ignore",
        message="ONNX Preprocess - Removing mutation from node aten::index_put_",
        category=UserWarning,
    )
    # No category: `TracerWarning` is not on `torch.jit`'s public stub.
    warnings.filterwarnings("ignore", message="Using len to get tensor shape")
    warnings.filterwarnings(
        "ignore", message="torch.tensor results are registered as constants"
    )
    _EXPORT_FILTERS_INSTALLED = True


GraphRunner = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


class _Graph(Protocol):
    onnx_bytes: bytes
    output_names: list[str]


def reference_runner(graph: _Graph) -> GraphRunner:
    """*graph*, run by ONNX's reference evaluator, which needs no runtime."""
    import onnx

    return _runner(onnx.load_from_string(graph.onnx_bytes), graph.output_names)


def _runner(model: Any, output_names: Sequence[str]) -> GraphRunner:
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    declared = set(evaluator.input_names)

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = evaluator.run(
            list(output_names), {k: v for k, v in feeds.items() if k in declared}
        )
        return {n: np.asarray(o) for n, o in zip(output_names, outs)}

    return run


class DivergentGraph(ValueError):
    """A traced graph that would not compute what its term computes in mjlab."""


_REPLACE = "register a trace-friendly replacement with the matching mjswan.register_* function."
_FIX = (
    "Compute it with tensor operations (`torch.where` for a branch), pass a fixed value "
    f"in the term's params, or {_REPLACE}"
)


@contextlib.contextmanager
def no_hidden_reads(what: str) -> Iterator[None]:
    """Fail if the body run inside swallowed a refused read mjlab's env would answer."""
    with watch_refused_reads() as refused:
        yield
    if refused:
        reads = ", ".join(f"`env.{read}`" for read in dict.fromkeys(refused))
        raise DivergentGraph(
            f"{what} read {reads}, which the tracer does not serve, and went on without "
            "it (a `getattr` default, `hasattr` or `except AttributeError`). mjlab's env "
            "has it, so the graph would follow a path the term does not take in mjlab. "
            "Read the value through what the tracer serves, pass it in the term's "
            f"params, or {_REPLACE}"
        )


#: Input sets per check, the trace inputs first; seeded, so a verdict is reproducible.
_CHECK_SAMPLES = 8

#: Ops that draw: such a graph draws apart from its body, so it is not compared.
_RANDOM_OPS = frozenset(
    {
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    }
)

#: Tracer warnings a single-env trace raises on sound code (see
#: :func:`_prepare_single_env_export`), left out of a disagreement's hints.
_SOUND_TRACER_WARNINGS = (
    "Using len to get tensor shape",
    "torch.tensor results are registered as constants",
)


def check_against_module(
    module: nn.Module,
    onnx_bytes: bytes,
    example: tuple[torch.Tensor, ...],
    *,
    input_names: Sequence[str],
    output_names: Sequence[str],
    what: str,
    columns: Sequence[tuple[str, int]] = (),
    rand_ranges: Sequence[Sequence[float]] | None = None,
    fixed: Collection[str] = (),
) -> None:
    """Refuse a graph that disagrees with the module it was exported from.

    The export records tensor operations only, so whatever the body decides in Python
    (``.item()``, an ``if`` on a tensor, a counter) is fixed at its trace-time value. The
    module, run as Python on the trace inputs and on perturbed ones, decides anew.
    *columns* names the term behind each span of the first output's last axis, for a
    fused group; *rand_ranges* bounds a trailing ``rand`` input; *fixed* names inputs held
    at their trace values: an event's model rows, which its body may check in Python, as
    mjlab's ``dr.geom_size`` checks its geom types.
    """
    import onnx

    model = onnx.load_from_string(onnx_bytes)
    _check_structure(model, what)
    if any(node.op_type in _RANDOM_OPS for node in model.graph.node):
        return
    run = _runner(model, output_names)
    held = [name in fixed for name in input_names]
    generator = torch.Generator().manual_seed(0)
    for sample in range(_CHECK_SAMPLES):
        inputs = (
            example
            if sample == 0
            else _perturbed(example, sample, generator, rand_ranges, held)
        )
        where = "on its trace inputs" if sample == 0 else "on perturbed inputs"
        feeds = {n: x.detach().cpu().numpy() for n, x in zip(input_names, inputs)}
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = run(feeds)
        except Exception:
            continue  # inputs the graph cannot run on leave nothing to compare
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                produced = module(*(x.clone() for x in inputs))
        except Exception as exc:
            # The graph ran, so the body failed on a read its trace never saw.
            raise DivergentGraph(
                f"{what} traced to a graph that disagrees with its own body {where}: "
                f"the body raised {type(exc).__name__}: {exc} {_FIX}"
            ) from exc
        expected = produced if isinstance(produced, tuple) else (produced,)
        details = (
            [f"the body gives {len(expected)} outputs, the graph {len(output_names)}"]
            if len(expected) != len(output_names)
            else [
                _apart(outputs[name], want, columns if i == 0 else ())
                for i, (name, want) in enumerate(zip(output_names, expected))
            ]
        )
        detail = next((d for d in details if d is not None), None)
        if detail is None:
            continue
        hints = _tracer_hints(module, example)
        raise DivergentGraph(
            f"{what} traced to a graph that disagrees with its own body {where} "
            f"({detail}). The export records tensor operations only, so a value the "
            "body works out in Python is fixed at its trace-time value: `.item()`, "
            "`float()` or `bool()` of a tensor, an `if` on a tensor, a counter or "
            "other state kept between calls."
            + (f" The tracer flagged {'; '.join(hints)}." if hints else "")
            + f" {_FIX}"
        )


def _perturbed(
    example: tuple[torch.Tensor, ...],
    sample: int,
    generator: torch.Generator,
    rand_ranges: Sequence[Sequence[float]] | None,
    held: Sequence[bool],
) -> tuple[torch.Tensor, ...]:
    """*example* redrawn for *sample* (1 on): a float negated, then scaled by a random
    factor of either sign; a bool flipped, then drawn; a draw at the low end of its range,
    just under the high end, then inside it, so a branch on it goes both ways. A float of
    whole numbers at least zero (a count, a flag) stays one: the export of ``any`` casts
    to an integer, exact only there. An integer, or an input *held*, is kept."""
    inputs = []
    for i, value in enumerate(example):
        drawn: torch.Tensor
        if rand_ranges is not None and i == len(example) - 1:
            bounds = torch.tensor(rand_ranges, dtype=torch.float32).reshape(-1, 2)
            edge = {1: 0.0, 2: 0.999}.get(sample)
            fraction = (
                torch.rand(len(bounds), generator=generator)
                if edge is None
                else torch.full((len(bounds),), edge)
            )
            drawn = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * fraction
        elif value.dtype == torch.bool:
            drawn = (
                ~value
                if sample == 1
                else torch.rand(value.shape, generator=generator) < 0.5
            )
        elif held[i] or not value.is_floating_point():
            drawn = value.clone()
        elif not _whole_counts(value):
            x = value.detach().cpu().to(torch.float32)
            if sample == 1:
                drawn = -x
            else:
                finite = x[x.isfinite()]
                spread = float(finite.abs().mean()) + 0.1 if finite.numel() else 0.1
                factor = torch.rand(x.shape, generator=generator) * 4 - 2
                noise = torch.randn(x.shape, generator=generator) * spread / 2
                drawn = x * factor + noise
        else:
            top = 2 * float(value.double().abs().max()) + 50 if value.numel() else 1
            drawn = torch.randint(
                0, int(min(top, 2**30)), value.shape, generator=generator
            )
        inputs.append(drawn.to(value.dtype).reshape(value.shape))
    return tuple(inputs)


def _whole_counts(value: torch.Tensor) -> bool:
    whole = (value >= 0) & (value == value.round()) & value.isfinite()
    return bool(value.numel()) and bool(whole.all())


def _apart(
    got: np.ndarray,
    want: torch.Tensor,
    columns: Sequence[tuple[str, int]],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> str | None:
    """Where the graph's *got* strays from the body's *want*, or ``None``.

    Each term's span is held to its own largest value, so float32 rounding in a large
    value is not taken for a disagreement, while a fixed one still is.
    """
    expected = want.detach().cpu().numpy().astype(np.float32)
    actual = np.asarray(got, dtype=np.float32)
    if actual.size != expected.size:
        return f"the graph gives {actual.size} values, the body {expected.size}"
    if not expected.size:
        return None
    width = expected.shape[-1] if expected.ndim else 1
    if sum(size for _, size in columns) != width:
        columns, width = [("", expected.size)], expected.size
    actual, expected = actual.reshape(-1, width), expected.reshape(-1, width)
    start = 0
    for name, size in columns:
        a, e = actual[:, start : start + size], expected[:, start : start + size]
        start += size
        finite = np.abs(e[np.isfinite(e)])
        bound = atol + rtol * (float(finite.max()) if finite.size else 0.0)
        with np.errstate(invalid="ignore"):
            within = np.abs(a - e) <= bound
        off = ~(within | (a == e) | (np.isnan(a) & np.isnan(e)))
        if off.any():
            at = tuple(np.argwhere(off)[0])
            term = f"term {name!r}, " if name else ""
            return f"{term}graph {a[at]:.6g}, body {e[at]:.6g}"
    return None


def _tracer_hints(module: nn.Module, example: tuple[torch.Tensor, ...]) -> list[str]:
    """``file:line (warning)`` for each place a trace of *module* says it fixed a value,
    at the innermost frame outside torch and this package: the term's own line."""
    hints: list[str] = []
    internal = (Path(torch.__file__).parent, Path(__file__).parent)

    def note(
        message: Warning | str, category: type[Warning], *_: Any, **__: Any
    ) -> None:
        text = str(message)
        if category.__name__ != "TracerWarning" or text.startswith(
            _SOUND_TRACER_WARNINGS
        ):
            return
        stack = traceback.extract_stack()
        # Only the frames this trace runs, not the build's that called it.
        outer = max(i for i, f in enumerate(stack) if f.name == "_tracer_hints")
        frames = [
            f
            for f in stack[outer + 1 :]
            if not f.filename.endswith("warnings.py")
            and not any(Path(f.filename).is_relative_to(d) for d in internal)
        ]
        if frames:
            where = f"{Path(frames[-1].filename).name}:{frames[-1].lineno}"
            hint = f"{where} ({text.partition('. ')[0]})"
            if hint not in hints:
                hints.append(hint)

    with warnings.catch_warnings(), mock.patch.object(warnings, "showwarning", note):
        warnings.simplefilter("always")
        with contextlib.suppress(Exception), torch.no_grad():
            torch.jit.trace(
                module,
                tuple(x.clone() for x in example),
                check_trace=False,
                strict=False,
            )
    return hints
