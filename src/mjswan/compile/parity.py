"""Numeric-parity harness: live mjlab env vs exported ONNX graphs.

Steps a live env with a seeded action sequence and, at each step, feeds the same raw
state through each exported graph via ``onnxruntime`` (not torch) and compares. Every
term must match at every step. Run headless with ``MUJOCO_GL=disable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Collection

import numpy as np
import torch

from ..envs.mdp.actions import is_traced
from ..envs.mdp.events import EventBinding
from ..envs.mdp.events import _custom_registry as _event_registry
from ..envs.mdp.observations import ObservationBinding
from ..envs.mdp.observations import _custom_registry as _observation_registry
from ..envs.mdp.registry import resolve
from ..envs.mdp.terminations import TerminationBinding
from ..envs.mdp.terminations import _custom_registry as _termination_registry
from .action import UntraceableAction, trace_action_term
from .event import (
    _env_ids,
    _event_outputs,
    event_writes_nothing_reason,
    trace_event_term,
)
from .record import WriteCaptures, _EventCaptureEnv
from .rng import DrawRecorder
from .slot import read_slot, slot_label, traced_action_state
from .term import ConstantTerm, TermExport, trace_term

#: Where an author's override of each kind of term is registered, and its TS binding.
_REGISTRIES: dict[str, tuple[dict[str, Any], type]] = {
    "observation": (_observation_registry, ObservationBinding),
    "termination": (_termination_registry, TerminationBinding),
}


@dataclass
class TermReport:
    name: str
    kind: str  # "observation" | "termination" | "event" | "action"
    representation: str  # "onnx" | "native"
    input_slots: list[str] = field(default_factory=list)
    constant_slots: list[str] = field(default_factory=list)
    max_abs_diff: float = 0.0
    steps_checked: int = 0
    passed: bool = True
    note: str = ""
    rand_dim: int = 0


@dataclass
class ParityReport:
    n_steps: int
    atol: float
    rtol: float
    terms: list[TermReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(t.passed for t in self.terms)

    def summary(self) -> str:
        lines = [
            f"Parity over {self.n_steps} steps (atol={self.atol}, rtol={self.rtol}):"
        ]
        for t in self.terms:
            status = "OK  " if t.passed else "FAIL"
            if t.representation == "native" or t.kind == "action":
                lines.append(f"  [{status}] {t.name:<16} {t.representation} ({t.note})")
            elif t.kind == "event":
                lines.append(
                    f"  [{status}] {t.name:<16} onnx-event  "
                    f"rand_dim={t.rand_dim} const={t.constant_slots} "
                    f"max|Δ|={t.max_abs_diff:.2e} over {t.steps_checked} draws"
                )
            else:
                lines.append(
                    f"  [{status}] {t.name:<16} onnx  "
                    f"in={t.input_slots} const={t.constant_slots} "
                    f"max|Δ|={t.max_abs_diff:.2e} over {t.steps_checked} steps"
                )
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def _declared_feeds(
    session: Any, feeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """*feeds* less anything the graph does not declare, as the browser does.

    A slot the body only *indexes* with is folded in as a constant, so the export
    prunes its input and ORT refuses the feed.
    """
    declared = {i.name for i in session.get_inputs()}
    return {name: value for name, value in feeds.items() if name in declared}


def _inputs(export: Any) -> list[tuple[str, Any, list[int] | None]]:
    """``(input name, slot, rows)`` per graph input; rows is None for a whole field."""
    rows = getattr(export, "input_rows", None) or []
    return [
        (name, slot, rows[i] if i < len(rows) else None)
        for i, (name, slot) in enumerate(zip(export.input_names, export.input_slots))
    ]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def _feed_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert to numpy preserving bool; float32 otherwise (ONNX input dtypes)."""
    arr = t.detach().cpu().numpy()
    return arr if arr.dtype == bool else arr.astype(np.float32)


def _iter_obs_terms(
    env: Any, group: str
) -> list[tuple[str, Callable[..., torch.Tensor], dict[str, Any]]]:
    om = env.observation_manager
    names = om.active_terms[group]
    out = []
    for term_name in names:
        cfg = om.get_term_cfg(group, term_name)
        out.append((term_name, cfg.func, dict(cfg.params)))
    return out


def _iter_termination_terms(
    env: Any,
) -> list[tuple[str, Callable[..., torch.Tensor], dict[str, Any]]]:
    tm = env.termination_manager
    out = []
    for term_name in tm.active_terms:
        cfg = tm.get_term_cfg(term_name)
        out.append((term_name, cfg.func, dict(cfg.params)))
    return out


def _iter_event_terms(
    env: Any, mode: str
) -> list[tuple[str, Callable[..., None], dict[str, Any], bool]]:
    em = env.event_manager
    names = em.active_terms.get(mode, [])
    out = []
    for term_name in names:
        cfg = em.get_term_cfg(term_name)
        out.append((term_name, cfg.func, dict(cfg.params), cfg.is_global_time))
    return out


def _trace_action_terms(env: Any) -> tuple[list[TermReport], dict[str, list[str]]]:
    """Trace each action term the build traces, which checks both graphs
    against the term's own bodies. Returns the reports, and each term's state for the
    observations that read it."""
    reports: list[TermReport] = []
    state: dict[str, list[str]] = {}
    manager = env.action_manager
    for name in manager.active_terms:
        cfg = manager.get_term(name).cfg
        if not is_traced(cfg):
            continue
        tr = TermReport(name=name, kind="action", representation="onnx")
        reports.append(tr)
        try:
            export = trace_action_term(cfg.build(env), name=name)
        except UntraceableAction as exc:
            tr.representation = "native"
            tr.passed = False
            tr.note = str(exc)
            continue
        tr.input_slots = [slot_label(k) for k in export.process.input_slots]
        tr.rand_dim = export.process.rand_dim
        tr.note = "process and reset match the term's own bodies"
        state[name] = [spec["name"] for spec in export.state_fields]
    return reports, state


def run_parity(
    env: Any,
    *,
    obs_group: str = "actor",
    n_steps: int = 64,
    seed: int = 0,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    event_modes: tuple[str, ...] = ("reset",),
    n_event_draws: int = 16,
    include_obs: bool = True,
    reader_fields: Collection[str] | None = None,
) -> ParityReport:
    """Trace a task's terms and assert live-vs-ONNX parity over ``n_steps``.

    ``env`` must be a freshly constructed mjlab env; this function resets it.
    Observation and termination terms are checked every step; ``reset``-mode Event
    terms are checked by replaying ``n_event_draws`` fresh recorded RNG draws. As
    in the build, a registered override is traced in place of its term and checked
    against mjlab's own, a TS binding goes unchecked, and a task's own action term is
    traced, its state served to the observations that read it.

    ``reader_fields`` is passed to :func:`trace_term`; an empty set traces every
    ``EntityData`` property through to raw sim slots, putting mjlab's own property math
    under this harness.
    """
    import onnxruntime as ort

    report = ParityReport(n_steps=n_steps, atol=atol, rtol=rtol)
    torch.manual_seed(seed)
    env.reset()
    action_reports, action_state = _trace_action_terms(env)
    report.terms.extend(action_reports)

    # --- Trace observation and termination terms; classify the rest. -----
    checks: list[
        tuple[TermReport, TermExport, Any, Callable[..., torch.Tensor], dict[str, Any]]
    ] = []
    terms = (
        [("observation", *t) for t in _iter_obs_terms(env, obs_group)]
        + [("termination", *t) for t in _iter_termination_terms(env)]
        if include_obs
        else []
    )
    for kind, term_name, func, params in terms:
        registry, binding = _REGISTRIES[kind]
        traced = resolve(func, term_name, registry, binding)
        if isinstance(traced, binding):
            report.terms.append(
                TermReport(
                    name=term_name,
                    kind=kind,
                    representation="native",
                    note=f"TS class {traced.ts_name}",
                )
            )
            continue
        try:
            with traced_action_state(action_state):
                export = trace_term(
                    traced, params, env, name=term_name, reader_fields=reader_fields
                )
        except (ValueError, AttributeError) as exc:
            report.terms.append(
                TermReport(
                    name=term_name,
                    kind=kind,
                    representation="native",
                    # The build bakes a constant observation; it refuses an untraceable
                    # term, and a constant termination.
                    passed=kind == "observation" and isinstance(exc, ConstantTerm),
                    note=str(exc).split(";")[0],
                )
            )
            continue
        tr = TermReport(
            name=term_name,
            kind=kind,
            representation="onnx",
            input_slots=[slot_label(k) for k in export.input_slots],
            constant_slots=[f"{e}.{f}" for e, f in export.constant_slots],
        )
        report.terms.append(tr)
        session = ort.InferenceSession(
            export.onnx_bytes, providers=["CPUExecutionProvider"]
        )
        checks.append((tr, export, session, func, params))

    action_dim = env.action_manager.total_action_dim

    # --- Step and compare every term every step. ------------------------
    for _ in range(n_steps):
        action = torch.rand((env.num_envs, action_dim)) * 2.0 - 1.0
        env.step(action)
        for tr, export, session, func, params in checks:
            feeds = _declared_feeds(
                session,
                {
                    in_name: _to_numpy(read_slot(env, slot, rows))
                    for in_name, slot, rows in _inputs(export)
                },
            )
            (onnx_out,) = session.run([export.output_name], feeds)
            live_out = _to_numpy(func(env, **params))
            diff = float(np.max(np.abs(onnx_out - live_out))) if live_out.size else 0.0
            tr.max_abs_diff = max(tr.max_abs_diff, diff)
            tr.steps_checked += 1
            if not np.allclose(onnx_out, live_out, atol=atol, rtol=rtol):
                tr.passed = False

    # --- Event terms: trace once, then replay fresh recorded RNG draws. -----
    for mode in event_modes:
        for term_name, func, params, is_global_time in _iter_event_terms(env, mode):
            tr = TermReport(name=term_name, kind="event", representation="onnx")
            report.terms.append(tr)
            bound = resolve(func, term_name, _event_registry, EventBinding)
            if isinstance(bound, EventBinding):
                tr.representation = "native"
                tr.note = f"TS class {bound.ts_name}"
                continue
            try:
                export = trace_event_term(
                    func,
                    params,
                    env,
                    name=term_name,
                    mode=mode,
                    is_global_time=is_global_time,
                )
            except Exception as exc:  # noqa: BLE001 — untraceable term → native fallback
                tr.representation = "native"
                tr.note = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
                # As the build, which fails on any other untraceable event.
                tr.passed = event_writes_nothing_reason(func, env, params) is not None
                continue
            tr.rand_dim = export.rand_dim
            tr.input_slots = [slot_label(k) for k in export.input_slots]
            tr.constant_slots = list(export.constant_slots)
            session = ort.InferenceSession(
                export.onnx_bytes, providers=["CPUExecutionProvider"]
            )
            env_ids = _env_ids(mode, is_global_time=is_global_time)
            for _ in range(n_event_draws):
                # Record a fresh reference invocation (real draws, no sim write).
                captures: WriteCaptures = {}
                proxy = _EventCaptureEnv(env, [], captures, model=True)
                with DrawRecorder(func) as rec:
                    func(proxy, env_ids, **params)
                _, ref_tensors = _event_outputs(captures, proxy.model)
                feeds = {"rand": _to_numpy(rec.rand_vector)}
                for in_name, slot, rows in zip(
                    export.input_names, export.input_slots, export.input_rows
                ):
                    feeds[in_name] = _to_numpy(read_slot(env, slot, rows))
                # A draw-free event has no `rand` input: the export prunes it.
                onnx_outs = session.run(
                    export.output_names, _declared_feeds(session, feeds)
                )
                for onnx_out, ref in zip(onnx_outs, ref_tensors):
                    ref_np = _to_numpy(ref)
                    tr.max_abs_diff = max(
                        tr.max_abs_diff, float(np.max(np.abs(onnx_out - ref_np)))
                    )
                    if not np.allclose(onnx_out, ref_np, atol=atol, rtol=rtol):
                        tr.passed = False
                tr.steps_checked += 1

    return report


def run_command_parity(
    term: Any,
    state_fields: list[str],
    *,
    name: str,
    command_field: str,
    n_draws: int = 16,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> TermReport:
    """Trace a stateful CommandTerm and check live-vs-ONNX parity.

    Traces ``_resample_command``+``_update_command`` once, then for ``n_draws`` random
    draws runs a resample step through ONNX Runtime and through the live term, chaining
    state, and compares the next state and any ``entity_write``. When the term's
    ``reset`` adds to mjlab's, every other step is also an episode reset. A final step
    without a resample checks that the state only updates.
    """
    import onnxruntime as ort

    from .command import (
        _reset_extra,
        _restore_state,
        _snapshot_state,
        compare_step,
        trace_command_term,
    )

    tr = TermReport(name=name, kind="command", representation="onnx")
    export = trace_command_term(
        term, state_fields, name=name, command_field=command_field
    )
    tr.rand_dim = export.rand_dim
    tr.input_slots = [slot_label(k) for k in export.input_slots]
    tr.note = f"state={[s['name'] for s in export.state_fields]} cmd={command_field}"
    session = ort.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = session.run(export.output_names, _declared_feeds(session, feeds))
        return {n: np.asarray(o) for n, o in zip(export.output_names, outs)}

    ranges = torch.tensor(export.rand_ranges, dtype=torch.float32).reshape(-1, 2)
    resets = _reset_extra(term) is not None
    snap = _snapshot_state(term)
    try:
        steps = [(True, resets and i % 2 == 0) for i in range(n_draws)]
        for i, (resample, reset) in enumerate([*steps, (False, False)]):
            rand = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
                len(ranges)
            )
            diff, note = compare_step(
                term,
                export,
                state_fields,
                run,
                rand,
                resample=resample,
                reset=reset,
                atol=atol,
                rtol=rtol,
            )
            tr.max_abs_diff = max(tr.max_abs_diff, diff)
            if note is not None:
                tr.passed = False
                tr.note += f" [step {i}: {note}]"
            if resample:
                tr.steps_checked += 1
    finally:
        _restore_state(term, snap)
    return tr
