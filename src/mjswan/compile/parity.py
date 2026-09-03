"""Numeric-parity harness: live mjlab env vs exported ONNX graphs.

Steps a live env with a seeded action sequence and, at each step, feeds the same raw
state through each exported graph via ``onnxruntime`` (not torch) and compares. Every
term must match at every step. Run headless with ``MUJOCO_GL=disable``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from .rng import DrawRecorder
from .tracer import (
    TermExport,
    WriteCaptures,
    _EventCaptureEnv,
    _flatten_captures,
    read_slot,
    slot_label,
    trace_event_term,
    trace_term,
)


@dataclass
class TermReport:
    name: str
    kind: str  # "observation" | "termination" | "event"
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
            if t.representation == "native":
                lines.append(f"  [{status}] {t.name:<16} native ({t.note})")
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


# Terms handled natively by the runtime (no ONNX graph); ADR 0005 §2 table.
_NATIVE_TERMINATIONS = {"time_out"}


def _declared_feeds(
    session: Any, feeds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """*feeds* less anything the graph does not declare, as the browser does.

    A slot the body only *indexes* with is folded in as a constant, so the export
    prunes its input and ORT refuses the feed.
    """
    declared = {i.name for i in session.get_inputs()}
    return {name: value for name, value in feeds.items() if name in declared}


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


def _iter_termination_terms(env: Any) -> list[tuple[str, Callable[..., torch.Tensor]]]:
    tm = env.termination_manager
    return [(name, tm.get_term_cfg(name).func) for name in tm.active_terms]


def _iter_event_terms(
    env: Any, mode: str
) -> list[tuple[str, Callable[..., None], dict[str, Any]]]:
    em = env.event_manager
    names = em.active_terms.get(mode, [])
    out = []
    for term_name in names:
        cfg = em.get_term_cfg(term_name)
        out.append((term_name, cfg.func, dict(cfg.params)))
    return out


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
) -> ParityReport:
    """Trace a task's terms and assert live-vs-ONNX parity over ``n_steps``.

    ``env`` must be a freshly constructed mjlab env; this function resets it.
    Observation terms are checked every step; ``reset``-mode Event terms are
    checked by replaying ``n_event_draws`` fresh recorded RNG draws (§2b).
    """
    import onnxruntime as ort

    report = ParityReport(n_steps=n_steps, atol=atol, rtol=rtol)
    torch.manual_seed(seed)
    env.reset()

    # --- Trace observation terms; classify the rest. --------------------
    exports: dict[str, TermExport] = {}
    sessions: dict[str, ort.InferenceSession] = {}
    term_meta: list[tuple[str, Callable[..., torch.Tensor], dict[str, Any]]] = []

    obs_terms = _iter_obs_terms(env, obs_group) if include_obs else []
    for term_name, func, params in obs_terms:
        try:
            export = trace_term(func, params, env, name=term_name)
        except ValueError as exc:
            report.terms.append(
                TermReport(
                    name=term_name,
                    kind="observation",
                    representation="native",
                    passed=True,
                    note=str(exc).split(";")[0],
                )
            )
            continue
        exports[term_name] = export
        sessions[term_name] = ort.InferenceSession(
            export.onnx_bytes, providers=["CPUExecutionProvider"]
        )
        term_meta.append((term_name, func, params))
        report.terms.append(
            TermReport(
                name=term_name,
                kind="observation",
                representation="onnx",
                input_slots=[slot_label(k) for k in export.input_slots],
                constant_slots=[f"{e}.{f}" for e, f in export.constant_slots],
            )
        )

    if include_obs:
        for term_name, func in _iter_termination_terms(env):
            native = term_name in _NATIVE_TERMINATIONS
            report.terms.append(
                TermReport(
                    name=term_name,
                    kind="termination",
                    representation="native" if native else "onnx",
                    passed=True,
                    note="elapsed_s >= episode_length_s" if native else "",
                )
            )
            # Non-native terminations would be traced here; Cartpole has only time_out.

    reports_by_name = {t.name: t for t in report.terms}
    action_dim = env.action_manager.total_action_dim

    # --- Step and compare every term every step. ------------------------
    for _ in range(n_steps):
        action = torch.rand((env.num_envs, action_dim)) * 2.0 - 1.0
        env.step(action)
        for term_name, func, params in term_meta:
            export = exports[term_name]
            session = sessions[term_name]
            feeds = _declared_feeds(
                session,
                {
                    in_name: _to_numpy(read_slot(env, slot))
                    for in_name, slot in zip(export.input_names, export.input_slots)
                },
            )
            (onnx_out,) = session.run([export.output_name], feeds)
            live_out = _to_numpy(func(env, **params))
            diff = float(np.max(np.abs(onnx_out - live_out))) if live_out.size else 0.0
            tr = reports_by_name[term_name]
            tr.max_abs_diff = max(tr.max_abs_diff, diff)
            tr.steps_checked += 1
            if not np.allclose(onnx_out, live_out, atol=atol, rtol=rtol):
                tr.passed = False

    # --- Event terms: trace once, then replay fresh recorded RNG draws. -----
    for mode in event_modes:
        for term_name, func, params in _iter_event_terms(env, mode):
            tr = TermReport(name=term_name, kind="event", representation="onnx")
            report.terms.append(tr)
            try:
                export = trace_event_term(func, params, env, name=term_name, mode=mode)
            except Exception as exc:  # noqa: BLE001 — untraceable term → native fallback
                tr.representation = "native"
                tr.note = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
                continue
            tr.rand_dim = export.rand_dim
            tr.input_slots = [slot_label(k) for k in export.input_slots]
            tr.constant_slots = list(export.constant_slots)
            session = ort.InferenceSession(
                export.onnx_bytes, providers=["CPUExecutionProvider"]
            )
            for _ in range(n_event_draws):
                # Record a fresh reference invocation (real draws, no sim write).
                captures: WriteCaptures = {}
                proxy = _EventCaptureEnv(env, [], captures)
                with DrawRecorder(func) as rec:
                    func(proxy, None, **params)
                _, ref_tensors = _flatten_captures(captures)
                feeds = {"rand": _to_numpy(rec.rand_vector)}
                for in_name, slot in zip(export.input_names, export.input_slots):
                    feeds[in_name] = _to_numpy(read_slot(env, slot))
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
    """Trace a stateful CommandTerm and check live-vs-ONNX parity (brief §3).

    Traces ``_resample_command``+``_update_command`` once, then for ``n_draws``
    replays fresh recorded RNG through the graph with ``resample_mask=True`` and
    compares the next state, command, and any ``entity_write`` against the live
    term. Also checks that ``resample_mask=False`` leaves the state unchanged.
    """
    import onnxruntime as ort

    from .tracer import (
        _entity_attrs,
        _flatten_captures,
        _RecordCommand,
        _restore_state,
        _snapshot_state,
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
    entity_attr_names = _entity_attrs(term)
    entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
    env_ids = torch.arange(term.num_envs)

    def _dyn_feeds() -> dict[str, np.ndarray]:
        # Dynamic slots read runtime state off the term's entity; feed live values.
        entity = getattr(term, entity_attr_names[0]) if entity_attr_names else None
        out = {}
        for in_name, (_ent, fld) in zip(export.input_names, export.input_slots):
            out[in_name] = _to_numpy(getattr(entity.data, fld))
        return out

    for _ in range(n_draws):
        snap = _snapshot_state(term)
        prev = {f: getattr(term, f).detach().clone() for f in state_fields}
        dyn_feeds = _dyn_feeds()  # read before the reference run mutates nothing
        with _RecordCommand(term, entity_attr_names, entity_name) as rec_env:
            with DrawRecorder(term._resample_command) as rec:
                term._resample_command(env_ids)
                term._update_command()
            ref_writes = _flatten_captures(dict(rec_env.captures))[1]
        ref_next = {f: getattr(term, f).detach().clone() for f in state_fields}
        _restore_state(term, snap)

        feeds = {f"prev_{f}": _feed_numpy(prev[f]) for f in state_fields}
        feeds.update(dyn_feeds)
        feeds["resample_mask"] = np.ones((term.num_envs,), dtype=bool)
        feeds["rand"] = _to_numpy(rec.rand_vector)
        outs = session.run(export.output_names, feeds)
        refs = [ref_next[f] for f in state_fields] + list(ref_writes)
        for out, ref in zip(outs, refs):
            ref_np = _to_numpy(ref)
            tr.max_abs_diff = max(tr.max_abs_diff, float(np.max(np.abs(out - ref_np))))
            if not np.allclose(out, ref_np, atol=atol, rtol=rtol):
                tr.passed = False
        tr.steps_checked += 1

    # resample_mask=False: no resample, but _update_command still runs on prev state.
    snap = _snapshot_state(term)
    prev = {f: getattr(term, f).detach().clone() for f in state_fields}
    dyn_feeds = _dyn_feeds()
    with _RecordCommand(term, entity_attr_names, entity_name):
        term._update_command()
    ref_false = {f: getattr(term, f).detach().clone() for f in state_fields}
    _restore_state(term, snap)

    feeds = {f"prev_{f}": _feed_numpy(prev[f]) for f in state_fields}
    feeds.update(dyn_feeds)
    feeds["resample_mask"] = np.zeros((term.num_envs,), dtype=bool)
    feeds["rand"] = _to_numpy(export.reference_rand)
    outs = session.run(export.output_names, feeds)
    # State fields only: the mask does not gate the write outputs, so there is no
    # reference to compare them against (`OnnxCommand`'s tests cover that half).
    for f, out in zip(state_fields, outs[: len(state_fields)]):
        if not np.allclose(out, _to_numpy(ref_false[f]), atol=atol, rtol=rtol):
            tr.passed = False
            tr.note += f" [mask=False mismatch on {f}]"
    return tr
