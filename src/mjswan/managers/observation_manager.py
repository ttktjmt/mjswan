"""Observation manager configuration for mjswan.

Provides ``ObservationTermCfg`` and ``ObservationGroupCfg`` with an API
compatible with ``mjlab.managers.observation_manager``.  mjswan only
runs inference in the browser, so training-only fields (noise, delay,
nan_policy, ...) are accepted for API compatibility but silently ignored
at build time.

Example (identical to mjlab)::

    from mjlab.envs.mdp import observations as obs_fns
    from mjlab.managers.scene_entity_cfg import SceneEntityCfg
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "base_lin_vel": ObservationTermCfg(func=obs_fns.base_lin_vel),
                "base_ang_vel": ObservationTermCfg(func=obs_fns.base_ang_vel),
                "projected_gravity": ObservationTermCfg(
                    func=obs_fns.projected_gravity
                ),
                "joint_pos": ObservationTermCfg(
                    func=obs_fns.joint_pos_rel, scale=0.5
                ),
                "joint_vel": ObservationTermCfg(func=obs_fns.joint_vel_rel),
                "last_action": ObservationTermCfg(func=obs_fns.last_action),
            },
            enable_corruption=True,
        ),
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..envs.mdp.observations import ObservationBinding


@dataclass
class ObservationTermCfg:
    """Configuration for a single observation term.

    Mirrors ``mjlab.managers.observation_manager.ObservationTermCfg``.

    Processing pipeline in mjlab: compute -> noise -> clip -> scale -> delay -> history.
    In mjswan the TS runtime handles scale and history; noise and delay are
    training-only and therefore accepted but ignored.

    ``func`` is either an :class:`ObservationBinding` (resolved to a TS class by name)
    or a plain ``func(env, **params)`` the build traces to ONNX.
    """

    func: ObservationBinding | Callable[..., Any]
    """Observation function — ObservationBinding sentinel (legacy) or a
    plain callable traced to ONNX (ADR 0005)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments forwarded to the TS observation constructor."""

    scale: tuple[float, ...] | float | None = None
    """Scaling factor(s) applied element-wise to the observation output."""

    clip: tuple[float, float] | None = None
    """(min, max) clipping range applied after scaling."""

    history_length: int = 0
    """Number of past frames to stack, newest first (``[x_t, x_{t-1}, …]``).
    0 = current only (no history).

    mjlab stacks the same frames in the opposite order — its buffer flattens
    chronologically — so a traced mjlab task arrives with ``history_steps`` spelling
    that order out, and never with a bare count."""

    history_steps: tuple[int, ...] | None = None
    """Look-back offsets to stack, in the order the policy reads them, instead of a
    count of frames.

    Two things a count cannot say. A *sparse* window — e.g. ``(0, 1, 2, 4, 8, 16)`` —
    reaches 17 frames back while contributing only 6, keeping the term's width at
    ``len(history_steps)`` where ``history_length`` would give 17. And an *order*:
    ``(3, 2, 1, 0)`` is mjlab's oldest-first stack of four frames, the reverse of
    ``history_length=4``. Takes precedence over ``history_length`` when both are set."""

    history_interleaved: bool = False
    """Isaac-style joint-major history layout: ``[a0_t, a0_t-1, ..., a1_t, ...]``
    instead of frame-major (``[a_t, a_t-1, ...]`` each the full vector). Only
    meaningful when ``history_length`` > 0."""

    flatten_history_dim: bool = True
    """Whether to flatten history into the feature dimension.
    Accepted for API compatibility; mjswan always flattens."""

    # --- mjlab training-only fields (accepted, ignored at build time) ---

    noise: Any = None
    """Noise config. Accepted for mjlab compatibility; ignored in mjswan."""

    delay_min_lag: int = 0
    delay_max_lag: int = 0
    delay_per_env: bool = True
    delay_hold_prob: float = 0.0
    delay_update_period: int = 0
    delay_per_env_phase: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize an ``ObservationBinding`` term.

        A plain-callable term needs a live env this method has no access to; the Builder
        calls ``mjswan._onnx_build.serialize_observation_group`` for those.
        """
        if isinstance(self.func, ObservationBinding):
            return self._to_dict_legacy()
        raise TypeError(
            f"ObservationTermCfg.to_dict() cannot serialize a plain callable "
            f"func ({self.func!r}) — it must be traced to ONNX against a live "
            f"env. Use mjswan._onnx_build.serialize_observation_group(group, "
            f"env, out_dir) instead (the Builder does this automatically)."
        )

    def _to_dict_legacy(self) -> dict[str, Any]:
        func: ObservationBinding = self.func  # type: ignore[assignment]
        entry: dict[str, Any] = {"name": func.ts_name}
        merged: dict[str, Any] = {**func.defaults, **self.params}
        if self.scale is not None:
            merged["scale"] = (
                list(self.scale) if isinstance(self.scale, tuple) else self.scale
            )
        if self.clip is not None:
            merged["clip"] = list(self.clip)
        if self.history_length > 0:
            merged["history_steps"] = self.history_length
        entry.update(merged)
        return entry


@dataclass
class ObservationGroupCfg:
    """Configuration for an observation group.

    Mirrors ``mjlab.managers.observation_manager.ObservationGroupCfg``.

    An observation group bundles multiple terms together.  The TS-side
    ``PolicyRunner`` concatenates term outputs in registration order.
    """

    terms: dict[str, ObservationTermCfg] = field(default_factory=dict)
    """Named observation terms, concatenated in registration order."""

    concatenate_terms: bool = True
    """Accepted for mjlab compatibility; mjswan always concatenates."""

    enable_corruption: bool = False
    """Accepted for mjlab compatibility; ignored (no training in browser)."""

    history_length: int | None = None
    """Group-level history override. If set, applies to all terms."""

    flatten_history_dim: bool = True
    """Accepted for mjlab compatibility; mjswan always flattens."""

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize the group's terms to a JSON-compatible list.

        If ``history_length`` is set at the group level, it overrides
        per-term settings (matching mjlab behaviour).
        """
        result = []
        for term_cfg in self.terms.values():
            d = term_cfg.to_dict()
            # Group-level history overrides term-level
            if self.history_length is not None and self.history_length > 0:
                d["history_steps"] = self.history_length
            result.append(d)
        return result


__all__ = ["ObservationTermCfg", "ObservationGroupCfg"]
