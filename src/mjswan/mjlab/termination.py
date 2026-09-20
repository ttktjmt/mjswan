"""mjlab termination terms as mjswan's.

Same resolution order as :mod:`.observation`: a ``TerminationBinding`` passes through,
then a ``register_termination`` override, then mjlab's own function, traced at build.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..envs.mdp.terminations import TerminationBinding
from ..envs.mdp.terminations import _custom_registry as _custom_term_registry
from ..managers.termination_manager import (
    TerminationTermCfg as MjswanTerminationTermCfg,
)
from .detect import is_from_mjlab


def _adapt_term_func(
    func: Any, term_name: str | None = None
) -> TerminationBinding | Callable[..., Any]:
    """Resolve the function a termination term's ONNX graph is traced from.

    Same resolution order as :func:`_adapt_obs_func`. *term_name* also covers closures,
    which can only be registered by their dict key.
    """
    if isinstance(func, TerminationBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_term_registry:
        return _custom_term_registry[name]
    if term_name and term_name in _custom_term_registry:
        return _custom_term_registry[term_name]
    return func


def _sanitize_termination_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-only termination params while keeping useful scope data."""
    if not params:
        return params

    result = {
        k: v for k, v in params.items() if k != "asset_cfg" and not is_from_mjlab(v)
    }
    asset_cfg = params.get("asset_cfg")
    if not is_from_mjlab(asset_cfg):
        return result

    entity_name = getattr(asset_cfg, "name", None)
    if entity_name:
        result["entity_name"] = entity_name

    body_names = getattr(asset_cfg, "body_names", None)
    if isinstance(body_names, (list, tuple)):
        result["body_names"] = [str(name) for name in body_names]
    elif isinstance(body_names, str):
        result["body_names"] = [body_names]

    return result


def _adapt_term_cfg(
    term: Any, term_name: str | None = None
) -> MjswanTerminationTermCfg:
    """Convert a single mjlab ``TerminationTermCfg`` to mjswan.

    As in :func:`_adapt_obs_term`, params are sanitized only for the binding path.
    """
    raw_params = dict(getattr(term, "params", None) or {})
    func = _adapt_term_func(term.func, term_name=term_name)
    params = (
        _sanitize_termination_params(raw_params)
        if isinstance(func, TerminationBinding)
        else raw_params
    )
    return MjswanTerminationTermCfg(
        func=func,
        params=params,
        time_out=getattr(term, "time_out", False),
    )


def adapt_terminations(
    terminations: dict[str, Any] | None,
) -> dict[str, MjswanTerminationTermCfg] | None:
    """Adapt termination configs, converting mjlab types if detected."""
    if terminations is None:
        return None
    return {
        key: term
        if isinstance(term, MjswanTerminationTermCfg)
        else _adapt_term_cfg(term, term_name=key)
        if is_from_mjlab(term)
        else term
        for key, term in terminations.items()
    }


__all__ = ["adapt_terminations"]
