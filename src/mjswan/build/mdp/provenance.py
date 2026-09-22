"""What a term entry says about where it came from.

The manifest says what a term reads and how wide it is; only the build knows which
function it is. :func:`term_provenance` records that on the entry, :func:`graph_meta`
stamps it into the graph itself, and :func:`resolved_params` fixes the params both are
traced and recorded with.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any


def param_json(value: Any) -> Any:
    """A term param as a viewer can read it; never the object itself."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (bool, int, float, str)) for v in value
    ):
        return list(value)
    # SceneEntityCfg, duck-typed. Its resolved ids are build-env indices and mean
    # nothing to a reader, so only the declared name patterns travel.
    if isinstance(getattr(value, "name", None), str):
        out: dict[str, Any] = {"entity": value.name}
        for key in ("joint_names", "body_names", "geom_names", "site_names"):
            names = getattr(value, key, None)
            if isinstance(names, str):
                out[key] = names
            elif isinstance(names, (list, tuple)) and names:
                out[key] = [str(n) for n in names]
        return out
    return type(value).__name__


def term_provenance(func: Any, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """``func`` / ``doc`` / ``params`` for a term entry.

    The manifest says what a term reads and how wide it is; only the build knows which
    function it is.
    """
    out: dict[str, Any] = {}
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if module and qualname:
        out["func"] = f"{module}:{qualname}"
    doc = inspect.getdoc(func)
    if doc:
        out["doc"] = doc.strip().splitlines()[0][:200]
    if params:
        out["params"] = {str(k): param_json(v) for k, v in params.items()}
    return out


def graph_meta(kind: str, term: str, func: Any = None) -> dict[str, str]:
    """The ``metadata_props`` a written graph carries; see :func:`.graph.stamp_provenance`."""
    meta = {"kind": kind, "term": term}
    path = term_provenance(func).get("func") if func is not None else None
    if path:
        meta["func"] = path
    return meta


def resolved_params(params: dict[str, Any], env: Any) -> dict[str, Any]:
    """Resolve every ``SceneEntityCfg`` in *params* against the live scene.

    mjlab's managers do this at ``_prepare_terms``, turning name patterns into concrete
    indices. The Builder serializes from the task config, whose cfgs are still
    unresolved (``site_ids=slice(None)``, every site), so tracing without this bakes a
    different function than mjlab runs.

    A copy is resolved, since resolution mutates the cfg. Duck-typed to keep mjlab a
    soft dependency.
    """
    resolved = dict(params)
    for key, value in params.items():
        if callable(getattr(value, "resolve", None)) and hasattr(value, "name"):
            entity_cfg = copy.deepcopy(value)
            entity_cfg.resolve(env.scene)
            resolved[key] = entity_cfg
    return resolved
