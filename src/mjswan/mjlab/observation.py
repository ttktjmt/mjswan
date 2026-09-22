"""mjlab observation groups as mjswan's.

mjlab's own term function is kept and traced at build time (ADR 0005); an author's
``register_observation`` override replaces it by function or term name. Params are
sanitized only for an ``ObservationBinding``, whose params go verbatim into the browser
JSON.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Any

from ..document.manifest import DEFAULT_IN_KEYS
from ..envs.mdp.observations import ObservationBinding, _custom_registry
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
)
from ..managers.observation_manager import (
    ObservationTermCfg as MjswanObservationTermCfg,
)
from .detect import is_from_mjlab


def _adapt_obs_func(
    func: Any, term_name: str | None = None
) -> ObservationBinding | Callable[..., Any]:
    """Resolve the function an observation term's ONNX graph is traced from.

    An ``ObservationBinding`` passes through, then any ``register_observation``
    override for this function or term name, then mjlab's own function, which the
    build traces directly.
    """
    if isinstance(func, ObservationBinding):
        return func
    name = getattr(func, "__name__", None)
    if name and name in _custom_registry:
        return _custom_registry[name]
    if term_name and term_name in _custom_registry:
        return _custom_registry[term_name]
    return func


def _sanitize_obs_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip mjlab-specific params that are not JSON-serializable.

    ``asset_cfg`` (a ``SceneEntityCfg``) is removed.  When it carries
    entity-scoping information, it is promoted into JSON-friendly fields so
    browser-side observation classes can resolve the correct MuJoCo entities
    at runtime.
    """
    if "asset_cfg" not in params:
        return params
    result = {k: v for k, v in params.items() if k != "asset_cfg"}
    asset_cfg = params["asset_cfg"]
    if is_from_mjlab(asset_cfg):
        entity_name = getattr(asset_cfg, "name", None)
        if entity_name:
            result["entity_name"] = entity_name
        joint_names = getattr(asset_cfg, "joint_names", None)
        if joint_names:
            names = (
                list(joint_names)
                if isinstance(joint_names, (list, tuple))
                else [joint_names]
            )
            result["joint_names"] = names
            if len(names) == 1:
                name = names[0]
                result["joint_name"] = f"{entity_name}/{name}" if entity_name else name
        site_names = getattr(asset_cfg, "site_names", None)
        if site_names:
            name = (
                site_names[0] if isinstance(site_names, (list, tuple)) else site_names
            )
            # mjlab namespaces entity sites as "{entity_name}/{site_name}"
            result["site_name"] = f"{entity_name}/{name}" if entity_name else name
    return result


def _adapt_obs_term(
    term: Any, term_name: str | None = None
) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan.

    Params are sanitized only for an ``ObservationBinding``, whose params go verbatim
    into the browser JSON. A traced func keeps the real ``SceneEntityCfg`` mjlab's own
    function expects.
    """
    raw_params = dict(getattr(term, "params", None) or {})
    func = _adapt_obs_func(term.func, term_name=term_name)
    params = (
        _sanitize_obs_params(raw_params)
        if isinstance(func, ObservationBinding)
        else raw_params
    )
    return MjswanObservationTermCfg(
        func=func,
        params=params,
        scale=getattr(term, "scale", None),
        clip=getattr(term, "clip", None),
        history_length=getattr(term, "history_length", 0) or 0,
    )


def _adapt_obs_group(group: Any) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    raw_terms = getattr(group, "terms", None) or {}
    terms = {
        name: _adapt_obs_term(cfg, term_name=name) for name, cfg in raw_terms.items()
    }
    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=getattr(group, "concatenate_terms", True),
        enable_corruption=getattr(group, "enable_corruption", False),
        history_length=getattr(group, "history_length", None),
    )


#: Groups belonging to networks that never leave training. Only the actor is exported,
#: so these have no input to feed and are dropped rather than traced and bundled.
_TRAINING_ONLY_OBS_GROUPS = frozenset({"critic"})

#: The slot a single observation group lands under, derived so it cannot drift from the
#: runtime's default ``in_keys``. It is mjlab's own name for the group its actor network
#: reads, so the common case needs no key at all (ADR 0006 §5). The TypeScript runtime
#: holds the same default, pinned to Python by ``default_slots.json``.
DEFAULT_OBS_GROUP_KEY = DEFAULT_IN_KEYS[0]

#: mjlab's name for the network whose group the exported policy reads. The same word as
#: the default slot, by design, but a different namespace: this one is looked up in
#: ``rl_cfg.obs_groups``, the other in ``in_keys``.
_MJLAB_ACTOR_NETWORK = "actor"


def _is_obs_group(value: Any) -> bool:
    """Whether *value* is a single observation group rather than a dict of them."""
    if isinstance(value, MjswanObservationGroupCfg):
        return True
    # A group carries `terms`; a dict of groups does not.
    return not isinstance(value, Mapping) and hasattr(value, "terms")


def _select_policy_group(
    observations: Mapping[str, Any],
    obs_groups: Mapping[str, tuple[str, ...]] | None,
) -> Mapping[str, Any]:
    """Reduce mjlab's network-keyed group dict to what the exported actor reads.

    mjlab keys ``env_cfg.observations`` by *network* (``"actor"``, ``"critic"``); mjswan
    keys it by *slot*, the name the policy's ``in_keys`` table uses. The default slot is
    also called ``actor``, so mjlab's own dict needs no renaming, only the other
    networks' groups dropped. A runner that names the actor's group something else
    (``obs_groups == {"actor": ("proprio",), ...}``) has that group moved to the default
    slot.

    Only groups the runner attributes to a network are touched. A key it does not know
    (``"command_"`` on a multi-input policy) is a slot the author added and stays, and a
    dict sharing no key with the actor's is not the task's dict and is left alone.
    """
    if not observations:
        return observations
    actor_groups = tuple((obs_groups or {}).get(_MJLAB_ACTOR_NETWORK) or ())
    if not actor_groups or set(actor_groups).isdisjoint(observations):
        return observations
    if len(actor_groups) != 1:
        # rsl-rl concatenates several groups per network; mjswan feeds one vector per
        # ONNX input, and taking the first would silently shorten it.
        raise ValueError(
            "The task's runner config feeds its actor network "
            f"{len(actor_groups)} concatenated observation groups "
            f"({', '.join(map(repr, actor_groups))}). mjswan feeds one group per "
            "ONNX input and cannot concatenate them, so pass the single group the "
            "exported policy actually takes: "
            "`observations=env_cfg.observations[<name>]`."
        )
    actor_key = actor_groups[0]
    network_groups = {g for groups in (obs_groups or {}).values() for g in groups}
    selected: dict[str, Any] = {}
    for key, group in observations.items():
        if key == actor_key:
            selected[DEFAULT_OBS_GROUP_KEY] = group
        elif key not in network_groups:
            selected[key] = group
    return selected


def adapt_observations(
    observations: Mapping[str, Any] | Any | None,
    *,
    obs_groups: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    Accepts three shapes, because the caller should not have to know which slot the
    runtime will feed:

    * a **single** group (mjlab's ``env_cfg.observations["actor"]``), which lands
      under :data:`DEFAULT_OBS_GROUP_KEY`;
    * mjlab's whole ``env_cfg.observations`` dict, from which the policy's group is
      selected (see :func:`_select_policy_group`) and the other networks' dropped;
    * a dict already keyed by slot name (the names the policy's ``in_keys`` table
      uses), passed through as-is.

    mjlab groups convert transparently, mjswan ones pass through, and a group named for
    a training-only network (:data:`_TRAINING_ONLY_OBS_GROUPS`) is dropped: silently
    beside an ``actor`` group, since the pair is mjlab's own dict, with a warning
    otherwise.

    *obs_groups* is the runner's ``rl_cfg.obs_groups``, consulted only for the dict form.
    """
    if observations is None:
        return None
    if _is_obs_group(observations):
        observations = {DEFAULT_OBS_GROUP_KEY: observations}
    else:
        observations = _select_policy_group(observations, obs_groups)

    # `Any`-valued while filling: the last branch passes a duck-typed group through.
    adapted: dict[str, Any] = {}
    for key, group in observations.items():
        if key in _TRAINING_ONLY_OBS_GROUPS:
            if DEFAULT_OBS_GROUP_KEY not in observations:
                warnings.warn(
                    f"Dropping observation group {key!r}: mjlab exports only the actor "
                    "network, so no ONNX input consumes it. Pass just the policy's own "
                    'group: `observations=env_cfg.observations["actor"]`.',
                    category=RuntimeWarning,
                    stacklevel=3,
                )
            continue
        if isinstance(group, MjswanObservationGroupCfg):
            adapted[key] = group
        elif is_from_mjlab(group):
            adapted[key] = _adapt_obs_group(group)
        else:
            adapted[key] = group
    return adapted


__all__ = ["DEFAULT_OBS_GROUP_KEY", "adapt_observations"]
