"""mjlab observation groups as mjswan's.

mjlab's own term function is kept and traced at build time (ADR 0005); an author's
``register_observation`` override replaces it by function or term name.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Collection, Mapping
from typing import Any

from ..document.manifest import DEFAULT_IN_KEYS
from ..envs.mdp.observations import ObservationBinding, _custom_registry
from ..envs.mdp.registry import resolve
from ..managers.observation_manager import (
    ObservationGroupCfg as MjswanObservationGroupCfg,
)
from ..managers.observation_manager import (
    ObservationTermCfg as MjswanObservationTermCfg,
)
from .detect import differs, field_default, is_from_mjlab


def _sanitize_obs_params(params: dict[str, Any]) -> dict[str, Any]:
    """Swap the non-JSON ``asset_cfg`` for the entity, joint and site names it holds."""
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


#: Fields each adapter carries over to mjswan's config.
_TERM_FIELDS = frozenset({"func", "params", "scale", "clip", "history_length"})
_GROUP_FIELDS = frozenset(
    {"terms", "concatenate_terms", "enable_corruption", "history_length"}
)

#: mjlab fields that only affect training, so the browser's observation ignores them.
_TRAINING_ONLY_FIELDS = frozenset(
    {
        "noise",
        "delay_min_lag",
        "delay_max_lag",
        "delay_per_env",
        "delay_hold_prob",
        "delay_update_period",
        "delay_per_env_phase",
        "nan_policy",
        "nan_check_per_term",
    }
)

#: Layout fields mjswan builds only at these values: history flattened term-major, and
#: terms concatenated on the feature axis (``0`` or ``-1`` for a flat observation).
_LAYOUT_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "flatten_history_dim": (True,),
    "concatenate_dim": (-1, 0),
}


def _refuse_dropped_fields(
    cfg: Any, carried: frozenset[str], where: str, *, has_history: bool
) -> None:
    """Raise if *cfg* sets a field the adapter drops to a non-default value.

    A dropped layout field (a subclass's ``history_ordering="time"``, say) would build
    clean and feed the policy a reordered observation of the same width. Duck-typed
    configs have no fields to walk and pass.
    """
    if not dataclasses.is_dataclass(cfg):
        return
    offending = []
    for f in dataclasses.fields(cfg):
        if f.name in carried or f.name in _TRAINING_ONLY_FIELDS:
            continue
        value = getattr(cfg, f.name, dataclasses.MISSING)
        if f.name in _LAYOUT_DEFAULTS:
            # The stack layout is moot without history.
            if f.name == "flatten_history_dim" and not has_history:
                continue
            if all(differs(value, ok) for ok in _LAYOUT_DEFAULTS[f.name]):
                offending.append(f"{f.name}={value!r}")
            continue
        default = field_default(f)
        if default is dataclasses.MISSING or differs(value, default):
            offending.append(f"{f.name}={value!r}")
    if offending:
        raise ValueError(
            f"Observation {where} sets {', '.join(offending)}, which mjswan does not "
            "carry: it stacks history term-major, flattened, and concatenated along the "
            "feature axis, so building would feed the policy an observation laid out "
            "differently from the one it was trained on. If the field does not change "
            "the observation, reset it to its default in the config passed to mjswan."
        )


def _adapt_obs_term(
    term: Any, term_name: str | None = None, *, group_history: int | None = None
) -> MjswanObservationTermCfg:
    """Convert a single mjlab ``ObservationTermCfg`` to mjswan.

    Params are sanitized only for an ``ObservationBinding``, whose params go verbatim
    into the browser JSON; a traced func needs the real ``SceneEntityCfg``.
    *group_history* is the group's history count, which overrides the term's count and
    ``flatten_history_dim`` when set, as in mjlab.
    """
    _refuse_dropped_fields(
        term,
        _TERM_FIELDS,
        f"term {term_name!r}" if term_name else "term",
        has_history=group_history is None and bool(getattr(term, "history_length", 0)),
    )
    raw_params = dict(getattr(term, "params", None) or {})
    func = resolve(term.func, term_name, _custom_registry, ObservationBinding)
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


def _adapt_obs_group(group: Any, name: str | None = None) -> MjswanObservationGroupCfg:
    """Convert a single mjlab ``ObservationGroupCfg`` to mjswan."""
    group_history = getattr(group, "history_length", None)
    _refuse_dropped_fields(
        group,
        _GROUP_FIELDS,
        f"group {name!r}" if name else "group",
        has_history=bool(group_history),
    )
    raw_terms = getattr(group, "terms", None) or {}
    terms = {
        term_name: _adapt_obs_term(
            cfg, term_name=term_name, group_history=group_history
        )
        for term_name, cfg in raw_terms.items()
        if cfg is not None
    }
    return MjswanObservationGroupCfg(
        terms=terms,
        concatenate_terms=getattr(group, "concatenate_terms", True),
        enable_corruption=getattr(group, "enable_corruption", False),
        history_length=getattr(group, "history_length", None),
    )


#: Groups of networks that never leave training: only the actor is exported, so no ONNX
#: input consumes them.
_TRAINING_ONLY_OBS_GROUPS = frozenset({"critic"})

#: The slot a single observation group lands under: the runtime's default ``in_keys``,
#: which is mjlab's name for the actor's group, so the common case needs no key
#: (ADR 0006 §5). The TypeScript copy is pinned to this one by ``default_slots.json``.
DEFAULT_OBS_GROUP_KEY = DEFAULT_IN_KEYS[0]

#: mjlab's name for the exported policy's network, a key of ``rl_cfg.obs_groups``. Same
#: word as the default slot by design, but that one is a key of ``in_keys``.
_MJLAB_ACTOR_NETWORK = "actor"


def _is_obs_group(value: Any) -> bool:
    """Whether *value* is a single observation group rather than a dict of them."""
    if isinstance(value, MjswanObservationGroupCfg):
        return True
    return not isinstance(value, Mapping) and hasattr(value, "terms")


def _select_policy_group(
    observations: Mapping[str, Any],
    obs_groups: Mapping[str, tuple[str, ...]] | None,
    image_groups: Collection[str] = (),
) -> Mapping[str, Any]:
    """Reduce mjlab's network-keyed group dict to what the exported actor reads.

    mjlab keys ``env_cfg.observations`` by *network* (``"actor"``, ``"critic"``), mjswan
    by *slot* (the policy's ``in_keys``). The default slot is also ``actor``, so mjlab's
    own dict only loses the other networks' groups; a runner naming the actor's group
    otherwise (``obs_groups == {"actor": ("proprio",), ...}``) has it moved there.

    A key the runner attributes to no network (``"command_"`` on a multi-input policy)
    is an author-added slot and stays, and a dict sharing no key with the actor's is not
    the task's and is left alone.

    An actor reading images as well (rsl-rl's CNN model) takes its one flat group as
    the input ``obs`` and each of *image_groups* as an input of the group's own name,
    so those keep their names beside the default slot.
    """
    if not observations:
        return observations
    actor_groups = tuple((obs_groups or {}).get(_MJLAB_ACTOR_NETWORK) or ())
    if not actor_groups or set(actor_groups).isdisjoint(observations):
        return observations
    flat_groups = [g for g in actor_groups if g not in image_groups]
    if len(flat_groups) != 1:
        # rsl-rl concatenates a network's flat groups into one input; mjswan feeds one
        # group per ONNX input, and taking the first would silently shorten it.
        raise ValueError(
            "The task's runner config feeds its actor network "
            f"{len(flat_groups)} concatenated observation groups "
            f"({', '.join(map(repr, flat_groups))}). mjswan feeds one group per "
            "ONNX input and cannot concatenate them, so pass the single group the "
            "exported policy actually takes: "
            "`observations=env_cfg.observations[<name>]`."
        )
    actor_key = flat_groups[0]
    actor_images = set(actor_groups) - {actor_key}
    network_groups = {g for groups in (obs_groups or {}).values() for g in groups}
    selected: dict[str, Any] = {}
    for key, group in observations.items():
        if key == actor_key:
            selected[DEFAULT_OBS_GROUP_KEY] = group
        elif key in actor_images or key not in network_groups:
            selected[key] = group
    return selected


def camera_groups(observations: Any, sensors: Any) -> frozenset[str]:
    """The groups of mjlab's ``env_cfg.observations`` with a term reading a camera.

    rsl-rl gives each image group an input of its own, by the tensor's rank; a camera
    is what makes a term's output an image, and naming one is all the config says.
    """
    if not isinstance(observations, Mapping):
        return frozenset()
    cameras = {
        sensor.name
        for sensor in sensors or ()
        if type(sensor).__name__ == "CameraSensorCfg"
    }
    return frozenset(
        key
        for key, group in observations.items()
        if any(
            (getattr(term, "params", None) or {}).get("sensor_name") in cameras
            for term in (getattr(group, "terms", None) or {}).values()
        )
    )


def adapt_observations(
    observations: Mapping[str, Any] | Any | None,
    *,
    obs_groups: Mapping[str, tuple[str, ...]] | None = None,
    image_groups: Collection[str] = (),
) -> dict[str, MjswanObservationGroupCfg] | None:
    """Adapt observation groups, converting mjlab types if detected.

    Accepts three shapes, so the caller need not know which slot the runtime feeds:

    * a **single** group (mjlab's ``env_cfg.observations["actor"]``), which lands
      under :data:`DEFAULT_OBS_GROUP_KEY`;
    * mjlab's whole ``env_cfg.observations`` dict, reduced by
      :func:`_select_policy_group`;
    * a dict already keyed by slot name (the policy's ``in_keys``), passed through.

    A group named for a training-only network (:data:`_TRAINING_ONLY_OBS_GROUPS`) is
    dropped: silently beside an ``actor`` group, since the pair is mjlab's own dict,
    with a warning otherwise.

    *obs_groups* is the runner's ``rl_cfg.obs_groups`` and *image_groups* what
    :func:`camera_groups` found, both read only for the dict form.
    """
    if observations is None:
        return None
    if _is_obs_group(observations):
        observations = {DEFAULT_OBS_GROUP_KEY: observations}
    else:
        observations = _select_policy_group(observations, obs_groups, image_groups)

    # `Any`-valued while filling: the last branch passes a duck-typed group through.
    adapted: dict[str, Any] = {}
    for key, group in observations.items():
        if group is None:
            continue
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
            adapted[key] = _adapt_obs_group(group, key)
        else:
            adapted[key] = group
    return adapted


__all__ = ["DEFAULT_OBS_GROUP_KEY", "adapt_observations", "camera_groups"]
