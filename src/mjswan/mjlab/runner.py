"""An mjlab task's *runner*: the rl-config fields playback needs, and its ONNX exporter.

Building the runner means building the task's env, so one export context serves every
checkpoint of every run.
"""

from __future__ import annotations

import contextlib
import copy
import io
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import onnx


class MjlabRunnerDefaults(NamedTuple):
    """What an mjlab task's *runner* config contributes to browser playback.

    Everything else on it is training-only or already inside the exported ONNX.
    """

    obs_groups: Mapping[str, tuple[str, ...]] | None
    """``rl_cfg.obs_groups``: which observation group(s) each network reads, keyed by
    network (``{"actor": ("actor",), "critic": ("critic",)}`` by default)."""

    clip_actions: float | None
    """The symmetric bound rsl-rl clamps the policy's raw output to."""


_NO_RUNNER_DEFAULTS = MjlabRunnerDefaults(obs_groups=None, clip_actions=None)


def resolve_runner_defaults(task_id: str | None) -> MjlabRunnerDefaults:
    """Read an mjlab task's runner config for the two fields playback needs.

    All-``None`` when the task is unknown or mjlab is absent, leaving the caller on the
    ``"actor"`` group name with the action unclamped.
    """
    if task_id is None:
        return _NO_RUNNER_DEFAULTS
    try:
        from mjlab.tasks.registry import load_rl_cfg
    except ImportError:
        return _NO_RUNNER_DEFAULTS
    try:
        rl_cfg = load_rl_cfg(task_id)
    except (KeyError, AttributeError):
        return _NO_RUNNER_DEFAULTS

    obs_groups = getattr(rl_cfg, "obs_groups", None)
    clip = getattr(rl_cfg, "clip_actions", None)
    return MjlabRunnerDefaults(
        obs_groups=(
            {network: tuple(groups) for network, groups in obs_groups.items()}
            if isinstance(obs_groups, Mapping)
            else None
        ),
        clip_actions=float(clip) if clip is not None else None,
    )


@dataclass
class PtOnnxExportContext:
    """Reusable mjlab export state for repeated PT->ONNX conversion."""

    env: Any
    runner: Any
    joint_names: list[str]
    default_joint_pos: list[float]
    encoder_bias: list[float]

    def close(self) -> None:
        self.env.close()


def create_pt_onnx_export_context(
    task_id: str, *, env_cfg: Any | None = None
) -> PtOnnxExportContext:
    """Build the task's env and runner once, to export any number of checkpoints."""
    try:
        import mjlab.tasks  # noqa: F401 (populates the task registry)
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

        from .env import build_mjlab_env
    except ImportError as e:
        raise ImportError(
            "mjlab and torch are required for only_latest=False. "
            "Install them with: pip install mjlab torch"
        ) from e

    env_cfg = (
        copy.deepcopy(env_cfg)
        if env_cfg is not None
        else load_env_cfg(task_id, play=True)
    )
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task_id)

    env = build_mjlab_env(env_cfg)

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # rsl-rl's constructor always prints; buffered so a failure can still show it.
    chatter = io.StringIO()
    try:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        with contextlib.redirect_stdout(chatter):
            runner = runner_cls(wrapped_env, asdict(agent_cfg), device="cpu")
    except Exception:
        print(chatter.getvalue(), end="")
        wrapped_env.close()
        raise

    # Joint names, default positions, and encoder bias from the action manager.
    joint_names: list[str] = []
    default_joint_pos: list[float] = []
    encoder_bias: list[float] = []

    inner_env = wrapped_env.env if hasattr(wrapped_env, "env") else wrapped_env
    action_mgr = getattr(inner_env, "action_manager", None)
    if action_mgr is not None:
        for term_name in action_mgr.active_terms:
            term = action_mgr.get_term(term_name)
            if not hasattr(term, "target_names"):
                continue
            entity_name = getattr(getattr(term, "cfg", None), "entity_name", None)
            prefix = f"{entity_name}/" if entity_name else ""
            joint_names = [f"{prefix}{n}" for n in term.target_names]
            if hasattr(term, "offset") and term.offset is not None:
                offset = term.offset
                if hasattr(offset, "tolist"):
                    flat = offset.flatten().tolist()
                elif hasattr(offset, "__iter__"):
                    flat = list(offset)
                else:
                    flat = [float(offset)] * len(joint_names)
                default_joint_pos = flat[: len(joint_names)]
            if entity_name:
                entity = inner_env.scene[entity_name]
                bias = entity.data.encoder_bias
                if hasattr(bias, "detach"):
                    bias = bias.detach()
                if hasattr(bias, "cpu"):
                    bias = bias.cpu()
                if hasattr(term.target_ids, "detach"):
                    target_ids = term.target_ids.detach()
                else:
                    target_ids = term.target_ids
                if hasattr(target_ids, "cpu"):
                    target_ids = target_ids.cpu()
                if hasattr(target_ids, "tolist"):
                    target_indices = target_ids.tolist()
                else:
                    target_indices = list(target_ids)
                if hasattr(bias, "__getitem__"):
                    selected_bias = bias[0, target_indices]
                    if hasattr(selected_bias, "tolist"):
                        encoder_bias = selected_bias.tolist()
                    else:
                        encoder_bias = list(selected_bias)
            break

    return PtOnnxExportContext(
        env=wrapped_env,
        runner=runner,
        joint_names=joint_names,
        default_joint_pos=default_joint_pos,
        encoder_bias=encoder_bias,
    )


def align_obs_normalizer(runner: Any, checkpoint: dict) -> None:
    """Match the runner's policy normalizer to the checkpoint about to be loaded.

    The runner is built from the task's *current* rl config, but a checkpoint carries
    whichever normalizer its run trained with. A mismatch either fails the strict load
    (config normalizes, checkpoint does not) or silently drops the trained statistics,
    and an untrained ``EmpiricalNormalization`` is not the identity: it still divides
    by ``std + eps``.
    """
    import torch
    from rsl_rl.modules import EmpiricalNormalization

    keys = list(checkpoint.get("actor_state_dict", {})) + [
        key.removeprefix("actor_") for key in checkpoint.get("model_state_dict", {})
    ]
    normalized = any(key.startswith("obs_normalizer.") for key in keys)

    policy = runner.alg.get_policy()
    if normalized == policy.obs_normalization:
        return
    policy.obs_normalization = normalized
    device = next(policy.parameters()).device
    policy.obs_normalizer = (
        EmpiricalNormalization(policy.obs_dim).to(device)
        if normalized
        else torch.nn.Identity()
    )


def export_checkpoint(context: PtOnnxExportContext, pt_path: Path) -> onnx.ModelProto:
    """One ``model_*.pt`` through the runner's own exporter, as an in-memory model.

    The observation normalizer is aligned to the checkpoint first, so a run whose
    ``rl_cfg`` moved on since training still exports what was trained.
    """
    import onnx
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_filename = f"{pt_path.stem}.onnx"
        align_obs_normalizer(
            context.runner,
            torch.load(str(pt_path), map_location="cpu", weights_only=False),
        )
        context.runner.load(
            str(pt_path),
            load_cfg={"actor": True},
            strict=True,
            map_location="cpu",
        )
        context.runner.export_policy_to_onnx(tmp_dir, onnx_filename)
        return onnx.load(str(Path(tmp_dir) / onnx_filename))


__all__ = [
    "MjlabRunnerDefaults",
    "PtOnnxExportContext",
    "align_obs_normalizer",
    "create_pt_onnx_export_context",
    "export_checkpoint",
    "resolve_runner_defaults",
]
