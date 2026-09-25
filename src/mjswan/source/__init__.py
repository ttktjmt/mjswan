"""Where remote assets come from: a reference in, local files out.

A source talks to one service and knows nothing of what the files mean. Each module
backs one ``add_<layer>_<source>()`` family: ``source.wandb`` for ``add_policy_wandb`` /
``add_motion_wandb``, ``source.hf`` for ``add_policy_hf`` / ``add_motion_hf`` /
``add_scene_hf`` / ``add_splat_hf``.
"""

from . import hf, wandb

__all__ = ["hf", "wandb"]
